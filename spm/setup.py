"""First-run setup for `python -m spm init` (brief P4).

Asks (or takes flags — Docker uses flags, `--yes` accepts the defaults) where the
model server lives, which chat model to use, and where embeddings come from;
probes the answers so a typo is visible immediately; then merges the keys it set
into the git-ignored config/config.local.json (holding only what the user
saved — the tracked config/config.json is never written) and pins the chat
model in .env as SPM_DEFAULT_MODEL. Interactive defaults come from the
effective settings: saved settings > .env (environment) > built-in defaults.

Probing NEVER fails setup: an offline server is a warning, not an error, so
init works before the LLM server has been started.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

DEFAULT_BACKEND_URL = "http://localhost:13305/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-GGUF-Q8_0"
PROBE_TIMEOUT_S = 5.0

ST_STEPS = """\
Next: connect SillyTavern (full guide: docs/st_extension_setup.md)
  1. API Connections -> Chat Completion -> source "Custom (OpenAI-compatible)"
  2. URL: http://127.0.0.1:5050/v1
  3. Copy st_extension/SPM-Chat-ID into SillyTavern's
     data/<user>/extensions/ folder (default user: data/default-user/extensions/)
  4. In Additional Headers add the X-SPM-* headers (see the doc above)."""


def load_json_file(path: Path) -> dict:
    """Parsed JSON object from `path`, or {} when missing/unreadable/not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update_env_line(env_path: Path, key: str, value: str) -> None:
    """Set `key=value` in `env_path`, appending or replacing that ONE line.

    Everything else in the file is kept verbatim; created when missing. utf-8
    explicitly (Path.write_text defaults to cp1252 on Windows)."""
    text = ""
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    prefix = f"{key}="
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = f"{prefix}{value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{prefix}{value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_ids(payload: Any) -> list[str]:
    """The `id` list from an OpenAI-style /models response ([] on any surprise)."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]


async def _get_json(http: httpx.AsyncClient, url: str) -> Any:
    try:
        resp = await http.get(url)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


async def run_setup(repo_root: Path, *,
                    backend_url: Optional[str] = None,
                    model: Optional[str] = None,
                    embedding_url: Optional[str] = None,
                    embedding_model: Optional[str] = None,
                    assume_yes: bool = False,
                    input_fn: Optional[Callable[[str], str]] = None) -> int:
    """Ask/probe/write as above; prints everything; returns the exit code (always 0
    from the setup itself — a failed probe only warns)."""
    config_dir = repo_root / "config"
    # Setup always writes the git-ignored local file (brief P4), even on a fresh
    # checkout where config.manager still reads config.json (no local file yet).
    local_path = config_dir / "config.local.json"
    env_path = repo_root / ".env"

    # Current effective settings become the interactive defaults (brief FX-C):
    # saved settings win, then the environment (os.environ plus the repo .env),
    # then the built-in defaults — the same rule config/manager.py applies, so
    # a `.env` backend URL is offered as the default and `--yes` keeps it.
    from spm.diagnostics import _Ctx, _effective_settings
    from spm.redact import Redactor

    env: dict = dict(os.environ)
    try:
        from dotenv import dotenv_values
        for k, v in dotenv_values(env_path).items():
            if v is not None and k not in env:
                env[k] = v
    except Exception:
        pass
    # SPM_CONFIG_PATH (environment or .env) names the file SPM reads, so the
    # answers go there too (third review, item 5).
    if env.get("SPM_CONFIG_PATH"):
        local_path = Path(env["SPM_CONFIG_PATH"]).expanduser()
    current: dict = _effective_settings(env, _Ctx(root=repo_root, redactor=Redactor()))[0]
    # SPM_DEFAULT_MODEL is not a settings key (no _ENV_VAR_MAP entry), so
    # _effective_settings does not carry it; the chat-model default needs it.
    current.setdefault("SPM_DEFAULT_MODEL", env.get("SPM_DEFAULT_MODEL", ""))

    # Ask for anything the flags did not supply.
    _input = input_fn or (lambda prompt: input(prompt))

    def ask(prompt: str, default: str) -> str:
        if assume_yes:
            return default
        try:
            answer = _input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        return answer or default

    url = backend_url or ask("\nModel server URL (OpenAI-compatible, ends in /v1)",
                             current.get("BACKEND_LLM_URL") or DEFAULT_BACKEND_URL)
    chat_model = model or ask("Chat model", current.get("SPM_DEFAULT_MODEL") or "")
    emb_url = embedding_url or ask("Embedding server URL (Enter = the model server)",
                                   current.get("EMBEDDING_URL") or url)
    emb_default = ("none" if str(current.get("EMBEDDING_PROVIDER", "")).lower() == "none"
                   else current.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL)
    emb_model = embedding_model if embedding_model is not None else \
        ask("Embedding model (or 'none')", emb_default)
    emb_model = (emb_model or "").strip()
    no_embeddings = emb_model.lower() == "none"

    # ── probes: warnings only, never fatal ──────────────────────────────────
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as http:
        models = await _get_json(http, url.rstrip("/") + "/models")
        if models is None:
            print(f"WARNING: could not read {url}/models — is the model server running? "
                  "Keeping your answer; setup continues.")
        else:
            ids = _model_ids(models)
            shown = ", ".join(ids[:20]) + (" …" if len(ids) > 20 else "")
            print(f"Models on {url}: {shown if ids else '(none listed)'}")
            if chat_model:
                from proxy.backend_client.lemonade_client import best_match
                if best_match(chat_model, ids) is None:
                    print(f"WARNING: chat model {chat_model!r} does not match any model "
                          "the server listed. It may need to be pulled or loaded first.")

        if not no_embeddings:
            payload = {"input": "hello", "model": emb_model}
            try:
                resp = await http.post(emb_url.rstrip("/") + "/embeddings", json=payload)
                ok = resp.status_code == 200
            except Exception:
                ok = False
            if not ok:
                print(f"WARNING: {emb_url}/embeddings failed for model {emb_model!r}. "
                      "If the server is Lemonade, pull it first: "
                      "lemonade pull Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")

        from proxy.backend_client.backend_profile import detect_window
        if chat_model:
            info = await detect_window(url, chat_model, http=http)
            if info.tokens:
                print(f"Context window for {chat_model}: {info.tokens} ({info.source})")
            else:
                print("Context window: unknown (the server will be probed again at runtime).")

    # ── write the local config (never the tracked config.json) ──────────────
    # Merge ONLY the keys this setup sets into whatever the local file already
    # holds (brief FX-C): the settings file holds only what the user saved, so
    # do not copy the tracked file's values into it wholesale.
    settings = load_json_file(local_path)
    settings["BACKEND_LLM_URL"] = url
    if no_embeddings:
        settings["EMBEDDING_PROVIDER"] = "none"
    else:
        settings["EMBEDDING_URL"] = emb_url
        settings["EMBEDDING_MODEL"] = emb_model
        if str(load_json_file(local_path).get("EMBEDDING_PROVIDER", "")).lower() == "none":
            settings["EMBEDDING_PROVIDER"] = "auto"  # the choice above must take effect
    local_path.parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote settings to {local_path}")
    if chat_model:
        update_env_line(env_path, "SPM_DEFAULT_MODEL", chat_model)
        print(f"Set SPM_DEFAULT_MODEL={chat_model} in {env_path}")

    print(ST_STEPS)
    return 0


def cmd_setup(repo_root: Path, **kwargs) -> int:
    """Sync entry point for `python -m spm init` (called after the database step).

    Setup must never crash a finished `init`: an unexpected error (probe helpers
    already swallow their own) prints a warning and returns success."""
    try:
        return asyncio.run(run_setup(repo_root, **kwargs))
    except Exception as err:
        print(f"WARNING: setup did not complete: {err}", file=sys.stderr)
        print("Re-run 'python -m spm init' (or edit config/config.local.json by hand).",
              file=sys.stderr)
        return 0
