#!/usr/bin/env python3
"""Generate .env.example from the code — the single config source of truth (brief P2).

Run `.venv/bin/python scripts/gen_env_example.py` to rewrite .env.example, or
`--check` to exit 1 with a diff when the checked-in file has drifted.

Two sources:
  1. config/manager.py `_ENV_VAR_MAP` + `_DEFAULT_VALUES`; the comment lines above a
     default in that file become the comment for the generated variable.
  2. DIRECT below — variables read with os.getenv/os.environ directly in production
     code (proxy/, evennia_world/, scripts/; NOT tests/ and NOT scripts/run_e2e_test.py).

Which variables were dropped from the DIRECT list and why (verified by grep,
2026-10-14, over proxy/, evennia_world/, core/, config/, scripts/):
  * SPM_PROXY_PORT   — setting only; nothing outside config/manager.py reads the env
                       var (the proxy binds SPM_PORT, listed in DIRECT). Kept in
                       manager.py as a setting; intentionally not in this file.
  * EVENNIA_LIAISON_URL — same: only config/manager.py reads it; the engine URL env
                       var the proxy actually uses is SPM_ENGINE_URL.
  * LLM_BACKEND_URL  — a legacy fallback alias in scripts/sleep_cycle.py only; the
                       canonical variable BACKEND_LLM_URL is generated (same value).
Test suite sets SPM_TEST_DB, RUN_LIVE_LLM_TESTS, RUN_LIVE_EMBED_TESTS, SPM_WORLD_RELOAD
and RUN_LIVE_LLM_TESTS — tests-only reads, so not in this file by design.
"""

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Variables read directly via os.getenv/os.environ in production code (verified by
# grep; see module docstring for drops). (name, default, one-line comment, section)
DIRECT = [
    ("SPM_HOST", "127.0.0.1",
     "Address the proxy binds to. Keep loopback unless you deliberately expose it.", "Server"),
    ("SPM_PORT", "5050",
     "Proxy port (SillyTavern connects here).", "Server"),
    ("SPM_LAN_ACCESS", "0",
     "Docker bundle only: 1 lets the dashboard answer requests from other devices "
     "(publish 0.0.0.0:5050:5050 too). Only on a trusted network: no login.", "Server"),
    ("SPM_ENGINE_HOST", "127.0.0.1",
     "Address the world engine binds to. Keep loopback.", "Server"),
    ("SPM_ENGINE_PORT", "4005",
     "World engine port.", "Server"),
    ("SPM_ENGINE_URL", "",
     "Override the engine API base URL (a second instance: tests, spikes, packaging).",
     "Development"),
    ("SPM_DEFAULT_MODEL", "Gemma-4-26B-A4B-it-GGUF",
     "Model served when a client asks for \"spm-sovereign-mesh\".", "Model backend"),
    ("POSTGRES_HOST", "localhost",
     "PostgreSQL + pgvector. docker compose -f config/docker-compose.yml up -d.", "Database"),
    ("POSTGRES_PORT", "5432", "Database port.", "Database"),
    ("POSTGRES_DB", "spm",
     "Database name (matches config/docker-compose.yml).", "Database"),
    ("POSTGRES_USER", "spm_user", "Database user.", "Database"),
    ("POSTGRES_PASSWORD", "spm_secure_password",
     "Database password (matches config/docker-compose.yml).", "Database"),
    ("SPM_WORLD_DB", "1",
     "World engine persistence: 0 runs the engine memory-only (no pool).", "Database"),
    ("SPM_DB_CONTAINER", "",
     "Name of a local Docker/Podman database container that `spm start` starts and "
     "`spm stop` stops, e.g. spm-postgres. Leave empty if you run PostgreSQL yourself.",
     "Database"),
    ("CONSOLIDATION_MODEL_NAME", "Gemma-4-E4B-it-GGUF",
     "Model used by the sleep-cycle memory consolidation.", "Behaviour"),
    ("SPM_CONFIG_PATH", "",
     "Settings file (what the dashboard and spm init saved; overrides .env). Default: config/config.local.json. The tracked config/config.json is read underneath as repo defaults and never written.", "Development"),
    ("SPM_RELOAD", "0",
     "1 = uvicorn autoreload for proxy development.", "Development"),
    ("SPM_LOG_HEADERS", "0",
     "1 = log incoming request headers (debugging).", "Development"),
]

SECTIONS = ["Server", "Model backend", "Embeddings", "Database", "Behaviour", "Development"]

# Section for each manager-derived env var (anything else falls to Behaviour).
MANAGER_SECTION = {
    "BACKEND_LLM_URL": "Model backend",
    "BACKEND_API_KEY": "Model backend",
    "BACKEND_MAX_TOKENS": "Model backend",
    "SPM_HARDWARE_TIER": "Model backend",
    "SPM_REASONING_MODE": "Model backend",
    "SPM_REASONING_EFFORT": "Model backend",
    "SPM_REASONING_MAX_TOKENS": "Model backend",
    "SPM_CONTEXT_WINDOW_OVERRIDE": "Model backend",
    "EMBEDDING_PROVIDER": "Embeddings",
    "EMBEDDING_URL": "Embeddings",
    "EMBEDDING_API_KEY": "Embeddings",
    "EMBEDDING_MODEL": "Embeddings",
    "EMBEDDING_DIM": "Embeddings",
    "EMBEDDING_TIMEOUT_S": "Embeddings",
    "EMBEDDING_ALLOW_REMOTE": "Embeddings",
    "EMBEDDING_MAX_COSINE_DISTANCE": "Embeddings",
    "SPM_GM_ACTIONS_MODE": "Behaviour",
    "SPM_GM_ACTIONS_CHANNEL": "Behaviour",
    "SPM_GM_ACTIONS_MAX_PER_TURN": "Behaviour",
    "SPM_GM_ACTIONS_MAX_ROOMS": "Behaviour",
    "SPM_PUBLIC_OUTPUT_DEFAULT": "Behaviour",
    "GATED_HISTORY_ENABLED": "Behaviour",
    "SPM_SLEEP_CYCLE_ENABLED": "Behaviour",
    "SPM_SLEEP_CYCLE_HOUR": "Behaviour",
}

# Shown when a manager default has no comment above it in config/manager.py.
# Keyed by the _DEFAULT_VALUES config key (not always the env var name).
COMMENT_FALLBACK = {
    "backend_max_tokens": ["Hard cap on completion tokens the backend may receive."],
    "gated_history_enabled": ["Gate chat history through the memory pipeline."],
    "BACKEND_LLM_URL": ["Any OpenAI-compatible server (Lemonade, LM Studio, llama-server, vLLM)."],
    "BACKEND_API_KEY": ["API key for BACKEND_LLM_URL (empty = no auth, typical for local)."],
    "SPM_HARDWARE_TIER": ["Hardware tier: SOVEREIGN, PERFORMANCE or EXPERIMENTAL."],
    "sleep_cycle_hour": ["Local hour (0-23) the sleep cycle runs at."],
    "sleep_cycle_enabled": ["Run the sleep-cycle memory consolidation inside the proxy scheduler (decision 15)."],
    "EMBEDDING_TIMEOUT_S": ["Per-request timeout (seconds) for the embedding backend."],
    "gm_actions_max_per_turn": ["Max GM actions extracted per turn."],
    "gm_actions_max_rooms_per_session": ["Max GM-created rooms tracked per session."],
}

HEADER = (
    "# GENERATED FILE — do not edit by hand.\n"
    "# Regenerate: .venv/bin/python scripts/gen_env_example.py\n"
    "#   (checks:      .venv/bin/python scripts/gen_env_example.py --check)\n"
    "#\n"
    "# Sovereign Persona Mesh (SPM) environment configuration.\n"
    "# Copy to .env and adjust. Defaults below match what the code uses when the\n"
    "# variable is unset.\n"
)

_ENTRY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)":')


def _manager_comments() -> dict:
    """env var -> comment lines: the comment above a _DEFAULT_VALUES entry
    in config/manager.py, skipping section-banner lines ("── ... ──")."""
    text = (ROOT / "config" / "manager.py").read_text(encoding="utf-8")
    block = text.split("_DEFAULT_VALUES", 1)[1].split("_INT_KEYS", 1)[0]
    comments = {}
    pending: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("# ").rstrip()
            if not body.startswith("──"):
                pending.append(body)
            continue
        m = _ENTRY_RE.match(stripped)
        if m:
            if pending:
                comments[m.group(1)] = list(pending)   # the whole comment, not a cut first line
            pending = []
        elif stripped:
            pending = []
    return comments


def generate() -> str:
    """The full .env.example text, computed from code defaults only."""
    from config.manager import _DEFAULT_VALUES, _ENV_VAR_MAP

    comments = _manager_comments()
    by_section: dict[str, list[tuple[str, str, list[str]]]] = {s: [] for s in SECTIONS}

    for key, env in _ENV_VAR_MAP.items():
        if env in ("SPM_PROXY_PORT", "EVENNIA_LIAISON_URL"):
            continue  # see module docstring; nothing outside manager.py reads them
        default = _DEFAULT_VALUES[key]
        value = str(default).lower() if isinstance(default, bool) else str(default)
        note = comments.get(key) or COMMENT_FALLBACK.get(key, [f"Config setting {key} (config/manager.py)."])
        if isinstance(note, str):
            note = [note]
        section = MANAGER_SECTION.get(env, "Behaviour")
        by_section[section].append((env, value, note))

    for name, value, comment, section in DIRECT:
        by_section[section].append((name, value, [comment]))

    out = [HEADER]
    for section in SECTIONS:
        out.append(f"\n# ── {section} ──\n")
        for name, value, note in by_section[section]:
            out.append(f"{name}={value}\n")
            for c in note:
                out.append(f"# {c}\n")
            out.append("\n")
    text = "".join(out)
    # One blank line after each comment block only.
    while "#\n\n" in text:
        text = text.replace("#\n\n", "#\n")
    return text.rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 with a diff if .env.example differs from generated")
    args = parser.parse_args()

    # Compute from CODE defaults only: a stale POSTGRES_* etc. in the ambient
    # environment (tests export them) must not leak into the generated file.
    saved = {}
    names = [v for _, v in _direct_and_map_names()]
    for name in names:
        if name in os.environ:
            saved[name] = os.environ.pop(name)
    try:
        generated = generate()
    finally:
        os.environ.update(saved)

    target = ROOT / ".env.example"
    current = target.read_text(encoding="utf-8") if target.exists() else ""

    if args.check:
        if current == generated:
            print(".env.example is up to date.")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(keepends=True), generated.splitlines(keepends=True),
            fromfile=".env.example (checked in)", tofile="generated")
        sys.stdout.writelines(diff)
        print("\n.env.example is stale — run: .venv/bin/python scripts/gen_env_example.py")
        return 1

    target.write_text(generated, encoding="utf-8")
    print(f"wrote {target}")
    return 0


def _direct_and_map_names():
    # Imported lazily so generate() stays import-safe for tests.
    from config.manager import _ENV_VAR_MAP
    return [(k, v) for k, v in _ENV_VAR_MAP.items()] + [(n, n) for n, *_ in DIRECT]


if __name__ == "__main__":
    sys.exit(main())
