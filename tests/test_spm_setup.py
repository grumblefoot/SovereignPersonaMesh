"""Tests for brief P4: config path choice and `spm init`'s first-run setup.

Everything runs against tmp_path repos and a fake model server (httpx
MockTransport): no live service is probed and the real repo's
config/config.local.json is never created or modified.
"""

import json
import re

import httpx
import pytest

import spm.setup as setup
from config.manager import default_config_path


# ── config path choice (config/manager.py) ──────────────────────────────────

def test_default_config_path_prefers_local_when_present(tmp_path):
    tracked = tmp_path / "config.json"
    local = tmp_path / "config.local.json"
    # No local file: the tracked file (unchanged behaviour).
    assert default_config_path(str(tmp_path)) == str(tracked)
    local.write_text("{}", encoding="utf-8")
    assert default_config_path(str(tmp_path)) == str(local)


def test_settings_manager_writes_the_chosen_local_file(tmp_path, monkeypatch):
    """Writes go to the same file the path choice picked."""
    import config.manager as cm
    local = tmp_path / "config.local.json"
    local.write_text('{"BACKEND_LLM_URL": "http://local/v1"}', encoding="utf-8")
    monkeypatch.setattr(cm, "_CONFIG_PATH", default_config_path(str(tmp_path)))
    mgr = cm.SettingsManager()
    assert mgr.config_path == str(local)
    mgr.write_settings({"SPM_HARDWARE_TIER": "TITAN"})
    assert json.loads(local.read_text(encoding="utf-8"))["SPM_HARDWARE_TIER"] == "TITAN"


# ── update_env_line ─────────────────────────────────────────────────────────

def test_update_env_line_creates_replaces_and_keeps(tmp_path):
    env = tmp_path / ".env"
    setup.update_env_line(env, "SPM_DEFAULT_MODEL", "A")           # create
    assert env.read_text(encoding="utf-8") == "SPM_DEFAULT_MODEL=A\n"
    env.write_text("POSTGRES_DB=x\nSPM_DEFAULT_MODEL=A\nOTHER=1\n", encoding="utf-8")
    setup.update_env_line(env, "SPM_DEFAULT_MODEL", "B")            # replace one line
    assert env.read_text(encoding="utf-8") == "POSTGRES_DB=x\nSPM_DEFAULT_MODEL=B\nOTHER=1\n"


# ── setup with a fake backend ───────────────────────────────────────────────

MODELS = {"data": [{"id": "Gemma-4-26B-A4B-it-GGUF"}, {"id": "Qwen3-8B"}]}


def install_fake_server(monkeypatch, *, models=MODELS, embeddings_ok=True,
                        reachable=True):
    """Route every httpx request run_setup makes through a MockTransport."""
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if not reachable:
            raise httpx.ConnectError("down", request=request)
        path = request.url.path
        if path.endswith("/models") and request.method == "GET":
            return httpx.Response(200, json=models)
        if path.endswith("/embeddings") and request.method == "POST":
            if not embeddings_ok:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"data": [{"embedding": [0.0, 1.0]}]})
        return httpx.Response(404, json={})

    real = httpx.AsyncClient

    def factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)

    monkeypatch.setattr(setup.httpx, "AsyncClient", factory)
    return requests_seen


@pytest.fixture()
def repo(tmp_path):
    """Temp repo root with a tracked config/config.json and no local file/.env."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.json").write_text(
        json.dumps({"SPM_HARDWARE_TIER": "SOVEREIGN",
                    "BACKEND_LLM_URL": "http://localhost:8000/v1"}),
        encoding="utf-8")
    return tmp_path


async def test_setup_with_flags_writes_local_config_and_env(repo, monkeypatch, capsys):
    seen = install_fake_server(monkeypatch)
    rc = await setup.run_setup(
        repo, backend_url="http://fake:13305/v1", model="Gemma-4-26B-A4B-it-GGUF",
        embedding_url="http://fake:13305/v1",
        embedding_model="Qwen3-Embedding-0.6B-GGUF-Q8_0", assume_yes=True)
    assert rc == 0

    tracked = repo / "config" / "config.json"
    local = repo / "config" / "config.local.json"
    # The tracked file is untouched; the local file holds ONLY the keys setup
    # set (brief FX-C) — the tracked file is no longer copied into it.
    assert json.loads(tracked.read_text(encoding="utf-8"))["BACKEND_LLM_URL"] == \
        "http://localhost:8000/v1"
    written = json.loads(local.read_text(encoding="utf-8"))
    assert "SPM_HARDWARE_TIER" not in written            # not saved: follows .env/defaults
    assert written["BACKEND_LLM_URL"] == "http://fake:13305/v1"
    assert written["EMBEDDING_URL"] == "http://fake:13305/v1"
    assert written["EMBEDDING_MODEL"] == "Qwen3-Embedding-0.6B-GGUF-Q8_0"

    env_line = (repo / ".env").read_text(encoding="utf-8")
    assert env_line == "SPM_DEFAULT_MODEL=Gemma-4-26B-A4B-it-GGUF\n"

    out = capsys.readouterr().out
    assert "Gemma-4-26B-A4B-it-GGUF" in out                     # models were listed
    assert "Custom (OpenAI-compatible)" in out                  # SillyTavern steps printed
    assert "http://127.0.0.1:5050/v1" in out
    assert "docs/st_extension_setup.md" in out
    # The embeddings probe actually ran with a one-word input.
    posts = [r for r in seen if r.method == "POST"]
    assert posts and json.loads(posts[0].content)["input"] == "hello"


async def test_setup_merges_existing_local_file(repo, monkeypatch):
    install_fake_server(monkeypatch)
    local = repo / "config" / "config.local.json"
    local.write_text('{"gm_actions_mode": "off"}', encoding="utf-8")
    rc = await setup.run_setup(repo, backend_url="http://fake/v1", model="Gemma-4-26B-A4B-it-GGUF",
                               embedding_model="none", assume_yes=True)
    assert rc == 0
    written = json.loads(local.read_text(encoding="utf-8"))
    assert written["gm_actions_mode"] == "off"                  # kept
    assert written["EMBEDDING_PROVIDER"] == "none"              # new choice
    assert "EMBEDDING_MODEL" not in written or written["EMBEDDING_PROVIDER"] == "none"


async def test_setup_yes_writes_env_backend_url(repo, monkeypatch):
    """Brief FX-C test: BACKEND_LLM_URL in the repo's .env and no local file —
    `--yes` writes that URL to config/config.local.json."""
    install_fake_server(monkeypatch)
    (repo / "config" / "config.json").unlink()   # fresh checkout: nothing saved yet
    (repo / ".env").write_text("BACKEND_LLM_URL=http://from-dotenv:1234/v1\n",
                               encoding="utf-8")
    monkeypatch.delenv("BACKEND_LLM_URL", raising=False)   # must come from the .env file
    rc = await setup.run_setup(repo, embedding_model="none", assume_yes=True)
    assert rc == 0
    written = json.loads((repo / "config" / "config.local.json").read_text(encoding="utf-8"))
    assert written["BACKEND_LLM_URL"] == "http://from-dotenv:1234/v1"


async def test_setup_offer_default_follows_env_file(repo, monkeypatch):
    """Brief FX-C: interactive defaults are the effective settings — with no
    local file and a BACKEND_LLM_URL in the repo's .env, the prompt's default
    is the .env value (not the built-in default), and answering Enter keeps it."""
    install_fake_server(monkeypatch)
    (repo / "config" / "config.json").unlink()   # fresh checkout: nothing saved yet
    (repo / ".env").write_text("BACKEND_LLM_URL=http://from-dotenv:1234/v1\n",
                               encoding="utf-8")
    monkeypatch.delenv("BACKEND_LLM_URL", raising=False)   # must come from the .env file
    offered = []

    def fake_input(prompt):
        m = re.search(r"\[(.*)\]:\s*$", prompt)
        if m:
            offered.append(m.group(1))
        return ""                                          # Enter everywhere

    rc = await setup.run_setup(repo, embedding_model="none", input_fn=fake_input)
    assert rc == 0
    assert "http://from-dotenv:1234/v1" in offered
    written = json.loads((repo / "config" / "config.local.json").read_text(encoding="utf-8"))
    assert written["BACKEND_LLM_URL"] == "http://from-dotenv:1234/v1"


async def test_setup_warns_on_unknown_chat_model(repo, monkeypatch, capsys):
    install_fake_server(monkeypatch)
    rc = await setup.run_setup(repo, backend_url="http://fake/v1", model="totally-made-up",
                               embedding_model="none", assume_yes=True)
    assert rc == 0
    assert "does not match any model" in capsys.readouterr().out


async def test_setup_warns_on_failing_embeddings_probe(repo, monkeypatch, capsys):
    install_fake_server(monkeypatch, embeddings_ok=False)
    rc = await setup.run_setup(repo, backend_url="http://fake/v1", model="Qwen3-8B",
                               embedding_model="nope-embedder", assume_yes=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "/embeddings failed" in out
    assert "pull" in out                                        # hint to pull the model


async def test_setup_survives_an_unreachable_backend(repo, monkeypatch, capsys):
    install_fake_server(monkeypatch, reachable=False)
    rc = await setup.run_setup(repo, backend_url="http://down/v1", model="Qwen3-8B",
                               embedding_model="Qwen3-Embedding-0.6B-GGUF-Q8_0",
                               assume_yes=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "could not read" in out
    # Answers are kept even though nothing could be probed.
    written = json.loads((repo / "config" / "config.local.json").read_text(encoding="utf-8"))
    assert written["BACKEND_LLM_URL"] == "http://down/v1"


async def test_setup_asks_when_no_flags(repo, monkeypatch, capsys):
    install_fake_server(monkeypatch)
    answers = ["http://asked/v1", "Qwen3-8B", "", "none"]
    rc = await setup.run_setup(repo, input_fn=lambda prompt: answers.pop(0))
    assert rc == 0
    written = json.loads((repo / "config" / "config.local.json").read_text(encoding="utf-8"))
    assert written["BACKEND_LLM_URL"] == "http://asked/v1"
    # Empty answer for the embedding URL defaults to the model server.
    assert written["EMBEDDING_PROVIDER"] == "none"
    assert (repo / ".env").read_text(encoding="utf-8").strip() == "SPM_DEFAULT_MODEL=Qwen3-8B"


async def test_cmd_setup_never_raises(repo, monkeypatch, capsys):
    """Even a totally broken inner setup only warns (init already succeeded)."""
    def explode(*a, **k):  # asyncio.run needs a coroutine: break run() instead
        raise RuntimeError("internal boom")
    monkeypatch.setattr(setup.asyncio, "run", explode)
    assert setup.cmd_setup(repo) == 0
    assert "did not complete" in capsys.readouterr().err


# ── argparse wiring (spm/__main__.py) ───────────────────────────────────────

def test_init_parser_accepts_setup_flags():
    from spm.__main__ import build_parser
    args = build_parser().parse_args([
        "init", "--backend-url", "http://h:1/v1", "--model", "M",
        "--embedding-url", "http://h:2/v1", "--embedding-model", "none",
        "--yes", "--no-setup"])
    assert args.backend_url == "http://h:1/v1" and args.model == "M"
    assert args.embedding_url == "http://h:2/v1" and args.embedding_model == "none"
    assert args.yes and args.no_setup


def test_main_runs_setup_after_database_and_skips_with_no_setup(monkeypatch):
    import spm.__main__ as m
    calls = []
    monkeypatch.setattr(m.db_init, "cmd_init", lambda root, existing=False: 0)
    monkeypatch.setattr(m.setup_mod, "cmd_setup",
                        lambda root, **kw: calls.append(kw) or 0)
    assert m.main(["init", "--yes", "--model", "M"]) == 0
    assert len(calls) == 1
    assert calls[0]["model"] == "M" and calls[0]["assume_yes"] is True

    calls.clear()
    assert m.main(["init", "--no-setup"]) == 0
    assert calls == []


def test_main_skips_setup_when_database_step_failed(monkeypatch):
    import spm.__main__ as m
    monkeypatch.setattr(m.db_init, "cmd_init", lambda root, existing=False: 1)
    monkeypatch.setattr(m.setup_mod, "cmd_setup",
                        lambda root, **kw: pytest.fail("setup must not run"))
    assert m.main(["init", "--yes"]) == 1
