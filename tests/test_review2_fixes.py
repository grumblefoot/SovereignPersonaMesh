"""Regression tests for the second pre-push review of v0.4 (2026-10-15).

Items covered here: 1 (the world engine refuses cross-site requests like the proxy),
4 (dashboard saves never land in the tracked config/config.json; a None value un-saves a
key; the tracked file is read underneath as repo defaults). Items 3 and 7 are tested in
test_report_api.py and test_shutdown.py.
"""
import json
import os

import httpx
import pytest


# ── item 1: engine origin guard ──────────────────────────────────────────

@pytest.fixture
def engine_client(monkeypatch):
    monkeypatch.setenv("SPM_WORLD_DB", "0")
    monkeypatch.delenv("SPM_ENGINE_HOST", raising=False)
    monkeypatch.delenv("SPM_IN_CONTAINER", raising=False)
    from evennia_world import app as world_app
    import time
    world_app.app.state.start_time = time.time()   # /health reads it (set by the lifespan)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=world_app.app),
                             base_url="http://127.0.0.1:4005")


@pytest.mark.asyncio
async def test_engine_refuses_cross_site_shutdown_and_mutations(engine_client):
    """A web page on another origin sends a simple POST (no preflight) with the user's
    loopback as peer: the engine must answer 403 before any handler runs. Only refused
    requests are sent here — an accepted shutdown would SIGTERM this process."""
    async with engine_client as c:
        evil = {"Origin": "http://evil.example"}
        r = await c.post("/api/v1/world/admin/shutdown", headers=evil)
        assert r.status_code == 403 and "cross-site" in r.text
        r = await c.post("/api/v1/world/configure", headers=evil,
                         json={"session_id": "s", "template_key": "dynamic"})
        assert r.status_code == 403
        r = await c.get("/api/v1/world/snapshot", params={"session_id": "s"}, headers=evil)
        assert r.status_code == 403
        # DNS rebinding: a foreign Host name is refused while the engine binds loopback.
        r = await c.get("/api/v1/world/snapshot", params={"session_id": "s"},
                        headers={"Host": "evil.example:4005"})
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_engine_allows_server_side_and_same_origin_callers(engine_client):
    async with engine_client as c:
        # The proxy, curl, the CLI: no Origin header.
        r = await c.get("/api/v1/world/snapshot", params={"session_id": "s_guard_ok"})
        assert r.status_code == 200
        # Same-origin / loopback alias.
        r = await c.get("/api/v1/world/snapshot", params={"session_id": "s_guard_ok"},
                        headers={"Origin": "http://localhost:4005"})
        assert r.status_code == 200
        # /health stays open (the launcher polls it).
        r = await c.get("/health", headers={"Origin": "http://evil.example"})
        assert r.status_code == 200


# ── item 4: settings file write path and layering ────────────────────────

def test_saves_go_to_local_file_never_tracked(tmp_path, monkeypatch):
    from config import manager
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    tracked = cfg_dir / "config.json"
    tracked.write_text(json.dumps({"periodic_review_cadence": 7}), encoding="utf-8")
    monkeypatch.setattr(manager, "_CONFIG_DIR", str(cfg_dir))

    # No local file yet: the default path is STILL the local file (it used to be the
    # tracked file, and a dashboard save dirtied the repo).
    assert manager.default_config_path() == str(cfg_dir / "config.local.json")

    mgr = manager.SettingsManager(manager.default_config_path())
    before = tracked.read_text(encoding="utf-8")
    merged = mgr.write_settings({"BACKEND_LLM_URL": "http://192.0.2.1:1234/v1"})
    assert tracked.read_text(encoding="utf-8") == before
    local = json.loads((cfg_dir / "config.local.json").read_text(encoding="utf-8"))
    assert local == {"BACKEND_LLM_URL": "http://192.0.2.1:1234/v1"}
    # The tracked file is read underneath as the repo's defaults.
    assert merged["periodic_review_cadence"] == 7
    assert merged["BACKEND_LLM_URL"] == "http://192.0.2.1:1234/v1"
    assert mgr.saved_keys() == ["BACKEND_LLM_URL"]


def test_none_unsaves_a_key_and_env_wins_again(tmp_path, monkeypatch):
    from config import manager
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr(manager, "_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("BACKEND_LLM_URL", "http://192.0.2.9:1234/v1")
    mgr = manager.SettingsManager(manager.default_config_path())

    mgr.write_settings({"BACKEND_LLM_URL": "http://192.0.2.1:1234/v1", "reasoning_mode": "none"})
    assert mgr.get_settings()["BACKEND_LLM_URL"] == "http://192.0.2.1:1234/v1"
    # A later change to the environment does not beat a saved value...
    monkeypatch.setenv("BACKEND_LLM_URL", "http://192.0.2.10:1234/v1")
    assert mgr.get_settings()["BACKEND_LLM_URL"] == "http://192.0.2.1:1234/v1"
    # ...until the key is un-saved with None.
    merged = mgr.write_settings({"BACKEND_LLM_URL": None})
    assert merged["BACKEND_LLM_URL"] == "http://192.0.2.10:1234/v1"
    assert mgr.saved_keys() == ["reasoning_mode"]
    on_disk = json.loads((cfg_dir / "config.local.json").read_text(encoding="utf-8"))
    assert "BACKEND_LLM_URL" not in on_disk
