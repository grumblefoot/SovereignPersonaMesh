"""Brief L1: a safe Quit button.

Proxy /admin/api/v1/shutdown: docker mode returns guidance and schedules
nothing; bare-metal mode asks the engine to quit itself, stops the managed
DB container, removes run/spm.pids, then SIGTERMs its own process — never
pkill. Engine /api/v1/world/admin/shutdown: loopback only. Nothing here may
touch the live services: every external call is patched.
"""

import asyncio
import inspect
import json as _json
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from proxy.main import app
import proxy.api.admin_routes as admin_routes
from proxy.api.routes import evennia_client


def _lb_request():
    """Brief FX-B: /shutdown now takes a Request and is loopback-only."""
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No inherited container/DB-container flags; the suite must decide these."""
    monkeypatch.delenv("SPM_IN_CONTAINER", raising=False)
    monkeypatch.delenv("SPM_DB_CONTAINER", raising=False)


# ── proxy endpoint ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shutdown_in_container_returns_guidance_and_schedules_nothing(monkeypatch):
    monkeypatch.setenv("SPM_IN_CONTAINER", "1")
    with patch("asyncio.create_task") as mock_task, patch("os.kill") as mock_kill:
        response = await admin_routes.shutdown_system(_lb_request())
    assert response.status_code == 200
    body = _json.loads(response.body)
    assert body["status"] == "docker"
    assert "scripts/docker/stop.sh" in body["message"]
    assert "scripts\\docker\\stop.cmd" in body["message"]
    assert "Your data is kept." in body["message"]
    mock_task.assert_not_called()
    mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_in_container_answers_docker_clients_before_the_loopback_rule(monkeypatch):
    """Second review, item 7: inside the bundle the browser's request arrives from
    the Docker bridge gateway (not loopback). The container check runs FIRST so the
    user gets the explanation, not a 403; still nothing is scheduled or killed."""
    monkeypatch.setenv("SPM_IN_CONTAINER", "1")
    bridge_client = SimpleNamespace(client=SimpleNamespace(host="172.18.0.1"))
    with patch("asyncio.create_task") as mock_task, patch("os.kill") as mock_kill:
        response = await admin_routes.shutdown_system(bridge_client)
    assert response.status_code == 200
    body = _json.loads(response.body)
    assert body["status"] == "docker"
    mock_task.assert_not_called()
    mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_bare_metal_rejects_non_loopback_clients(monkeypatch):
    """Outside Docker the loopback rule still applies: a LAN client gets 403 and
    nothing is scheduled."""
    from fastapi import HTTPException
    lan_client = SimpleNamespace(client=SimpleNamespace(host="192.168.1.20"))
    with patch("asyncio.create_task") as mock_task, patch("os.kill") as mock_kill:
        with pytest.raises(HTTPException) as exc:
            await admin_routes.shutdown_system(lan_client)
    assert exc.value.status_code == 403
    mock_task.assert_not_called()
    mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_bare_metal_engine_then_db_then_pids_then_sigterm(tmp_path):
    calls: list[str] = []
    # conftest redirects admin_routes._pid_file to tmp_path/run/spm.pids: create it
    # there so the test proves the background task removes it.
    pid_file = admin_routes._pid_file()
    assert pid_file == tmp_path / "run" / "spm.pids", "conftest must redirect the PID file"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(_json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    engine_client = MagicMock()
    engine_client.is_closed = False  # else EvenniaWorldClient rebuilds a real client

    async def fake_post(url, **kw):
        calls.append(f"engine:{url}")
        return MagicMock(raise_for_status=lambda: None)

    engine_client.post = fake_post
    evennia_client._client = engine_client

    def fake_stop_db(env):
        calls.append("stop_db")
        return True

    def fake_kill(pid, sig):
        calls.append(f"kill:{pid}:{int(sig)}")
        assert not pid_file.exists(), "PID file must be gone before the self-SIGTERM"

    with patch("spm.run.stop_db_container", fake_stop_db), \
            patch("os.kill", fake_kill), \
            patch("asyncio.create_task") as mock_task:
        response = await admin_routes.shutdown_system(_lb_request())
        assert response.status_code == 200
        body = _json.loads(response.body)
        assert body["status"] == "success"
        mock_task.assert_called_once()
        assert pid_file.exists()  # nothing removed synchronously: the task does it
        shutdown_coro = mock_task.call_args[0][0]
        await shutdown_coro  # drive the background task to completion

    expected_engine_url = f"{evennia_client.base_url}/world/admin/shutdown"
    assert calls == [f"engine:{expected_engine_url}", "stop_db",
                     f"kill:{__import__('os').getpid()}:{int(signal.SIGTERM)}"]
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_shutdown_proceeds_when_engine_call_fails():
    """A dead engine must not strand the proxy: errors are ignored, the proxy
    still ends itself."""
    calls: list[str] = []

    async def boom(url, **kw):
        raise RuntimeError("engine is down")

    dead_engine = MagicMock(post=boom)
    dead_engine.is_closed = False
    evennia_client._client = dead_engine
    with patch("spm.run.stop_db_container", lambda env: calls.append("stop_db")), \
            patch("os.kill", lambda pid, sig: calls.append("kill")), \
            patch("asyncio.create_task") as mock_task:
        response = await admin_routes.shutdown_system(_lb_request())
        assert response.status_code == 200
        await mock_task.call_args[0][0]
    assert calls == ["stop_db", "kill"]


def test_no_pkill_anywhere_in_proxy():
    """The old shutdown ran `pkill -f` patterns: no pkill may remain in proxy/."""
    from pathlib import Path
    root = Path(admin_routes.__file__).resolve().parents[2] / "proxy"
    offenders = [str(p) for p in root.rglob("*.py") if "pkill" in p.read_text(encoding="utf-8")]
    assert not offenders, f"pkill still in: {offenders}"


# ── engine endpoint ─────────────────────────────────────────────────────

@pytest.fixture()
def engine_client_factory():
    from evennia_world.app import app as world_app
    return world_app


def _world_client(world_app, host: str) -> TestClient:
    """TestClient with a spoofed peer address (client= is supported by the
    installed starlette; see the guard below)."""
    assert "client" in inspect.signature(TestClient.__init__).parameters, \
        "starlette TestClient lost the client= peer-address parameter"
    return TestClient(world_app, client=(host, 54321))


def test_engine_shutdown_rejects_non_loopback(engine_client_factory):
    from evennia_world.app import app as world_app
    async def _no_startup():
        pass
    with patch("os.kill") as mock_kill, \
            patch("evennia_world.app.startup_event", _no_startup), \
            patch("evennia_world.app.shutdown_event", _no_startup):
        client = _world_client(world_app, "203.0.113.7")
        response = client.post("/api/v1/world/admin/shutdown")
    assert response.status_code == 403
    mock_kill.assert_not_called()


def test_engine_shutdown_loopback_succeeds_and_terminates(engine_client_factory):
    from evennia_world.app import app as world_app
    async def _no_startup():
        pass
    with patch("os.kill") as mock_kill, \
            patch("evennia_world.app.startup_event", _no_startup), \
            patch("evennia_world.app.shutdown_event", _no_startup):
        client = _world_client(world_app, "127.0.0.1")
        response = client.post("/api/v1/world/admin/shutdown")
    assert response.status_code == 200
    assert response.json() == {"success": True}
    # Starlette runs background tasks before the TestClient call returns.
    import os as _os
    mock_kill.assert_called_once_with(_os.getpid(), signal.SIGTERM)
