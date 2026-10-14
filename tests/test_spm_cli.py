"""Tests for the `spm` launcher (brief P3a). No real servers are started:
subprocess.Popen and the health check are patched."""

import json
import os
import signal
import sys

import pytest

from spm import run as spm_run


class FakeProc:
    """Stands in for subprocess.Popen: records the call, fixed pid, poll()=None."""

    instances = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 1000 + len(FakeProc.instances)
        self.returncode = None
        FakeProc.instances.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def _reset_procs():
    FakeProc.instances = []
    yield


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """Temp repo root with a .env; module pointed at it (and at the temp dir)
    so nothing touches the real repo's logs/ or run/. The real environment's
    port variables are cleared so the .env values under test actually apply."""
    for var in ("SPM_ENGINE_PORT", "SPM_PORT"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        "SPM_ENGINE_PORT=4105\nSPM_PORT=5105\nSPM_ONLY_FROM_DOTENV=yes\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def patch_popen(monkeypatch):
    monkeypatch.setattr(spm_run.subprocess, "Popen", FakeProc)
    return FakeProc


def _health_urls(*, up_engine=True, up_proxy=True, healthy_before_spawn=False):
    """health_ok replacement answering per-service (port 4105 = engine). With
    healthy_before_spawn=False it answers down while no child has been spawned
    yet, so cmd_start's initial 'already running?' check does not short-circuit
    the spawn path under test."""

    def _ok(url):
        if not healthy_before_spawn and not FakeProc.instances:
            return False
        if ":4105" in url:
            return up_engine
        return up_proxy

    return _ok


# ── start ────────────────────────────────────────────────────────────────


def test_start_spawns_engine_before_proxy_and_writes_pids(repo, patch_popen, monkeypatch, capsys):
    monkeypatch.setattr(spm_run, "health_ok", _health_urls())
    rc = spm_run.cmd_start(repo, poll_s=0.01)
    assert rc == 0

    procs = patch_popen.instances
    assert [p.cmd[-1] for p in procs] == ["evennia_world.app", "proxy.main"]

    # engine spawned before proxy became healthy, proxy before its own spawn:
    # both children got the .env values (engine does not load .env itself)
    for p in procs:
        assert p.kwargs["env"]["SPM_ONLY_FROM_DOTENV"] == "yes"
        assert p.kwargs["env"]["SPM_ENGINE_PORT"] == "4105"
        assert p.kwargs["cwd"] == str(repo)
    # engine only waited on engine health before spawning the proxy
    assert procs[0].kwargs["stderr"] is spm_run.subprocess.STDOUT

    pid_file = repo / "run" / "spm.pids"
    assert json.loads(pid_file.read_text(encoding="utf-8")) == {
        "engine": procs[0].pid, "proxy": procs[1].pid}

    out = capsys.readouterr().out
    assert "http://127.0.0.1:5105/admin" in out
    assert "http://127.0.0.1:5105/v1" in out


def test_start_logs_append_to_logs_dir(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run, "health_ok", _health_urls())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    assert (repo / "logs").is_dir()
    for name in ("spm_engine.out", "spm_proxy.out"):
        assert (repo / "logs" / name).exists()


def test_start_when_both_healthy_spawns_nothing(repo, patch_popen, monkeypatch, capsys):
    monkeypatch.setattr(spm_run, "health_ok", _health_urls(healthy_before_spawn=True))
    rc = spm_run.cmd_start(repo, poll_s=0.01)
    assert rc == 0
    assert patch_popen.instances == []
    assert "already running" in capsys.readouterr().out


def test_start_health_timeout_stops_and_exits_1(repo, patch_popen, monkeypatch, capsys):
    # engine never becomes healthy
    monkeypatch.setattr(spm_run, "health_ok", _health_urls(up_engine=False))
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    # leave a log tail behind to prove it gets printed
    (repo / "logs").mkdir()
    (repo / "logs" / "spm_engine.out").write_text(
        "\n".join(f"line {i}" for i in range(30)), encoding="utf-8")

    rc = spm_run.cmd_start(repo, health_timeout_s=0.05, poll_s=0.01)
    assert rc == 1
    assert killed == [patch_popen.instances[0].pid]  # engine killed, proxy never spawned
    assert not (repo / "run" / "spm.pids").exists()
    err = capsys.readouterr().err
    assert "line 29" in err and "\nline 5\n" not in err  # last 20 lines only (10..29)
    assert "line 10\n" in err


def test_start_proxy_timeout_stops_engine_too(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run, "health_ok", _health_urls(up_proxy=False))
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    rc = spm_run.cmd_start(repo, health_timeout_s=0.05, poll_s=0.01)
    assert rc == 1
    # proxy (started second) is stopped first, then the engine
    assert killed == [patch_popen.instances[1].pid, patch_popen.instances[0].pid]


# ── stop ─────────────────────────────────────────────────────────────────


def test_stop_no_pid_file(repo, capsys):
    assert spm_run.cmd_stop(repo) == 0
    assert "not running (no PID file)" in capsys.readouterr().out


def test_stop_kills_proxy_then_engine_and_removes_file(repo, monkeypatch):
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")

    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    # health goes down immediately after the kill
    monkeypatch.setattr(spm_run, "health_ok", lambda url: False)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == [222, 111]  # proxy first, engine second
    assert not pid_file.exists()


def test_stop_dead_pid_is_fine(repo, monkeypatch):
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 2147483646, "proxy": 2147483645}), encoding="utf-8")
    monkeypatch.setattr(spm_run, "health_ok", lambda url: False)
    # real os.kill on a nonexistent pid must not raise (ProcessLookupError swallowed)
    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert not pid_file.exists()


# ── status ───────────────────────────────────────────────────────────────


def test_status_both_up_exit_0(repo, monkeypatch, capsys):
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    monkeypatch.setattr(spm_run, "health_ok", lambda url: True)
    assert spm_run.cmd_status(repo) == 0
    out = capsys.readouterr().out
    assert "engine" in out and "up" in out and "111" in out


def test_status_one_down_exit_3(repo, monkeypatch, capsys):
    monkeypatch.setattr(spm_run, "health_ok", _health_urls(up_engine=True, up_proxy=False))
    assert spm_run.cmd_status(repo) == 3
    out = capsys.readouterr().out
    assert "pid=-" in out  # no PID file -> "-"


# ── platform flags ───────────────────────────────────────────────────────


def test_posix_uses_start_new_session(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run.os, "name", "posix")
    monkeypatch.setattr(spm_run, "health_ok", _health_urls())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    for p in patch_popen.instances:
        assert p.kwargs.get("start_new_session") is True
        assert "creationflags" not in p.kwargs


def test_windows_uses_detached_creation_flags(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run.os, "name", "nt")
    monkeypatch.setattr(spm_run, "health_ok", _health_urls())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    for p in patch_popen.instances:
        assert p.kwargs.get("creationflags") == (
            spm_run.CREATE_NEW_PROCESS_GROUP | spm_run.DETACHED_PROCESS)
        assert "start_new_session" not in p.kwargs


# ── sanity: the module's own constants match the services ────────────────


def test_default_ports():
    assert spm_run.DEFAULT_ENGINE_PORT == 4005
    assert spm_run.DEFAULT_PROXY_PORT == 5050
    assert spm_run.HEALTH_TIMEOUT_S == 60.0


def test_health_ok_requests_the_health_endpoint(monkeypatch):
    """Live check 2026-10-14: start passed the bare service URL; the request went to '/'
    (404) and start timed out although the engine was up."""
    import spm.run as run
    seen = []

    class R:
        status_code = 200
    monkeypatch.setattr(run.httpx, "get", lambda url, timeout: seen.append(url) or R())
    assert run.health_ok("http://127.0.0.1:4005")
    assert run.health_ok("http://127.0.0.1:5050/")
    assert seen == ["http://127.0.0.1:4005/health", "http://127.0.0.1:5050/health"]
