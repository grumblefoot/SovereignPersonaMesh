"""Tests for the `spm` launcher (brief P3a). No real servers are started:
subprocess.Popen and the health check are patched."""

import json
import os
import signal
import subprocess
import sys

import pytest

from spm import run as spm_run


class FakeProc:
    """Stands in for subprocess.Popen: records the call, fixed pid, poll()=None.
    FakeProc.returncodes: optional list of exit codes assigned to spawned
    children in order (None = stays running) — the --foreground crash test."""

    instances = []
    returncodes: list = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 1000 + len(FakeProc.instances)
        i = len(FakeProc.instances)
        self.returncode = (FakeProc.returncodes[i]
                           if i < len(FakeProc.returncodes) else None)
        FakeProc.instances.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def _reset_procs():
    FakeProc.instances = []
    FakeProc.returncodes = []
    yield


class FakeRun:
    """Stands in for subprocess.run: records argv, canned result."""

    def __init__(self, returncode=0, error=None):
        self.calls = []
        self.returncode = returncode
        self.error = error

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr="boom")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """Temp repo root with a .env; module pointed at it (and at the temp dir)
    so nothing touches the real repo's logs/ or run/. The real environment's
    port variables are cleared so the .env values under test actually apply.

    Container isolation (second review): a developer's .env is in os.environ
    for the whole session (proxy.main runs load_dotenv at collection) and
    cmd_start/cmd_stop run a REAL `docker start/stop <SPM_DB_CONTAINER>` when
    that variable is set. Clear the container variables and replace
    spm.run.subprocess.run with a recording fake by default; tests of the
    container path set SPM_DB_CONTAINER and their own fake explicitly."""
    for var in ("SPM_ENGINE_PORT", "SPM_PORT", "SPM_DB_CONTAINER",
                "SPM_IN_CONTAINER", "SPM_LAN_ACCESS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spm_run.subprocess, "run", FakeRun())
    (tmp_path / ".env").write_text(
        "SPM_ENGINE_PORT=4105\nSPM_PORT=5105\nSPM_ONLY_FROM_DOTENV=yes\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def fake_docker(monkeypatch):
    """A container CLI 'installed' without a real binary on PATH: the container
    tests must not depend on docker being present on the machine running them."""
    monkeypatch.setattr(spm_run.shutil, "which",
                        lambda c: "/fake/bin/docker" if c == "docker" else None)


@pytest.fixture()
def patch_popen(monkeypatch):
    monkeypatch.setattr(spm_run.subprocess, "Popen", FakeProc)
    return FakeProc


def _health_bodies(*, up_engine=True, up_proxy=True, healthy_before_spawn=False,
                   engine_pid=None, proxy_pid=None):
    """health_json replacement answering per-service (port 4105 = engine). With
    healthy_before_spawn=False it answers down while no child has been spawned
    yet, so cmd_start's initial 'already running?' check does not short-circuit
    the spawn path under test. pids: what /health reports as the live process."""

    def _json(url):
        if not healthy_before_spawn and not FakeProc.instances:
            return None
        if ":4105" in url:
            if not up_engine:
                return None
            return {"status": "ok"} | ({"pid": engine_pid} if engine_pid else {})
        if not up_proxy:
            return None
        return {"status": "healthy"} | ({"pid": proxy_pid} if proxy_pid else {})

    return _json


# ── start ────────────────────────────────────────────────────────────────


def test_start_spawns_engine_before_proxy_and_writes_pids(repo, patch_popen, monkeypatch, capsys):
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    # Record, at each spawn, which children already exist: the proxy must be
    # spawned only after the engine (its /health answers once it has been spawned).
    existing_at_spawn: list[list[str]] = []
    real_spawn = spm_run._spawn

    def spawn(root, module, log_path, env):
        existing_at_spawn.append([p.cmd[-1] for p in FakeProc.instances])
        return real_spawn(root, module, log_path, env)
    monkeypatch.setattr(spm_run, "_spawn", spawn)

    rc = spm_run.cmd_start(repo, poll_s=0.01)
    assert rc == 0

    procs = patch_popen.instances
    # Spawn order: the engine first, the proxy second (the proxy needs the engine).
    assert [p.cmd[-1] for p in procs] == ["evennia_world.app", "proxy.main"]
    assert existing_at_spawn == [[], ["evennia_world.app"]]

    # both children got the .env values (engine does not load .env itself)
    for p in procs:
        assert p.kwargs["env"]["SPM_ONLY_FROM_DOTENV"] == "yes"
        assert p.kwargs["env"]["SPM_ENGINE_PORT"] == "4105"
        assert p.kwargs["cwd"] == str(repo)
        # both log streams go to the per-service log file
        assert p.kwargs["stderr"] is spm_run.subprocess.STDOUT

    pid_file = repo / "run" / "spm.pids"
    assert json.loads(pid_file.read_text(encoding="utf-8")) == {
        "engine": procs[0].pid, "proxy": procs[1].pid}

    out = capsys.readouterr().out
    assert "http://127.0.0.1:5105/admin" in out
    assert "http://127.0.0.1:5105/v1" in out


def test_start_logs_append_to_logs_dir(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    assert (repo / "logs").is_dir()
    for name in ("spm_engine.out", "spm_proxy.out"):
        assert (repo / "logs" / name).exists()


def test_start_when_both_healthy_spawns_nothing(repo, patch_popen, monkeypatch, capsys):
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(healthy_before_spawn=True))
    rc = spm_run.cmd_start(repo, poll_s=0.01)
    assert rc == 0
    assert patch_popen.instances == []
    assert "already running" in capsys.readouterr().out


def test_start_health_timeout_stops_and_exits_1(repo, patch_popen, monkeypatch, capsys):
    # engine never becomes healthy
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(up_engine=False))
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
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(up_proxy=False))
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

    # health answers with the matching pid until that service is killed
    state = {"engine": 111, "proxy": 222}

    def health(url):
        live = state["engine" if ":4105" in url else "proxy"]
        return {"pid": live} if live is not None else None
    monkeypatch.setattr(spm_run, "health_json", health)

    def kill(pid):
        killed.append(pid)
        for k, v in list(state.items()):
            if v == pid:
                state[k] = None
    monkeypatch.setattr(spm_run, "_kill", kill)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == [222, 111]  # proxy first, engine second
    assert not pid_file.exists()


def test_stop_dead_pid_health_down_no_kill(repo, monkeypatch, capsys):
    """Brief FX-D: a recorded PID with no health answer may belong to another
    program after a reboot — do NOT kill, say so, and clean the PID file."""
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 2147483646, "proxy": 2147483645}),
                        encoding="utf-8")
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    monkeypatch.setattr(spm_run, "health_json", lambda url: None)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == []  # nothing killed
    out = capsys.readouterr().out
    assert "proxy not running (stale PID file entry)" in out
    assert "engine not running (stale PID file entry)" in out
    assert not pid_file.exists()  # both entries stale -> file removed


def test_stop_health_matching_pid_kills_recorded(repo, monkeypatch):
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    monkeypatch.setattr(spm_run, "health_json",
                        lambda url: {"pid": 111 if ":4105" in url else 222})
    monkeypatch.setattr(spm_run, "_wait_unhealthy", lambda url, t, p: True)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == [222, 111]
    assert not pid_file.exists()


def test_stop_health_reports_foreign_pid_kills_that_one(repo, monkeypatch, capsys):
    """PID file says 111/222 but health reports 999/888: those are the real
    SPM processes (the PID file is stale); kill what health reports."""
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    monkeypatch.setattr(spm_run, "health_json",
                        lambda url: {"pid": 999 if ":4105" in url else 888})
    monkeypatch.setattr(spm_run, "_wait_unhealthy", lambda url, t, p: True)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == [888, 999]
    out = capsys.readouterr().out
    assert "real SPM process" in out
    assert not pid_file.exists()


def test_stop_keeps_pid_file_when_service_still_answers(repo, monkeypatch, capsys):
    """After the kill the engine still answers /health: keep the PID file so a
    retry has something to work with."""
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    # proxy goes down, engine stubbornly answers with its matching pid
    monkeypatch.setattr(spm_run, "health_json",
                        lambda url: {"pid": 111} if ":4105" in url else None)
    monkeypatch.setattr(spm_run, "_wait_unhealthy",
                        lambda url, t, p: ":4105" not in url)

    assert spm_run.cmd_stop(repo, poll_s=0.01) == 1
    assert killed == [111]  # proxy was stale (health down, not killed); engine was killed
    assert pid_file.exists()
    assert "still answers" in capsys.readouterr().err


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
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(up_engine=True, up_proxy=False))
    assert spm_run.cmd_status(repo) == 3
    out = capsys.readouterr().out
    assert "pid=-" in out  # no PID file -> "-"


# ── platform flags ───────────────────────────────────────────────────────


def test_posix_uses_start_new_session(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run.os, "name", "posix")
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    for p in patch_popen.instances:
        assert p.kwargs.get("start_new_session") is True
        assert "creationflags" not in p.kwargs


def test_windows_uses_detached_creation_flags(repo, patch_popen, monkeypatch):
    monkeypatch.setattr(spm_run.os, "name", "nt")
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    for p in patch_popen.instances:
        assert p.kwargs.get("creationflags") == (
            spm_run.CREATE_NEW_PROCESS_GROUP | spm_run.DETACHED_PROCESS)
        assert "start_new_session" not in p.kwargs


# ── optional database container (brief L1) ──────────────────────────────


def test_container_cli_prefers_docker_then_podman(monkeypatch):
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: c if c == "docker" else None)
    assert spm_run._container_cli() == "docker"
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: c if c == "podman" else None)
    assert spm_run._container_cli() == "podman"
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: None)
    assert spm_run._container_cli() is None


def test_db_container_empty_env_is_noop(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    assert spm_run.start_db_container({}) is False
    assert spm_run.stop_db_container({}) is False
    assert fake.calls == []


def test_start_runs_cli_start_before_spawning_engine(repo, patch_popen, fake_docker, monkeypatch):
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    monkeypatch.setattr(spm_run, "_wait_db_port", lambda env, t: True)
    monkeypatch.setenv("SPM_DB_CONTAINER", "spm-postgres")
    order: list[str] = []
    real_spawn = spm_run._spawn

    def spawn(root, module, log_path, env):
        order.append(f"spawn:{module}")
        return real_spawn(root, module, log_path, env)
    monkeypatch.setattr(spm_run, "_spawn", spawn)

    class OrderedRun(FakeRun):
        def __call__(self, cmd, **kwargs):
            order.append("cli:" + " ".join(cmd))
            return super().__call__(cmd, **kwargs)
    fake = OrderedRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    assert [c[0] for c in fake.calls] == [["docker", "start", "spm-postgres"]]
    assert fake.calls[0][1]["timeout"] == 60
    # the container was started before any service was spawned
    assert order == ["cli:docker start spm-postgres",
                     "spawn:evennia_world.app", "spawn:proxy.main"]


def test_start_without_db_container_never_calls_cli(repo, patch_popen, fake_docker, monkeypatch):
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    monkeypatch.delenv("SPM_DB_CONTAINER", raising=False)
    fake = FakeRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    assert fake.calls == []


def test_stop_runs_cli_stop_after_services(repo, fake_docker, monkeypatch, capsys):
    pid_file = repo / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": 111, "proxy": 222}), encoding="utf-8")
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))
    monkeypatch.setattr(spm_run, "health_json", lambda url: None)
    monkeypatch.setenv("SPM_DB_CONTAINER", "spm-postgres")
    fake = FakeRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    assert spm_run.cmd_stop(repo, poll_s=0.01) == 0
    assert killed == []  # health down -> stale entries, never killed
    assert [c[0] for c in fake.calls] == [["docker", "stop", "spm-postgres"]]
    assert not pid_file.exists()
    assert "SPM stopped." in capsys.readouterr().out


def test_inherited_db_container_from_dotenv_never_reaches_the_cli(tmp_path, monkeypatch, capsys):
    """Second review: with SPM_DB_CONTAINER in the process environment (a
    developer's .env, loaded at collection) cmd_stop ran a real `docker stop`.
    The suite-level fixtures clear it; prove the default `repo` isolation holds
    even when the variable is set AFTER the fixtures ran, by a leaking test."""
    monkeypatch.setenv("SPM_DB_CONTAINER", "spm-review-inherited-name")
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: "/fake/bin/docker")
    fake = FakeRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    pid_file = tmp_path / "run" / "spm.pids"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"engine": None, "proxy": None}), encoding="utf-8")
    monkeypatch.setattr(spm_run, "health_json", lambda url: None)
    assert spm_run.cmd_stop(tmp_path, poll_s=0.01) == 0
    # The call reached our fake, not a real CLI, with the inherited name.
    assert [c[0] for c in fake.calls] == [["docker", "stop", "spm-review-inherited-name"]]


def test_db_container_cli_failure_never_raises(monkeypatch, capsys):
    # CLI blows up mid-call (missing binary, timeout, permissions)
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: "docker")
    fake = FakeRun(error=OSError("docker: is the daemon running?"))
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    assert spm_run.start_db_container({"SPM_DB_CONTAINER": "spm-postgres"}) is False
    assert "daemon running" in capsys.readouterr().err
    # CLI answers non-zero: reported, still no raise
    fake2 = FakeRun(returncode=1)
    monkeypatch.setattr(spm_run.subprocess, "run", fake2)
    assert spm_run.stop_db_container({"SPM_DB_CONTAINER": "spm-postgres"}) is False
    assert "boom" in capsys.readouterr().err
    # no container CLI installed at all
    monkeypatch.setattr(spm_run.shutil, "which", lambda c: None)
    assert spm_run.start_db_container({"SPM_DB_CONTAINER": "spm-postgres"}) is False
    assert "neither docker nor podman" in capsys.readouterr().err


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
        @staticmethod
        def json():
            return {"status": "ok"}
    monkeypatch.setattr(run.httpx, "get", lambda url, timeout: seen.append(url) or R())
    assert run.health_ok("http://127.0.0.1:4005")
    assert run.health_ok("http://127.0.0.1:5050/")
    assert seen == ["http://127.0.0.1:4005/health", "http://127.0.0.1:5050/health"]


def test_container_rewrites_loopback_model_urls(tmp_path, monkeypatch):
    import spm.run as run
    (tmp_path / ".env").write_text("BACKEND_LLM_URL=http://127.0.0.1:13305/v1\n"
                                   "EMBEDDING_URL=http://localhost:13305/v1\n", encoding="utf-8")
    for k in ("BACKEND_LLM_URL", "EMBEDDING_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SPM_IN_CONTAINER", "1")
    env = run.child_env(tmp_path)
    assert env["BACKEND_LLM_URL"] == "http://host.docker.internal:13305/v1"
    assert env["EMBEDDING_URL"] == "http://host.docker.internal:13305/v1"
    monkeypatch.setenv("BACKEND_LLM_URL", "http://192.168.1.9:1234/v1")
    assert run.child_env(tmp_path)["BACKEND_LLM_URL"] == "http://192.168.1.9:1234/v1"
    monkeypatch.delenv("SPM_IN_CONTAINER")
    assert run.child_env(tmp_path)["EMBEDDING_URL"] == "http://localhost:13305/v1"


# ── brief FX-D: launcher robustness ──────────────────────────────────────


def test_start_engine_already_up_spawns_only_the_proxy(repo, patch_popen, monkeypatch, capsys):
    """Engine answers /health with pid 999 before start; proxy is down.
    cmd_start must spawn ONLY the proxy and record engine=999 + the new pid."""
    spawned = {"proxy_up": False}

    def health(url):
        if ":4105" in url:
            return {"status": "ok", "pid": 999}
        # proxy comes up only once it has been spawned and polled
        return {"status": "healthy", "pid": 555} if spawned["proxy_up"] else None
    monkeypatch.setattr(spm_run, "health_json", health)

    real_spawn = spm_run._spawn

    def fake_spawn(root, module, log_path, env):
        spawned["proxy_up"] = True
        return real_spawn(root, module, log_path, env)
    monkeypatch.setattr(spm_run, "_spawn", fake_spawn)

    assert spm_run.cmd_start(repo, poll_s=0.01) == 0
    procs = patch_popen.instances
    assert [p.cmd[-1] for p in procs] == ["proxy.main"]
    out = capsys.readouterr().out
    assert "engine is already running" in out and "not spawning another" in out
    pid_file = repo / "run" / "spm.pids"
    assert json.loads(pid_file.read_text(encoding="utf-8")) == {"engine": 999, "proxy": 555}


def test_start_failed_stops_container_it_started(repo, patch_popen, fake_docker, monkeypatch, capsys):
    """This cmd_start call started the DB container; the engine then never gets
    healthy -> the container must be stopped again."""
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(up_engine=False))
    monkeypatch.setattr(spm_run, "_wait_db_port", lambda env, t: True)
    monkeypatch.setenv("SPM_DB_CONTAINER", "spm-postgres")
    fake = FakeRun()
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    monkeypatch.setattr(spm_run, "_kill", lambda pid: None)

    assert spm_run.cmd_start(repo, health_timeout_s=0.05, poll_s=0.01) == 1
    assert [c[0] for c in fake.calls] == [["docker", "start", "spm-postgres"],
                                          ["docker", "stop", "spm-postgres"]]


def test_start_failed_keeps_container_it_did_not_start(repo, patch_popen, fake_docker, monkeypatch):
    """The container was already running (docker start answered non-zero, e.g.
    already up -> our helper returns False): a failed start must NOT stop it."""
    monkeypatch.setattr(spm_run, "health_json", _health_bodies(up_engine=False))
    monkeypatch.setattr(spm_run, "_wait_db_port", lambda env, t: True)
    monkeypatch.setenv("SPM_DB_CONTAINER", "spm-postgres")
    fake = FakeRun(returncode=1)
    monkeypatch.setattr(spm_run.subprocess, "run", fake)
    monkeypatch.setattr(spm_run, "_kill", lambda pid: None)

    assert spm_run.cmd_start(repo, health_timeout_s=0.05, poll_s=0.01) == 1
    assert [c[0] for c in fake.calls] == [["docker", "start", "spm-postgres"]]


def test_foreground_returns_crashed_child_return_code(repo, patch_popen, monkeypatch, capsys):
    """Brief FX-D: the engine dies with code 3 -> --foreground must exit 3,
    not 0, so docker/systemd sees the failure."""
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    FakeProc.returncodes = [3]  # engine (first child) exits 3; proxy keeps running
    killed = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))

    rc = spm_run.cmd_start(repo, foreground=True, poll_s=0.01)
    assert rc == 3
    assert "exited with code 3" in capsys.readouterr().err
    assert not (repo / "run" / "spm.pids").exists()


def test_foreground_clean_run_returns_zero(repo, patch_popen, monkeypatch):
    """Ctrl+C path: no crash recorded -> foreground returns 0, stops the proxy
    then the engine and removes the PID file. No real signal is sent (on
    Windows os.kill(SIGINT) is TerminateProcess on pytest itself): the
    launcher's own handler, captured from its signal.signal() call, is invoked
    from the fake child's poll() after a few loop iterations."""
    monkeypatch.setattr(spm_run, "health_json", _health_bodies())
    killed: list[int] = []
    monkeypatch.setattr(spm_run, "_kill", lambda pid: killed.append(pid))

    installed: dict[int, object] = {}
    restored: dict[int, object] = {}

    def fake_signal(signum, handler):
        if signum not in installed:
            installed[signum] = handler
        else:
            restored[signum] = handler
        return signal.SIG_DFL  # what the launcher must restore afterwards
    # cmd_start does `import signal as _signal` -> the same stdlib module object.
    monkeypatch.setattr(signal, "signal", fake_signal)

    polls: list[int] = []
    real_poll = FakeProc.poll

    def poll(self):
        polls.append(self.pid)
        if len(polls) >= 6:
            # a "Ctrl+C" arrives: run the launcher's handler, not a real signal
            installed[signal.SIGINT](signal.SIGINT, None)
        return real_poll(self)
    monkeypatch.setattr(FakeProc, "poll", poll)

    rc = spm_run.cmd_start(repo, foreground=True, poll_s=0.01)
    assert rc == 0
    assert signal.SIGINT in installed and signal.SIGTERM in installed
    # the handler was restored to what signal() returned, for both signals
    assert restored == {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
    engine, proxy = patch_popen.instances
    assert [p.cmd[-1] for p in (engine, proxy)] == ["evennia_world.app", "proxy.main"]
    assert killed == [proxy.pid, engine.pid]  # proxy first, then the engine
    assert not (repo / "run" / "spm.pids").exists()
    assert len(polls) >= 6  # the loop really ran until the handler fired


def test_child_env_forces_pythonutf8_and_no_reload(tmp_path, monkeypatch):
    """Brief FX-D: children log to files (Windows would use cp1252) and must
    never inherit uvicorn's reload supervisor — even when .env says otherwise
    for SPM_RELOAD... except .env must not be able to re-enable reload."""
    (tmp_path / ".env").write_text("SPM_RELOAD=1\nMY_VAR=keep\n", encoding="utf-8")
    for k in ("SPM_RELOAD", "PYTHONUTF8", "MY_VAR"):
        monkeypatch.delenv(k, raising=False)
    env = spm_run.child_env(tmp_path)
    assert env["PYTHONUTF8"] == "1"
    assert env["SPM_RELOAD"] == "0"  # .env's SPM_RELOAD=1 must NOT win
    assert env["MY_VAR"] == "keep"
    # an explicit shell PYTHONUTF8 is respected (setdefault)
    monkeypatch.setenv("PYTHONUTF8", "0")
    assert spm_run.child_env(tmp_path)["PYTHONUTF8"] == "0"
