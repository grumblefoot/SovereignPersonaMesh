"""start / stop / status logic for the `spm` command (brief P3a).

Starts the world engine (`evennia_world.app`) and the proxy (`proxy.main`) as
detached children with cwd = repo root, waits on their /health endpoints, and
tracks their PIDs in run/spm.pids. Works on Linux, macOS and Windows without
nc, podman, notify-send, xdg-open or a shell.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

# Windows process-creation flags. The numeric values are the documented Win32
# constants; subprocess only exposes the names on Windows, so define them here
# too to keep the code (and its tests) importable everywhere.
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

DEFAULT_ENGINE_PORT = 4005
DEFAULT_PROXY_PORT = 5050
HEALTH_TIMEOUT_S = 60.0
STOP_WAIT_S = 10.0
POLL_INTERVAL_S = 0.5


def default_repo_root() -> Path:
    """Repo root = the parent of the spm/ package directory."""
    return Path(__file__).resolve().parent.parent


def health_ok(base_url: str) -> bool:
    """True if <base_url>/health answers 200 (short timeout; any error counts as down).
    Callers pass the service's base URL; the bare base answered 404 and start waited
    60 s for nothing (Claude's live check, 2026-10-14)."""
    try:
        return httpx.get(base_url.rstrip("/") + "/health", timeout=2.0).status_code == 200
    except Exception:
        return False


def child_env(repo_root: Path) -> dict:
    """os.environ plus the repo-root .env, WITHOUT overriding already-set
    variables. The engine does not load .env itself, so the launcher must pass
    the merged environment to both children."""
    env = dict(os.environ)
    for key, value in dotenv_values(repo_root / ".env").items():
        if value is not None and key not in env:
            env[key] = value
    return env


def _port(env: dict, var: str, default: int) -> int:
    try:
        return int(env.get(var) or default)
    except ValueError:
        return default


def _engine_url(env: dict) -> str:
    return f"http://127.0.0.1:{_port(env, 'SPM_ENGINE_PORT', DEFAULT_ENGINE_PORT)}"


def _proxy_url(env: dict) -> str:
    return f"http://127.0.0.1:{_port(env, 'SPM_PORT', DEFAULT_PROXY_PORT)}"


def _spawn(repo_root: Path, module: str, log_path: Path, env: dict) -> subprocess.Popen:
    """Start `python -m <module>` detached, appending stdout+stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    if os.name == "nt":
        return subprocess.Popen(
            [sys.executable, "-m", module], cwd=str(repo_root),
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
            creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
    return subprocess.Popen(
        [sys.executable, "-m", module], cwd=str(repo_root),
        stdout=log_file, stderr=subprocess.STDOUT, env=env,
        start_new_session=True)


def _wait_healthy(url: str, timeout_s: float, poll_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if health_ok(url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(poll_s, 0.01))


def _wait_unhealthy(url: str, timeout_s: float, poll_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if not health_ok(url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(poll_s, 0.01))


def _tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _kill(pid: int) -> None:
    # SIGTERM on POSIX; on Windows os.kill(SIGTERM) is TerminateProcess —
    # an immediate, NON-graceful kill (no shutdown handlers run).
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already gone — fine
    except OSError:
        pass  # e.g. permission problems: nothing else we can do here


def _write_pid_file(repo_root: Path, engine_pid: int, proxy_pid: int) -> Path:
    run_dir = repo_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / "spm.pids"
    pid_file.write_text(json.dumps({"engine": engine_pid, "proxy": proxy_pid}),
                        encoding="utf-8")
    return pid_file


def _read_pid_file(repo_root: Path) -> dict | None:
    pid_file = repo_root / "run" / "spm.pids"
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cmd_start(repo_root: Path, foreground: bool = False,
              health_timeout_s: float = HEALTH_TIMEOUT_S,
              poll_s: float = POLL_INTERVAL_S) -> int:
    env = child_env(repo_root)
    engine_url, proxy_url = _engine_url(env), _proxy_url(env)

    if health_ok(engine_url) and health_ok(proxy_url):
        print("SPM is already running (both services answer /health).")
        return 0

    procs: dict[str, subprocess.Popen] = {}
    services = [
        ("engine", "evennia_world.app", repo_root / "logs" / "spm_engine.out", engine_url),
        ("proxy", "proxy.main", repo_root / "logs" / "spm_proxy.out", proxy_url),
    ]
    for name, module, log_path, url in services:
        print(f"Starting {name} ({module})...")
        procs[name] = _spawn(repo_root, module, log_path, env)
        if not _wait_healthy(url, health_timeout_s, poll_s):
            print(f"ERROR: {name} did not become healthy at {url}/health "
                  f"within {health_timeout_s:.0f}s.", file=sys.stderr)
            # Stop what was started (proxy first if it got that far, then engine).
            for stop_name in reversed(list(procs)):
                _kill(procs[stop_name].pid)
            tail = _tail(log_path)
            if tail:
                print(f"--- last lines of {log_path} ---", file=sys.stderr)
                print(tail, file=sys.stderr)
            return 1

    pid_file = _write_pid_file(repo_root, procs["engine"].pid, procs["proxy"].pid)
    print(f"PIDs written to {pid_file} (engine={procs['engine'].pid}, proxy={procs['proxy'].pid})")
    print(f"Admin dashboard:  {proxy_url}/admin")
    print(f"SillyTavern API:  {proxy_url}/v1")

    if not foreground:
        return 0

    # --foreground: stay attached (this is what a container runs). On Ctrl+C or
    # SIGTERM, stop the proxy first, then the engine.
    stopping: list[int] = []

    def _handler(signum, _frame):
        stopping.append(signum)

    import signal as _signal
    saved = {}
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            saved[sig] = _signal.signal(sig, _handler)
        except ValueError:
            pass  # not the main thread: Ctrl+C handling unavailable, poll only
    try:
        while not stopping:
            if any(p.poll() is not None for p in procs.values()):
                break
            time.sleep(max(poll_s, 0.05))
    finally:
        for sig, old in saved.items():
            _signal.signal(sig, old)
    print("Stopping...")
    for name in ("proxy", "engine"):
        _kill(procs[name].pid)
        procs[name].wait()
    pid_file.unlink(missing_ok=True)
    return 0


def cmd_stop(repo_root: Path, wait_s: float = STOP_WAIT_S,
             poll_s: float = POLL_INTERVAL_S) -> int:
    pid_file = repo_root / "run" / "spm.pids"
    pids = _read_pid_file(repo_root)
    if pids is None:
        print("not running (no PID file)")
        return 0

    env = child_env(repo_root)
    # Proxy first, engine second (mirror of startup order reversed).
    for name, url in (("proxy", _proxy_url(env)), ("engine", _engine_url(env))):
        pid = pids.get(name)
        if pid is None:
            continue
        _kill(int(pid))
        if not _wait_unhealthy(url, wait_s, poll_s):
            print(f"WARNING: {name} (pid {pid}) still answers {url}/health "
                  f"after {wait_s:.0f}s.", file=sys.stderr)
    pid_file.unlink(missing_ok=True)
    print("SPM stopped.")
    return 0


def cmd_status(repo_root: Path) -> int:
    env = child_env(repo_root)
    pids = _read_pid_file(repo_root) or {}
    rows = [
        ("engine", pids.get("engine"), _engine_url(env)),
        ("proxy", pids.get("proxy"), _proxy_url(env)),
    ]
    all_up = True
    for name, pid, url in rows:
        up = health_ok(url)
        all_up = all_up and up
        print(f"{name:<7} pid={pid if pid is not None else '-'}\t{'up' if up else 'down'}\t{url}/health")
    return 0 if all_up else 3
