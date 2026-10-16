"""start / stop / status logic for the `spm` command (brief P3a).

Starts the world engine (`evennia_world.app`) and the proxy (`proxy.main`) as
detached children with cwd = repo root, waits on their /health endpoints, and
tracks their PIDs in run/spm.pids. Works on Linux, macOS and Windows without
nc, podman, notify-send, xdg-open or a shell.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import re

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


def health_json(base_url: str) -> dict | None:
    """GET <base_url>/health; return the JSON body on 200, None when down.

    The body carries the service's `pid` (brief FX-D): the launcher must verify
    it is signalling the real SPM process, not a PID recycled after a reboot.
    """
    try:
        res = httpx.get(base_url.rstrip("/") + "/health", timeout=2.0)
        if res.status_code != 200:
            return None
        body = res.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return None


def health_ok(base_url: str) -> bool:
    """True if <base_url>/health answers 200 (short timeout; any error counts as down).
    Callers pass the service's base URL; the bare base answered 404 and start waited
    60 s for nothing (Claude's live check, 2026-10-14)."""
    return health_json(base_url) is not None


def child_env(repo_root: Path) -> dict:
    """os.environ plus the repo-root .env, WITHOUT overriding already-set
    variables. The engine does not load .env itself, so the launcher must pass
    the merged environment to both children."""
    env = dict(os.environ)
    for key, value in dotenv_values(repo_root / ".env").items():
        if value is not None and key not in env:
            env[key] = value
    # Children log to files: without UTF-8, Windows would use cp1252 and die on
    # non-ASCII output (brief FX-D).
    env.setdefault("PYTHONUTF8", "1")
    # uvicorn's reload supervisor forks children the launcher cannot track,
    # which breaks `spm stop` and the dashboard Quit — never let it inherit on.
    env["SPM_RELOAD"] = "0"
    if env.get("SPM_IN_CONTAINER") == "1":
        _container_hosts(env)
    return env


_LOOPBACK = re.compile(r"^(https?://)(localhost|127\.0\.0\.1)([:/]|$)")


def _container_hosts(env: dict) -> None:
    """Inside a container, localhost is the container itself: a model or embeddings
    server on the host is host.docker.internal (compose maps it to the host gateway).
    A host .env written for a bare-metal run says 127.0.0.1 (Claude's review of P5)."""
    for key in ("BACKEND_LLM_URL", "EMBEDDING_URL"):
        value = env.get(key) or ""
        new = _LOOPBACK.sub(r"\1host.docker.internal\3", value)
        if new != value:
            print(f"[spm] {key}: {value} -> {new} (container)")
            env[key] = new


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


def _wait_healthy_body(url: str, timeout_s: float, poll_s: float) -> dict | None:
    """Poll /health until it answers; return its JSON body, or None on timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        body = health_json(url)
        if body is not None:
            return body
        if time.monotonic() >= deadline:
            return None
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


# ── Optional database container (brief L1) ──────────────────────────────
# SPM_DB_CONTAINER names a local Docker/Podman container that `spm start`
# starts and `spm stop` stops. Empty (default) = do nothing: people who run
# PostgreSQL themselves, and the Docker bundle (compose manages its own DB),
# are untouched.

def _container_cli() -> str | None:
    """First container CLI on PATH: docker, then podman; None if neither."""
    for cli in ("docker", "podman"):
        if shutil.which(cli):
            return cli
    return None


def _db_container_action(action: str, env: dict) -> bool:
    """Run `<cli> <action> <name>` for SPM_DB_CONTAINER. Prints what happened;
    never raises. Returns True only when the container was actually handled."""
    name = (env.get("SPM_DB_CONTAINER") or "").strip()
    if not name:
        return False
    cli = _container_cli()
    if cli is None:
        print(f"[spm] SPM_DB_CONTAINER={name} but neither docker nor podman is "
              f"installed; not {action} the database container.", file=sys.stderr)
        return False
    try:
        res = subprocess.run([cli, action, name], capture_output=True, text=True,
                             timeout=60)
    except Exception as e:  # CLI missing mid-run, timeout, permissions: never raise
        print(f"[spm] {cli} {action} {name} failed: {e}", file=sys.stderr)
        return False
    if res.returncode == 0:
        print(f"[spm] {cli} {action} {name}: ok")
        return True
    err = (res.stderr or res.stdout or "").strip().splitlines()
    print(f"[spm] {cli} {action} {name} failed (exit {res.returncode}): "
          f"{err[-1] if err else 'no output'}", file=sys.stderr)
    return False


def start_db_container(env: dict) -> bool:
    """Start the configured database container (no-op unless SPM_DB_CONTAINER)."""
    return _db_container_action("start", env)


def stop_db_container(env: dict) -> bool:
    """Stop the configured database container (no-op unless SPM_DB_CONTAINER)."""
    return _db_container_action("stop", env)


def _wait_db_port(env: dict, timeout_s: float, poll_s: float = 0.5) -> bool:
    """Wait for a TCP connection to POSTGRES_HOST:POSTGRES_PORT to succeed."""
    host = env.get("POSTGRES_HOST") or "localhost"
    try:
        port = int(env.get("POSTGRES_PORT") or 5432)
    except ValueError:
        port = 5432
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(poll_s, 0.05))


def _kill(pid: int) -> None:
    # SIGTERM on POSIX; on Windows os.kill(SIGTERM) is TerminateProcess —
    # an immediate, NON-graceful kill (no shutdown handlers run).
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already gone — fine
    except OSError:
        pass  # e.g. permission problems: nothing else we can do here


def _write_pid_file(repo_root: Path, engine_pid: int | None,
                    proxy_pid: int | None) -> Path:
    """Record the per-service PIDs (None = service is up but /health did not
    report a pid; cmd_stop skips those entries — health is the source of truth)."""
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
    services = [
        ("engine", "evennia_world.app", repo_root / "logs" / "spm_engine.out", engine_url),
        ("proxy", "proxy.main", repo_root / "logs" / "spm_proxy.out", proxy_url),
    ]

    # Per service, not all-or-nothing (brief FX-D): if the engine is already up
    # and only the proxy is missing, `spm start` must spawn just the proxy.
    bodies = {name: health_json(url) for name, _, _, url in services}
    if all(b is not None for b in bodies.values()):
        print("SPM is already running (both services answer /health).")
        return 0

    # Optional managed database container (brief L1): start it first and give
    # Postgres up to 30 s to accept connections before spawning the engine.
    # If this call started it and the start then fails, stop it again.
    container_started_by_us = start_db_container(env)
    if container_started_by_us:
        print("Waiting for the database port...")
        if not _wait_db_port(env, 30.0):
            print("WARNING: database port not reachable after 30s; starting anyway.",
                  file=sys.stderr)

    def _fail_container() -> None:
        if container_started_by_us:
            print("[spm] start failed; stopping the database container we started.")
            stop_db_container(env)

    procs: dict[str, subprocess.Popen] = {}
    pids: dict[str, int] = {}
    for name, module, log_path, url in services:
        body = bodies[name]
        if body is not None:
            live_pid = _reported_pid(body)
            print(f"{name} is already running at {url} — not spawning another.")
            if live_pid is not None:
                pids[name] = live_pid
            continue
        print(f"Starting {name} ({module})...")
        procs[name] = _spawn(repo_root, module, log_path, env)
        body = _wait_healthy_body(url, health_timeout_s, poll_s)
        if body is None:
            print(f"ERROR: {name} did not become healthy at {url}/health "
                  f"within {health_timeout_s:.0f}s.", file=sys.stderr)
            # Stop what was started (proxy first if it got that far, then engine).
            for stop_name in reversed(list(procs)):
                _kill(procs[stop_name].pid)
            _fail_container()
            tail = _tail(log_path)
            if tail:
                print(f"--- last lines of {log_path} ---", file=sys.stderr)
                print(tail, file=sys.stderr)
            return 1
        # Prefer the PID the service itself reports; fall back to the child we
        # spawned when /health does not carry one (older service build).
        pids[name] = _reported_pid(body) or procs[name].pid

    pid_file = _write_pid_file(repo_root, pids.get("engine"), pids.get("proxy"))
    print(f"PIDs written to {pid_file} "
          f"(engine={pids.get('engine', '-')}, proxy={pids.get('proxy', '-')})")
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
    crashed: tuple[str, int] | None = None
    try:
        while not stopping:
            for pname, p in procs.items():
                rc = p.poll()
                if rc is not None:
                    crashed = (pname, rc)
                    break
            if crashed:
                break
            time.sleep(max(poll_s, 0.05))
    finally:
        for sig, old in saved.items():
            _signal.signal(sig, old)
    if crashed:
        pname, rc = crashed
        print(f"ERROR: {pname} exited with code {rc}.", file=sys.stderr)
    print("Stopping...")
    for name in ("proxy", "engine"):
        if name in procs:
            _kill(procs[name].pid)
            procs[name].wait()
    pid_file.unlink(missing_ok=True)
    # Exit with the crashed child's non-zero code — a supervisor (docker
    # restart, systemd) must see the failure, not 0 (brief FX-D).
    return crashed[1] if crashed and crashed[1] != 0 else 0


def _runs_from(pid: int, repo_root: Path) -> bool:
    """True when process `pid` runs with `repo_root` as its working directory
    (spm start spawns both services with cwd=repo_root). Unknown -> False: on
    systems without /proc (Windows, macOS) a mismatched PID is never killed."""
    try:
        cwd = os.readlink(f"/proc/{int(pid)}/cwd")
    except (OSError, ValueError):
        return False
    try:
        return Path(cwd).resolve() == Path(repo_root).resolve()
    except OSError:
        return False


def _reported_pid(body: dict) -> int | None:
    """The `pid` a /health body reports, or None (missing/garbage)."""
    try:
        return int(body.get("pid"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def cmd_stop(repo_root: Path, wait_s: float = STOP_WAIT_S,
             poll_s: float = POLL_INTERVAL_S) -> int:
    pid_file = repo_root / "run" / "spm.pids"
    pids = _read_pid_file(repo_root)
    if pids is None:
        print("not running (no PID file)")
        return 0

    env = child_env(repo_root)
    entries = 0
    still_up = False
    # Proxy first, engine second (mirror of startup order reversed).
    for name, url in (("proxy", _proxy_url(env)), ("engine", _engine_url(env))):
        recorded = pids.get(name)
        if recorded is None:
            continue
        entries += 1
        recorded = int(recorded)
        body = health_json(url)
        if body is None:
            # Health is down: the recorded PID may belong to another program
            # after a reboot — never kill blindly (brief FX-D).
            print(f"{name} not running (stale PID file entry)")
            continue
        live_pid = _reported_pid(body)
        pid_to_kill = recorded if live_pid in (None, recorded) else live_pid
        if live_pid not in (None, recorded):
            # Another SPM answers on this port. Stop it only when it runs from
            # THIS checkout: a stale PID file in a second checkout must never
            # stop the instance the user actually runs (third review, item 7).
            if not _runs_from(live_pid, repo_root):
                print(f"{name}: {url} is served by pid {live_pid}, which is not the "
                      f"SPM started from {repo_root} (PID file says {recorded}). Not "
                      f"stopping it; run `python -m spm stop` in that SPM's own folder.",
                      file=sys.stderr)
                still_up = True
                continue
            print(f"{name}: PID file says {recorded} but {url}/health reports "
                  f"{live_pid} (same folder); stopping {live_pid}.")
        _kill(pid_to_kill)
        if not _wait_unhealthy(url, wait_s, poll_s):
            print(f"WARNING: {name} (pid {pid_to_kill}) still answers {url}/health "
                  f"after {wait_s:.0f}s.", file=sys.stderr)
            still_up = True

    if still_up:
        # Keep the PID file: something is still serving, the user can retry.
        return 1
    if entries:
        pid_file.unlink(missing_ok=True)
    # All recorded services are down: now it is safe to stop the managed DB
    # container (no-op unless SPM_DB_CONTAINER is set). The --foreground exit
    # path does NOT stop it — inside Docker, compose manages its own database.
    stop_db_container(env)
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
