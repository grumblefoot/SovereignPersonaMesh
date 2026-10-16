"""Read-only diagnostic checks for bug reports (brief R1b).

Every check is strictly read-only: nothing here writes files, starts or stops
services, loads a model, or sends a generation request. Each socket call has an
explicit timeout, and every check additionally runs under the hard per-check
cap of ``run_checks`` so one hanging dependency degrades one check and never
the whole report. Imports of asyncpg/httpx happen inside the checks, so a
missing package becomes a ``skipped`` check instead of a crash, and all error
messages pass through the ``Redactor`` before they leave this module.

Never point these checks at production by accident: the checks only read
configuration from the ``env`` dict the caller passes (merged os.environ +
repo .env, see spm/run.py child_env).

Design notes ("Checks", "Developer use", "Changes after
Hermes plan review") for the policy this encodes.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .redact import Redactor

__all__ = ["Check", "CHECKS", "EXTRA_CHECKS", "run_checks", "name_table"]

CHECKS = ("install", "services", "config", "model_server", "embeddings", "database",
          "recent_errors", "scheduler", "system")
EXTRA_CHECKS = ("recent_errors_full", "chat_overview", "backend_models_full")

# Per-request socket timeouts (brief R1b "The checks"); the per-check cap in
# run_checks is a second, outer safety net.
_HEALTH_TIMEOUT_S = 2.0
_SERVER_TIMEOUT_S = 5.0
_GIT_TIMEOUT_S = 3.0
_SYSINFO_TIMEOUT_S = 2.0
# recent_errors reads only the last window of each log file (a tail, so a huge
# log is never read whole, like report._tail_file).
_SCAN_WINDOW_BYTES = 512 * 1024

# Config keys removed entirely from the report (secrets never appear, not even
# masked — plan "Changes after Hermes plan review" item 4 / privacy rules).
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|password|passwd|secret|token|authorization)")

# Grouping keys for recent_errors: numbers and long hex ids normalise away so
# the same error recurring with different ids counts as one group.
_HEX_RUN_RE = re.compile(r"\b[0-9A-Fa-f]{8,}\b")
_DIGIT_RUN_RE = re.compile(r"\d+")

_SPM_TABLES_SQL = (
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = 'public' AND ("
    "table_name LIKE 'spm\\_%' OR table_name LIKE 'csa\\_memory\\_%' "
    "OR table_name = 'world_state_sessions') ORDER BY table_name"
)


@dataclass
class Check:
    name: str
    status: str            # ok | warn | fail | skipped
    summary: str           # one line
    details: dict          # JSON-serialisable, already redacted


@dataclass
class _Ctx:
    root: Path
    redactor: Redactor


# ── seams for tests (all network/process access goes through these) ─────────

async def _make_http(**kwargs):
    """Create an httpx.AsyncClient (import inside so a missing httpx is 'skipped')."""
    import httpx
    return httpx.AsyncClient(**kwargs)


async def _connect(cfg: dict):
    """Connect to PostgreSQL with the mandatory 5 s timeout."""
    import asyncpg
    return await asyncpg.connect(**cfg, timeout=5)


def _subprocess_run(args, timeout: float):
    """Run a read-only helper command (git / nvidia-smi / wmic ...) capturing output.

    Blocking: checks call it through asyncio.to_thread so the per-check cap in
    run_checks can pre-empt a hung command (FX-G 5b)."""
    import subprocess
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)


def _pid_alive(pid: int) -> bool:
    """Does this pid exist? Read-only: signal 0 on POSIX, OpenProcess on Windows
    (os.kill on Windows would TERMINATE the process, so it must never be used)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
        except Exception:
            return False
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _redact_tree(obj: Any, redactor: Redactor) -> Any:
    """Apply Redactor.text() to every string in a JSON-ish structure."""
    if isinstance(obj, str):
        return redactor.text(obj)
    if isinstance(obj, dict):
        # Keys too: table names such as csa_memory_<character> carry names.
        return {redactor.text(str(k)): _redact_tree(v, redactor) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_tree(v, redactor) for v in obj]
    return obj


def _err_details(err: Exception, *, password: Optional[str] = None) -> dict:
    msg = str(err)
    if password:
        msg = msg.replace(str(password), "***")
    return {"error": f"{type(err).__name__}: {msg}"}


def _model_ids(data: Any) -> List[str]:
    """Model ids from an OpenAI-style /models payload; guards every shape."""
    if not isinstance(data, dict):
        return []
    ids = []
    for m in data.get("data") or []:
        if isinstance(m, dict) and m.get("id"):
            ids.append(str(m["id"]))
    return ids


# ── the checks ───────────────────────────────────────────────────────────────

async def check_install(env: dict, ctx: _Ctx) -> Check:
    import platform

    version = "unknown"
    try:
        vf = ctx.root / "VERSION"
        if vf.is_file():
            version = vf.read_text(encoding="utf-8", errors="replace").strip() or "unknown"
    except OSError:
        pass

    commit = None
    try:
        res = await asyncio.to_thread(
            _subprocess_run,
            ["git", "-C", str(ctx.root), "rev-parse", "--short", "HEAD"],
            timeout=_GIT_TIMEOUT_S)
        if res.returncode == 0:
            commit = res.stdout.strip() or None
    except Exception:
        commit = None

    packages: Dict[str, str] = {}
    for pkg in ("fastapi", "uvicorn", "pydantic", "asyncpg", "httpx", "python-dotenv"):
        try:
            import importlib.metadata
            packages[pkg] = importlib.metadata.version(pkg)
        except Exception:
            packages[pkg] = "missing"

    details = {
        "version": version,
        "git_commit": commit,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "in_container": (env.get("SPM_IN_CONTAINER") or "").strip() == "1",
        "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "packages": packages,
    }
    return Check("install", "ok", f"SPM {version} on Python {details['python']}", details)


async def check_services(env: dict, ctx: _Ctx) -> Check:
    from spm.run import _engine_url, _proxy_url, _read_pid_file

    engine_url, proxy_url = _engine_url(env), _proxy_url(env)
    details: Dict[str, Any] = {"engine_url": engine_url, "proxy_url": proxy_url}
    up: Dict[str, bool] = {}
    async with await _make_http(timeout=_HEALTH_TIMEOUT_S) as client:
        for name, url in (("engine", engine_url), ("proxy", proxy_url)):
            started = time.monotonic()
            try:
                resp = await client.get(url.rstrip("/") + "/health")
                up[name] = resp.status_code == 200
            except Exception:
                up[name] = False
            details[f"{name}_latency_ms"] = round((time.monotonic() - started) * 1000)

    pids: Dict[str, Any] = {}
    for key, value in (_read_pid_file(ctx.root) or {}).items():
        try:
            pids[str(key)] = {"pid": int(value), "alive": _pid_alive(int(value))}
        except (TypeError, ValueError):
            pids[str(key)] = {"pid": str(value), "alive": None}
    details["pid_file"] = str(ctx.root / "run" / "spm.pids")
    details["pids"] = pids
    details["log_files"] = {n: str(ctx.root / "logs" / n)
                            for n in ("spm_engine.out", "spm_proxy.out")}

    n_up = sum(1 for v in up.values() if v)
    status = "ok" if n_up == 2 else ("warn" if n_up == 1 else "fail")
    summary = ", ".join(f"{n} {'up' if up[n] else 'down'}" for n in ("engine", "proxy"))
    return Check("services", status, summary, details)


def _effective_settings(env: dict, ctx: "_Ctx"):
    """Settings as SPM resolves them: the settings file wins over the environment, which
    wins over the built-in defaults (config/manager.py). A real run checked Lemonade from
    .env while the file pointed SPM at another server (2026-10-14).
    Returns (settings, path, file_error_or_None, file_exists)."""
    import json

    from config.manager import _DEFAULT_VALUES, _ENV_VAR_MAP, default_config_path

    path = env.get("SPM_CONFIG_PATH") or default_config_path(str(ctx.root / "config"))
    merged: Dict[str, Any] = dict(_DEFAULT_VALUES)
    for key, var in _ENV_VAR_MAP.items():
        if var in env:
            merged[key] = env[var]
    p = Path(path)
    error, exists = None, p.is_file()
    if exists:
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged.update(loaded)
            else:
                error = "config file is not a JSON object"
        except (json.JSONDecodeError, OSError) as err:
            error = f"config file unreadable: {type(err).__name__}"
    return merged, path, error, exists


def _setting(env: dict, ctx: "_Ctx", key: str) -> str:
    return str(_effective_settings(env, ctx)[0].get(key) or "")


async def check_config(env: dict, ctx: _Ctx) -> Check:
    import json

    from config.manager import _DEFAULT_VALUES, _ENV_VAR_MAP, default_config_path

    merged, path, error, exists = _effective_settings(env, ctx)
    status, summary = "ok", f"config: {path}"
    if error:
        status, summary = "warn", error
    elif not exists:
        status, summary = "warn", "config file not found (defaults in use)"

    shown = {k: v for k, v in merged.items() if not _SECRET_KEY_RE.search(str(k))}
    details = {
        "config_path": str(path),
        "settings": shown,
        "reasoning_mode": merged.get("reasoning_mode"),
        "gm_actions_mode": merged.get("gm_actions_mode"),
        "context_window_override": merged.get("context_window_override"),
    }
    return Check("config", status, summary, details)


async def check_model_server(env: dict, ctx: _Ctx) -> Check:
    from proxy.backend_client.backend_profile import detect_window
    from proxy.backend_client.lemonade_client import DEFAULT_CHAT_MODEL, best_match
    from proxy.core.reasoning import _is_local

    base = _setting(env, ctx, "BACKEND_LLM_URL").strip().rstrip("/")
    if not base:
        return Check("model_server", "warn", "BACKEND_LLM_URL is not set", {})
    chat_model = (env.get("SPM_DEFAULT_MODEL") or DEFAULT_CHAT_MODEL)
    details: Dict[str, Any] = {"url": base, "chat_model": chat_model}
    try:
        details["local"] = bool(_is_local(base))
    except Exception:
        details["local"] = None

    # GET /models only — NEVER a generation request (brief R1b).
    try:
        async with await _make_http(timeout=_SERVER_TIMEOUT_S) as client:
            try:
                resp = await client.get(f"{base}/models")
            except Exception as err:
                return Check("model_server", "fail",
                             f"model server unreachable at {base}",
                             _err_details(err))
            ids: List[str] = []
            parse_error = None
            if resp.status_code == 200:
                try:
                    ids = _model_ids(resp.json())
                except Exception as err:
                    parse_error = f"{type(err).__name__}: {err}"
            else:
                parse_error = f"HTTP {resp.status_code}"
            details["model_ids"] = ids
            match = best_match(chat_model, ids)
            details["matched"] = match
            try:
                info = await detect_window(base, chat_model, http=client)
                details["window"] = info.tokens
                details["window_source"] = info.source
            except Exception as err:
                details["window"] = None
                details["window_source"] = "error"
                parse_error = parse_error or _err_details(err)["error"]
    except Exception:
        raise  # missing package etc. -> run_checks turns it into skipped/fail

    if parse_error:
        return Check("model_server", "warn", f"model server answered oddly: {parse_error}",
                     details)
    if match:
        return Check("model_server", "ok",
                     f"{base} reachable, chat model matches as '{match}'", details)
    return Check("model_server", "warn",
                 f"{base} reachable but chat model '{chat_model}' not found", details)


async def check_embeddings(env: dict, ctx: _Ctx) -> Check:
    provider = (_setting(env, ctx, "EMBEDDING_PROVIDER") or "auto").strip().lower()
    if provider == "none":
        return Check("embeddings", "skipped", "embedding provider is 'none'", {})
    base = (_setting(env, ctx, "EMBEDDING_URL") or _setting(env, ctx, "BACKEND_LLM_URL")).strip().rstrip("/")
    if not base:
        return Check("embeddings", "warn", "no embedding URL configured",
                     {"provider": provider})

    async with await _make_http(timeout=_SERVER_TIMEOUT_S) as client:
        try:
            resp = await client.get(f"{base}/models")
        except Exception as err:
            return Check("embeddings", "fail", f"embedding server unreachable at {base}",
                         _err_details(err))
        ids = _model_ids(resp.json()) if resp.status_code == 200 else []
        embed_ids = [i for i in ids if "embed" in i.lower()]
        details: Dict[str, Any] = {"url": base, "provider": provider,
                                   "embedding_ids": embed_ids}
        if resp.status_code != 200:
            return Check("embeddings", "fail",
                         f"embedding server answered HTTP {resp.status_code}", details)
        if not embed_ids:
            # Never probe /embeddings here: a JIT-capable server would load a
            # chat model for a request it never needed (plan review item 3).
            return Check("embeddings", "warn",
                         "no embedding model id listed by the server", details)

        model = None
        wanted = _setting(env, ctx, "EMBEDDING_MODEL").strip()
        if wanted:
            try:
                from proxy.backend_client.lemonade_client import best_match
                model = best_match(wanted, embed_ids)
            except Exception:
                model = None
        model = model or embed_ids[0]
        details["probed_model"] = model
        started = time.monotonic()
        try:
            # A cold embedding model can take several seconds to answer the first
            # request (a real run failed at 5 s, then answered in 37 ms): allow 9 s.
            probe = await client.post(f"{base}/embeddings",
                                      json={"model": model, "input": "ping"}, timeout=9.0)
        except Exception as err:
            import httpx as _httpx
            if isinstance(err, _httpx.TimeoutException):
                return Check("embeddings", "warn",
                             "embedding probe slow (over 9 s; the model may still be loading)",
                             {**details, **_err_details(err)})
            return Check("embeddings", "fail", "embedding probe failed",
                         {**details, **_err_details(err)})
        details["latency_ms"] = round((time.monotonic() - started) * 1000)
        dimension = None
        if probe.status_code == 200:
            try:
                payload = probe.json()
                emb = payload["data"][0]["embedding"]
                dimension = len(emb)
            except Exception:
                dimension = None
        if dimension:
            details["dimension"] = dimension
            return Check("embeddings", "ok",
                         f"embedding model '{model}' answers ({dimension} dims, "
                         f"{details['latency_ms']} ms)", details)
        return Check("embeddings", "fail",
                     f"embedding probe returned HTTP {probe.status_code} or a bad shape",
                     details)


def _migration_files(ctx: _Ctx):
    """(sorted *.sql names, manual-only set) for the migration ledger diff."""
    mig_dir = ctx.root / "scripts" / "migrations"
    manual: set = set()
    try:
        root_str = str(ctx.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from scripts.apply_migrations import MANUAL_ONLY_MIGRATIONS, MIGRATIONS_DIR
        manual = set(MANUAL_ONLY_MIGRATIONS)
        if not mig_dir.is_dir():
            mig_dir = Path(MIGRATIONS_DIR)
    except Exception:
        pass
    files = sorted(p.name for p in mig_dir.glob("*.sql")) if mig_dir.is_dir() else []
    return files, manual


async def check_database(env: dict, ctx: _Ctx) -> Check:
    from spm.db import db_config_from_env

    cfg = db_config_from_env(env)
    if cfg is None:
        return Check("database", "fail", "POSTGRES_DB is not set", {})
    try:
        conn = await _connect(cfg)
    except Exception as err:
        return Check("database", "fail", "cannot connect to the database",
                     _err_details(err, password=cfg.get("password")))
    try:
        server_version = str(await conn.fetchval("SELECT version()") or "")
        server_version = server_version.splitlines()[0] if server_version else ""
        vector = bool(await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        try:
            rows = await conn.fetch(
                "SELECT filename FROM spm_schema_migrations ORDER BY filename")
            ledger = [str(r["filename"]) for r in rows]
            ledger_error = None
        except Exception as err:
            ledger = []
            ledger_error = f"{type(err).__name__}: {err}"
        files, manual_only = await asyncio.to_thread(_migration_files, ctx)
        pending = [f for f in files if f not in set(ledger) and f not in manual_only]

        counts: Dict[str, int] = {}
        try:
            for r in await conn.fetch(_SPM_TABLES_SQL):
                table = str(r["table_name"])
                try:
                    counts[table] = int(await conn.fetchval(
                        f'SELECT count(*) FROM "{table}"'))
                except Exception:
                    counts[table] = -1
        except Exception as err:
            counts = {}
            ledger_error = ledger_error or f"table list: {type(err).__name__}"
    finally:
        try:
            await conn.close()
        except Exception:
            pass

    problems = []
    if not vector:
        problems.append("vector extension missing")
    if pending:
        problems.append(f"{len(pending)} pending migration(s)")
    if ledger_error:
        problems.append("migration ledger unavailable")
    status = "warn" if problems else "ok"
    summary = f"connected to database" + (
        ": " + ", ".join(problems) if problems else "; schema up to date")
    details = {
        "database": cfg["database"],
        "host": cfg["host"],
        "server_version": server_version,
        "vector_extension": vector,
        "ledger": {"applied": ledger,
                   "pending": pending,
                   "manual_only": sorted(set(files) & manual_only),
                   "error": ledger_error},
        "row_counts": counts,
    }
    return Check("database", status, summary, details)


def _tail_lines(path: Path, max_bytes: int = _SCAN_WINDOW_BYTES) -> List[str]:
    """The lines in the last `max_bytes` of a file (a partial first line dropped)."""
    size = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()
        return fh.read().splitlines()


def _scan_logs(root: Path, limit: int) -> List[str]:
    """The last `limit` WARNING/ERROR/Traceback lines across logs/*.out|*.log,
    looking only at the last _SCAN_WINDOW_BYTES of each file."""
    matched: List[str] = []
    files = sorted(list(root.glob("logs/*.out")) + list(root.glob("logs/*.log")))
    for path in files:
        try:
            lines = _tail_lines(path)
        except OSError:
            continue
        matched.extend(line for line in lines
                       if "WARNING" in line or "ERROR" in line
                       or line.startswith("Traceback"))
    return matched[-limit:]


async def _recent_errors(ctx: _Ctx, name: str, limit: int) -> Check:
    # File scan and redaction off the event loop, so the per-check cap applies.
    raw = await asyncio.to_thread(_scan_logs, ctx.root, limit)
    kept = await asyncio.to_thread(ctx.redactor.lines, raw)  # public mode: whitelist + full rule set
    groups: Dict[str, Dict[str, Any]] = {}
    for line in kept:
        key = _DIGIT_RUN_RE.sub("N", _HEX_RUN_RE.sub("#", line))
        group = groups.setdefault(key, {"message": key, "count": 0})
        group["count"] += 1
    ordered = sorted(groups.values(), key=lambda g: (-g["count"], g["message"]))
    details = {"limit": limit,
               "lines_matched": len(raw),
               "lines_after_filter": len(kept),
               "groups": ordered}
    if not ordered:
        return Check(name, "ok", "no recent errors or warnings in logs/", details)
    total = sum(g["count"] for g in ordered)
    return Check(name, "warn",
                 f"{total} recent error/warning line(s) in {len(ordered)} group(s)",
                 details)


async def check_recent_errors(env: dict, ctx: _Ctx) -> Check:
    return await _recent_errors(ctx, "recent_errors", 50)


async def check_recent_errors_full(env: dict, ctx: _Ctx) -> Check:
    return await _recent_errors(ctx, "recent_errors_full", 300)


async def check_scheduler(env: dict, ctx: _Ctx) -> Check:
    from spm.run import _proxy_url

    proxy_url = _proxy_url(env).rstrip("/")
    async with await _make_http(timeout=_HEALTH_TIMEOUT_S) as client:
        try:
            health = await client.get(proxy_url + "/health")
            if health.status_code != 200:
                return Check("scheduler", "skipped", "proxy is not answering /health",
                             {})
        except Exception:
            return Check("scheduler", "skipped", "proxy is not answering /health", {})
        try:
            resp = await client.get(proxy_url + "/admin/api/v1/scheduler")
        except Exception as err:
            return Check("scheduler", "fail", "scheduler snapshot failed",
                         _err_details(err))
    if resp.status_code != 200:
        return Check("scheduler", "warn",
                     f"scheduler endpoint answered HTTP {resp.status_code}",
                     {"response_text": (resp.text or "")[:2000]})
    try:
        snapshot = resp.json()
    except Exception as err:
        return Check("scheduler", "warn", "scheduler snapshot is not JSON",
                     _err_details(err))
    return Check("scheduler", "ok", "scheduler snapshot captured",
                 {"snapshot": snapshot})


# wmic is gone on Windows 11 24H2: PowerShell/CIM is the fallback.
_WMIC_GPU_CMD = ["wmic", "path", "win32_VideoController", "get", "name"]
_POWERSHELL_GPU_CMD = ["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_VideoController).Name"]


def _gpu_names() -> List[str]:
    """GPU name lines only (no user fields); 2 s per probe. Blocking: called
    through asyncio.to_thread."""
    if os.name == "nt":
        cmds = [_WMIC_GPU_CMD, _POWERSHELL_GPU_CMD]
    else:
        cmds = [["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                ["rocm-smi"]]
    names: List[str] = []
    for cmd in cmds:
        try:
            res = _subprocess_run(cmd, timeout=_SYSINFO_TIMEOUT_S)
        except Exception:
            continue
        if res.returncode != 0:
            continue
        for line in (res.stdout or "").splitlines():
            s = line.strip()[:120]
            if not s or "=" in s or s.lower() in ("name", "gpu"):
                continue
            names.append(s)
        if names:
            break
    return names[:4]


async def check_system(env: dict, ctx: _Ctx) -> Check:
    import platform
    import shutil

    details: Dict[str, Any] = {"cpu_count": os.cpu_count(),
                               "processor": platform.processor() or None}

    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
            mem = {}
            for m in re.finditer(r"^(MemTotal|MemFree|MemAvailable):\s+(\d+)\s+kB",
                                 text, re.MULTILINE):
                mem[m.group(1).lower()] = int(m.group(2)) * 1024
            details["memory_bytes"] = mem
        except OSError as err:
            details["memory_bytes"] = None
            details["memory_error"] = str(err)
    elif os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MemoryStatusEx()
            ms.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                details["memory_bytes"] = {"total": ms.ullTotalPhys,
                                           "free": ms.ullAvailPhys}
            else:
                details["memory_bytes"] = None
        except Exception as err:
            details["memory_bytes"] = None
            details["memory_error"] = str(err)
    else:
        details["memory_bytes"] = None
        details["memory_note"] = "memory information not supported on this platform"

    try:
        details["disk_free_bytes"] = shutil.disk_usage(ctx.root).free
    except OSError:
        details["disk_free_bytes"] = None

    details["gpu"] = await asyncio.to_thread(_gpu_names)
    return Check("system", "ok", "system information collected", details)


_OVERVIEW_QUERIES = (
    ("turns", "SELECT session_id, count(*) AS n FROM spm_perception GROUP BY session_id"),
    ("rules", "SELECT session_id, count(*) AS n FROM spm_chat_rules GROUP BY session_id"),
    ("rooms", "SELECT session_id, count(DISTINCT room_id) AS n "
              "FROM world_state_sessions GROUP BY session_id"),
)


async def check_chat_overview(env: dict, ctx: _Ctx) -> Check:
    """Per-chat counts only — never chat text (plan: 'no text')."""
    from spm.db import db_config_from_env

    cfg = db_config_from_env(env)
    if cfg is None:
        return Check("chat_overview", "fail", "POSTGRES_DB is not set", {})
    try:
        conn = await _connect(cfg)
    except Exception as err:
        return Check("chat_overview", "fail", "cannot connect to the database",
                     _err_details(err, password=cfg.get("password")))
    sessions: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    try:
        for field, sql in _OVERVIEW_QUERIES:
            try:
                for r in await conn.fetch(sql):
                    sid = str(r["session_id"])
                    row = sessions.setdefault(sid, {"session": sid,
                                                    "turns": 0, "rules": 0, "rooms": 0})
                    row[field] = int(r["n"])
            except Exception as err:
                errors[field] = f"{type(err).__name__}: {err}"
    finally:
        try:
            await conn.close()
        except Exception:
            pass
    status = "ok" if not errors else ("warn" if sessions else "fail")
    return Check("chat_overview", status,
                 f"{len(sessions)} chat session(s) summarised (counts only)",
                 {"sessions": [sessions[k] for k in sorted(sessions)], "errors": errors})


async def check_backend_models_full(env: dict, ctx: _Ctx) -> Check:
    base = _setting(env, ctx, "BACKEND_LLM_URL").strip().rstrip("/")
    if not base:
        return Check("backend_models_full", "warn", "BACKEND_LLM_URL is not set", {})
    async with await _make_http(timeout=_SERVER_TIMEOUT_S) as client:
        try:
            resp = await client.get(f"{base}/models")
        except Exception as err:
            return Check("backend_models_full", "fail",
                         f"model server unreachable at {base}", _err_details(err))
    if resp.status_code != 200:
        return Check("backend_models_full", "fail",
                     f"model server answered HTTP {resp.status_code}",
                     {"response_text": (resp.text or "")[:2000]})
    try:
        payload = resp.json()
    except Exception as err:
        return Check("backend_models_full", "warn",
                     "model list is not JSON", _err_details(err))
    return Check("backend_models_full", "ok",
                 f"full /models response from {base}", {"response": payload})


_REGISTRY: Dict[str, Callable] = {
    "install": check_install,
    "services": check_services,
    "config": check_config,
    "model_server": check_model_server,
    "embeddings": check_embeddings,
    "database": check_database,
    "recent_errors": check_recent_errors,
    "scheduler": check_scheduler,
    "system": check_system,
    "recent_errors_full": check_recent_errors_full,
    "chat_overview": check_chat_overview,
    "backend_models_full": check_backend_models_full,
}


# ── public entry points ──────────────────────────────────────────────────────

async def run_checks(env: dict, *, names=CHECKS, system_info=False, repo_root=None,
                     redactor=None, per_check_timeout=10.0) -> List[Check]:
    """Run the named checks in order; ALWAYS returns one Check per name.

    Each check runs under a hard per-check cap (asyncio.wait_for): a timeout
    becomes a fail Check, an exception becomes a fail Check (skipped when it is
    a missing dependency), never an exception out of this function. 'system'
    runs only when system_info=True. Every details dict (and every error
    message) passes through the Redactor before being returned.
    """
    from .run import default_repo_root

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    ctx = _Ctx(root=root, redactor=redactor if redactor is not None else Redactor())
    results: List[Check] = []
    for name in names:
        if name == "system" and not system_info:
            results.append(Check("system", "skipped", "not requested", {}))
            continue
        fn = _REGISTRY.get(name)
        if fn is None:
            results.append(Check(name, "fail", "unknown check name",
                                 {"known": list(CHECKS) + list(EXTRA_CHECKS)}))
            continue
        try:
            check = await asyncio.wait_for(fn(env, ctx), per_check_timeout)
        except asyncio.TimeoutError:
            check = Check(name, "fail", f"check timed out after {per_check_timeout}s",
                          {"error": f"timed out after {per_check_timeout}s"})
        except ModuleNotFoundError as err:
            check = Check(name, "skipped",
                          f"missing dependency: {err.name or err}", _err_details(err))
        except Exception as err:
            check = Check(name, "fail",
                          f"{type(err).__name__}: {str(err).splitlines()[0][:140] if str(err) else ''}",
                          _err_details(err))
        check.details = _redact_tree(check.details, ctx.redactor)
        check.summary = ctx.redactor.text(check.summary)
        results.append(check)
    return results


async def _values(conn, sql: str) -> List[str]:
    try:
        return [str(r["v"]) for r in await conn.fetch(sql)]
    except Exception:
        return []


async def _name_lists(env: dict) -> Optional[Dict[str, List[str]]]:
    """Raw names by kind from the database (chat / char / player); None when no
    database is configured. Raises when the database does not answer."""
    from .db import db_config_from_env

    cfg = db_config_from_env(env)
    if cfg is None:
        return None
    conn = await _connect(cfg)
    try:
        chats = await _values(conn,
                              "SELECT DISTINCT session_id AS v FROM spm_perception ORDER BY v")
        people = await _values(conn,
                               "SELECT DISTINCT v FROM (SELECT recipient_id AS v FROM spm_perception "
                               "UNION SELECT actor_id AS v FROM spm_perception) t ORDER BY v")
        chars = await _values(conn,
                              "SELECT DISTINCT character_id AS v FROM spm_chat_rules "
                              "WHERE character_id IS NOT NULL ORDER BY v")
        players = await _values(conn,
                                "SELECT DISTINCT player_name AS v FROM spm_chat_settings "
                                "WHERE player_name IS NOT NULL ORDER BY v")
    finally:
        try:
            await conn.close()
        except Exception:
            pass
    all_chars = list(people) + list(chars)
    return {"chat": list(chats), "char": all_chars,
            "player": list(players) + _legacy_personas(chats, all_chars)}


def _names_cache_path(root: Optional[Path] = None) -> Path:
    """The last good name lists (git-ignored run/), used when the database is
    down: a dead database is a prime reason to file a report (third review, 4).
    Seam: tests point it at a temporary folder."""
    from .run import default_repo_root
    return Path(root or default_repo_root()) / "run" / "names.json"


def _read_names_cache(path: Path) -> Dict[str, List[str]]:
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: [str(v) for v in data.get(k) or [] if isinstance(v, str)]
            for k in ("chat", "char", "player")}


def _write_names_cache(path: Path, lists: Dict[str, List[str]]) -> None:
    import json
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lists, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _table_from_lists(lists: Dict[str, List[str]]) -> Dict[str, str]:
    table: Dict[str, str] = {}

    def add(values: List[str], prefix: str) -> None:
        n = 0
        for v in values:
            v = str(v).strip()
            if not v or v.lower() == "user" or v in table:
                continue
            n += 1
            table[v] = f"{prefix}#{n}"

    add(lists.get("chat") or [], "chat")
    add(lists.get("char") or [], "char")
    add(lists.get("player") or [], "player")
    return table


def _legacy_personas(chats: List[str], chars: List[str]) -> List[str]:
    """Persona names inside legacy session ids "st_<persona>_<character>": the
    persona is stored nowhere else, so strip a known character id off the end."""
    out: List[str] = []
    for sid in chats:
        if not sid.startswith("st_") or sid.startswith("st_chat_"):
            continue
        for ch in sorted((c for c in chars if c), key=len, reverse=True):
            if sid.endswith("_" + ch) and len(sid) > len(ch) + 4:
                persona = sid[3:-(len(ch) + 1)]
                if persona and persona not in out:
                    out.append(persona)
                break
    return out


async def name_table(env: dict, root: Optional[Path] = None) -> Dict[str, str]:
    """Replacement table for Redactor.names: session ids -> chat#N, character /
    world ids -> char#N, player names -> player#N (5 s cap). The names are also
    kept in run/names.json (merged with what was there), and that file is the
    fallback when the database does not answer; empty when neither works."""
    path = _names_cache_path(root)
    cached = _read_names_cache(path)
    try:
        fresh = await asyncio.wait_for(_name_lists(env), 5.0)
    except Exception:
        fresh = None
    if not fresh:
        return _table_from_lists(cached)
    merged = {k: list(dict.fromkeys((fresh.get(k) or []) + (cached.get(k) or [])))
              for k in ("chat", "char", "player")}
    _write_names_cache(path, merged)
    return _table_from_lists(merged)
