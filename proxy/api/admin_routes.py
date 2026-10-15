"""
Admin REST API Endpoints for SPM (Port 5050).

Mounted under /admin/api/v1/:
  - GET  /stats: Real-time telemetry metrics, queue depth, DB size, active sessions.
  - GET  /config: Fetch current configuration parameters.
  - POST /config: Update configuration parameters.
  - DELETE /sessions/{session_id}: Hard delete all memory and spatial records for a session.
  - DELETE /factory_reset: Truncate all character memory tables, bulk imports, cold archives, and Evennia state.
"""

import logging
import re
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from config.manager import get_settings_manager
from proxy.core.telemetry import get_telemetry_collector
from core.resource_manager import strings
from core.identifiers import safe_char_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/api/v1", tags=["admin"])


class AdminState:
    """Encapsulates admin module state (db pool, lazy-init logic)."""

    _db_pool: Any = None

    @classmethod
    def get_db_pool(cls) -> Any:
        if cls._db_pool is not None:
            return cls._db_pool
        from proxy.api.routes import _db_pool
        if _db_pool is not None:
            cls._db_pool = _db_pool
            return cls._db_pool
        return None

    @classmethod
    def set_db_pool(cls, pool: Any) -> None:
        cls._db_pool = pool

    @classmethod
    async def ensure_db_pool(cls) -> Any:
        if cls._db_pool is not None:
            return cls._db_pool
        import os, asyncio, asyncpg
        from proxy.api.routes import set_db_pool
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "litellm_postgres")
        user = os.getenv("POSTGRES_USER", "spm_user")
        pwd = os.getenv("POSTGRES_PASSWORD", "spm_secure_password")
        dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
        try:
            pool = await asyncio.wait_for(asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5), timeout=2.0)
            cls.set_db_pool(pool)
            set_db_pool(pool)
            logger.info("[AdminAPI] Lazy database connection pool successfully established.")
            return pool
        except Exception as e:
            logger.warning(f"[AdminAPI] Failed lazy DB connection attempt: {e}")
            return None


def set_admin_db_pool(pool) -> None:
    """Backward-compatible setter — delegates to AdminState."""
    AdminState.set_db_pool(pool)


def get_admin_db_pool() -> Any:
    """Backward-compatible getter — delegates to AdminState."""
    return AdminState.get_db_pool()


async def ensure_db_pool() -> Any:
    """Backward-compatible lazy-init — delegates to AdminState."""
    return await AdminState.ensure_db_pool()


@router.get("/stats")
async def get_admin_stats():
    """Return live system telemetry, active sessions, and database size."""
    telemetry = get_telemetry_collector()
    stats = telemetry.get_metrics(AdminState.get_db_pool())

    # If DB pool is available, fetch actual DB table stats
    db_size_mb = 0.0
    pool = AdminState.get_db_pool()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                size_bytes = await conn.fetchval(strings.get("sql.fetch_db_size"))
                if size_bytes:
                    db_size_mb = round(size_bytes / (1024 * 1024), 2)
        except Exception as e:
            logger.warning(f"[AdminAPI] Could not query DB size: {e}")

    stats["db_size_mb"] = db_size_mb
    return JSONResponse(content=stats)


@router.get("/config")
async def get_config():
    """Fetch current dynamic settings."""
    mgr = get_settings_manager()
    return JSONResponse(content=mgr.get_settings())


@router.get("/logs")
async def get_logs(session_id: Optional[str] = None, level: Optional[str] = None, limit: int = 50):
    """Fetch recent log entries from the circular log buffer."""
    telemetry = get_telemetry_collector()
    logs = telemetry.get_logs(session_id=session_id, level=level, limit=limit)
    return JSONResponse(content={"logs": logs})


@router.get("/sessions/{session_id}/trace")
async def get_session_trace(session_id: str):
    """Agent-readable historical decision trace endpoint for an SPM session."""
    telemetry = get_telemetry_collector()
    trace = telemetry.get_session_trace(session_id)
    return JSONResponse(content={"session_id": session_id, "trace": trace})


@router.get("/stream/thoughts")
async def stream_thoughts():
    """SSE event stream broadcasting live inner monologue thoughts and routing decisions."""
    import json, asyncio
    from fastapi.responses import StreamingResponse

    telemetry = get_telemetry_collector()
    q = asyncio.Queue()

    # Replay recent thought history so UI immediately populates
    recent_thoughts = telemetry.get_recent_thoughts()
    for t in recent_thoughts:
        q.put_nowait(t)

    telemetry.subscribe_thoughts(q)

    async def thought_event_generator():
        try:
            yield "data: {\"event\": \"connected\"}\n\n"
            while True:
                data = await q.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            telemetry.unsubscribe_thoughts(q)

    return StreamingResponse(thought_event_generator(), media_type="text/event-stream")


@router.get("/reasoning/status")
async def reasoning_status():
    """Native-reasoning spike: configured mode, what 'auto' resolved to per probed backend,
    and whether the backend is local. A non-local backend gets a cost notice: hidden
    thinking is billed as output tokens there."""
    from proxy.core import reasoning
    s = get_settings_manager().get_settings()
    url = str(s.get("BACKEND_LLM_URL") or "")
    local = reasoning._is_local(url)
    resolved = {f"{u} ({m})": mode for (u, m), (_, mode, _ttl) in reasoning._probe_cache.items()}
    notice = "" if local else (
        "This backend looks remote and may charge per token. Hidden reasoning is billed as "
        "output tokens, and SPM's background jobs (lore review, memory consolidation) make "
        "extra calls. 'Auto' uses no reasoning for remote backends; choose Native only if "
        "you accept the extra cost.")
    return JSONResponse(content={
        "configured": reasoning.configured_mode(s), "effort": reasoning.effort(s),
        "reasoning_max_tokens": int(s.get("reasoning_max_tokens", 3000) or 3000),
        "backend_url": url, "local": local, "resolved": resolved, "notice": notice})


@router.get("/backend/window")
async def backend_window():
    """Brief W1b (OPEN-024): what context window SPM believes the configured backend
    has for the default chat model — the detected value, the manual override, and
    which one wins (budget.context_window precedence)."""
    from proxy.backend_client import backend_profile
    from proxy.backend_client.lemonade_client import DEFAULT_CHAT_MODEL
    from config.hardware_tiers import get_hardware_config
    from proxy.rag import budget
    from proxy.api.routes import lemonade_client
    s = get_settings_manager().get_settings()
    win = await backend_profile.window_for(lemonade_client, DEFAULT_CHAT_MODEL)
    tier = str(s.get("SPM_HARDWARE_TIER") or "SOVEREIGN")
    hw = get_hardware_config(tier)
    override = s.get("context_window_override")
    try:
        override = int(override) if override else None
    except (TypeError, ValueError):
        override = None
    return JSONResponse(content={
        "tokens": win.tokens, "source": win.source, "detail": win.detail,
        "override": override,
        "effective": budget.context_window(s, hw, detected=win.tokens,
                                           detected_source=win.source),
        "tier": tier, "model": DEFAULT_CHAT_MODEL,
        "backend_url": lemonade_client.base_url})


@router.post("/config")
async def update_config(new_settings: Dict[str, Any]):
    """Update dynamic settings in config.json."""
    mgr = get_settings_manager()
    updated = mgr.write_settings(new_settings)
    return JSONResponse(content={"status": "success", "settings": updated})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Hard delete all records for a specific session across all memory tables."""
    pool = get_admin_db_pool()
    if pool is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": strings.get("api.errors.db_unavailable")}
        )

    deleted_count = 0
    try:
        async with pool.acquire() as conn:
            # Delete from spm_chat_imports
            await conn.execute(strings.get("sql.delete_session_imports"), session_id)
            # Delete from spm_cold_archives
            await conn.execute(strings.get("sql.delete_session_archives"), session_id)
            # Delete from world_state_sessions
            await conn.execute(strings.get("sql.delete_session_world_state"), session_id)

            # Chat rules now live in spm_chat_rules (brief C1b): a deleted chat
            # leaves no chat-scoped rules behind (OPEN-007).
            res = await conn.execute(
                "DELETE FROM spm_chat_rules WHERE session_id = $1;", session_id)
            try:
                deleted_count += int(res.split()[-1])
            except (ValueError, IndexError):
                pass

            tables = list(await conn.fetch(strings.get("sql.list_memory_tables")))
            for t in tables:
                table_name = t["table_name"]
                res = await conn.execute(
                    f"DELETE FROM {table_name} WHERE session_id = $1;", session_id
                )
                # Format: "DELETE 5"
                try:
                    num = int(res.split()[-1])
                    deleted_count += num
                except (ValueError, IndexError):
                    pass

        telemetry = get_telemetry_collector()
        telemetry.dismiss_session(session_id)

        logger.info(f"[AdminAPI] Deleted session {session_id}: {deleted_count} memory rows removed")
        return JSONResponse(content={
            "status": "success",
            "session_id": session_id,
            "deleted_records": deleted_count
        })
    except Exception as e:
        logger.error(f"[AdminAPI] Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/factory_reset")
async def factory_reset():
    """Truncate all character memory tables, bulk imports, cold archives, and reset telemetry metrics."""
    pool = get_admin_db_pool()
    if pool is None:
        pool = await ensure_db_pool()
    if pool is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": strings.get("api.errors.db_unavailable")}
        )

    try:
        async with pool.acquire() as conn:
            # Truncate tracking tables
            await conn.execute(strings.get("sql.truncate_imports"))
            await conn.execute(strings.get("sql.truncate_archives"))
            await conn.execute(strings.get("sql.truncate_world_state"))

            # Truncate all csa_memory tables (brief C1b: list_tables_info now
            # lists only csa_memory_%) plus the shared chat-rules table.
            await conn.execute("TRUNCATE TABLE spm_chat_rules RESTART IDENTITY CASCADE;")
            tables = await conn.fetch(
                strings.get("sql.list_tables_info")
            )
            for t in tables:
                table_name = t["table_name"]
                await conn.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")

        # Reset telemetry metrics, active sessions, and request logs
        telemetry = get_telemetry_collector()
        telemetry.reset()

        # Clear the world engine's in-memory state through the SHARED, configured
        # client. This used its own client with a hardcoded http://localhost:4005, which
        # bypassed the test suite's redirect: every test run that exercised factory
        # reset wiped all chat worlds on the LIVE engine (QA 2026-10-05).
        from proxy.api.routes import evennia_client
        try:
            resp = await evennia_client.client.delete(f"{evennia_client.base_url}/world/admin/reset")
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"[AdminAPI] Failed to reset Evennia cache: {e}")

        logger.warning("[AdminAPI] FACTORY RESET TRIGGERED — all memory tables truncated and telemetry reset.")
        return JSONResponse(content={
            "status": "success",
            "message": "Factory reset complete. All character memories, session data, and telemetry stats truncated."
        })
    except Exception as e:
        logger.error(f"[AdminAPI] Error executing factory reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _pid_file():
    """The launcher's PID file (a function so tests can point it at a temp dir: a test
    once deleted the live SPM's PID file through this path, 2026-10-14)."""
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "run" / "spm.pids"

@router.post("/shutdown")
async def shutdown_system():
    """Quit this SPM install — never with a machine-wide process-name kill (brief L1).

    The old implementation killed by process-name pattern: on Windows there is
    no such tool (the button did nothing) and on Linux it killed every
    matching process on the machine, not only this SPM. It also stopped the
    spm-postgres container unconditionally. Now each process ends itself.
    """
    import os, signal, asyncio
    logger.warning("[AdminAPI] Received shutdown request. Shutting down services...")

    if os.environ.get("SPM_IN_CONTAINER") == "1":
        # Inside the Docker bundle a self-kill just restarts the container and
        # strands the database container. Point at the bundle's stop instead.
        return JSONResponse(content={
            "status": "docker",
            "message": ("SPM runs in Docker. Stop SPM and its database together with "
                        "scripts\\docker\\stop.cmd (Windows) or scripts/docker/stop.sh "
                        "(Linux/macOS), or the Stop button of the 'spm' group in "
                        "Docker Desktop. Your data is kept.")})

    # Engine URL from the existing, configured evennia client (its base_url ends
    # in /api/v1, so the new engine shutdown endpoint hangs off it directly).
    from proxy.api.routes import evennia_client
    engine_shutdown_url = f"{evennia_client.base_url}/world/admin/shutdown"
    pid_file = _pid_file()

    async def _shutdown():
        await asyncio.sleep(1)
        try:  # ask the engine to end itself; a dead engine is not our problem
            resp = await evennia_client.client.post(engine_shutdown_url, timeout=5.0)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"[AdminAPI] Engine shutdown request failed: {e}")
        import spm.run
        spm.run.stop_db_container(dict(os.environ))
        try:
            if pid_file.exists():
                pid_file.unlink()
        except OSError as e:
            logger.warning(f"[AdminAPI] Could not remove {pid_file}: {e}")
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_shutdown())

    return JSONResponse(content={
        "status": "success",
        "message": "Shutting down SPM and Game services gracefully."
    })

@router.post("/lore/test_connection")
async def test_lore_connection(body: dict):
    """Test connection to the alternate model."""
    from proxy.backend_client.lemonade_client import LemonadeLLMClient
    model_name = body.get("model_name", "")
    
    client = LemonadeLLMClient()
    resolved = await client._resolve_model(model_name)
    await client.close()
    
    if resolved:
        return JSONResponse(content={"status": "success", "resolved_model": resolved})
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Model not found or unreachable"})


@router.get("/sessions/{session_id}/perception")
async def session_perception(session_id: str, recipient_id: str = "", turn_id: str = "",
                             limit: int = 100):
    """Gating phase 5 observability: what each character actually perceived, straight
    from spm_perception. Filter by recipient and/or turn. This is the dev window into
    'what did X see at turn N' — read-only, like the thoughts tab."""
    pool = AdminState.get_db_pool()
    if pool is None:
        return JSONResponse(status_code=503, content={"error": strings.get("api.errors.db_unavailable")})
    limit = max(1, min(int(limit), 500))
    where = ["session_id = $1"]
    args: list = [session_id]
    if recipient_id:
        args.append(recipient_id)
        where.append(f"recipient_id = ${len(args)}")
    if turn_id:
        args.append(turn_id)
        where.append(f"turn_id = ${len(args)}")
    args.append(limit)
    sql = f"""
        SELECT tick, turn_id, actor_id, action_type, recipient_id, gating_level,
               perceived_text, distance_ft, barriers, created_at
        FROM spm_perception WHERE {' AND '.join(where)}
        ORDER BY tick DESC, id DESC LIMIT ${len(args)}
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return JSONResponse(content={"session_id": session_id, "rows": [
            {**dict(r), "created_at": r["created_at"].isoformat() if r["created_at"] else None,
             "distance_ft": float(r["distance_ft"]) if r["distance_ft"] is not None else None}
            for r in rows]})
    except Exception as e:
        logger.error(f"[AdminAPI] perception view failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler")
async def scheduler_snapshot():
    """LLM scheduler telemetry (fifo_queue.md phase 6): queue depths, running jobs,
    wait percentiles, dispatch/preempt/supersede counters, loaded model."""
    from proxy.core.llm_scheduler import get_scheduler
    return JSONResponse(content=get_scheduler().snapshot())


# ── Brief R4: "Report a problem" from the admin dashboard ───────────────────
# spm/report.py (briefs R2/R3) does the real work; nothing here starts a
# service or talks to a model server. Tests patch spm.report.build_report.

_REPORT_BUNDLE_RE = re.compile(r"spm-report-[A-Za-z0-9-]+\.zip")


def _reports_dir():
    """The bundle directory (a function so tests can point it at a temp dir)."""
    from pathlib import Path
    from spm.run import default_repo_root
    return Path(default_repo_root()) / "reports"


@router.post("/report")
async def create_report(body: dict):
    """Build a bug report bundle (see `spm report`); Enhanced may take a minute.

    Body: {description, system_info, full_logs, enhanced, confirm_remote}.
    With `enhanced` and a remote model server, the redacted diagnostics would
    leave this machine, so without `confirm_remote` this returns 409 and the UI
    asks first (same rule as the CLI's --enhanced consent).
    """
    from spm import report as spm_report
    from spm.run import child_env, default_repo_root

    from proxy.core.reasoning import _is_local

    description = str(body.get("description") or "")
    system_info = bool(body.get("system_info"))
    full_logs = bool(body.get("full_logs"))
    enhanced = bool(body.get("enhanced"))
    confirm_remote = bool(body.get("confirm_remote"))

    env = child_env(default_repo_root())
    if enhanced and not confirm_remote:
        base = spm_report._backend_url(env)
        if base and not _is_local(base):
            import urllib.parse
            host = urllib.parse.urlparse(base).hostname or base
            return JSONResponse(
                status_code=409,
                content={"needs_confirmation": True,
                         "message": (f"Enhanced analysis sends the redacted "
                                     f"diagnostics to {host}, a remote service "
                                     f"(this may cost tokens). Continue?")})

    try:
        report = await spm_report.build_report(
            env, description=description, system_info=system_info,
            full_logs=full_logs, enhanced=enhanced)
    except Exception as e:
        logger.error(f"[AdminAPI] report build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    content = {
        "markdown": report.markdown,
        "checks": [{"name": c.name, "status": c.status, "summary": c.summary}
                   for c in report.checks],
        "file": report.path.name if report.path else None,
        "issue_url": report.issue_url,
        "private": report.private,
    }
    if isinstance(report.markdown, str) and \
            "Enhanced analysis unavailable:" in report.markdown:
        for line in report.markdown.splitlines():
            if line.startswith("Enhanced analysis unavailable:"):
                content["enhanced_error"] = \
                    line[len("Enhanced analysis unavailable:"):].strip()
                break
    return JSONResponse(content=content)


@router.get("/report/{name}")
async def download_report(name: str):
    """Download a report bundle from reports/ — only a plain
    `spm-report-*.zip` name, never a path (no traversal)."""
    from fastapi.responses import FileResponse

    if not _REPORT_BUNDLE_RE.fullmatch(name or ""):
        raise HTTPException(status_code=404, detail="report not found")
    path = _reports_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(path, media_type="application/zip", filename=name)


@router.get("/sessions/{session_id}/world")
async def session_world(session_id: str):
    """The REAL world for a chat (rooms, occupants, edges, tick) for the admin
    spatial map. That panel used to invent its contents from log strings and a
    hardcoded 'Arvenia' (QA 2026-10-05). Served through the proxy so the browser
    never calls the engine port directly."""
    from proxy.api.routes import evennia_client
    try:
        snap = await evennia_client.get_snapshot(session_id=session_id, template_key="")
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"world engine unavailable: {e}"})
    return JSONResponse(content=snap)


@router.get("/sessions/{session_id}/settings")
async def get_chat_settings(session_id: str):
    """Per-chat settings (QA F26). narrator_mode: auto | on | off."""
    pool = AdminState.get_db_pool()
    if pool is None:
        return JSONResponse(status_code=503, content={"error": strings.get("api.errors.db_unavailable")})
    async with pool.acquire() as conn:
        mode = await conn.fetchval(
            "SELECT narrator_mode FROM spm_chat_settings WHERE session_id = $1", session_id)
    return JSONResponse(content={"session_id": session_id, "narrator_mode": mode or "auto"})


@router.put("/sessions/{session_id}/settings")
async def put_chat_settings(session_id: str, body: dict):
    """Set per-chat settings. Body: {"narrator_mode": "auto" | "on" | "off"}."""
    mode = str(body.get("narrator_mode", "")).lower()
    if mode not in ("auto", "on", "off"):
        return JSONResponse(status_code=400, content={"error": "narrator_mode must be auto, on or off"})
    pool = AdminState.get_db_pool()
    if pool is None:
        return JSONResponse(status_code=503, content={"error": strings.get("api.errors.db_unavailable")})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO spm_chat_settings (session_id, narrator_mode, updated_at) "
            "VALUES ($1, $2, CURRENT_TIMESTAMP) ON CONFLICT (session_id) DO UPDATE "
            "SET narrator_mode = EXCLUDED.narrator_mode, updated_at = CURRENT_TIMESTAMP",
            session_id, mode)
    return JSONResponse(content={"session_id": session_id, "narrator_mode": mode})


# ── Brief C1a: per-chat rules admin API ─────────────────────────────────────
# Chat-level rules live in spm_chat_rules (via proxy/rag/chat_rules.py); room
# rules live in the world engine. 'location' is the API-level level for the
# engine path; chat_rules itself knows only its three DB levels.

_CHAT_RULE_LEVELS = ("world", "character_internal", "character_external", "location")
_DB_RULE_FIELDS = ("rule_text", "rule_type", "level", "character_id", "status")
_ROOM_RULE_FIELDS = ("rule_text", "rule_type", "status")


def _rules_db_503() -> JSONResponse:
    return JSONResponse(status_code=503,
                        content={"error": strings.get("api.errors.db_unavailable")})


def _engine_error_response(e: Exception, not_found_detail: str) -> JSONResponse:
    """Map an engine client failure: 404 -> 404, 422 -> 422, anything else -> 500."""
    import httpx
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 404:
            return JSONResponse(status_code=404, content={"error": not_found_detail})
        if code == 422:
            return JSONResponse(status_code=422, content={"error": str(e)})
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/chats")
async def list_chats():
    """All chats for the rules admin, newest activity first. A session shows up if
    it has perception rows OR chat rules (a chat that only ever got a manual rule
    still needs an entry point). Characters = who perceived something, minus the
    player; rule counts from spm_chat_rules; last activity = max created_at over
    both tables."""
    pool = get_admin_db_pool()
    if pool is None:
        return _rules_db_503()
    sql = """
        WITH sessions AS (
            SELECT DISTINCT session_id FROM spm_perception
            UNION
            SELECT DISTINCT session_id FROM spm_chat_rules
        ),
        chars AS (
            SELECT session_id,
                   array_agg(DISTINCT recipient_id ORDER BY recipient_id) AS characters
            FROM spm_perception
            WHERE recipient_id <> 'user'
            GROUP BY session_id
        ),
        counts AS (
            SELECT session_id,
                   COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                   COUNT(*) FILTER (WHERE status = 'active') AS active
            FROM spm_chat_rules
            GROUP BY session_id
        ),
        activity AS (
            SELECT session_id, MAX(created_at) AS last_activity FROM (
                SELECT session_id, created_at FROM spm_perception
                UNION ALL
                SELECT session_id, created_at FROM spm_chat_rules
            ) all_rows
            GROUP BY session_id
        )
        SELECT s.session_id, COALESCE(c.characters, '{}'::text[]) AS characters,
               COALESCE(n.pending, 0) AS pending, COALESCE(n.active, 0) AS active,
               a.last_activity
        FROM sessions s
        LEFT JOIN chars c USING (session_id)
        LEFT JOIN counts n USING (session_id)
        LEFT JOIN activity a USING (session_id)
        ORDER BY a.last_activity DESC NULLS LAST
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
    except Exception as e:
        logger.error(f"[AdminAPI] list chats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content=[
        {
            "session_id": r["session_id"],
            "characters": list(r["characters"]),
            "rules": {"pending": r["pending"], "active": r["active"]},
            "last_activity": r["last_activity"].isoformat() if r["last_activity"] else None,
        }
        for r in rows])


@router.get("/chats/{session_id}/rules")
async def get_chat_rules_view(session_id: str):
    """One chat's rules, grouped for the admin UI: world rules, per-character
    internal/external rules, and the room rules from the world engine snapshot
    (all rooms, including rooms without rules). An engine outage still returns
    the DB rules with locations=[] and an engine_error message."""
    pool = get_admin_db_pool()
    if pool is None:
        return _rules_db_503()
    from proxy.rag import chat_rules
    from proxy.api.routes import evennia_client
    try:
        async with pool.acquire() as conn:
            rules = await chat_rules.list_rules(conn, session_id)   # includes created_at
    except Exception as e:
        logger.error(f"[AdminAPI] rules view failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    world: list = []
    characters: Dict[str, Dict[str, list]] = {}
    for rule in rules:
        if rule["level"] == "world":
            world.append(rule)
            continue
        bucket = characters.setdefault(rule["character_id"],
                                       {"internal": [], "external": []})
        bucket["internal" if rule["level"] == "character_internal"
               else "external"].append(rule)

    content: Dict[str, Any] = {"world": world, "characters": characters,
                               "locations": []}
    try:
        snap = await evennia_client.get_snapshot(session_id=session_id,
                                                 template_key="")
        content["locations"] = [
            {"room_id": room.get("room_id"), "name": room.get("name"),
             "rules": list(room.get("rules") or [])}
            for room in snap.get("rooms", [])]
    except Exception as e:
        logger.warning(f"[AdminAPI] world engine unavailable for {session_id}: {e}")
        content["engine_error"] = str(e)
    return JSONResponse(content=content)


@router.post("/chats/{session_id}/rules")
async def add_chat_rule(session_id: str, body: dict):
    """Add a manual rule to a chat (status='active', source='manual'). Level
    'location' goes to the world engine as a room rule; the other three levels
    go to spm_chat_rules. Manual rules carry no embedding — trigger rules then
    match only via text-free paths (that is accepted)."""
    level = str(body.get("level") or "")
    if level not in _CHAT_RULE_LEVELS:
        return JSONResponse(status_code=422, content={
            "error": f"level must be one of {list(_CHAT_RULE_LEVELS)}"})
    rule_text = str(body.get("rule_text") or "")
    if not rule_text.strip():
        return JSONResponse(status_code=422, content={"error": "rule_text must not be empty"})
    rule_type = str(body.get("rule_type") or "invariant")
    from proxy.rag import chat_rules
    if rule_type not in chat_rules.RULE_TYPES:
        return JSONResponse(status_code=422, content={
            "error": f"rule_type must be one of {list(chat_rules.RULE_TYPES)}"})

    if level == "location":
        room_id = body.get("room_id")
        if not room_id:
            return JSONResponse(status_code=422,
                                content={"error": "level 'location' requires a room_id"})
        from proxy.api.routes import evennia_client
        try:
            res = await evennia_client.add_room_rule(
                str(room_id), rule_text, session_id, rule_type=rule_type,
                status="active", source="manual", origin="user")
        except Exception as e:
            return _engine_error_response(e, f"room {room_id} not found")
        if res.get("duplicate"):
            return JSONResponse(status_code=409,
                                content={"error": "duplicate of a rule already on this room"})
        return JSONResponse(content={"id": res.get("rule_id"), "level": "location"})

    pool = get_admin_db_pool()
    if pool is None:
        return _rules_db_503()
    try:
        async with pool.acquire() as conn:
            rule_id = await chat_rules.add_rule(
                conn, session_id=session_id, level=level,
                character_id=body.get("character_id"), rule_text=rule_text,
                rule_type=rule_type, status="active", source="manual")
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:
        logger.error(f"[AdminAPI] add chat rule failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if rule_id is None:
        return JSONResponse(status_code=409,
                            content={"error": "duplicate of an existing rule in this chat"})
    return JSONResponse(content={"id": rule_id, "level": level})


@router.put("/chats/{session_id}/rules/{rule_id}")
async def edit_chat_rule(session_id: str, rule_id: str, body: dict,
                         room_id: Optional[str] = None):
    """Edit a rule. Body may carry any of rule_text, rule_type, level,
    character_id, status. Pass ?room_id= to edit a room rule in the world
    engine instead of a chat rule."""
    from proxy.rag import chat_rules
    if room_id:
        fields = {k: body[k] for k in _ROOM_RULE_FIELDS if k in body}
        if not fields:
            return JSONResponse(status_code=422, content={
                "error": f"nothing to update; expected any of {list(_ROOM_RULE_FIELDS)}"})
        from proxy.api.routes import evennia_client
        try:
            await evennia_client.update_room_rule(room_id, rule_id, session_id, **fields)
        except Exception as e:
            return _engine_error_response(e, "room rule not found")
        return JSONResponse(content={"status": "success", "id": rule_id,
                                     "level": "location"})

    pool = get_admin_db_pool()
    if pool is None:
        return _rules_db_503()
    fields = {k: body[k] for k in _DB_RULE_FIELDS if k in body}
    if not fields:
        return JSONResponse(status_code=422, content={
            "error": f"nothing to update; expected any of {list(_DB_RULE_FIELDS)}"})
    try:
        async with pool.acquire() as conn:
            ok = await chat_rules.update_rule(conn, session_id, rule_id, **fields)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:
        logger.error(f"[AdminAPI] update chat rule failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if not ok:
        return JSONResponse(status_code=404, content={"error": "rule not found"})
    return JSONResponse(content={"status": "success", "id": rule_id})


@router.delete("/chats/{session_id}/rules/{rule_id}")
async def remove_chat_rule(session_id: str, rule_id: str,
                           room_id: Optional[str] = None):
    """Delete a rule. Pass ?room_id= to delete a room rule from the world
    engine instead of a chat rule."""
    if room_id:
        from proxy.api.routes import evennia_client
        try:
            await evennia_client.delete_room_rule(room_id, rule_id, session_id)
        except Exception as e:
            return _engine_error_response(e, "room rule not found")
        return JSONResponse(content={"status": "success"})

    pool = get_admin_db_pool()
    if pool is None:
        return _rules_db_503()
    from proxy.rag import chat_rules
    try:
        async with pool.acquire() as conn:
            deleted = await chat_rules.delete_rule(conn, session_id, rule_id)
    except Exception as e:
        logger.error(f"[AdminAPI] delete chat rule failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "rule not found"})
    return JSONResponse(content={"status": "success"})
