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
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from config.manager import get_settings_manager
from proxy.core.telemetry import get_telemetry_collector
from core.resource_manager import strings

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
                size_bytes = await conn.fetchval(
                    "SELECT pg_database_size(current_database());"
                )
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
            await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id)
            # Delete from spm_cold_archives
            await conn.execute("DELETE FROM spm_cold_archives WHERE session_id = $1;", session_id)
            # Delete from world_state_sessions
            await conn.execute("DELETE FROM world_state_sessions WHERE session_id = $1;", session_id)

            # Query all csa_memory tables
            tables = await conn.fetch(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_name LIKE 'csa_memory_%';"""
            )
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
            await conn.execute("TRUNCATE TABLE spm_chat_imports RESTART IDENTITY CASCADE;")
            await conn.execute("TRUNCATE TABLE spm_cold_archives RESTART IDENTITY CASCADE;")
            await conn.execute("TRUNCATE TABLE world_state_sessions RESTART IDENTITY CASCADE;")

            # Truncate all csa_memory tables
            tables = await conn.fetch(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_name LIKE 'csa_memory_%';"""
            )
            for t in tables:
                table_name = t["table_name"]
                await conn.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")

        # Reset telemetry metrics, active sessions, and request logs
        telemetry = get_telemetry_collector()
        telemetry.reset()

        # Clear Evennia in-memory cache
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.delete("http://localhost:4005/api/v1/world/admin/reset")
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

@router.post("/shutdown")
async def shutdown_system():
    """Cleanly shut down the SPM proxy and Evennia game services."""
    import os, signal, asyncio, subprocess
    logger.warning("[AdminAPI] Received shutdown request. Shutting down services...")

    # Run a background task to cleanly exit after responding
    async def _shutdown():
        await asyncio.sleep(1)
        # Stop Postgres Docker container
        subprocess.run(['docker', 'stop', 'spm-postgres'], check=False)
        # Send SIGTERM to evennia_world.app
        subprocess.run(['pkill', '-TERM', '-f', r'python -m evennia_world.app'], check=False)
        # Send SIGTERM to proxy.main (kills master process)
        subprocess.run(['pkill', '-TERM', '-f', r'python -m proxy.main'], check=False)
        # Also kill any orphaned Uvicorn multiprocessing workers
        subprocess.run(['pkill', '-TERM', '-f', r'multiprocessing.spawn'], check=False)

    asyncio.create_task(_shutdown())
    
    return JSONResponse(content={
        "status": "success",
        "message": "Shutting down SPM and Game services gracefully."
    })
