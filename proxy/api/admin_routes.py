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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/api/v1", tags=["admin"])

# Global db_pool reference passed from routes.py / main.py
_admin_db_pool = None


def set_admin_db_pool(pool):
    global _admin_db_pool
    _admin_db_pool = pool


@router.get("/stats")
async def get_admin_stats():
    """Return live system telemetry, active sessions, and database size."""
    telemetry = get_telemetry_collector()
    stats = telemetry.get_metrics(_admin_db_pool)

    # If DB pool is available, fetch actual DB table stats
    db_size_mb = 0.0
    if _admin_db_pool is not None:
        try:
            async with _admin_db_pool.acquire() as conn:
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


@router.post("/config")
async def update_config(new_settings: Dict[str, Any]):
    """Update dynamic settings in config.json."""
    mgr = get_settings_manager()
    updated = mgr.write_settings(new_settings)
    return JSONResponse(content={"status": "success", "settings": updated})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Hard delete all records for a specific session across all memory tables."""
    if _admin_db_pool is None:
        return JSONResponse(
            status_code=530,
            content={"status": "error", "message": "Database connection unavailable"}
        )

    deleted_count = 0
    try:
        async with _admin_db_pool.acquire() as conn:
            # Delete from spm_chat_imports
            await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id)
            # Delete from spm_cold_archives
            await conn.execute("DELETE FROM spm_cold_archives WHERE session_id = $1;", session_id)

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
    """Truncate all character memory tables, bulk imports, and cold archives."""
    if _admin_db_pool is None:
        return JSONResponse(
            status_code=530,
            content={"status": "error", "message": "Database connection unavailable"}
        )

    try:
        async with _admin_db_pool.acquire() as conn:
            # Truncate tracking tables
            await conn.execute("TRUNCATE TABLE spm_chat_imports RESTART IDENTITY CASCADE;")
            await conn.execute("TRUNCATE TABLE spm_cold_archives RESTART IDENTITY CASCADE;")

            # Truncate all csa_memory tables
            tables = await conn.fetch(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_name LIKE 'csa_memory_%';"""
            )
            for t in tables:
                table_name = t["table_name"]
                await conn.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")

        logger.warning("[AdminAPI] FACTORY RESET TRIGGERED — all memory tables truncated.")
        return JSONResponse(content={
            "status": "success",
            "message": "Factory reset complete. All character memories and session data truncated."
        })
    except Exception as e:
        logger.error(f"[AdminAPI] Error executing factory reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
