"""
Main Entry Point for Sovereign Persona Mesh (SPM) FastAPI Proxy (Port 5050).
"""

import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from proxy.api.routes import router as api_router
from config.hardware_tiers import get_hardware_config
from proxy.core.logger import setup_spm_logging
from proxy.core.telemetry import TelemetryCollector
from proxy.api.routes import router as api_router
from proxy.api.admin_routes import router as admin_router
from config.hardware_tiers import get_hardware_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SPMProxyMain")

app = FastAPI(
    title="Sovereign Persona Mesh (SPM) Proxy",
    description="High-performance, model-agnostic multi-agent OpenAI emulation proxy with sensory gating and private memory isolation.",
    version="0.2.0"
)

# CORS middleware for SillyTavern web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(admin_router)

_UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
if os.path.exists(_UI_DIR):
    app.mount("/proxy/ui", StaticFiles(directory=_UI_DIR), name="static_ui")

@app.get("/admin")
async def serve_admin_dashboard():
    index_file = os.path.join(_UI_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"error": "Admin UI index.html not found"}


@app.get("/")
async def root():
    tier_name = os.getenv("SPM_HARDWARE_TIER", "SOVEREIGN")
    hw_config = get_hardware_config(tier_name)
    return {
        "status": "online",
        "service": "Sovereign Persona Mesh Proxy",
        "port": 5050,
        "hardware_tier": tier_name,
        "max_context_tokens": hw_config.max_context_tokens,
        "gtt_vram_budget_gb": hw_config.gtt_vram_budget_gb
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    import asyncpg
    from proxy.api.routes import set_db_pool, _db_pool, _db_pool_explicitly_set
    from proxy.api.admin_routes import set_admin_db_pool, _admin_db_pool

    if _db_pool_explicitly_set or (_db_pool is not None and _admin_db_pool is not None):
        return

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "litellm_postgres")
    user = os.getenv("POSTGRES_USER", "spm_user")
    pwd = os.getenv("POSTGRES_PASSWORD", "spm_secure_password")
    dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    try:
        pool = await asyncio.wait_for(asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5), timeout=2.0)
        app.state.db_pool = pool
        if _db_pool is None:
            set_db_pool(pool)
        if _admin_db_pool is None:
            set_admin_db_pool(pool)
        
        # Hydrate telemetry buffers from the DB
        from proxy.core.telemetry import get_telemetry_collector
        await get_telemetry_collector().hydrate_from_db(pool)
        
        logger.info("[SPMProxyMain] Database connection pool initialized and injected into routes & admin routes.")
    except Exception as e:
        logger.warning(f"[SPMProxyMain] Could not initialize DB pool on startup: {e}")



@app.on_event("shutdown")
async def shutdown_event():
    import proxy.api.routes
    import proxy.api.admin_routes
    if hasattr(app.state, "db_pool") and app.state.db_pool:
        try:
            await app.state.db_pool.close()
        except Exception:
            pass
        app.state.db_pool = None
    proxy.api.routes._db_pool = None
    proxy.api.routes._db_pool_explicitly_set = False
    proxy.api.admin_routes._admin_db_pool = None
    logger.info("[SPMProxyMain] Database pool closed.")



if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SPM_HOST", "0.0.0.0")
    port = int(os.getenv("SPM_PORT", 5050))
    logger.info(f"Starting SPM Proxy Server on http://{host}:{port}")
    uvicorn.run("proxy.main:app", host=host, port=port, reload=True)
