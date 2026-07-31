"""
Main Entry Point for Sovereign Persona Mesh (SPM) FastAPI Proxy (Port 5050).
"""

import os
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
    version="1.0.0"
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
        "hardware_tier": hw_config.tier.value,
        "max_context_tokens": hw_config.max_context_tokens,
        "gtt_vram_budget_gb": hw_config.gtt_vram_budget_gb
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SPM_HOST", "0.0.0.0")
    port = int(os.getenv("SPM_PORT", 5050))
    logger.info(f"Starting SPM Proxy Server on http://{host}:{port}")
    uvicorn.run("proxy.main:app", host=host, port=port, reload=True)
