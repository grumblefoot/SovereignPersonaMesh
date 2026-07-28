"""
Main Entry Point for Sovereign Persona Mesh (SPM) FastAPI Proxy (Port 5050).
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from proxy.api.routes import router as api_router
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
