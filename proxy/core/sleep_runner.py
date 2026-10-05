"""
Sleep cycle inside the proxy scheduler (decision 15 / fifo_queue.md phase 5).

The nightly memory consolidation used to be a systemd timer running
scripts/sleep_cycle.py with its own HTTP client, able to collide with live chat
for the single backend slot. Here it runs as a proxy background task whose LLM
calls go through the shared scheduler at P3 (the lowest lane): a chat arrival
preempts a running consolidation call, and the coalescing key means a preempted
batch simply re-runs on the next pass. The systemd path still works for
operators who prefer it; set sleep_cycle_enabled=false here if the timer stays.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SLEEP_SESSION = "spm-sleep-cycle"


async def run_sleep_cycle_once() -> None:
    """One consolidation pass, LLM calls routed through the scheduler."""
    from config.manager import get_settings_manager
    from proxy.core.llm_scheduler import get_scheduled_client
    from scripts.sleep_cycle import MemoryConsolidationWorker

    settings = get_settings_manager().get_settings()
    model = os.getenv("CONSOLIDATION_MODEL_NAME",
                      settings.get("alternate_extraction_model_name") or "Gemma-4-E4B-it-GGUF")
    client = get_scheduled_client()

    async def scheduled_call(prompt: str) -> str:
        chunks = []
        async for c in client.generate_stream(
                messages=[{"role": "user", "content": prompt}],
                model=model, max_tokens=256, temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                job_kind="sleep", session_id=SLEEP_SESSION,
                coalesce_key="sleep-cycle"):
            chunks.append(c)
        return "".join(chunks)

    worker = MemoryConsolidationWorker(db_config_from_env(), settings.get("BACKEND_LLM_URL", ""),
                                       llm_call=scheduled_call)
    await worker.run()


def db_config_from_env() -> dict:
    """The SAME POSTGRES_* variables and defaults the proxy (proxy/main.py) and the
    standalone scripts/sleep_cycle.py use. The first version of this runner read
    DB_HOST/DB_PORT with a 5435 default, so the nightly pass could never connect
    (QA 2026-10-05)."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "spm_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "spm_secure_password"),
        "database": os.getenv("POSTGRES_DB", "litellm_postgres"),
    }


def _seconds_until(hour: int) -> float:
    now = datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def sleep_cycle_loop() -> None:
    """Lifespan task: run the consolidation daily at sleep_cycle_hour."""
    from config.manager import get_settings_manager
    while True:
        settings = get_settings_manager().get_settings()
        if not settings.get("sleep_cycle_enabled", True):
            await asyncio.sleep(3600)
            continue
        hour = int(settings.get("sleep_cycle_hour", 4))
        wait = _seconds_until(hour)
        logger.info(f"[SleepCycle] next consolidation pass in {wait/3600:.1f} h (at {hour:02d}:00)")
        await asyncio.sleep(wait)
        try:
            logger.info("[SleepCycle] starting consolidation pass (scheduler P3 lane)")
            await run_sleep_cycle_once()
            logger.info("[SleepCycle] consolidation pass complete")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[SleepCycle] consolidation pass failed: {e}")
