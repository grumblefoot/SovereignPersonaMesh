"""
Nightly Memory Consolidation Script ("The Sleep Cycle").
Executed via systemd user timer daily at 3:00 AM.
Summarizes subjective sensory logs into single-sentence episodic memory nodes using Gemma 9B,
commits core memory nodes with is_core_memory=TRUE, and prunes expired volatile logs.
"""

import os
import asyncio
import logging
import asyncpg
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SLEEP_CYCLE_PROMPT_TEMPLATE = """<boss>You are the Memory Consolidation Engine for the character {character_id}.
Review the chronological logs of their sensory inputs and inner monologues from the past 24 hours. Synthesize them into a single-sentence episodic memory node.
- Maintain a first-person perspective ("I felt...", "I saw...").
- Keep emotional states, specific locations, and key discoveries intact.
- Avoid generic summaries.

Daily Logs:
{daily_logs}

Summary Node:
<idle>"""


class MemoryConsolidationWorker:
    def __init__(self, db_config: Dict[str, Any], consolidation_model_url: str):
        self.db_config = db_config
        self.consolidation_model_url = consolidation_model_url

    async def _call_consolidation_model(self, prompt: str) -> str:
        """Call Gemma 9B via OpenAI-compatible chat completions endpoint."""
        url = f"{self.consolidation_model_url}/v1/chat/completions"
        payload = {
            "model": "google/gemma-4-9B-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            generated = data["choices"][0]["message"]["content"].strip()
        # Extract only the inner monologue / first sentence of consolidated memory
        # Strip any tags like <boss>, <idle>, etc.
        import re
        cleaned = re.sub(r"<[^>]+>", "", generated).strip()
        # Take first sentence only
        first_sentence = cleaned.split(".")[0] + "." if "." in cleaned else cleaned
        return first_sentence

    async def get_active_character_tables(self, conn: asyncpg.Connection) -> List[str]:
        """Fetch all csa_memory_* tables in PostgreSQL."""
        rows = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name LIKE 'csa_memory_%';
        """)
        return [r['table_name'] for r in rows]

    async def process_character_sleep_cycle(self, conn: asyncpg.Connection, table_name: str):
        """Run consolidation loop for a single character subagent table."""
        char_id = table_name.replace('csa_memory_', '')
        logger.info(f"[Sleep Cycle] Starting memory consolidation for character: {char_id} ({table_name})")

        # 1. Extraction: Fetch un-consolidated volatile memories from past 24h
        records = await conn.fetch(f"""
            SELECT id, sensory_input, inner_monologue, timestamp 
            FROM {table_name} 
            WHERE is_core_memory = FALSE 
              AND timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp ASC;
        """)

        if not records:
            logger.info(f"[Sleep Cycle] No un-consolidated logs found for {char_id}.")
            return

        # 2. Format Daily Logs
        log_lines = []
        for r in records:
            log_lines.append(f"[{r['timestamp']}] Sensory: {r['sensory_input']}")
            if r['inner_monologue']:
                log_lines.append(f"[{r['timestamp']}] Inner Thought: {r['inner_monologue']}")
        daily_logs_str = "\n".join(log_lines)

        # 3. Summarization: dispatch to Gemma 9B via HTTP
        logger.info(f"[Sleep Cycle] Dispatching {len(records)} log entries for {char_id} to Gemma 9B...")
        try:
            prompt = SLEEP_CYCLE_PROMPT_TEMPLATE.format(
                character_id=char_id,
                daily_logs=daily_logs_str,
            )
            summary_node = await self._call_consolidation_model(prompt)
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"[Sleep Cycle] Gemma 9B request failed for {char_id}: "
                f"{exc.response.status_code} {exc.response.text}"
            )
            summary_node = f"I processed sensory inputs from the last 24 hours."
        except Exception as exc:
            logger.warning(
                f"[Sleep Cycle] Gemma 9B call failed for {char_id}, using fallback: {exc}"
            )
            summary_node = f"I processed sensory inputs from the last 24 hours."

        # 4. Commit core memory node & Prune volatile entries
        async with conn.transaction():
            # Insert core memory summary node
            await conn.execute(f"""
                INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, is_core_memory, is_subjective, importance_score)
                VALUES ('sleep_cycle_consolidated', $1, $2, TRUE, TRUE, 8);
            """, summary_node, f"Nightly consolidation summary for {char_id}")

            # Prune volatile logs (NEVER prune is_core_memory = TRUE)
            prune_count = await conn.execute(f"""
                DELETE FROM {table_name} 
                WHERE is_core_memory = FALSE 
                  AND timestamp < NOW() - INTERVAL '24 hours';
            """)
            logger.info(f"[Sleep Cycle] Successfully consolidated {char_id}. Pruned old volatile logs: {prune_count}")

    async def run(self):
        """Main execution entry point."""
        logger.info("Starting SPM Nightly Memory Consolidation Pipeline...")
        conn = await asyncpg.connect(**self.db_config)
        try:
            tables = await self.get_active_character_tables(conn)
            for table in tables:
                await self.process_character_sleep_cycle(conn, table)
        finally:
            await conn.close()
        logger.info("SPM Nightly Memory Consolidation Pipeline Complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_conf = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", 5432),
        "user": os.getenv("POSTGRES_USER", "spm_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "spm_secure_password"),
        "database": os.getenv("POSTGRES_DB", "litellm_postgres"),
    }
    worker = MemoryConsolidationWorker(db_conf, os.getenv("LLM_BACKEND_URL", "http://localhost:13305/v1"))
    asyncio.run(worker.run())
