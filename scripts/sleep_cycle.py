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

        # 3. Summarization (Stub: Hermes will attach async HTTP request to Gemma 9B)
        logger.info(f"[Sleep Cycle] Dispatching {len(records)} log entries for {char_id} to Gemma 9B...")
        summary_node = f"I explored the location and processed sensory inputs from the last 24 hours."

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
