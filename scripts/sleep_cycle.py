"""
Nightly Memory Consolidation Script ("The Sleep Cycle").
Executed via systemd user timer daily at 3:00 AM.
Summarizes volatile logs older than 24h into single-sentence core memory nodes (is_core_memory=TRUE)
per session and day, then deletes only the logs that were summarized. Failed summaries change nothing.
Run as a module from the repo root: python -m scripts.sleep_cycle
"""

import os
import asyncio
import logging
import asyncpg
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from core.resource_manager import strings
from scripts.onnx_embedder import CPUEmbeddingEngine


class MemoryConsolidationWorker:
    def __init__(self, db_config: Dict[str, Any], consolidation_model_url: str, embedder=None):
        self.db_config = db_config
        self.consolidation_model_url = consolidation_model_url
        self.embedder = embedder or CPUEmbeddingEngine()

    async def _call_consolidation_model(self, prompt: str) -> str:
        """Call Consolidation Model via OpenAI-compatible chat completions endpoint."""
        base = self.consolidation_model_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        payload = {
            "model": os.getenv("CONSOLIDATION_MODEL_NAME", "Gemma-4-E4B-it-GGUF"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.3,
            # Gemma 4 GGUF reasons by default and spends the whole 256-token budget on reasoning_content,
            # leaving content empty. The summary is one sentence; no reasoning needed.
            "chat_template_kwargs": {"enable_thinking": False},
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
        """Consolidate one character's volatile logs into core memory nodes.

        Volatile logs stay raw for a 24h hot window. After that, each (session, day) batch is summarised
        into one core memory node in the same session, and only the logs in that batch are deleted, in
        the same transaction. If the summary fails, nothing is written or deleted and the batch is
        retried on the next run, so a failed night can never lose memories.
        """
        char_id = table_name.replace('csa_memory_', '')
        logger.info(f"[Sleep Cycle] Starting memory consolidation for character: {char_id} ({table_name})")

        # 1. Extraction: every volatile memory that has left the 24h hot window (includes any backlog)
        records = await conn.fetch(f"""
            SELECT id, session_id, sensory_input, inner_monologue, timestamp
            FROM {table_name}
            WHERE is_core_memory = FALSE
              AND timestamp < NOW() - INTERVAL '24 hours'
            ORDER BY timestamp ASC;
        """)

        if not records:
            logger.info(f"[Sleep Cycle] No un-consolidated logs found for {char_id}.")
            return

        # 2. Group by session (FR-001 isolation) and calendar day
        batches: Dict[tuple, List[Any]] = {}
        for r in records:
            batches.setdefault((r['session_id'], r['timestamp'].date()), []).append(r)

        for (session_id, day), batch in batches.items():
            await self._consolidate_batch(conn, table_name, char_id, session_id, day, batch)

    async def _consolidate_batch(self, conn, table_name, char_id, session_id, day, batch) -> bool:
        """Summarise one (session, day) batch. Returns True if it was committed."""
        log_lines = []
        for r in batch:
            log_lines.append(f"[{r['timestamp']}] Sensory: {r['sensory_input']}")
            if r['inner_monologue']:
                log_lines.append(f"[{r['timestamp']}] Inner Thought: {r['inner_monologue']}")

        # 3. Summarization via the consolidation model
        logger.info(f"[Sleep Cycle] Summarising {len(batch)} log entries for {char_id} (session={session_id}, day={day})...")
        try:
            prompt = strings.get("scripts.sleep_cycle.prompt_template",
                character_id=char_id,
                daily_logs="\n".join(log_lines),
            )
            summary_node = await self._call_consolidation_model(prompt)
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"[Sleep Cycle] Consolidation request failed for {char_id}: "
                f"{exc.response.status_code} {exc.response.text}. Keeping {len(batch)} logs for the next run."
            )
            return False
        except Exception as exc:
            logger.error(f"[Sleep Cycle] Consolidation call failed for {char_id}: {exc}. Keeping {len(batch)} logs for the next run.")
            return False

        if not summary_node or not summary_node.strip(". \n"):
            logger.error(f"[Sleep Cycle] Empty summary for {char_id}. Keeping {len(batch)} logs for the next run.")
            return False
        if "memory consolidation engine" in summary_node.lower():
            logger.error(f"[Sleep Cycle] Model echoed the instructions for {char_id}. Keeping {len(batch)} logs for the next run.")
            return False

        # Embed the summary so the retriever (which requires an embedding and the session) can find it
        embedding = await self.embedder.generate_embedding(summary_node)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        # 4. Commit the core memory node and delete exactly the logs it summarises
        async with conn.transaction():
            await conn.execute(f"""
                INSERT INTO {table_name} (session_id, timestamp, sensory_input, inner_monologue, episodic_embedding,
                                          is_core_memory, is_subjective, importance_score)
                VALUES ($1, $2, $3, $4, $5::vector, TRUE, TRUE, 8);
            """, session_id, batch[-1]['timestamp'], summary_node,
                f"Nightly consolidation summary for {char_id} ({day})", embedding_str)

            pruned = await conn.execute(
                f"DELETE FROM {table_name} WHERE id = ANY($1::uuid[]) AND is_core_memory = FALSE;",
                [r['id'] for r in batch],
            )
        logger.info(f"[Sleep Cycle] Consolidated {char_id} (session={session_id}, day={day}). Pruned: {pruned}")
        return True

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
