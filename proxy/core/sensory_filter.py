"""
Observer Inference Gating & Bypass Protocol.
Bypasses active LLM generation for characters in Blackout or Null gating states,
committing deterministic ambient logs directly to PostgreSQL with zero GPU inference cost.
"""

import logging
import asyncpg
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ObserverInferenceGatingFilter:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def evaluate_and_bypass(
        self,
        character_id: str,
        session_id: str,
        gating_level: str,
        sensory_feed: str
    ) -> bool:
        """
        Evaluates whether character turn can bypass active LLM inference.
        Returns True if bypassed (committed directly to DB with zero inference cost), False if LLM generation required.
        """
        gating_lower = gating_level.lower()
        if gating_lower in ["null", "blackout"]:
            logger.info(f"[InferenceBypass] Character {character_id} is in {gating_level} state. Bypassing LLM generation.")
            await self._commit_ambient_log(character_id, session_id, sensory_feed or "No sensory feed observed.")
            return True
        return False

    async def _commit_ambient_log(self, character_id: str, session_id: str, ambient_text: str):
        """Commits deterministic ambient log directly to csa_memory_{character_id} without LLM inference."""
        table_name = f"csa_memory_{character_id.lower()}"
        async with self.db_pool.acquire() as conn:
            # Ensure table exists
            await conn.execute("SELECT create_csa_memory_table($1);", character_id.lower())
            await conn.execute(f"""
                INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, is_core_memory, is_subjective, importance_score)
                VALUES ($1, $2, NULL, FALSE, TRUE, 1);
            """, session_id, ambient_text)
            logger.info(f"[InferenceBypass] Successfully committed deterministic ambient log to {table_name}.")
