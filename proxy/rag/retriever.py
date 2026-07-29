"""
Episodic Memory RAG Retriever & Game AI-Inspired Decay Scoring Engine.
Combines pgvector cosine distance search (<=> operator < 0.35) with exponential time decay and importance scoring.
"""

import math
import logging
import asyncpg
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EpisodicRAGRetriever:
    def __init__(self, db_pool: asyncpg.Pool, decay_lambda: float = 0.01):
        self.db_pool = db_pool
        self.decay_lambda = decay_lambda

    async def retrieve_memories(
        self,
        character_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        max_cosine_distance: float = 0.35,
        session_id: str = "default_session"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top K relevant episodic memory nodes from csa_memory_{character_id}.
        Strictly filters by session_id for zero-bleed session isolation.
        Calculates RAG Score = (1 - cosine_dist) * exp(-lambda * delta_t) * (1 + importance/10) * access_count.
        """
        table_name = f"csa_memory_{character_id.lower()}"
        async with self.db_pool.acquire() as conn:
            # Ensure table exists
            await conn.execute("SELECT create_csa_memory_table($1);", character_id.lower())

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            query = f"""
                SELECT id, sensory_input, inner_monologue, is_core_memory, importance_score, access_count,
                       (episodic_embedding <=> $1::vector) AS cosine_distance,
                       EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0 AS delta_hours
                FROM {table_name}
                WHERE episodic_embedding IS NOT NULL
                  AND (episodic_embedding <=> $1::vector) < $2
                  AND session_id = $3
                ORDER BY cosine_distance ASC
                LIMIT 20;
            """
            records = await conn.fetch(query, embedding_str, max_cosine_distance, session_id)

            scored_nodes = []
            for r in records:
                cosine_dist = float(r['cosine_distance'])
                delta_t = float(r['delta_hours'])
                importance = int(r['importance_score'])
                access_count = int(r['access_count'])

                # Compute Game AI-inspired RAG decay score
                sim_score = max(0.0, 1.0 - cosine_dist)
                time_decay = math.exp(-self.decay_lambda * delta_t)
                importance_weight = 1.0 + (importance / 10.0)
                access_mult = 1.0 + math.log1p(access_count)

                rag_score = sim_score * time_decay * importance_weight * access_mult

                scored_nodes.append({
                    "id": str(r['id']),
                    "sensory_input": r['sensory_input'],
                    "inner_monologue": r['inner_monologue'],
                    "is_core_memory": r['is_core_memory'],
                    "cosine_distance": cosine_dist,
                    "rag_score": rag_score
                })

            # Sort by RAG score descending and return Top K
            scored_nodes.sort(key=lambda x: x["rag_score"], reverse=True)
            top_memories = scored_nodes[:top_k]

            # Update access counts asynchronously
            if top_memories:
                node_ids = [m["id"] for m in top_memories]
                await conn.execute(f"""
                    UPDATE {table_name}
                    SET access_count = access_count + 1, last_accessed_at = NOW()
                    WHERE id = ANY($1::uuid[]);
                """, node_ids)

            logger.info(f"[RAGRetriever] Retrieved {len(top_memories)} memory nodes for {character_id} (session={session_id}).")
            return top_memories
