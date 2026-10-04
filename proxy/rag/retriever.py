"""Episodic Memory RAG Retriever & Game AI-Inspired Decay Scoring Engine.

Two recall paths (embeddings plan phase 3):

- **Vector** (a query embedding exists): pgvector cosine distance, filtered to
  the active embedding space — rows with a NULL embedding or a vector from
  another space are invisible, so vectors from different models are never
  compared.
- **Full-text fallback** (no query embedding: provider 'none' or an outage):
  Postgres full-text search (`fts` generated column, 'simple' config) over the
  same session, ranked by ts_rank and recency, returning the same node shape
  with rag_score derived from the normalised rank.

Both paths apply the exponential time decay and importance scoring.
"""

import math
import logging
import asyncpg
from typing import List, Dict, Any, Optional
from core.resource_manager import strings
from core.identifiers import safe_char_id

logger = logging.getLogger(__name__)


class EpisodicRAGRetriever:
    def __init__(self, db_pool: asyncpg.Pool, decay_lambda: float = 0.01):
        self.db_pool = db_pool
        self.decay_lambda = decay_lambda

    def _score(self, cosine_or_rank_sim: float, delta_hours: float,
               importance: int, access_count: int) -> float:
        """RAG Score = similarity * exp(-lambda * dt) * (1 + importance/10) * log-access."""
        time_decay = math.exp(-self.decay_lambda * delta_hours)
        importance_weight = 1.0 + (importance / 10.0)
        access_mult = 1.0 + math.log1p(access_count)
        return cosine_or_rank_sim * time_decay * importance_weight * access_mult

    async def _bump_access(self, conn, table_name: str, nodes: List[Dict[str, Any]]) -> None:
        if not nodes:
            return
        node_ids = [m["id"] for m in nodes]
        await conn.execute(f"""
            UPDATE {table_name}
            SET access_count = access_count + 1, last_accessed_at = NOW()
            WHERE id = ANY($1::uuid[]);
        """, node_ids)

    async def retrieve_memories(
        self,
        character_id: str,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        max_cosine_distance: float = 0.35,
        session_id: str = "default_session",
        query_text: str = "",
        embedding_space_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top K relevant episodic memory nodes from csa_memory_{character_id}.
        Strictly filters by session_id for zero-bleed session isolation.
        Filters out query_text exact matches to prevent regenerated turn bleedthrough.

        With a query embedding: cosine recall within `embedding_space_id`
        (None matches only legacy rows whose space is NULL).
        Without one: full-text fallback over `query_text`; empty query_text
        returns [] (nothing to match on).
        """
        clean_query = query_text.strip().lower() if query_text else ""
        if query_embedding is None:
            if not clean_query:
                logger.info(
                    f"[RAGRetriever] No query embedding and no query text; skipping recall for {character_id}."
                )
                return []
            return await self._retrieve_memories_fts(
                character_id, clean_query, top_k, session_id
            )

        table_name = f"csa_memory_{safe_char_id(character_id)}"
        async with self.db_pool.acquire() as conn:
            # Ensure table exists (and is lazily upgraded to the current shape)
            await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(character_id))

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            # The inner query (OFFSET 0 fence) filters by space BEFORE the
            # distance operator runs: a dimension-free vector column may hold
            # different dimensions per space, and <=> errors on a mismatch.
            query = f"""
                SELECT id, sensory_input, inner_monologue, is_core_memory, importance_score, access_count,
                       (episodic_embedding <=> $1::vector) AS cosine_distance,
                       delta_hours
                FROM (
                    SELECT id, sensory_input, inner_monologue, is_core_memory, importance_score,
                           access_count, episodic_embedding,
                           EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0 AS delta_hours
                    FROM {table_name}
                    WHERE episodic_embedding IS NOT NULL
                      AND embedding_space_id IS NOT DISTINCT FROM $5
                      AND session_id = $3
                      AND ($4 = '' OR LOWER(sensory_input) != $4)
                    OFFSET 0
                ) candidates
                WHERE (episodic_embedding <=> $1::vector) < $2
                ORDER BY cosine_distance ASC
                LIMIT 20;
            """
            records = await conn.fetch(
                query, embedding_str, max_cosine_distance, session_id, clean_query,
                embedding_space_id,
            )

            scored_nodes = []
            for r in records:
                cosine_dist = float(r['cosine_distance'])
                sim_score = max(0.0, 1.0 - cosine_dist)
                rag_score = self._score(
                    sim_score, float(r['delta_hours']),
                    int(r['importance_score']), int(r['access_count']),
                )
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

            await self._bump_access(conn, table_name, top_memories)

            logger.info(f"[RAGRetriever] Retrieved {len(top_memories)} memory nodes for {character_id} (session={session_id}).")
            return top_memories

    async def _retrieve_memories_fts(
        self,
        character_id: str,
        clean_query: str,
        top_k: int,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Degraded recall: lexical full-text search over the same session.

        Uses the 'simple' config (language-agnostic) over the stored turn
        (sensory_input + public_response). rag_score reuses the decay formula
        with the batch-normalised ts_rank as the similarity term; FTS scores
        are not comparable with cosine scores, so cosine_distance is None.
        """
        table_name = f"csa_memory_{safe_char_id(character_id)}"
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(character_id))
            query = f"""
                SELECT id, sensory_input, inner_monologue, is_core_memory, importance_score, access_count,
                       ts_rank(fts, plainto_tsquery('simple', $1)) AS rank,
                       EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0 AS delta_hours
                FROM {table_name}
                WHERE session_id = $2
                  AND fts @@ plainto_tsquery('simple', $1)
                  AND LOWER(sensory_input) != $1
                ORDER BY rank DESC, timestamp DESC
                LIMIT 20;
            """
            records = await conn.fetch(query, clean_query, session_id)

            max_rank = max((float(r['rank']) for r in records), default=0.0)
            scored_nodes = []
            for r in records:
                rank = float(r['rank'])
                sim_score = (rank / max_rank) if max_rank > 0 else 0.0
                rag_score = self._score(
                    sim_score, float(r['delta_hours']),
                    int(r['importance_score']), int(r['access_count']),
                )
                scored_nodes.append({
                    "id": str(r['id']),
                    "sensory_input": r['sensory_input'],
                    "inner_monologue": r['inner_monologue'],
                    "is_core_memory": r['is_core_memory'],
                    "cosine_distance": None,
                    "rag_score": rag_score,
                })

            scored_nodes.sort(key=lambda x: x["rag_score"], reverse=True)
            top_memories = scored_nodes[:top_k]

            await self._bump_access(conn, table_name, top_memories)

            logger.info(
                f"[RAGRetriever] Degraded recall (full-text) returned {len(top_memories)} nodes "
                f"for {character_id} (session={session_id})."
            )
            return top_memories

    async def retrieve_lore_rules(
        self,
        character_id: str,
        query_embedding: Optional[List[float]],
        max_cosine_distance: float = 0.35,
        embedding_space_id: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves active lore rules for a character.
        Returns all 'invariant' rules, and 'conditional_trigger'/'game_over' rules that match
        the query embedding within the active space. Trigger matching stays vector-only:
        without a query embedding (provider 'none' or an outage) the invariants still
        apply and triggers are skipped.
        """
        table_name = f"csa_lore_rules_{safe_char_id(character_id)}"
        async with self.db_pool.acquire() as conn:
            # Ensure table exists (create if missing)
            await conn.execute("SELECT create_csa_lore_rules_table($1);", safe_char_id(character_id))

            # Fetch Invariants (never depend on vectors)
            invariants_query = f"SELECT id, rule_text, rule_type FROM {table_name} WHERE rule_type = 'invariant' AND status = 'active';"
            invariant_records = await conn.fetch(invariants_query)

            trigger_records = []
            if query_embedding is not None:
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
                # Same space filter + OFFSET 0 fence as memory recall.
                triggers_query = f"""
                    SELECT id, rule_text, rule_type,
                           (rule_embedding <=> $1::vector) AS cosine_distance
                    FROM (
                        SELECT id, rule_text, rule_type, rule_embedding
                        FROM {table_name}
                        WHERE rule_type IN ('conditional_trigger', 'game_over')
                          AND rule_embedding IS NOT NULL
                          AND embedding_space_id IS NOT DISTINCT FROM $3
                          AND status = 'active'
                        OFFSET 0
                    ) candidates
                    WHERE (rule_embedding <=> $1::vector) < $2
                    ORDER BY cosine_distance ASC;
                """
                trigger_records = await conn.fetch(
                    triggers_query, embedding_str, max_cosine_distance, embedding_space_id
                )
            else:
                logger.info(
                    f"[RAGRetriever] No query embedding: lore triggers skipped for {character_id} "
                    f"(invariants still apply)."
                )

            return {
                "invariants": [dict(r) for r in invariant_records],
                "triggers": [dict(r) for r in trigger_records]
            }
