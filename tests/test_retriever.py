"""
Unit tests for Episodic RAG Retriever (Phase 1: Database & Container Infrastructure).
Tests real PostgreSQL/pgvector with the litellm_postgres container.
"""
import numpy as np
import asyncpg


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


def _make_vector(n=3584):
    """Generate a normalized random vector.

    Returns a tuple: (pgvector_string, list_for_retriever)
    The string is used for direct SQL INSERT; the list is passed to the retriever.
    """
    np.random.seed(np.random.randint(0, 2**31))
    v = np.random.randn(n).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    pgv_str = "[" + ",".join(map(str, v)) + "]"
    return pgv_str, v.tolist()


# ── Phase 1: DB Infrastructure Tests ──────────────────────────────────────

def test_pool_connectivity():
    """Verify we can create a pool and execute queries via the pool."""
    import asyncio

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
            assert result == 1
        finally:
            await pool.close()

    asyncio.run(_test())


def test_csa_memory_tables_exist():
    """Verify csa_memory_* tables were created for all demo characters."""
    import asyncio

    async def _test():
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                "AND tablename LIKE 'csa_memory_%' ORDER BY tablename"
            )
        finally:
            await conn.close()

        table_names = [r["tablename"] for r in tables]
        expected = ["csa_memory_domino", "csa_memory_luna", "csa_memory_rowan", "csa_memory_seamus"]
        for name in expected:
            assert name in table_names, f"Missing table: {name}"

    asyncio.run(_test())


def test_create_csa_memory_table_function():
    """Verify the create_csa_memory_table() function creates tables dynamically."""
    import asyncio

    async def _test():
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            await conn.execute("SELECT create_csa_memory_table($1);", "test_phase1")
            exists = await conn.fetchval(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_tables "
                "WHERE schemaname='public' AND tablename='csa_memory_test_phase1'"
                ")"
            )
            assert exists is True
            await conn.execute("DROP TABLE IF EXISTS csa_memory_test_phase1;")
        finally:
            await conn.close()

    asyncio.run(_test())


def test_retriever_insert_and_retrieve():
    """Insert a memory node and verify it can be retrieved via the retriever."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)
            pgv, query_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_insert")
                await conn.execute(
                    """INSERT INTO csa_memory_test_insert
                       (session_id, sensory_input, inner_monologue, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4, $5)""",
                    "test", "Test sensory input", "Test inner thought", pgv, 7,
                )

            results = await retriever.retrieve_memories("test_insert", query_vec, top_k=5)
            assert len(results) >= 1
            assert results[0]["sensory_input"] == "Test sensory input"
            assert results[0]["cosine_distance"] < 0.35
        finally:
            await pool.close()

    asyncio.run(_test())


def test_retriever_respects_cosine_threshold():
    """Verify retriever filters out memories above the max cosine distance."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)
            pgv, _ = _make_vector()
            diff_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_threshold")
                await conn.execute(
                    """INSERT INTO csa_memory_test_threshold
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "test", "Far away memory", pgv, 1,
                )

            results = await retriever.retrieve_memories("test_threshold", diff_vec[1], top_k=5, max_cosine_distance=0.001)
            assert len(results) == 0
        finally:
            await pool.close()

    asyncio.run(_test())


def test_rag_score_formula():
    """Verify the RAG score formula components."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool, decay_lambda=0.01)
            pgv, query_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_score")
                await conn.execute(
                    """INSERT INTO csa_memory_test_score
                       (session_id, sensory_input, episodic_embedding, importance_score, access_count)
                       VALUES ($1, $2, $3, $4, $5)""",
                    "test", "Scoring test", pgv, 8, 3,
                )

            results = await retriever.retrieve_memories("test_score", query_vec, top_k=5)
            assert len(results) >= 1

            r = results[0]
            cosine_dist = r["cosine_distance"]
            sim_score = max(0.0, 1.0 - cosine_dist)
            assert sim_score > 0.5
            assert r["rag_score"] > 0
        finally:
            await pool.close()

    asyncio.run(_test())


def test_retriever_no_embedding_returns_empty():
    """Verify records with NULL embeddings are excluded from retrieval."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_null_emb")
                await conn.execute(
                    """INSERT INTO csa_memory_test_null_emb
                       (session_id, sensory_input, inner_monologue, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, NULL, $4)""",
                    "test", "No embedding", "Should not appear", 5,
                )

            _, query_vec = _make_vector()
            results = await retriever.retrieve_memories("test_null_emb", query_vec, top_k=5)
            assert len(results) == 0
        finally:
            await pool.close()

    asyncio.run(_test())


def test_retriever_access_count_increment():
    """Verify that retrieval increments access_count on returned nodes."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)
            pgv, vec_list = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_access")
                await conn.execute(
                    """INSERT INTO csa_memory_test_access
                       (session_id, sensory_input, episodic_embedding, importance_score, access_count)
                       VALUES ($1, $2, $3, $4, $5)""",
                    "test", "Access test", pgv, 5, 1,
                )

            results1 = await retriever.retrieve_memories("test_access", vec_list, top_k=5)
            assert len(results1) == 1
            score1 = results1[0]["rag_score"]
            assert score1 > 0

            results2 = await retriever.retrieve_memories("test_access", vec_list, top_k=5)
            assert len(results2) == 1
            score2 = results2[0]["rag_score"]
            assert score2 > score1, f"Score should increase with access count: {score1} -> {score2}"
        finally:
            await pool.close()

    asyncio.run(_test())


def test_retriever_character_table_auto_creation():
    """Verify that retrieve_memories auto-creates the character table if it doesn't exist."""
    import asyncio
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)
            _, query_vec = _make_vector()

            results = await retriever.retrieve_memories("fresh_character", query_vec, top_k=5)
            assert isinstance(results, list)

            async with pool.acquire() as conn:
                table_exists = await conn.fetchval(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_tables "
                    "WHERE schemaname='public' AND tablename='csa_memory_fresh_character'"
                    ")"
                )
                assert table_exists is True
        finally:
            await pool.close()

    asyncio.run(_test())
