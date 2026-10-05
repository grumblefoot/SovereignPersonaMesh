"""Sprint 1 Track B: embedding provider layer, migration, FTS fallback, re-embed job.

Covers embeddings plan phases 1-3 (docs/plans/embeddings.md) plus the lore
session/scope columns that ride in migration 001 (SPRINT_PLAN §1.5).
Runs against the throwaway DB only (tests/_testdb.py).
"""

import asyncio
import math
from pathlib import Path

import asyncpg
import httpx
import os
import pytest

from proxy.embeddings import (
    DeterministicEmbedder,
    EmbeddingService,
    FakeEmbeddingProvider,
    create_embedding_service,
    get_embedding_service,
    reset_embedding_service,
    resolve_provider_name,
)
from scripts.apply_migrations import MIGRATIONS_DIR, apply_migrations
from scripts.reembed import reembed_all
from tests._testdb import TEST_DB_CONFIG

MIGRATION_001 = MIGRATIONS_DIR / "001_embedding_spaces_and_lore_scope.sql"


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def _vec_str(vec):
    return "[" + ",".join(map(str, vec)) + "]"


# ──────────────────────────────────────────────────────────────────────────
# Provider resolution matrix (decision 5: cloud is never chosen implicitly)
# ──────────────────────────────────────────────────────────────────────────

LEMONADE = "http://127.0.0.1:13305/v1"
OPENAI = "https://api.openai.com/v1"


@pytest.mark.parametrize("settings,expected", [
    # auto on a Lemonade-like loopback backend -> openai_compat
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": LEMONADE}, "openai_compat"),
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": "http://localhost:8000/v1"}, "openai_compat"),
    # LAN addresses and hostnames count as local
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": "http://192.168.1.50:5001/v1"}, "openai_compat"),
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": "http://strix.lan:8080"}, "openai_compat"),
    # EMBEDDING_URL overrides the chat backend (llama.cpp second instance)
    ({"EMBEDDING_PROVIDER": "auto", "EMBEDDING_URL": "http://127.0.0.1:8081",
      "BACKEND_LLM_URL": OPENAI}, "openai_compat"),
    # auto on api.openai.com -> none (cloud never chosen implicitly)
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": OPENAI}, "none"),
    # ... unless the explicit remote opt-in flag is set
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": OPENAI,
      "EMBEDDING_ALLOW_REMOTE": True}, "openai_compat"),
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": OPENAI,
      "EMBEDDING_ALLOW_REMOTE": "true"}, "openai_compat"),
    # explicit overrides are honoured as-is
    ({"EMBEDDING_PROVIDER": "openai_compat", "BACKEND_LLM_URL": OPENAI}, "openai_compat"),
    ({"EMBEDDING_PROVIDER": "none", "BACKEND_LLM_URL": LEMONADE}, "none"),
    ({"EMBEDDING_PROVIDER": "fake"}, "fake"),
    # no backend at all -> none
    ({"EMBEDDING_PROVIDER": "auto", "BACKEND_LLM_URL": ""}, "none"),
    # unknown value degrades to none, not an exception
    ({"EMBEDDING_PROVIDER": "banana"}, "none"),
])
def test_provider_resolution_matrix(settings, expected):
    assert resolve_provider_name(settings) == expected


def test_onnx_local_is_a_phase4_stub():
    with pytest.raises(NotImplementedError, match="phase 4"):
        resolve_provider_name({"EMBEDDING_PROVIDER": "onnx_local"})


def test_defaults_resolve_openai_compat_on_local_backend():
    """config/manager.py defaults: auto + the Lemonade backend URL."""
    from config.manager import get_default_settings
    settings = get_default_settings()
    settings.pop("EMBEDDING_PROVIDER", None)  # conftest forces 'none' via env
    settings["EMBEDDING_PROVIDER"] = "auto"
    settings["BACKEND_LLM_URL"] = LEMONADE
    assert resolve_provider_name(settings) == "openai_compat"
    # Default switched after the decision-4 bake-off (docs/plans/BAKEOFF_2026-10-05.md).
    assert settings["EMBEDDING_MODEL"] == "Qwen3-Embedding-0.6B-GGUF-Q8_0"
    assert settings["EMBEDDING_DIM"] == 0
    assert settings["EMBEDDING_TIMEOUT_S"] == 3


def test_singleton_defaults_to_none_under_tests():
    """conftest forces EMBEDDING_PROVIDER=none: the shared service stores NULLs."""
    service = get_embedding_service()
    assert service.provider_name == "none"
    assert service.available is False
    assert service.space is None


# ──────────────────────────────────────────────────────────────────────────
# openai_compat contract (httpx.MockTransport; no live services)
# ──────────────────────────────────────────────────────────────────────────

def _compat_service(handler, dim=0, model="embed-gemma-300m-FLM"):
    settings = {
        "EMBEDDING_PROVIDER": "openai_compat",
        "BACKEND_LLM_URL": LEMONADE,
        "EMBEDDING_MODEL": model,
        "EMBEDDING_DIM": dim,
        "EMBEDDING_TIMEOUT_S": 3,
    }
    return create_embedding_service(settings, transport=httpx.MockTransport(handler))


async def test_openai_compat_happy_path_and_normalisation():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        import json
        body = json.loads(request.content)
        seen["body"] = body
        data = [
            {"index": i, "embedding": [3.0, 4.0] + [0.0] * 766}
            for i in range(len(body["input"]))
        ]
        return httpx.Response(200, json={"data": data})

    service = _compat_service(handler)
    vec = await service.generate_embedding("the sword is cursed")
    assert seen["url"] == "http://127.0.0.1:13305/v1/embeddings"
    assert seen["body"]["model"] == "embed-gemma-300m-FLM"
    assert seen["body"]["input"] == ["the sword is cursed"]
    assert len(vec) == 768
    # L2-normalised: 3-4-5 triangle -> 0.6 / 0.8
    assert abs(vec[0] - 0.6) < 1e-9 and abs(vec[1] - 0.8) < 1e-9
    assert service.space.dim == 768


async def test_openai_compat_5xx_returns_none():
    service = _compat_service(lambda req: httpx.Response(500, text="boom"))
    assert await service.generate_embedding("hello") is None


async def test_openai_compat_timeout_returns_none():
    def handler(request):
        raise httpx.ConnectTimeout("timed out")
    service = _compat_service(handler)
    assert await service.generate_embedding("hello") is None


async def test_openai_compat_malformed_response_returns_none():
    service = _compat_service(lambda req: httpx.Response(200, json={"nope": 1}))
    assert await service.generate_embedding("hello") is None


async def test_dimension_mismatch_drops_vector():
    """EMBEDDING_DIM=768 but the provider answers 5 dims: nothing is stored."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3, 4, 5]}]})
    service = _compat_service(handler, dim=768)
    assert await service.generate_embedding("hello") is None


async def test_nan_vector_dropped():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [float("nan"), 1.0]}]})
    service = _compat_service(handler)
    assert await service.generate_embedding("hello") is None


async def test_circuit_breaker_stops_calls_after_three_failures():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    service = _compat_service(handler)
    for _ in range(5):
        assert await service.generate_embedding(f"text {_}") is None
    # breaker opened after 3 consecutive failures; later calls never hit the wire
    assert calls["n"] == 3


async def test_empty_text_embeds_to_none_without_a_request():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("empty text must not reach the provider")
    service = _compat_service(handler)
    assert await service.generate_embedding("") is None
    assert await service.generate_embedding("   ") is None
    assert await service.batch_generate_embeddings([]) == []


async def test_cache_hits_skip_the_wire():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    service = _compat_service(handler)
    v1 = await service.generate_embedding("same text")
    v2 = await service.generate_embedding("same text")
    assert v1 == v2
    assert calls["n"] == 1


async def test_network_call_blocked_under_pytest_without_optin():
    """A real (transport-less) openai_compat provider never dials out in tests."""
    service = create_embedding_service({
        "EMBEDDING_PROVIDER": "openai_compat",
        "BACKEND_LLM_URL": "http://127.0.0.1:9/v1",  # closed port either way
        "EMBEDDING_MODEL": "m",
    })
    assert await service.generate_embedding("hello") is None


# ──────────────────────────────────────────────────────────────────────────
# fake provider / DeterministicEmbedder shim
# ──────────────────────────────────────────────────────────────────────────

async def test_fake_is_deterministic_and_word_sensitive():
    a = EmbeddingService(FakeEmbeddingProvider(dim=64), "fake", "fake-wordhash", 64)
    b = EmbeddingService(FakeEmbeddingProvider(dim=64), "fake", "fake-wordhash", 64)
    v1 = await a.generate_embedding("the sword is cursed")
    v2 = await b.generate_embedding("the sword is cursed")
    assert v1 == v2  # salt-free: stable across instances (and processes)

    near = await a.generate_embedding("the sword is cursed tonight")
    far = await a.generate_embedding("warm tavern fire and ale")
    assert _cosine(v1, near) > _cosine(v1, far)


async def test_deterministic_embedder_shim_keeps_old_interface():
    emb = DeterministicEmbedder()
    assert emb.available is True
    assert emb.dimension == 3584
    vec = await emb.generate_embedding("hello world")
    assert len(vec) == 3584
    batch = await emb.batch_generate_embeddings(["a", "b"])
    assert len(batch) == 2 and all(v is not None for v in batch)


async def test_cpu_embedding_engine_shim_delegates_to_service(monkeypatch):
    """scripts.onnx_embedder.CPUEmbeddingEngine is a deprecated delegate."""
    from scripts.onnx_embedder import CPUEmbeddingEngine
    # Default under tests: provider none -> unavailable, returns None.
    eng = CPUEmbeddingEngine()
    assert eng.available is False
    assert await eng.generate_embedding("some text") is None
    # Flip the configured provider to fake: the same shim starts embedding.
    from config.manager import get_settings_manager
    get_settings_manager().write_settings({"EMBEDDING_PROVIDER": "fake"})
    reset_embedding_service()
    assert eng.available is True
    vec = await eng.generate_embedding("some text")
    assert vec is not None and len(vec) == 768
    reset_embedding_service()


# ──────────────────────────────────────────────────────────────────────────
# Migration 001: idempotent, upgrades old-shape tables, tracked by
# apply_migrations. conftest already applied the new-shape init_db.sql, so
# running 001 on top proves idempotency over it.
# ──────────────────────────────────────────────────────────────────────────

OLD_MEMORY_SHAPE = """
CREATE TABLE IF NOT EXISTS csa_memory_oldshape (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL DEFAULT 'default_session',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sensory_input TEXT NOT NULL,
    inner_monologue TEXT,
    episodic_embedding VECTOR(3584),
    importance_score INT DEFAULT 5,
    is_core_memory BOOLEAN DEFAULT FALSE,
    is_subjective BOOLEAN DEFAULT TRUE,
    access_count INT DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

OLD_LORE_SHAPE = """
CREATE TABLE IF NOT EXISTS csa_lore_rules_oldshape (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_text TEXT NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    rule_embedding VECTOR(3584),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""


async def _column_names(conn, table):
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=$1;", table)
    return {r["column_name"] for r in rows}


async def _vector_typmod(conn, table, column):
    return await conn.fetchval(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass($1) "
        "AND attname = $2 AND NOT attisdropped;", table, column)


def test_migration_001_idempotent_and_upgrades_all_tables():
    async def _test():
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        try:
            # Seed tables in the pre-Track-B shape (VECTOR(3584), no new columns),
            # plus one legacy row, to prove the dynamic catalog sweep covers them.
            await conn.execute(OLD_MEMORY_SHAPE)
            await conn.execute(OLD_LORE_SHAPE)
            await conn.execute(
                "INSERT INTO csa_lore_rules_oldshape (rule_text, rule_type) "
                "VALUES ('The vault never opens at night', 'invariant');")

            sql = MIGRATION_001.read_text()
            await conn.execute(sql)
            await conn.execute(sql)  # idempotent: second run is harmless

            # Old-shape tables got every new column.
            mem_cols = await _column_names(conn, "csa_memory_oldshape")
            assert {"embedding_space_id", "public_response", "fts"} <= mem_cols
            lore_cols = await _column_names(conn, "csa_lore_rules_oldshape")
            assert {"embedding_space_id", "session_id", "scope", "fts"} <= lore_cols

            # VECTOR(3584) became dimension-free vector (atttypmod -1).
            assert await _vector_typmod(conn, "csa_memory_oldshape", "episodic_embedding") == -1
            assert await _vector_typmod(conn, "csa_lore_rules_oldshape", "rule_embedding") == -1

            # Existing rows kept, with the schema-only lore defaults
            # (session_id 'legacy_global', scope 'chat' — Part B §B.4).
            row = await conn.fetchrow("SELECT session_id, scope FROM csa_lore_rules_oldshape;")
            assert row["session_id"] == "legacy_global"
            assert row["scope"] == "chat"

            # Fresh-install tables from init_db.sql were swept too and already match.
            assert await _vector_typmod(conn, "csa_memory_luna", "episodic_embedding") == -1
            assert "scope" in await _column_names(conn, "csa_lore_rules_luna")

            # spm_embedding_spaces exists with its uniqueness contract.
            await conn.execute(
                "INSERT INTO spm_embedding_spaces (provider, model, dim) VALUES ('fake','m',8) "
                "ON CONFLICT (provider, model, dim) DO NOTHING;")
            await conn.execute(
                "INSERT INTO spm_embedding_spaces (provider, model, dim) VALUES ('fake','m',8) "
                "ON CONFLICT (provider, model, dim) DO NOTHING;")
            n = await conn.fetchval(
                "SELECT count(*) FROM spm_embedding_spaces WHERE provider='fake' AND model='m' AND dim=8;")
            assert n == 1

            # The helper functions now produce the new shape for future tables.
            await conn.execute("SELECT create_csa_memory_table('future_char');")
            future_cols = await _column_names(conn, "csa_memory_future_char")
            assert {"embedding_space_id", "fts"} <= future_cols
            await conn.execute("SELECT create_csa_lore_rules_table('future_char');")
            future_lore = await _column_names(conn, "csa_lore_rules_future_char")
            assert {"embedding_space_id", "session_id", "scope", "fts"} <= future_lore

            # apply_migrations records it once, then is a no-op.
            first = await apply_migrations(conn)
            assert MIGRATION_001.name in first
            second = await apply_migrations(conn)
            assert second == []
        finally:
            await conn.execute("DROP TABLE IF EXISTS csa_memory_oldshape, csa_lore_rules_oldshape, "
                               "csa_memory_future_char, csa_lore_rules_future_char;")
            await conn.close()

    asyncio.run(_test())


# ──────────────────────────────────────────────────────────────────────────
# Retriever: space-aware vector recall + full-text fallback
# ──────────────────────────────────────────────────────────────────────────

async def _seed_space(conn, provider, model, dim):
    await conn.execute(
        "INSERT INTO spm_embedding_spaces (provider, model, dim) VALUES ($1,$2,$3) "
        "ON CONFLICT (provider, model, dim) DO NOTHING;", provider, model, dim)
    return await conn.fetchval(
        "SELECT id FROM spm_embedding_spaces WHERE provider=$1 AND model=$2 AND dim=$3;",
        provider, model, dim)


def test_vector_recall_respects_embedding_spaces():
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
        service = EmbeddingService(FakeEmbeddingProvider(dim=32), "fake", "space-test", 32)
        try:
            retriever = EpisodicRAGRetriever(pool)
            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table('spacetest');")
                space_a = await service.ensure_space(conn)
                space_b = await _seed_space(conn, "fake", "space-test-other", 32)
                vec = await service.generate_embedding("the dragon sleeps in the keep")

                for space, label in ((space_a, "in space A"), (space_b, "in space B"), (None, "legacy NULL space")):
                    await conn.execute(
                        "INSERT INTO csa_memory_spacetest (session_id, sensory_input, "
                        "episodic_embedding, embedding_space_id) VALUES ($1,$2,$3::vector,$4);",
                        "s1", f"the dragon sleeps {label}", _vec_str(vec), space)

            # Active space A: only the space-A row is visible to vector search.
            res = await retriever.retrieve_memories(
                "spacetest", vec, session_id="s1", embedding_space_id=space_a)
            assert [r["sensory_input"] for r in res] == ["the dragon sleeps in space A"]

            # No active space (legacy callers): only the NULL-space row.
            res = await retriever.retrieve_memories("spacetest", vec, session_id="s1")
            assert [r["sensory_input"] for r in res] == ["the dragon sleeps legacy NULL space"]
        finally:
            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS csa_memory_spacetest;")
            await pool.close()

    asyncio.run(_test())


def test_fulltext_fallback_finds_row_in_right_session_only():
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
        try:
            retriever = EpisodicRAGRetriever(pool)
            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table('ftstest');")
                for session, text in (
                    ("sess_a", "The crimson sigil glows on the vault door"),
                    ("sess_a", "A quiet morning in the garden"),
                    ("sess_b", "The crimson sigil was seen in another chat"),
                ):
                    await conn.execute(
                        "INSERT INTO csa_memory_ftstest (session_id, sensory_input, "
                        "public_response, episodic_embedding) VALUES ($1,$2,$3,NULL);",
                        session, text, "noted.")

            # No query embedding (provider none / outage): FTS over the same session.
            res = await retriever.retrieve_memories(
                "ftstest", None, session_id="sess_a", query_text="crimson sigil")
            assert len(res) == 1
            node = res[0]
            assert node["sensory_input"] == "The crimson sigil glows on the vault door"
            # Same node shape as the vector path; rank-derived score, no cosine.
            assert set(node) == {"id", "sensory_input", "inner_monologue",
                                 "is_core_memory", "cosine_distance", "rag_score"}
            assert node["cosine_distance"] is None
            assert node["rag_score"] > 0

            # Words that only exist in the other session never cross over.
            res_b = await retriever.retrieve_memories(
                "ftstest", None, session_id="sess_b", query_text="crimson sigil")
            assert [r["sensory_input"] for r in res_b] == [
                "The crimson sigil was seen in another chat"]

            # Full-text also matches words that live in public_response.
            res_none = await retriever.retrieve_memories(
                "ftstest", None, session_id="sess_a", query_text="xyzzy nothing matches")
            assert res_none == []

            # No embedding and no query text -> [] without touching the DB path.
            assert await retriever.retrieve_memories("ftstest", None, session_id="sess_a") == []
        finally:
            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS csa_memory_ftstest;")
            await pool.close()

    asyncio.run(_test())


def test_lore_invariants_survive_without_vectors_and_triggers_respect_space():
    from proxy.rag.retriever import EpisodicRAGRetriever

    async def _test():
        pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
        service = EmbeddingService(FakeEmbeddingProvider(dim=32), "fake", "lore-space", 32)
        try:
            retriever = EpisodicRAGRetriever(pool)
            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_lore_rules_table('loretest');")
                space_id = await service.ensure_space(conn)
                trig_vec = await service.generate_embedding("the cursed sword awakens")
                # Lore is chat-scoped since B2 (OPEN-007): seed into the session
                # the retrieval below uses (its default, 'default_session').
                await conn.execute(
                    "INSERT INTO csa_lore_rules_loretest (rule_text, rule_type, status, session_id) "
                    "VALUES ('Luna never lies', 'invariant', 'active', 'default_session');")
                await conn.execute(
                    "INSERT INTO csa_lore_rules_loretest (rule_text, rule_type, status, "
                    "rule_embedding, embedding_space_id, session_id) VALUES "
                    "('Game over if the cursed sword awakens', 'game_over', 'active', $1::vector, $2, 'default_session');",
                    _vec_str(trig_vec), space_id)

            # Degraded mode: invariants still apply, trigger matching is skipped.
            res = await retriever.retrieve_lore_rules("loretest", None)
            assert [r["rule_text"] for r in res["invariants"]] == ["Luna never lies"]
            assert res["triggers"] == []

            # Vector mode in the right space finds the trigger...
            res = await retriever.retrieve_lore_rules(
                "loretest", trig_vec, embedding_space_id=space_id)
            assert len(res["triggers"]) == 1
            assert res["triggers"][0]["rule_type"] == "game_over"

            # ...and other spaces are invisible.
            res = await retriever.retrieve_lore_rules("loretest", trig_vec, embedding_space_id=None)
            assert res["triggers"] == []
            assert len(res["invariants"]) == 1
        finally:
            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS csa_lore_rules_loretest;")
            await pool.close()

    asyncio.run(_test())


# ──────────────────────────────────────────────────────────────────────────
# Re-embed job: fills NULLs, stamps the space, no-op on rerun
# ──────────────────────────────────────────────────────────────────────────

def test_reembed_fills_nulls_stamps_space_and_is_resumable():
    async def _test():
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        service = EmbeddingService(FakeEmbeddingProvider(dim=16), "fake", "reembed-test", 16)
        try:
            await conn.execute("SELECT create_csa_memory_table('reembed');")
            await conn.execute("SELECT create_csa_lore_rules_table('reembed');")
            for i in range(3):
                await conn.execute(
                    "INSERT INTO csa_memory_reembed (session_id, sensory_input, public_response) "
                    "VALUES ('s1', $1, 'reply');", f"memory number {i}")
            for i in range(2):
                await conn.execute(
                    "INSERT INTO csa_lore_rules_reembed (rule_text, rule_type) "
                    "VALUES ($1, 'invariant');", f"rule number {i}")

            # batch_size=2 forces several batches (interrupt/resume granularity).
            only = ["csa_memory_reembed", "csa_lore_rules_reembed"]
            totals = await reembed_all(conn, service, batch_size=2, allow_fake=True, only_tables=only)
            assert totals["updated"] == 5
            assert totals["space_id"] is not None

            n_mem = await conn.fetchval(
                "SELECT count(*) FROM csa_memory_reembed WHERE episodic_embedding IS NOT NULL "
                "AND embedding_space_id = $1;", totals["space_id"])
            n_lore = await conn.fetchval(
                "SELECT count(*) FROM csa_lore_rules_reembed WHERE rule_embedding IS NOT NULL "
                "AND embedding_space_id = $1;", totals["space_id"])
            assert (n_mem, n_lore) == (3, 2)

            # Rerun: nothing left to do.
            again = await reembed_all(conn, service, batch_size=2, allow_fake=True, only_tables=only)
            assert again["updated"] == 0

            # Simulate an interrupted run: one row loses its vector; only it is redone.
            await conn.execute(
                "UPDATE csa_memory_reembed SET episodic_embedding = NULL, embedding_space_id = NULL "
                "WHERE sensory_input = 'memory number 1';")
            resumed = await reembed_all(conn, service, batch_size=2, allow_fake=True, only_tables=only)
            assert resumed["updated"] == 1
        finally:
            await conn.execute("DROP TABLE IF EXISTS csa_memory_reembed, csa_lore_rules_reembed;")
            await conn.close()

    asyncio.run(_test())


def test_reembed_is_noop_for_none_and_unflagged_fake():
    async def _test():
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        try:
            none_service = EmbeddingService(None, "none")
            totals = await reembed_all(conn, none_service)
            assert totals == {"updated": 0, "skipped": 0, "space_id": None, "tables": 0}

            fake_service = EmbeddingService(FakeEmbeddingProvider(dim=8), "fake", "f", 8)
            totals = await reembed_all(conn, fake_service)  # no allow_fake
            assert totals["updated"] == 0 and totals["space_id"] is None
        finally:
            await conn.close()

    asyncio.run(_test())


# ──────────────────────────────────────────────────────────────────────────
# Opt-in live check against Lemonade (run after merge, not in CI):
#   RUN_LIVE_LLM_TESTS=1 SPM_TEST_DB=... pytest tests/test_embeddings_track_b.py -k live
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
                    reason="live Lemonade embeddings test is opt-in (RUN_LIVE_LLM_TESTS=1)")
async def test_live_lemonade_embed_gemma_768_dims():
    service = create_embedding_service({
        "EMBEDDING_PROVIDER": "openai_compat",
        "BACKEND_LLM_URL": "http://localhost:13305/v1",
        "EMBEDDING_MODEL": "embed-gemma-300m-FLM",
        "EMBEDDING_TIMEOUT_S": 120,  # first call may swap the model in
    })
    cursed = await service.generate_embedding("the sword is cursed")
    assert cursed is not None and len(cursed) == 768
    blade = await service.generate_embedding("my blade carries a curse")
    tavern = await service.generate_embedding("the tavern is warm tonight")
    assert _cosine(cursed, blade) > _cosine(cursed, tavern)


def test_deterministic_embedder_registers_a_fake_space():
    """Write paths stamp embedding_space_id via space_id_for(); the test
    double registers and reuses a 'fake' space like the real service."""
    from proxy.embeddings import space_id_for

    async def _test():
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        try:
            emb = DeterministicEmbedder(dimension=32)
            sid1 = await space_id_for(emb, conn)
            sid2 = await space_id_for(emb, conn)
            assert isinstance(sid1, int) and sid1 == sid2
            # Objects without ensure_space (plain mocks) degrade to None.
            assert await space_id_for(object(), conn) is None
        finally:
            await conn.close()

    asyncio.run(_test())
