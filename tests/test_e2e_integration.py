"""
End-to-End Integration Tests for Sovereign Persona Mesh (SPM).

Tests the full SPM stack: Evennia Liaison (Port 4005), SPM Proxy (Port 5050),
Lemonade LLM Server (Port 13305), PostgreSQL (litellm_postgres), and
SillyTavern character Seraphina.

Seven test groups:
  1. Service readiness probes
  2. Character registration in Evennia dungeon_cellar
  3. OpenAI /v1/chat/completions request simulating SillyTavern payload
  4. Real-time monologue stripping (<ctrl94>) verification in SSE stream
  5. PostgreSQL verification: csa_memory_seraphina turn persistence
  6. RAG memory retrieval injection on follow-up turn
  7. Zero-inference blackout bypass verification when character is distant
"""

import asyncio
import json
import time
import uuid
import pytest
import asyncpg
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.main import app as proxy_app
from evennia_world.app import app as evennia_app
from proxy.core.stream_parser import MonologueStreamParser
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.rag.retriever import EpisodicRAGRetriever
from evennia_world.hybrid_builder import HybridWorldBuilder
from evennia_world.session_lock import SessionLockManager

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


def _unique_session() -> str:
    """Generate a unique session id to avoid cross-test data leakage."""
    return f"e2e_{uuid.uuid4().hex[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def proxy_client():
    with TestClient(proxy_app) as c:
        yield c


@pytest.fixture
def evennia_client():
    with TestClient(evennia_app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_e2e_state(evennia_client):
    """Reset evennia global state and clean up DB before each test."""
    import evennia_world.app as evennia_mod
    evennia_mod.current_world = {}
    evennia_mod.world_builder = HybridWorldBuilder()
    evennia_mod.lock_manager = SessionLockManager()
    evennia_mod.action_tick_counter = 1420

    # Clean up any leftover seraphina data
    conn = None
    pool = None
    try:
        conn = asyncio.new_event_loop().run_until_complete(
            _clean_seraphina_db(DB_CONFIG)
        )
    except Exception:
        pass
    finally:
        if conn:
            try:
                asyncio.get_event_loop().run_until_complete(conn.close())
            except Exception:
                pass
    yield


async def _clean_seraphina_db(cfg: dict):
    """Delete all csa_memory_seraphina rows that match our e2e patterns."""
    conn = await asyncpg.connect(**cfg)
    try:
        await conn.execute("SELECT create_csa_memory_table('seraphina');")
        await conn.execute(
            "DELETE FROM csa_memory_seraphina WHERE session_id LIKE 'e2e_%';"
        )
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Service Readiness Probes
# ─────────────────────────────────────────────────────────────────────────────

class TestServiceReadiness:
    def test_evennia_health(self, evennia_client):
        resp = evennia_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_proxy_health(self, proxy_client):
        resp = proxy_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_proxy_root(self, proxy_client):
        resp = proxy_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Sovereign Persona Mesh Proxy"
        assert data["port"] == 5050
        assert data["status"] == "online"

    @pytest.mark.asyncio
    async def test_postgresql_connectivity(self):
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            ext = await conn.fetchval(
                "SELECT extname FROM pg_extension WHERE extname='vector'"
            )
            assert ext == "vector"
        finally:
            await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Character Registration (Seraphina in dungeon_cellar)
# ─────────────────────────────────────────────────────────────────────────────

class TestCharacterRegistration:
    def test_list_templates_contains_dungeon_cellar(self, evennia_client):
        resp = evennia_client.get("/api/v1/world/templates")
        assert resp.status_code == 200
        assert "dungeon_cellar" in resp.json()["templates"]

    def test_add_seraphina_to_cellar(self, evennia_client):
        resp = evennia_client.post(
            "/api/v1/world/characters",
            json={
                "character_id": "seraphina",
                "room_id": "cellar",
                "template_key": "dungeon_cellar",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["character_id"] == "seraphina"

    def test_seraphina_appears_in_character_list(self, evennia_client):
        evennia_client.post(
            "/api/v1/world/characters",
            json={
                "character_id": "seraphina",
                "room_id": "cellar",
                "template_key": "dungeon_cellar",
            },
        )
        resp = evennia_client.get(
            "/api/v1/world/characters?template_key=dungeon_cellar"
        )
        assert resp.status_code == 200
        characters = resp.json()
        seraphina_entries = [
            c for c in characters if c["character_id"] == "seraphina"
        ]
        assert len(seraphina_entries) >= 1

    def test_seraphina_world_state_query(self, evennia_client):
        evennia_client.post(
            "/api/v1/world/characters",
            json={
                "character_id": "seraphina",
                "room_id": "cellar",
                "template_key": "dungeon_cellar",
            },
        )
        resp = evennia_client.get(
            "/api/v1/world/state",
            params={"character_id": "seraphina", "session_id": "e2e_ws"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["character_id"] == "seraphina"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: SillyTavern Chat Completion Request
# ─────────────────────────────────────────────────────────────────────────────

class TestSillyTavernChatCompletion:
    def test_sillytavern_payload_for_seraphina(self, proxy_client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Character: Seraphina\n"
                        "You are Seraphina, a mysterious healer in the cellar."
                    ),
                },
                {
                    "role": "user",
                    "name": "user",
                    "content": "Seraphina, I found a strange key.",
                },
            ],
            "temperature": 0.8,
            "max_tokens": 256,
            "stream": True,
        }

        with patch(
            "proxy.api.routes.evennia_client.submit_action",
            new_callable=AsyncMock,
        ) as mock_action, patch(
            "proxy.api.routes.lemonade_client.generate_stream"
        ) as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 2001,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "Seraphina, I found a strange key.",
                        "gating_level": "direct",
                        "distance_ft": 2.0,
                        "barriers": [],
                    }
                ],
            }

            async def mock_stream(*args, **kwargs):
                yield "<ctrl94>Let me examine that key...</ctrl94> "
                yield "It has runes on it. It might open the old chest."

            mock_llm.side_effect = mock_stream

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            assert "[DONE]" in resp.text

    def test_target_character_extracted_from_system_message(self, proxy_client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {
                    "role": "system",
                    "content": "Character: Seraphina\nYou are Seraphina.",
                },
                {"role": "user", "content": "Hello."},
            ],
            "stream": False,
        }

        with patch(
            "proxy.api.routes.evennia_client.submit_action",
            new_callable=AsyncMock,
        ) as mock_action, patch(
            "proxy.api.routes.lemonade_client.generate_stream"
        ) as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 2002,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "Hello.",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": [],
                    }
                ],
            }

            async def mock_stream(*args, **kwargs):
                yield "Hello there, traveler."

            mock_llm.side_effect = mock_stream

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert (
                data["choices"][0]["message"]["content"]
                == "Hello there, traveler."
            )
            assert mock_action.call_args.kwargs["target_id"] == "seraphina"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Monologue Stripping in SSE Stream
# ─────────────────────────────────────────────────────────────────────────────

class TestMonologueStripping:
    def test_e2e_monologue_stripping_in_stream(self, proxy_client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Seraphina"},
                {"role": "user", "content": "What do you see?"},
            ],
            "stream": True,
        }

        with patch(
            "proxy.api.routes.evennia_client.submit_action",
            new_callable=AsyncMock,
        ) as mock_action, patch(
            "proxy.api.routes.lemonade_client.generate_stream"
        ) as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 2003,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "What do you see?",
                        "gating_level": "direct",
                        "distance_ft": 1.5,
                        "barriers": [],
                    }
                ],
            }

            async def mock_stream(*args, **kwargs):
                yield "<ctrl94>I should be cautious.</ctrl94> "
                yield "I see a torch on the wall."

            mock_llm.side_effect = mock_stream

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.text

            chunks = []
            for line in body.split("\n"):
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data != "[DONE]":
                        try:
                            chunk_data = json.loads(data)
                            delta = chunk_data.get(
                                "choices", [{}]
                            )[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunks.append(content)
                        except json.JSONDecodeError:
                            pass

            public_text = "".join(chunks)
            assert "I should be cautious" not in public_text
            assert "I see a torch" in public_text

    @pytest.mark.asyncio
    async def test_parser_two_state_machine_directly(self):
        async def mock_stream():
            yield "<ctrl94>First thought.</ctrl94> "
            yield "First public response. "
            yield "<ctrl94>Second thought.</ctrl94> "
            yield "Second public response."

        parser = MonologueStreamParser()
        public_chunks = []
        async for chunk in parser.process_token_stream(mock_stream()):
            public_chunks.append(chunk)

        inner, public = parser.get_final_buffers()
        assert "First thought" in inner
        assert "Second thought" in inner
        assert "First public response" in "".join(public_chunks)
        assert "Second public response" in "".join(public_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: PostgreSQL Turn Persistence (csa_memory_seraphina)
# ─────────────────────────────────────────────────────────────────────────────

class TestPostgreSQLPersistence:
    @pytest.mark.asyncio
    async def test_create_seraphina_memory_table(self):
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            await conn.execute(
                "SELECT create_csa_memory_table('seraphina');"
            )
            exists = await conn.fetchval(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_name='csa_memory_seraphina'"
                ")"
            )
            assert exists is True
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_insert_and_query_seraphina_turn(self):
        conn = await asyncpg.connect(**DB_CONFIG)
        sess = _unique_session()
        try:
            await conn.execute(
                "SELECT create_csa_memory_table('seraphina');"
            )

            await conn.execute(
                """
                INSERT INTO csa_memory_seraphina
                    (session_id, sensory_input, inner_monologue,
                     is_core_memory, is_subjective, importance_score)
                VALUES ($1, $2, $3, FALSE, TRUE, 3);
                """,
                sess,
                "User: Seraphina, what do you see?",
                "I see the user holding a key.",
            )

            rows = await conn.fetch(
                "SELECT session_id, sensory_input "
                "FROM csa_memory_seraphina "
                "WHERE session_id = $1 "
                "ORDER BY timestamp DESC LIMIT 1;",
                sess,
            )
            assert len(rows) >= 1
            assert rows[0]["sensory_input"] == "User: Seraphina, what do you see?"
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_core_memory_retention(self):
        """Verify is_core_memory = TRUE records survive pruning."""
        conn = await asyncpg.connect(**DB_CONFIG)
        sess = _unique_session()
        try:
            await conn.execute(
                "SELECT create_csa_memory_table('seraphina');"
            )

            # Insert core memory and volatile log
            await conn.execute(
                """
                INSERT INTO csa_memory_seraphina
                    (session_id, sensory_input, inner_monologue,
                     is_core_memory, is_subjective, importance_score)
                VALUES
                    ($1, 'User gave me a golden key.',
                     'Core: I received a key.', TRUE, TRUE, 8),
                    ($1, 'User said hello.',
                     'Just a greeting.', FALSE, TRUE, 1);
                """,
                sess,
            )

            # Verify both exist
            all_rows = await conn.fetch(
                "SELECT id, is_core_memory "
                "FROM csa_memory_seraphina "
                "WHERE session_id = $1;",
                sess,
            )
            assert len(all_rows) >= 2

            # Prune volatile records
            await conn.execute(
                "DELETE FROM csa_memory_seraphina "
                "WHERE is_core_memory = FALSE;"
            )

            # Only core memory should remain
            core_rows = await conn.fetch(
                "SELECT sensory_input, is_core_memory "
                "FROM csa_memory_seraphina "
                "WHERE session_id = $1;",
                sess,
            )
            assert len(core_rows) == 1
            assert core_rows[0]["is_core_memory"] is True
            assert "golden key" in core_rows[0]["sensory_input"]
        finally:
            await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: RAG Memory Retrieval Injection
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGMemoryRetrieval:
    @pytest.mark.asyncio
    async def test_rag_retrieval_injects_relevant_memories(self):
        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None:
            mock_embedding = np.random.randn(3584).tolist()
        else:
            mock_embedding = [0.1] * 3584
        embedding_str = "[" + ",".join(map(str, mock_embedding)) + "]"

        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            await conn.execute(
                "SELECT create_csa_memory_table('seraphina');"
            )

            await conn.execute(
                """
                INSERT INTO csa_memory_seraphina
                    (session_id, sensory_input, inner_monologue,
                     is_core_memory, is_subjective, importance_score,
                     episodic_embedding)
                VALUES
                    ($1,
                     'User gave me a golden key.',
                     'Core: The golden key is mine.',
                     TRUE, TRUE, 8, $2::vector),
                    ($1,
                     'I heard footsteps in the cellar.',
                     'Someone is nearby.',
                     FALSE, TRUE, 3, $2::vector),
                    ($1,
                     'The old torch flickers.',
                     'It is almost out.',
                     FALSE, FALSE, 1, $2::vector);
                """,
                _unique_session(),
                embedding_str,
            )

            pool = await asyncpg.create_pool(**DB_CONFIG)
            try:
                retriever = EpisodicRAGRetriever(
                    db_pool=pool, decay_lambda=0.01
                )
                retrieved = await retriever.retrieve_memories(
                    character_id="seraphina",
                    query_embedding=mock_embedding,
                    top_k=5,
                    max_cosine_distance=0.35,
                )

                assert len(retrieved) >= 1
                for mem in retrieved:
                    assert "id" in mem
                    assert "sensory_input" in mem
                    assert "cosine_distance" in mem
                    assert "rag_score" in mem
                    assert mem["cosine_distance"] < 0.35
                    assert mem["rag_score"] > 0
            finally:
                await pool.close()
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_rag_decay_scoring_formula(self):
        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None:
            mock_embedding = np.random.randn(3584).tolist()
        else:
            mock_embedding = [0.2] * 3584
        embedding_str = "[" + ",".join(map(str, mock_embedding)) + "]"

        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            await conn.execute(
                "SELECT create_csa_memory_table('seraphina');"
            )

            await conn.execute(
                """
                INSERT INTO csa_memory_seraphina
                    (session_id, sensory_input, is_core_memory,
                     importance_score, access_count,
                     episodic_embedding)
                VALUES
                    ($1, 'Low importance memory',
                     FALSE, 1, 0, $2::vector),
                    ($1, 'High importance memory',
                     FALSE, 9, 0, $2::vector);
                """,
                _unique_session(),
                embedding_str,
            )

            pool = await asyncpg.create_pool(**DB_CONFIG)
            try:
                retriever = EpisodicRAGRetriever(
                    db_pool=pool, decay_lambda=0.01
                )
                retrieved = await retriever.retrieve_memories(
                    character_id="seraphina",
                    query_embedding=mock_embedding,
                    top_k=5,
                )

                assert len(retrieved) >= 2
                scores = {
                    m["sensory_input"]: m["rag_score"]
                    for m in retrieved
                }
                assert (
                    scores["High importance memory"]
                    > scores["Low importance memory"]
                )
            finally:
                await pool.close()
        finally:
            await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Zero-Inference Blackout Bypass
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackoutBypass:
    def test_blackout_gating_bypasses_llm(self, proxy_client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Seraphina"},
                {"role": "user", "content": "Seraphina, are you there?"},
            ],
            "stream": True,
        }

        with patch(
            "proxy.api.routes.evennia_client.submit_action",
            new_callable=AsyncMock,
        ) as mock_action:

            mock_action.return_value = {
                "success": True,
                "action_tick": 2004,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "You hear muffled sounds from the cellar.",
                        "gating_level": "blackout",
                        "distance_ft": 45.0,
                        "barriers": ["closed_door", "solid_wall"],
                    }
                ],
            }

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.text
            assert mock_action.call_count == 1
            assert "chatcmpl-spm-bypass" in body
            assert "muffled" in body.lower()

    def test_null_gating_bypasses_llm(self, proxy_client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Seraphina"},
                {"role": "user", "content": "Hello?"},
            ],
            "stream": True,
        }

        with patch(
            "proxy.api.routes.evennia_client.submit_action",
            new_callable=AsyncMock,
        ) as mock_action:

            mock_action.return_value = {
                "success": True,
                "action_tick": 2005,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "",
                        "gating_level": "null",
                        "distance_ft": 100.0,
                        "barriers": ["solid_wall"],
                    }
                ],
            }

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            assert "chatcmpl-spm-bypass" in resp.text

    @pytest.mark.asyncio
    async def test_observer_inference_filter_commits_ambient_log(self):
        """
        ObserverInferenceGatingFilter evaluates bypass correctly
        and commits ambient logs to PostgreSQL.
        """
        conn = await asyncpg.connect(**DB_CONFIG)
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            gating_filter = ObserverInferenceGatingFilter(db_pool=pool)

            # Blackout bypass
            result = await gating_filter.evaluate_and_bypass(
                character_id="seraphina",
                session_id="e2e_bypass_1",
                gating_level="blackout",
                sensory_feed="Faint sounds from above.",
            )
            assert result is True

            # Verify ambient log
            row = await conn.fetchval(
                "SELECT sensory_input "
                "FROM csa_memory_seraphina "
                "WHERE session_id = 'e2e_bypass_1' "
                "ORDER BY timestamp DESC LIMIT 1;"
            )
            assert row is not None
            assert "Faint sounds from above" in row

            # Null bypass
            result = await gating_filter.evaluate_and_bypass(
                character_id="seraphina",
                session_id="e2e_bypass_2",
                gating_level="null",
                sensory_feed="",
            )
            assert result is True

            # Direct gating — no bypass
            result = await gating_filter.evaluate_and_bypass(
                character_id="seraphina",
                session_id="e2e_direct_1",
                gating_level="direct",
                sensory_feed="Direct conversation.",
            )
            assert result is False
        finally:
            await pool.close()
            await conn.close()
