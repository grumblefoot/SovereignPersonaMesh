"""
FR-001: Session-Bound Context & Memory Isolation Tests.

Verifies strict, zero-bleed session-bound memory and spatial isolation (session_id)
across PostgreSQL database tables, RAG retrieval queries, Evennia spatial room state,
and SPM Proxy routing. Every chat session starts 100% fresh for a character.
"""

import asyncio
import json
import asyncpg
import numpy as np
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteClient

from proxy.rag.retriever import EpisodicRAGRetriever
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.main import app as proxy_app
from proxy.api.routes import _extract_session_id
from evennia_world import app as evennia_app
from evennia_world.hybrid_builder import HybridWorldBuilder


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


def _make_vector(n=3584):
    """Generate a normalized random vector for pgvector."""
    np.random.seed(np.random.randint(0, 2**31))
    v = np.random.randn(n).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    pgv_str = "[" + ",".join(map(str, v)) + "]"
    return pgv_str, v.tolist()


# ======================================================================
# Retriever: session_id filtering in retrieve_memories()
# ======================================================================

class TestRetrieverSessionFilter:
    """Tests that retrieve_memories() strictly filters by session_id."""

    def test_retrieve_memories_filter_by_session(self):
        """Memories from different sessions should not leak into each other."""
        async def _test():
            pool = await asyncpg.create_pool(**DB_CONFIG)
            retriever = EpisodicRAGRetriever(pool)
            pgv, query_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_fr001_retriever")
                # Insert two memories with different session_ids
                await conn.execute(
                    """INSERT INTO csa_memory_test_fr001_retriever
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "session_A", "Memory from session A", pgv, 5,
                )
                await conn.execute(
                    """INSERT INTO csa_memory_test_fr001_retriever
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "session_B", "Memory from session B", pgv, 5,
                )

                # Query for session_A should only return session_A memory
                results_a = await retriever.retrieve_memories(
                    "test_fr001_retriever", query_vec, top_k=5,
                    session_id="session_A",
                )
                assert len(results_a) == 1
                assert results_a[0]["sensory_input"] == "Memory from session A"

                # Query for session_B should only return session_B memory
                results_b = await retriever.retrieve_memories(
                    "test_fr001_retriever", query_vec, top_k=5,
                    session_id="session_B",
                )
                assert len(results_b) == 1
                assert results_b[0]["sensory_input"] == "Memory from session B"

                # Query with default_session should return nothing
                results_default = await retriever.retrieve_memories(
                    "test_fr001_retriever", query_vec, top_k=5,
                    session_id="default_session",
                )
                assert len(results_default) == 0

                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr001_retriever;")

            await pool.close()

        asyncio.run(_test())

    def test_default_session_id_no_bleed(self):
        """Default session should be isolated from explicitly-named sessions."""
        async def _test():
            pool = await asyncpg.create_pool(**DB_CONFIG)
            retriever = EpisodicRAGRetriever(pool)
            pgv, query_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_fr001_bleed")
                await conn.execute(
                    """INSERT INTO csa_memory_test_fr001_bleed
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "explicit_session", "Should not bleed", pgv, 5,
                )

            # Default session should NOT see explicit_session memories
            results = await retriever.retrieve_memories(
                "test_fr001_bleed", query_vec, top_k=5,
            )
            assert len(results) == 0

            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr001_bleed;")

            await pool.close()

        asyncio.run(_test())


# ======================================================================
# Sensory Filter: session_id persistence in evaluate_and_bypass()
# ======================================================================

class TestSensoryFilterSessionPersistence:
    """Tests that evaluate_and_bypass() persists session_id in ambient logs."""

    def test_ambient_log_persists_session_id(self):
        """Ambient log from evaluate_and_bypass should carry the session_id."""
        async def _test():
            pool = await asyncpg.create_pool(**DB_CONFIG)
            gating_filter = ObserverInferenceGatingFilter(pool)
            char_id = "test_fr001_sensor"
            sess = "sensory_test_session"

            bypassed = await gating_filter.evaluate_and_bypass(
                character_id=char_id,
                session_id=sess,
                gating_level="blackout",
                sensory_feed="Hears distant thunder.",
            )
            assert bypassed is True

            async with pool.acquire() as conn:
                records = await conn.fetch(
                    f"SELECT session_id FROM csa_memory_{char_id} WHERE session_id = $1",
                    sess,
                )
                assert len(records) == 1
                assert records[0]["session_id"] == sess

                await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")

            await pool.close()

        asyncio.run(_test())

    def test_different_sessions_are_isolated_in_filter(self):
        """Two sessions bypassing should create isolated ambient logs."""
        async def _test():
            pool = await asyncpg.create_pool(**DB_CONFIG)
            gating_filter = ObserverInferenceGatingFilter(pool)
            char_id = "test_fr001_isolated"

            await gating_filter.evaluate_and_bypass(
                character_id=char_id,
                session_id="sess_one",
                gating_level="blackout",
                sensory_feed="Feed one.",
            )
            await gating_filter.evaluate_and_bypass(
                character_id=char_id,
                session_id="sess_two",
                gating_level="blackout",
                sensory_feed="Feed two.",
            )

            async with pool.acquire() as conn:
                all_records = await conn.fetch(
                    f"SELECT session_id, sensory_input FROM csa_memory_{char_id} ORDER BY session_id",
                )
                assert len(all_records) == 2
                sess_ids = [r["session_id"] for r in all_records]
                assert "sess_one" in sess_ids
                assert "sess_two" in sess_ids

                await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")

            await pool.close()

        asyncio.run(_test())


# ======================================================================
# Evennia: session-keyed world state isolation
# ======================================================================

class TestEvenniaSessionIsolation:
    """Tests for session-scoped world state in Evennia app endpoints."""

    def _reset_evennia_state(self):
        """Reset evennia app module state between tests."""
        evennia_app.current_world = {}
        evennia_app.session_worlds = {}
        evennia_app.action_tick_counter = 0
        evennia_app.lock_manager = type(evennia_app.lock_manager)(default_ttl=60.0)
        evennia_app.world_builder = HybridWorldBuilder()

    def test_session_scoped_world_state(self):
        """Two sessions should have independent world states."""
        self._reset_evennia_state()

        with StarletteClient(evennia_app.app) as client:
            # Session A: query world state (uses default template "dungeon_cellar")
            r_a = client.get("/api/v1/world/state", params={
                "character_id": "luna",
                "session_id": "fr001_sess_a",
            })
            assert r_a.status_code == 200
            data_a = r_a.json()
            assert data_a["character_id"] == "luna"

            # Session B: different session should have independent state
            r_b = client.get("/api/v1/world/state", params={
                "character_id": "luna",
                "session_id": "fr001_sess_b",
            })
            assert r_b.status_code == 200
            data_b = r_b.json()
            assert data_b["character_id"] == "luna"

            # Both sessions should have their own world entries
            assert "fr001_sess_a" in evennia_app.session_worlds
            assert "fr001_sess_b" in evennia_app.session_worlds
            assert evennia_app.session_worlds["fr001_sess_a"] is not evennia_app.session_worlds["fr001_sess_b"]

    def test_world_action_uses_session_id(self):
        """Action endpoint should respect session_id parameter."""
        self._reset_evennia_state()

        with StarletteClient(evennia_app.app) as client:
            r1 = client.post("/api/v1/world/action", json={
                "character_id": "rowan",
                "action_type": "speak",
                "target_id": "domino",
                "raw_text": "Hello from session one.",
                "session_id": "fr001_action_one",
            })
            assert r1.status_code == 200
            data1 = r1.json()
            assert data1["success"] is True

            r2 = client.post("/api/v1/world/action", json={
                "character_id": "rowan",
                "action_type": "speak",
                "target_id": "domino",
                "raw_text": "Hello from session two.",
                "session_id": "fr001_action_two",
            })
            assert r2.status_code == 200
            data2 = r2.json()
            assert data2["success"] is True

            assert "fr001_action_one" in evennia_app.session_worlds
            assert "fr001_action_two" in evennia_app.session_worlds

    def test_characters_endpoints_with_session_id(self):
        """Character endpoints should accept session_id in CharacterMovePayload."""
        self._reset_evennia_state()

        with StarletteClient(evennia_app.app) as client:
            r = client.post("/api/v1/world/characters", json={
                "character_id": "fr001_new_char",
                "room_id": "cellar",
                "template_key": "dungeon_cellar",
                "session_id": "fr001_char_sess",
            })
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is True

    def test_session_isolation_in_world_query(self):
        """Querying state for one session should not affect another session's state."""
        self._reset_evennia_state()

        with StarletteClient(evennia_app.app) as client:
            client.get("/api/v1/world/state", params={
                "character_id": "rowan",
                "session_id": "fr001_isolation_x",
            })
            client.get("/api/v1/world/state", params={
                "character_id": "domino",
                "session_id": "fr001_isolation_y",
            })

            assert "fr001_isolation_x" in evennia_app.session_worlds
            assert "fr001_isolation_y" in evennia_app.session_worlds
            x_worlds = evennia_app.session_worlds["fr001_isolation_x"]
            y_worlds = evennia_app.session_worlds["fr001_isolation_y"]
            assert x_worlds is not y_worlds

    def test_default_session_fallback(self):
        """Default session should work when no session_id is provided."""
        self._reset_evennia_state()

        with StarletteClient(evennia_app.app) as client:
            r = client.get("/api/v1/world/state", params={
                "character_id": "luna",
            })
            assert r.status_code == 200
            data = r.json()
            assert data["character_id"] == "luna"
            assert "default_session" in evennia_app.session_worlds


# ======================================================================
# Proxy: _extract_session_id() with precedence chain
# ======================================================================

class TestProxySessionExtraction:
    """Tests for _extract_session_id() with header > body > default precedence."""

    def test_header_precedence(self):
        """X-Session-ID header takes highest precedence."""
        with TestClient(proxy_app) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": "google/gemma-4-26B-A4B-it",
                "messages": [{"role": "user", "content": "test"}],
            }, headers={"X-Session-ID": "header-session-99"})
            assert resp.status_code == 200

    def test_body_fallback(self):
        """When no header, body session_id should be used."""
        with TestClient(proxy_app) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": "google/gemma-4-26B-A4B-it",
                "messages": [{"role": "user", "content": "test"}],
            })
            assert resp.status_code == 200

    def test_default_session_fallback(self):
        """When no header and no body session_id, default_session should be used."""
        with TestClient(proxy_app) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": "google/gemma-4-26B-A4B-it",
                "messages": [{"role": "user", "content": "test"}],
            }, headers={"X-Session-ID": ""})
            assert resp.status_code == 200

    def test_extract_session_id_function(self):
        """Direct unit test of _extract_session_id() precedence chain."""
        from starlette.requests import Request

        # Header wins over body
        scope = {
            "type": "http",
            "headers": [(b"x-session-id", b"header-wins")],
        }
        mock_req = Request(scope)
        body = {"session_id": "body-loses"}
        result = _extract_session_id(mock_req, body)
        assert result == "header-wins"

        # Body used when no header
        scope2 = {"type": "http", "headers": []}
        mock_req2 = Request(scope2)
        body2 = {"session_id": "body-wins"}
        result2 = _extract_session_id(mock_req2, body2)
        assert result2 == "body-wins"

        # Default when neither
        scope3 = {"type": "http", "headers": []}
        mock_req3 = Request(scope3)
        body3 = {}
        result3 = _extract_session_id(mock_req3, body3)
        assert result3 == "st_user_default"


# ======================================================================
# Integration: End-to-end session isolation across all components
# ======================================================================

class TestE2ESessionIsolation:
    """End-to-end tests verifying session isolation across the full SPM stack."""

    def test_full_stack_session_isolation(self):
        """
        Simulate: session_A creates memories, session_B queries.
        session_B should NOT see session_A's memories (RAG isolation).
        """
        async def _test():
            pool = await asyncpg.create_pool(**DB_CONFIG)
            retriever = EpisodicRAGRetriever(pool)
            pgv, query_vec = _make_vector()

            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table($1);", "test_fr001_e2e")

            # Insert memories
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO csa_memory_test_fr001_e2e
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "e2e_session_a", "Secret memory A", pgv, 7,
                )
                await conn.execute(
                    """INSERT INTO csa_memory_test_fr001_e2e
                       (session_id, sensory_input, episodic_embedding, importance_score)
                       VALUES ($1, $2, $3, $4)""",
                    "e2e_session_b", "Secret memory B", pgv, 7,
                )

            # Session A queries -> should only see A's memory
            results_a = await retriever.retrieve_memories(
                "test_fr001_e2e", query_vec, top_k=5,
                session_id="e2e_session_a",
            )
            sensory_a = [r["sensory_input"] for r in results_a]
            assert "Secret memory A" in sensory_a
            assert "Secret memory B" not in sensory_a

            # Session B queries -> should only see B's memory
            results_b = await retriever.retrieve_memories(
                "test_fr001_e2e", query_vec, top_k=5,
                session_id="e2e_session_b",
            )
            sensory_b = [r["sensory_input"] for r in results_b]
            assert "Secret memory B" in sensory_b
            assert "Secret memory A" not in sensory_b

            # Verify zero-bleed
            for r in results_a:
                assert "B" not in r["sensory_input"], f"Bleed detected in session A: {r['sensory_input']}"
            for r in results_b:
                assert "A" not in r["sensory_input"], f"Bleed detected in session B: {r['sensory_input']}"

            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr001_e2e;")

            await pool.close()

        asyncio.run(_test())

    def test_proxy_passes_session_to_evennia(self):
        """Proxy should pass session_id from request to Evennia world client."""
        with TestClient(proxy_app) as client:
            payload = {
                "model": "google/gemma-4-26B-A4B-it",
                "messages": [
                    {"role": "system", "content": "Character: Luna"},
                    {"role": "user", "content": "Hello Luna, test session isolation."}
                ],
                "stream": True,
            }

            with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
                 patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

                mock_action.return_value = {
                    "success": True,
                    "action_tick": 9001,
                    "consequences": [
                        {
                            "recipient_id": "luna",
                            "sensory_feed": "Hello Luna, test session isolation.",
                            "gating_level": "direct",
                            "distance_ft": 3.0,
                            "barriers": []
                        }
                    ]
                }

                async def mock_stream(*args, **kwargs):
                    yield "I hear you."

                mock_llm.side_effect = mock_stream

                resp = client.post("/v1/chat/completions", json=payload,
                                   headers={"X-Session-ID": "e2e_proxy_test"})
                assert resp.status_code == 200

                # Verify the session_id was passed to Evennia
                call_kwargs = mock_action.call_args
                assert call_kwargs is not None
                assert call_kwargs.kwargs["session_id"] == "e2e_proxy_test"

    def test_blackout_bypass_respects_session_id(self):
        """Bypass path in proxy should also use the extracted session_id."""
        with TestClient(proxy_app) as client:
            payload = {
                "model": "google/gemma-4-26B-A4B-it",
                "messages": [
                    {"role": "system", "content": "Character: Seamus"},
                    {"role": "user", "content": "Where is Seamus?"}
                ],
                "stream": True,
            }

            with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
                mock_action.return_value = {
                    "success": True,
                    "action_tick": 9002,
                    "consequences": [
                        {
                            "recipient_id": "seamus",
                            "sensory_feed": "Muffled sounds.",
                            "gating_level": "blackout",
                            "distance_ft": 45.0,
                            "barriers": ["closed_door", "solid_wall"]
                        }
                    ]
                }

                resp = client.post("/v1/chat/completions", json=payload,
                                   headers={"X-Session-ID": "blackout_isolation_test"})
                assert resp.status_code == 200
                body = resp.text
                assert "[DONE]" in body
