"""
FR-001: Session-Bound Context & Memory Isolation Test Suite.
Verifies that character memories, spatial locations, and proxy routes are 100% isolated by session_id.
"""

import json
import pytest
import asyncpg
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.main import app as proxy_app
from evennia_world.app import app as liaison_app
from proxy.rag.retriever import EpisodicRAGRetriever

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


@pytest.fixture
def proxy_client():
    with TestClient(proxy_app) as client:
        yield client


@pytest.fixture
def liaison_client():
    with TestClient(liaison_app) as client:
        yield client


class TestFR001SessionIsolation:
    """Test suite for FR-001 Session Isolation."""

    @pytest.mark.asyncio
    async def test_rag_session_memory_isolation(self):
        """Insert memories for domino under session_A and session_B. Query session_A returns ONLY session_A memories."""
        pool = await asyncpg.create_pool(**DB_CONFIG)
        vec_str = json.dumps([0.1] + [0.0] * 3583)

        try:
            async with pool.acquire() as conn:
                await conn.execute("SELECT create_csa_memory_table('domino');")
                # Insert memory under session_A
                await conn.execute("""
                    INSERT INTO csa_memory_domino (session_id, sensory_input, inner_monologue, is_core_memory, episodic_embedding, importance_score)
                    VALUES ('session_A', 'Domino sees a dragon in session A.', 'I must prepare my bow.', FALSE, $1::vector, 5);
                """, vec_str)
                # Insert memory under session_B
                await conn.execute("""
                    INSERT INTO csa_memory_domino (session_id, sensory_input, inner_monologue, is_core_memory, episodic_embedding, importance_score)
                    VALUES ('session_B', 'Domino meets a merchant in session B.', 'I need potions.', FALSE, $1::vector, 5);
                """, vec_str)

            retriever = EpisodicRAGRetriever(db_pool=pool)
            dummy_query = [0.1] + [0.0] * 3583

            # Query under session_A
            memories_a = await retriever.retrieve_memories("domino", dummy_query, top_k=5, session_id="session_A")
            sensory_texts_a = [m["sensory_input"] for m in memories_a]
            assert any("dragon" in s for s in sensory_texts_a)
            assert not any("merchant" in s for s in sensory_texts_a)

            # Query under session_B
            memories_b = await retriever.retrieve_memories("domino", dummy_query, top_k=5, session_id="session_B")
            sensory_texts_b = [m["sensory_input"] for m in memories_b]
            assert any("merchant" in s for s in sensory_texts_b)
            assert not any("dragon" in s for s in sensory_texts_b)
        finally:
            await pool.close()

    def test_evennia_spatial_session_isolation(self, liaison_client):
        """Move character luna to cellar in session_A, and tavern_upstairs in session_B. Verify isolation."""
        r1 = liaison_client.post("/api/v1/world/characters", json={
            "character_id": "luna",
            "room_id": "cellar",
            "template_key": "dungeon_cellar",
            "session_id": "session_A",
        })
        assert r1.status_code == 200

        r2 = liaison_client.post("/api/v1/world/characters", json={
            "character_id": "luna",
            "room_id": "tavern_upstairs",
            "template_key": "dungeon_cellar",
            "session_id": "session_B",
        })
        assert r2.status_code == 200

        # Query session_A state
        state_a = liaison_client.get("/api/v1/world/state", params={"character_id": "luna", "session_id": "session_A"}).json()
        assert state_a["current_room"]["room_id"] == "cellar"

        # Query session_B state
        state_b = liaison_client.get("/api/v1/world/state", params={"character_id": "luna", "session_id": "session_B"}).json()
        assert state_b["current_room"]["room_id"] == "tavern_upstairs"

    def test_proxy_header_session_routing(self, proxy_client):
        """Send requests to /v1/chat/completions with header X-Session-ID vs body session_id vs default."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [{"role": "user", "content": "Seraphina, hello!"}],
            "stream": False
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1500,
                "consequences": []
            }

            async def mock_stream(*args, **kwargs):
                yield "Greetings traveler."

            mock_llm.side_effect = mock_stream

            # Header X-Session-ID
            resp = proxy_client.post("/v1/chat/completions", json=payload, headers={"X-Session-ID": "custom_header_sess"})
            assert resp.status_code == 200
            assert mock_action.called
            call_session_id = mock_action.call_args.kwargs.get("session_id")
            assert call_session_id == "custom_header_sess"
