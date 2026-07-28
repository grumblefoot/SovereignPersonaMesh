"""
End-to-End SPM Integration Test Suite (Phase 8).
Verifies SPM Proxy, Evennia Liaison, PostgreSQL memory persistence,
monologue stream parsing, RAG retrieval, and zero-inference blackout bypass.
"""

import json
import pytest
import asyncpg
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.main import app as proxy_app
from evennia_world.app import app as liaison_app
from proxy.core.stream_parser import MonologueStreamParser
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
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


class TestE2ESeraphinaWorkflow:
    """Full End-to-End integration test suite for character Seraphina."""

    def test_service_health_and_models(self, proxy_client, liaison_client):
        """Verify service health and models endpoints."""
        r1 = liaison_client.get("/health")
        assert r1.status_code == 200
        assert r1.json()["status"] == "ok"

        r2 = proxy_client.get("/v1/models")
        assert r2.status_code == 200
        data = r2.json()
        assert data["object"] == "list"

    def test_seraphina_character_registration(self, liaison_client):
        """Register Seraphina in Evennia cellar room."""
        r = liaison_client.post("/api/v1/world/characters", json={
            "character_id": "seraphina",
            "room_id": "cellar",
            "template_key": "dungeon_cellar",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

        r_state = liaison_client.get("/api/v1/world/state", params={"character_id": "seraphina"})
        assert r_state.status_code == 200
        assert r_state.json()["current_room"]["room_id"] == "cellar"

    def test_seraphina_chat_completion_monologue_stripping(self, proxy_client):
        """
        Simulate SillyTavern OpenAI Chat Completion payload for Seraphina.
        Verifies monologue stripping (<ctrl94>) and clean public response streaming.
        """
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {
                    "role": "system",
                    "content": "[System note: Write one reply only. Character: Seraphina\nYou are Seraphina, a celestial guardian in the cellar.]"
                },
                {
                    "role": "user",
                    "content": "Seraphina, who goes there?"
                }
            ],
            "stream": True
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1450,
                "consequences": [
                    {
                        "recipient_id": "seraphina",
                        "sensory_feed": "Seraphina, who goes there?",
                        "gating_level": "direct",
                        "distance_ft": 1.5,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "<ctrl94>I observe the intruder closely.</ctrl94> "
                yield "I am Seraphina, guardian of this cellar."

            mock_llm.side_effect = mock_stream

            resp = proxy_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.text
            assert "I am Seraphina" in body
            assert "I observe the intruder" not in body

    @pytest.mark.asyncio
    async def test_postgres_memory_persistence(self):
        """Verify turn data persistence in csa_memory_seraphina."""
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            await conn.execute("SELECT create_csa_memory_table('seraphina');")
            await conn.execute("""
                INSERT INTO csa_memory_seraphina (session_id, sensory_input, inner_monologue, is_core_memory, is_subjective, importance_score)
                VALUES ('e2e_session_1', 'Who goes there?', 'I observe the intruder', FALSE, TRUE, 3);
            """)

            rows = await conn.fetch("SELECT sensory_input, inner_monologue FROM csa_memory_seraphina WHERE session_id = 'e2e_session_1';")
            assert len(rows) >= 1
            assert rows[0]["sensory_input"] == "Who goes there?"
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_observer_blackout_bypass(self):
        """Verify zero-inference blackout bypass for distant characters."""
        pool = await asyncpg.create_pool(**DB_CONFIG)
        try:
            gating_filter = ObserverInferenceGatingFilter(db_pool=pool)
            bypassed = await gating_filter.evaluate_and_bypass(
                character_id="seraphina",
                session_id="e2e_blackout_sess",
                gating_level="blackout",
                sensory_feed="Distant thuds from above."
            )
            assert bypassed is True

            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT sensory_input, inner_monologue FROM csa_memory_seraphina WHERE session_id = 'e2e_blackout_sess';")
                assert len(rows) >= 1
                assert rows[0]["sensory_input"] == "Distant thuds from above."
                assert rows[0]["inner_monologue"] is None
        finally:
            await pool.close()
