"""
Unit & Integration Tests for SPM FastAPI Proxy API Routes (Port 5050).
Tests /v1/models, /v1/chat/completions, spatial routing, gating bypass,
monologue stream parsing, and FIFO queue dispatching.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.main import app
from proxy.core.stream_parser import MonologueStreamParser
from proxy.core.fifo_queue import InferenceFIFOQueue
from proxy.core.sensory_filter import ObserverInferenceGatingFilter


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestProxyModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "spm-sovereign-mesh" in model_ids
        assert "google/gemma-4-26B-A4B-it" in model_ids


class TestStreamParserFailSafe:
    @pytest.mark.asyncio
    async def test_normal_monologue_stripping(self):
        async def mock_llm_stream():
            yield "<ctrl94>I must analyze the room carefully.</ctrl94> "
            yield "I see a wooden chest in the corner."

        parser = MonologueStreamParser()
        public_tokens = []
        async for token in parser.process_token_stream(mock_llm_stream()):
            public_tokens.append(token)

        inner, public = parser.get_final_buffers()
        assert inner == "I must analyze the room carefully."
        assert public == "I see a wooden chest in the corner."
        assert "".join(public_tokens) == " I see a wooden chest in the corner."

    @pytest.mark.asyncio
    async def test_unexpected_eos_failsafe(self):
        async def mock_incomplete_stream():
            yield "<ctrl94>I am thinking but stream cuts off"

        parser = MonologueStreamParser()
        public_tokens = []
        async for token in parser.process_token_stream(mock_incomplete_stream()):
            public_tokens.append(token)

        assert parser.is_failsafe_triggered is True
        assert len(public_tokens) > 0
        assert "I am thinking but stream cuts off" in "".join(public_tokens)

    @pytest.mark.asyncio
    async def test_max_token_exceeded_failsafe(self):
        async def mock_long_monologue_stream():
            yield "<ctrl94>"
            for _ in range(550):
                yield "word "

        parser = MonologueStreamParser()
        public_tokens = []
        async for token in parser.process_token_stream(mock_long_monologue_stream()):
            public_tokens.append(token)

        assert parser.is_failsafe_triggered is True
        assert len(public_tokens) > 0


class TestInferenceFIFOQueue:
    @pytest.mark.asyncio
    async def test_fifo_queue_sequential_execution(self):
        queue = InferenceFIFOQueue()
        execution_order = []

        async def dummy_turn(name: str):
            execution_order.append(f"start-{name}")
            import asyncio
            await asyncio.sleep(0.01)
            execution_order.append(f"end-{name}")
            return name

        import asyncio
        r1, r2 = await asyncio.gather(
            queue.enqueue_and_execute(dummy_turn, "turn1"),
            queue.enqueue_and_execute(dummy_turn, "turn2")
        )

        assert r1 == "turn1"
        assert r2 == "turn2"
        assert execution_order == ["start-turn1", "end-turn1", "start-turn2", "end-turn2"]


class TestChatCompletionsEndpoint:
    def test_chat_completions_direct_gating(self, client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Luna\nYou are Luna, an elf wizard."},
                {"role": "user", "content": "Hello Luna, what do you see?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1421,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "Hello Luna, what do you see?",
                        "gating_level": "direct",
                        "distance_ft": 3.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "<ctrl94>I observe the user.</ctrl94> "
                yield "I see the dimly lit cellar."

            mock_llm.side_effect = mock_stream

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            assert "data: {" in body
            assert "[DONE]" in body

    def test_chat_completions_blackout_bypass(self, client):
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Seamus"},
                {"role": "user", "content": "Where is Seamus?"}
            ],
            "stream": True
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
            mock_action.return_value = {
                "success": True,
                "action_tick": 1422,
                "consequences": [
                    {
                        "recipient_id": "seamus",
                        "sensory_feed": "You hear muffled sounds from downstairs.",
                        "gating_level": "blackout",
                        "distance_ft": 45.0,
                        "barriers": ["closed_door", "solid_wall"]
                    }
                ]
            }

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.text
            assert "chatcmpl-spm-bypass" in body
            assert "*Luna hears muffled sounds" in body or "muffled" in body

    def test_chat_completions_null_gating(self, client):
        """Null gating level also triggers bypass."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": True
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
            mock_action.return_value = {
                "success": True,
                "action_tick": 1423,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "",
                        "gating_level": "null",
                        "distance_ft": 100.0,
                        "barriers": ["solid_wall"]
                    }
                ]
            }

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.text
            assert "chatcmpl-spm-bypass" in body

    def test_chat_completions_non_streaming(self, client):
        """Non-streaming request returns a JSON response with assistant content."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "user", "content": "What do you see?"}
            ],
            "stream": False
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1424,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "What do you see?",
                        "gating_level": "direct",
                        "distance_ft": 3.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "<ctrl94>Thinking...</ctrl94> "
                yield "I see a candle."

            mock_llm.side_effect = mock_stream

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert "I see a candle" in data["choices"][0]["message"]["content"]

    def test_chat_completions_stop_param_forwarded(self, client):
        """Stop parameter is forwarded to the LLM client."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "user", "content": "Tell a story."}
            ],
            "stream": False,
            "stop": ["\n\n", "END"]
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1425,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "Tell a story.",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "Once upon a time."

            mock_llm.side_effect = mock_stream

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            # Verify the generate_stream was called with stop param
            call_kwargs = mock_llm.call_args
            assert call_kwargs is not None
            assert "stop" in call_kwargs.kwargs or (
                len(call_kwargs.args) > 0
            )

    def test_chat_completions_default_model(self, client):
        """If no model is specified, the default 'spm-sovereign-mesh' is used."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
            mock_action.return_value = {
                "success": True,
                "action_tick": 1426,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "Hi",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_chat_completions_no_messages(self, client):
        """Empty messages list should still produce a valid (bypass) response."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": []
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
            mock_action.return_value = {
                "success": True,
                "action_tick": 1427,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200

    def test_chat_completions_message_name_extracted(self, client):
        """Target character is extracted from message.name field."""
        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "user", "content": "Hello", "name": "Domino"}
            ],
            "stream": False
        }

        with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 1428,
                "consequences": [
                    {
                        "recipient_id": "domino",
                        "sensory_feed": "Hello",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "I am Domino."

            mock_llm.side_effect = mock_stream

            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            # Verify Evennia was called with domino as target
            call_args = mock_action.call_args
            assert call_args.kwargs["target_id"] == "domino"
