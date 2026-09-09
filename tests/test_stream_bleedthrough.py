import pytest
import asyncio
import json
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from proxy.main import app

@pytest.mark.asyncio
async def test_integration_stream_bleedthrough(monkeypatch):
    """
    Tests the stream parsing flow against the full proxy app to reproduce the
    issue where GM_ACTION tags bleed through if the LLM forgets the </think> tag.
    """
    # 1. Load the SillyTavern chat payload
    with open("tests/fixtures/test_chat_payload.json", "r") as f:
        payload = json.load(f)

    # 2. Mock the LemonadeLLMClient.generate_stream to return the failing output in fragmented chunks
    async def mock_generate_stream(*args, **kwargs):
        # We fragment the GM_ACTION tag specifically to test the parser's robustness
        # and we intentionally omit the </think> tag at the end of the monologue.
        chunks = [
            "<th", "ink>\n[GM_ACT", "ION: {\"type\": \"CREATE_ROOM\"",
            ", \"room_id\": \"test_suite_123\", \"name\": \"Test Suite\", \"desc\": \"A test room.\"}]\n",
            "[GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"Vardus\", \"room_id\": \"test_suite_123\"}]\n\n",
            "The user is playing as Vardus. I am describing the environment.\n\n",
            "The heavy oak doors click shut, sealing out the echoing silence..."
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.001)

    # Use monkeypatch for all patches to ensure they survive the ASGI transport context
    monkeypatch.setattr("proxy.api.routes.lemonade_client.generate_stream", mock_generate_stream)

    evennia_submit = AsyncMock(return_value={
        "success": True,
        "action_tick": 1,
        "consequences": [
            {"recipient_id": "default", "sensory_feed": "test", "gating_level": "direct", "distance_ft": 0, "barriers": []}
        ]
    })
    monkeypatch.setattr("proxy.api.routes.evennia_client.submit_action", evennia_submit)

    # Patch asyncpg.create_pool to avoid real DB connection on startup
    monkeypatch.setattr("asyncpg.create_pool", AsyncMock())

    # 3. Stream the request through the FastAPI app (Integration Test)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"

        full_text = ""
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    if "content" in data.get("choices", [{}])[0].get("delta", {}):
                        full_text += data["choices"][0]["delta"]["content"]
                except json.JSONDecodeError:
                    pass

    # 4. Wait for background _dispatch_gm_actions task to complete
    await asyncio.sleep(0.2)

    # 5. Assertions
    # If the parser works correctly, [GM_ACTION] shouldn't bleed into the text.
    # Under the current state machine parser, this should FAIL, proving the bug exists.
    assert "[GM_ACTION" not in full_text, "Bleedthrough detected! GM_ACTION tags leaked to the frontend."
