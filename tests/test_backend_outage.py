"""When Lemonade is down, SPM must say so and save nothing, not invent an in-character reply."""
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from proxy.backend_client.lemonade_client import LLMBackendError
from proxy.main import app

PAYLOAD = {
    "model": "spm-sovereign-mesh",
    "messages": [
        {"role": "system", "content": "[Character: Mira] A tavern keeper."},
        {"role": "user", "content": "Got a room?"},
    ],
}


async def _down(*args, **kwargs):
    raise LLMBackendError("LLM backend unreachable: connection refused")
    yield  # pragma: no cover  (makes this an async generator)


EVENNIA_OK = {
    "success": True, "action_tick": 1,
    "consequences": [{"recipient_id": "mira", "sensory_feed": "Got a room?", "gating_level": "direct",
                      "distance_ft": 0, "barriers": []}],
}


def _run(stream):
    with patch("proxy.api.routes.lemonade_client.generate_stream", _down), \
         patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock, return_value=EVENNIA_OK), \
         patch("proxy.api.routes._dispatch_lore_extraction") as lore, \
         patch("proxy.api.routes._dispatch_gm_actions", new_callable=AsyncMock) as gm, \
         TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    return resp, lore, gm


def test_streaming_outage_shows_notice_and_saves_nothing():
    resp, lore, gm = _run(stream=True)
    assert resp.status_code == 200
    text = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert "LLM backend is unavailable" in text
    assert "I am ready" not in text
    assert resp.text.rstrip().endswith("data: [DONE]")
    lore.assert_not_called()
    gm.assert_not_called()


def test_non_streaming_outage_returns_502():
    resp, lore, gm = _run(stream=False)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "llm_backend_error"
    lore.assert_not_called()
    gm.assert_not_called()


def test_blackout_bypass_names_the_target_character_not_luna():
    blackout = {**EVENNIA_OK, "consequences": [{**EVENNIA_OK["consequences"][0], "gating_level": "blackout"}]}
    with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock, return_value=blackout), \
         TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
    assert "Mira hears only muffled sounds" in resp.text
    assert "Luna" not in resp.text


def test_world_engine_outage_degrades_to_ungated_turn():
    """OPEN-010: with Evennia unreachable (conftest points it at a closed port), the chat
    route must still serve the turn ungated instead of returning a raw 500."""

    async def fake_stream(*args, **kwargs):
        yield "<think>planning</think>"
        yield "The tavern is quiet tonight."

    with patch("proxy.api.routes.lemonade_client.generate_stream", fake_stream), \
         patch("proxy.api.routes._dispatch_lore_extraction"), \
         TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})

    assert resp.status_code == 200
    text = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert "tavern is quiet" in text
    assert "think" not in text
