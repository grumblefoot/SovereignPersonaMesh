import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from proxy.api.routes import (
    _extract_target_char, 
    _extract_location_from_messages,
    ChatCompletionMessage
)
from proxy.main import app

client = TestClient(app)

def test_extract_target_char_patterns():
    # Pattern 1
    m1 = [ChatCompletionMessage(role="system", content="[Arvenia's Personality=brave]")]
    assert _extract_target_char(m1) == "arvenia"
    
    # Pattern 2
    m2 = [ChatCompletionMessage(role="system", content="[Character: Bob]")]
    assert _extract_target_char(m2) == "bob"
    
    # Pattern 3
    m3 = [ChatCompletionMessage(role="system", content="[Alice: says hi]")]
    assert _extract_target_char(m3) == "alice"
    
    # Name attribute
    m4 = [ChatCompletionMessage(role="user", content="hi", name="Charlie")]
    assert _extract_target_char(m4) == "charlie"
    
    # Default
    assert _extract_target_char([]) == "default"

def test_extract_location_from_messages():
    # Location kw
    loc = _extract_location_from_messages([], system_prompt="Location: The Dark Forest")
    assert loc == "The Dark Forest"
    
    # LOC tag
    loc2 = _extract_location_from_messages([], system_prompt="[LOC: Castle]")
    assert loc2 == "Castle"
    
    # Keyword fallback
    loc3 = _extract_location_from_messages([], system_prompt="We are in a dungeon.")
    assert loc3 == "Underground Prison"
    
    # Default
    assert _extract_location_from_messages([], system_prompt="") == "Starting Location"

@pytest.mark.asyncio
async def test_non_streaming_chat_completion():
    # This hits the non-streaming path of /v1/chat/completions
    req_body = {
        "model": "google/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    }
    
    # We mock evennia_client and lemonade_client to avoid actual network calls
    with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_evennia, \
         patch("proxy.api.routes.lemonade_client.generate_stream") as mock_lemonade:
        
        mock_evennia.return_value = {"consequences": []}
        
        async def mock_stream(*args, **kwargs):
            yield "Hello non-streaming world"
            
        mock_lemonade.side_effect = mock_stream
        
        response = client.post("/v1/chat/completions", json=req_body)
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "Hello non-streaming world" in data["choices"][0]["message"]["content"]

@pytest.mark.asyncio
async def test_bypass_null_gating_completion():
    req_body = {
        "model": "google/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "user", "content": "You are far away."}
        ],
        "stream": True
    }
    
    with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_evennia:
        # Force gating_level to null to trigger the bypass
        mock_evennia.return_value = {
            "consequences": [
                {
                    "recipient_id": "default",
                    "sensory_feed": "...",
                    "gating_level": "null"
                }
            ]
        }
        
        response = client.post("/v1/chat/completions", json=req_body)
        
        assert response.status_code == 200
        # It's an SSE stream, read the text
        content = response.text
        assert "chatcmpl-spm-bypass" in content
        assert "muffled sounds" in content
