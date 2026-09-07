import pytest
import json
from unittest.mock import AsyncMock, patch
from proxy.backend_client.lemonade_client import LemonadeLLMClient

@pytest.fixture
def lemonade():
    return LemonadeLLMClient()

class MockResponse:
    def __init__(self, status_code, json_data=None, lines=None):
        self.status_code = status_code
        self._json_data = json_data
        self._lines = lines or []
        
    def json(self):
        return self._json_data
        
    async def aiter_lines(self):
        for line in self._lines:
            yield line

class MockStreamContext:
    def __init__(self, response):
        self.response = response
        
    async def __aenter__(self):
        return self.response
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_resolve_model_exact(lemonade):
    mock_client = AsyncMock()
    mock_client.get.return_value = MockResponse(200, {"data": [{"id": "exact_model"}]})
    assert await lemonade._resolve_model("exact_model", mock_client) == "exact_model"

@pytest.mark.asyncio
async def test_resolve_model_substring(lemonade):
    mock_client = AsyncMock()
    mock_client.get.return_value = MockResponse(200, {"data": [{"id": "google/gemma-4"}]})
    assert await lemonade._resolve_model("gemma-4", mock_client) == "google/gemma-4"

@pytest.mark.asyncio
async def test_resolve_model_fallback(lemonade):
    mock_client = AsyncMock()
    mock_client.get.return_value = MockResponse(200, {"data": [{"id": "other"}]})
    assert await lemonade._resolve_model("unknown", mock_client) == "other"

@pytest.mark.asyncio
async def test_resolve_model_error(lemonade):
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Network")
    assert await lemonade._resolve_model("requested", mock_client) == "requested"

@pytest.mark.asyncio
async def test_generate_stream_success(lemonade):
    with patch("proxy.backend_client.lemonade_client.httpx.AsyncClient") as mock_httpx:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        mock_client.get.return_value = MockResponse(200, {"data": [{"id": "m1"}]})
        
        # Mock stream
        lines = [
            'data: {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}',
            'data: {"choices": [{"delta": {"content": "hello"}}]}',
            'data: [DONE]'
        ]
        mock_client.stream.return_value = MockStreamContext(MockResponse(200, lines=lines))
        
        outputs = []
        async for chunk in lemonade.generate_stream(prompt="Hi"):
            outputs.append(chunk)
            
        assert outputs == ["<thinking>", "thinking...", "</thinking>", "hello"]

@pytest.mark.asyncio
async def test_generate_stream_404_fallback(lemonade):
    with patch("proxy.backend_client.lemonade_client.httpx.AsyncClient") as mock_httpx:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        mock_client.get.return_value = MockResponse(200, {"data": [{"id": "m1"}]})
        
        # 1st call 404, 2nd call 200 with completions format
        lines = [
            'data: {"choices": [{"text": "fallback text"}]}',
            'data: [DONE]'
        ]
        mock_client.stream.side_effect = [
            MockStreamContext(MockResponse(404)),
            MockStreamContext(MockResponse(200, lines=lines))
        ]
        
        outputs = []
        async for chunk in lemonade.generate_stream(prompt="Hi"):
            outputs.append(chunk)
            
        assert outputs == ["fallback text"]

@pytest.mark.asyncio
async def test_generate_stream_error_status(lemonade):
    with patch("proxy.backend_client.lemonade_client.httpx.AsyncClient") as mock_httpx:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        mock_client.get.return_value = MockResponse(200, {"data": []})
        mock_client.stream.return_value = MockStreamContext(MockResponse(500))
        
        outputs = []
        async for chunk in lemonade.generate_stream():
            outputs.append(chunk)
            
        assert outputs == ["Error from LLM Backend: 500"]

@pytest.mark.asyncio
async def test_generate_stream_exception_fallback(lemonade):
    with patch("proxy.backend_client.lemonade_client.httpx.AsyncClient") as mock_httpx:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        mock_client.stream.side_effect = Exception("Connection Refused")
        
        outputs = []
        async for chunk in lemonade.generate_stream():
            outputs.append(chunk)
            
        assert "I hear movements nearby." in "".join(outputs)
