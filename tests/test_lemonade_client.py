import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from proxy.backend_client.lemonade_client import LemonadeLLMClient

@pytest.fixture
def client():
    return LemonadeLLMClient(base_url="http://test-server/v1")

@pytest.mark.asyncio
async def test_resolve_model_found(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "google/gemma-4-26B-A4B-it"}]}
    
    mock_httpx = AsyncMock()
    mock_httpx.get.return_value = mock_resp
    
    result = await client._resolve_model("google/gemma-4-26B-A4B-it", mock_httpx)
    assert result == "google/gemma-4-26B-A4B-it"

@pytest.mark.asyncio
async def test_resolve_model_fallback(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "some-other-model"}]}
    
    mock_httpx = AsyncMock()
    mock_httpx.get.return_value = mock_resp
    
    result = await client._resolve_model("unknown-model", mock_httpx)
    assert result == "some-other-model"

@pytest.mark.asyncio
async def test_generate_stream_success(client):
    # Mocking httpx.AsyncClient to yield SSE events
    mock_client_instance = AsyncMock()
    
    # We need a mock context manager for the client
    class AsyncClientContextMock:
        async def __aenter__(self):
            return mock_client_instance
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # We need a mock stream response
    mock_stream_response = AsyncMock()
    mock_stream_response.status_code = 200

    async def mock_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
        yield 'data: [DONE]'

    mock_stream_response.aiter_lines = mock_aiter_lines

    class StreamContextMock:
        async def __aenter__(self):
            return mock_stream_response
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client_instance.stream = MagicMock(return_value=StreamContextMock())
    # Also mock the get for resolve_model
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {"data": [{"id": "test-model"}]}
    mock_client_instance.get.return_value = mock_get_resp

    with patch('httpx.AsyncClient', return_value=AsyncClientContextMock()):
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi", model="test-model"):
            tokens.append(chunk)

        assert tokens == ["Hello", " World"]

@pytest.mark.asyncio
async def test_generate_stream_404_fallback(client):
    # Mocking httpx.AsyncClient to yield 404 then 200
    mock_client_instance = AsyncMock()
    
    class AsyncClientContextMock:
        async def __aenter__(self):
            return mock_client_instance
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_stream_response_404 = AsyncMock()
    mock_stream_response_404.status_code = 404

    mock_stream_response_200 = AsyncMock()
    mock_stream_response_200.status_code = 200

    async def mock_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "Fallback"}}]}'
        yield 'data: [DONE]'

    mock_stream_response_200.aiter_lines = mock_aiter_lines

    class StreamContextMock404:
        async def __aenter__(self):
            return mock_stream_response_404
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class StreamContextMock200:
        async def __aenter__(self):
            return mock_stream_response_200
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Return 404 on first stream call, 200 on second stream call
    mock_client_instance.stream = MagicMock(side_effect=[StreamContextMock404(), StreamContextMock200()])
    
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {"data": [{"id": "test-model"}]}
    mock_client_instance.get.return_value = mock_get_resp

    with patch('httpx.AsyncClient', return_value=AsyncClientContextMock()):
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi", model="test-model"):
            tokens.append(chunk)

        assert tokens == ["Fallback"]

@pytest.mark.asyncio
async def test_generate_stream_decode_error(client):
    mock_client_instance = AsyncMock()
    
    class AsyncClientContextMock:
        async def __aenter__(self):
            return mock_client_instance
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_stream_response = AsyncMock()
    mock_stream_response.status_code = 200

    async def mock_aiter_lines():
        yield 'data: invalid json'
        yield 'data: {"choices": [{"delta": {"content": "Valid"}}]}'
        yield 'data: [DONE]'

    mock_stream_response.aiter_lines = mock_aiter_lines

    class StreamContextMock:
        async def __aenter__(self):
            return mock_stream_response
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client_instance.stream = MagicMock(return_value=StreamContextMock())
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {"data": [{"id": "test-model"}]}
    mock_client_instance.get.return_value = mock_get_resp

    with patch('httpx.AsyncClient', return_value=AsyncClientContextMock()):
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi"):
            tokens.append(chunk)

        assert tokens == ["Valid"]
