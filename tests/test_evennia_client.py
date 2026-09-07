import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from proxy.backend_client.evennia_client import EvenniaWorldClient

@pytest.fixture
def client():
    return EvenniaWorldClient(base_url="http://test-server/api/v1")

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        
    def json(self):
        return self._json_data

# For mocking httpx.AsyncClient context manager
class AsyncClientMockContext:
    def __init__(self, mock_instance):
        self.mock_instance = mock_instance
    async def __aenter__(self):
        return self.mock_instance
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_submit_action_success(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.return_value = MockResponse(200, {"success": True, "action_tick": 1422})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.submit_action("char1", "move", "go north")
        assert result == {"success": True, "action_tick": 1422}
        mock_httpx.post.assert_called_once()
        args, kwargs = mock_httpx.post.call_args
        assert "world/action" in args[0]
        assert kwargs["json"]["raw_text"] == "go north"

@pytest.mark.asyncio
async def test_submit_action_failure_fallback(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.return_value = MockResponse(500, {})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.submit_action("char1", "move", "go north")
        assert result["success"] is True
        assert result["action_tick"] == 1421
        assert result["consequences"][0]["sensory_feed"] == "go north"

@pytest.mark.asyncio
async def test_submit_action_exception_fallback(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.side_effect = Exception("Connection refused")
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.submit_action("char1", "move", "go north")
        assert result["success"] is True

@pytest.mark.asyncio
async def test_get_character_state_success(client):
    mock_httpx = AsyncMock()
    mock_httpx.get.return_value = MockResponse(200, {"character_id": "char1", "sensory_feed": "test"})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.get_character_state("char1")
        assert result == {"character_id": "char1", "sensory_feed": "test"}
        mock_httpx.get.assert_called_once()
        assert "world/state" in mock_httpx.get.call_args[0][0]

@pytest.mark.asyncio
async def test_get_character_state_failure_fallback(client):
    mock_httpx = AsyncMock()
    mock_httpx.get.return_value = MockResponse(404, {})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.get_character_state("char1")
        assert result["character_id"] == "char1"
        assert result["sensory_feed"] == "Standing in room."

@pytest.mark.asyncio
async def test_move_character_success(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.return_value = MockResponse(200, {"status": "moved"})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.move_character("char1", "room1", idempotency_key="key1")
        assert result == {"status": "moved"}
        mock_httpx.post.assert_called_once()
        args, kwargs = mock_httpx.post.call_args
        assert "world/move" in args[0]
        assert kwargs["json"]["room_id"] == "room1"
        assert kwargs["headers"]["X-Idempotency-Key"] == "key1"

@pytest.mark.asyncio
async def test_move_character_failure(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.return_value = MockResponse(500, {})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.move_character("char1", "room1")
        assert result == {}

@pytest.mark.asyncio
async def test_create_room_success(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.return_value = MockResponse(200, {"room_id": "room2"})
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.create_room("room2", "Name", "Desc", idempotency_key="key2")
        assert result == {"room_id": "room2"}
        mock_httpx.post.assert_called_once()
        args, kwargs = mock_httpx.post.call_args
        assert "world/rooms" in args[0]
        assert kwargs["json"]["room_id"] == "room2"
        assert kwargs["json"]["room_name"] == "Name"
        assert kwargs["headers"]["X-Idempotency-Key"] == "key2"

@pytest.mark.asyncio
async def test_create_room_failure(client):
    mock_httpx = AsyncMock()
    mock_httpx.post.side_effect = httpx.ConnectError("Network error")
    
    with patch("httpx.AsyncClient", return_value=AsyncClientMockContext(mock_httpx)):
        result = await client.create_room("room2", "Name", "Desc")
        assert result == {}
