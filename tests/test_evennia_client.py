import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from proxy.backend_client.evennia_client import EvenniaWorldClient

@pytest.fixture
async def client():
    c = EvenniaWorldClient(base_url="http://test-server/api/v1")
    yield c
    await c.close()

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        
    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("Error", request=MagicMock(), response=self)

@pytest.mark.asyncio
async def test_submit_action_success(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(200, {"success": True, "action_tick": 1422})
        result = await client.submit_action("char1", "move", "go north")
        assert result == {"success": True, "action_tick": 1422}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "world/action" in args[0]
        assert kwargs["json"]["raw_text"] == "go north"

@pytest.mark.asyncio
async def test_submit_action_failure_fallback(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(500, {})
        with pytest.raises(httpx.HTTPStatusError):
            await client.submit_action("char1", "move", "go north")

@pytest.mark.asyncio
async def test_submit_action_exception_fallback(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection refused")
        with pytest.raises(httpx.RequestError):
            await client.submit_action("char1", "move", "go north")

@pytest.mark.asyncio
async def test_get_character_state_success(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MockResponse(200, {"character_id": "char1", "sensory_feed": "test"})
        result = await client.get_character_state("char1")
        assert result == {"character_id": "char1", "sensory_feed": "test"}
        mock_get.assert_called_once()
        assert "world/state" in mock_get.call_args[0][0]

@pytest.mark.asyncio
async def test_get_character_state_failure_fallback(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MockResponse(404, {})
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_character_state("char1")

@pytest.mark.asyncio
async def test_move_character_success(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(200, {"status": "moved"})
        result = await client.move_character("char1", "room1", idempotency_key="key1")
        assert result == {"status": "moved"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "world/move" in args[0]
        assert kwargs["json"]["room_id"] == "room1"
        assert kwargs["headers"]["X-Idempotency-Key"] == "key1"

@pytest.mark.asyncio
async def test_move_character_failure(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(500, {})
        with pytest.raises(httpx.HTTPStatusError):
            await client.move_character("char1", "room1")

@pytest.mark.asyncio
async def test_create_room_success(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(200, {"room_id": "room2"})
        result = await client.create_room("room2", "Name", "Desc", idempotency_key="key2")
        assert result == {"room_id": "room2"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "world/rooms" in args[0]
        assert kwargs["json"]["room_id"] == "room2"
        assert kwargs["json"]["room_name"] == "Name"
        assert kwargs["headers"]["X-Idempotency-Key"] == "key2"

@pytest.mark.asyncio
async def test_create_room_failure(client):
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Network error")
        with pytest.raises(httpx.ConnectError):
            await client.create_room("room2", "Name", "Desc")
