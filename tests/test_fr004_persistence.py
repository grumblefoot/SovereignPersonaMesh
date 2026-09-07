import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from evennia_world.app import app, _persist_room, _log_objective_action, DB_CONFIG, RoomMetadata

@pytest.fixture
def sample_room():
    return RoomMetadata(
        room_id="test_persist_room",
        room_name="Persist Room",
        description="A room for testing persistence",
        lighting="normal",
        exits=[],
        present_characters=["test_actor"],
        nearby_objects=[]
    )

class AsyncContextManagerMock:
    def __init__(self, conn):
        self.conn = conn
    async def __aenter__(self):
        return self.conn
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_persist_room(sample_room):
    # Mock the database pool and connection
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value = AsyncContextManagerMock(mock_conn)
    
    with patch("evennia_world.app._db_pool", mock_pool):
        await _persist_room("test_session", "dungeon_cellar", "test_persist_room", sample_room)
        
        # Verify execute was called
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args[0]
        assert "INSERT INTO world_state_sessions" in args[0]
        assert "ON CONFLICT (session_id, template_key, room_id)" in args[0]
        assert args[1] == "test_session"
        assert args[2] == "dungeon_cellar"
        assert args[3] == "test_persist_room"
        
        # Verify JSON
        room_data = json.loads(args[4])
        assert room_data["room_id"] == "test_persist_room"
        assert "test_actor" in room_data["present_characters"]

@pytest.mark.asyncio
async def test_log_objective_action():
    # Mock the database pool and connection
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value = AsyncContextManagerMock(mock_conn)
    
    with patch("evennia_world.app._db_pool", mock_pool):
        await _log_objective_action(
            "test_session",
            1421,
            "test_actor",
            "tavern",
            "speak",
            "Hello test"
        )
        
        # Verify execute was called
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args[0]
        assert "INSERT INTO objective_world_log" in args[0]
        assert args[1] == "test_session"
        assert args[2] == 1421
        assert args[3] == "test_actor"
        assert args[4] == "tavern"
        assert args[5] == "speak"
        assert args[6] == "Hello test"
