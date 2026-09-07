import pytest
import asyncio
from unittest.mock import patch, MagicMock

from proxy.core.stream_parser import MonologueStreamParser
from proxy.api.routes import _dispatch_gm_actions, evennia_client

@pytest.mark.asyncio
async def test_gm_actions_dispatch_flow():
    # Setup the stream parser with some actions in the monologue
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = """
    I'll move to the tavern.
    [GM_ACTION: {"type": "MOVE", "entity": "Player1", "room_id": "tavern_main"}]
    And I'll create a backroom.
    [GM_ACTION: {"type": "CREATE_ROOM", "room_id": "tavern_backroom", "name": "Backroom", "desc": "A shady place."}]
    """
    
    with patch.object(evennia_client, 'move_character', new_callable=MagicMock) as mock_move, \
         patch.object(evennia_client, 'create_room', new_callable=MagicMock) as mock_create:
        
        # Make mocks awaitable
        async def mock_async_move(*args, **kwargs): pass
        async def mock_async_create(*args, **kwargs): pass
        mock_move.side_effect = mock_async_move
        mock_create.side_effect = mock_async_create
        
        # Dispatch actions
        _dispatch_gm_actions(parser, session_id="test_session", target_char="Player1")
        
        # Yield to event loop to allow asyncio.create_task to run
        await asyncio.sleep(0.01)
        
        # Assert evennia_client methods were called with expected arguments
        mock_move.assert_called_once()
        move_kwargs = mock_move.call_args[1]
        assert move_kwargs["character_id"] == "Player1"
        assert move_kwargs["room_id"] == "tavern_main"
        assert move_kwargs["session_id"] == "test_session"
        assert "idempotency_key" in move_kwargs
        
        mock_create.assert_called_once()
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["room_id"] == "tavern_backroom"
        assert create_kwargs["name"] == "Backroom"
        assert create_kwargs["desc"] == "A shady place."
        assert create_kwargs["session_id"] == "test_session"
        assert "idempotency_key" in create_kwargs
