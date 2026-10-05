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
         patch.object(evennia_client, 'create_room', new_callable=MagicMock) as mock_create, \
         patch.object(evennia_client, 'get_snapshot', new_callable=MagicMock) as mock_snap:
        
        # Make mocks awaitable
        async def mock_async_move(*args, **kwargs): pass
        async def mock_async_create(*args, **kwargs): pass
        # The validator (A4) fetches one world snapshot to check entities and rooms.
        async def mock_async_snap(*args, **kwargs):
            return {"rooms": [{"room_id": "tavern_main"}],
                    "occupants": [{"entity_id": "Player1", "room_id": "tavern_main"}]}
        mock_move.side_effect = mock_async_move
        mock_create.side_effect = mock_async_create
        mock_snap.side_effect = mock_async_snap
        
        # Dispatch actions (this is an async function — await it directly)
        await _dispatch_gm_actions(parser, session_id="test_session", target_char="Player1")
        
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


# ── A3: dispatch gated by gm_actions_mode ───────────────────────────────────

def _loaded_parser():
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = (
        '[GM_ACTION: {"type": "MOVE", "entity": "Player1", "room_id": "tavern_main"}]'
        '[GM_ACTION: {"type": "CREATE_ROOM", "room_id": "annex", "name": "Annex", "desc": "x"}]')
    return parser


def _settings_patch(monkeypatch, mode):
    import proxy.api.routes as routes

    class _Mgr:
        def get_settings(self):
            return {"gm_actions_mode": mode, "gm_actions_max_per_turn": 4,
                    "gm_actions_max_rooms_per_session": 40}
    monkeypatch.setattr(routes, "get_settings_manager", lambda: _Mgr())


@pytest.mark.asyncio
async def test_gm_mode_off_makes_zero_engine_calls(monkeypatch):
    _settings_patch(monkeypatch, "off")
    with patch.object(evennia_client, 'move_character', new_callable=MagicMock) as mock_move, \
         patch.object(evennia_client, 'create_room', new_callable=MagicMock) as mock_create, \
         patch.object(evennia_client, 'get_snapshot', new_callable=MagicMock) as mock_snap:
        await _dispatch_gm_actions(_loaded_parser(), session_id="s", target_char="Player1")
        assert mock_move.call_count == 0
        assert mock_create.call_count == 0
        assert mock_snap.call_count == 0          # off never even reads the world


@pytest.mark.asyncio
async def test_gm_mode_move_only_drops_create_room(monkeypatch):
    _settings_patch(monkeypatch, "move_only")
    with patch.object(evennia_client, 'move_character', new_callable=MagicMock) as mock_move, \
         patch.object(evennia_client, 'create_room', new_callable=MagicMock) as mock_create, \
         patch.object(evennia_client, 'get_snapshot', new_callable=MagicMock) as mock_snap:
        async def snap(*a, **k):
            return {"rooms": [{"room_id": "tavern_main"}],
                    "occupants": [{"entity_id": "Player1", "room_id": "tavern_main"}]}
        async def ok(*a, **k):
            return {}
        mock_snap.side_effect = snap
        mock_move.side_effect = ok
        mock_create.side_effect = ok
        await _dispatch_gm_actions(_loaded_parser(), session_id="s", target_char="Player1")
        assert mock_move.call_count == 1
        assert mock_create.call_count == 0


@pytest.mark.asyncio
async def test_gm_validation_blocks_unknown_entity_and_room(monkeypatch):
    _settings_patch(monkeypatch, "full")
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = (
        '[GM_ACTION: {"type": "MOVE", "entity": "intruder", "room_id": "tavern_main"}]'
        '[GM_ACTION: {"type": "MOVE", "entity": "Player1", "room_id": "nowhere"}]')
    with patch.object(evennia_client, 'move_character', new_callable=MagicMock) as mock_move, \
         patch.object(evennia_client, 'get_snapshot', new_callable=MagicMock) as mock_snap:
        async def snap(*a, **k):
            return {"rooms": [{"room_id": "tavern_main"}],
                    "occupants": [{"entity_id": "Player1", "room_id": "tavern_main"}]}
        mock_snap.side_effect = snap
        await _dispatch_gm_actions(parser, session_id="s", target_char="Player1")
        assert mock_move.call_count == 0
