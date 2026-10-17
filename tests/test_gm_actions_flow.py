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


@pytest.mark.asyncio
async def test_n13_withheld_reply_dispatches_no_gm_actions(monkeypatch):
    """QA N13 (2026-10-12): a planning-only reply was withheld ("not saved"), yet its
    CREATE_ROOM + MOVE still ran and the narrator followed. Tainted turn: no GM, no sync."""
    import proxy.api.routes as routes
    from proxy.core.stream_parser import MonologueStreamParser
    calls = []

    async def fake_dispatch(*a, **k):
        calls.append("dispatch")

    async def fake_sync(*a, **k):
        calls.append("sync")

    async def fake_record(*a, tainted=False, **k):
        calls.append(("record", tainted))

    monkeypatch.setattr(routes, "_dispatch_gm_actions", fake_dispatch)
    monkeypatch.setattr(routes, "_sync_narrator_with_player", fake_sync)
    monkeypatch.setattr(routes, "_record_reply_action", fake_record)

    p = MonologueStreamParser(initial_state=0)
    p._monologue_sections.append('[GM_ACTION: {"type": "MOVE", "entity": "user", "room_id": "office"}]')
    p.is_tainted = True
    await routes._gm_then_reply(p, "s", "narr", "", "s:1", turn_index=1, persona="", narrator=True)
    assert calls == [("record", True)]

    calls.clear()
    p.is_tainted = False
    await routes._gm_then_reply(p, "s", "narr", "reply", "s:1", turn_index=1, persona="", narrator=True)
    assert calls == ["dispatch", "sync", ("record", False)]


@pytest.mark.asyncio
async def test_q10_offscene_move_needs_the_story_to_mention_the_character(monkeypatch):
    """QA Q10 (2026-10-13): Lian's turn moved Mei in from the kitchen with no narrative
    cause. A move of a character who is elsewhere is applied only when the player's
    message or the reply names them; moves within the scene are unaffected."""
    import proxy.api.routes as routes
    from proxy.core.stream_parser import MonologueStreamParser
    snap = {"rooms": [{"room_id": r} for r in ("living", "kitchen", "garden")],
            "occupants": [{"entity_id": "user", "room_id": "living"},
                          {"entity_id": "lian", "room_id": "living"},
                          {"entity_id": "mei", "room_id": "kitchen"}]}
    moved = []

    async def fake_snapshot(**kw):
        return snap

    async def fake_move(**kw):
        moved.append(kw["character_id"])

    monkeypatch.setattr(routes.evennia_client, "get_snapshot", fake_snapshot)
    monkeypatch.setattr(routes.evennia_client, "move_character", fake_move)
    monkeypatch.setattr(routes, "resolve_gm_actions_mode", lambda s: "full")

    def parser():
        p = MonologueStreamParser(initial_state=0)
        p._monologue_sections.append(
            '[GM_ACTION: {"type": "MOVE", "entity": "lian", "room_id": "garden"}]\n'
            '[GM_ACTION: {"type": "MOVE", "entity": "user", "room_id": "garden"}]\n'
            '[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "garden"}]\n'
            '[GM_ACTION: {"type": "NONE"}]')
        return p

    await routes._dispatch_gm_actions(parser(), "s", "lian", turn_index=1, persona="Vardus",
                                      story_text='"Would you show me the garden?" Lian took his arm.')
    assert sorted(moved) == ["lian", "user"]          # Mei stays in the kitchen

    # review B4: separators don't matter ("mei_lin" vs "Mei-Lin")
    snap["occupants"].append({"entity_id": "mei_lin", "room_id": "kitchen"})
    moved.clear()
    p2 = MonologueStreamParser(initial_state=0)
    p2._monologue_sections.append('[GM_ACTION: {"type": "MOVE", "entity": "mei_lin", "room_id": "garden"}]')
    await routes._dispatch_gm_actions(p2, "s", "lian", turn_index=3, persona="Vardus",
                                      story_text="Mei-Lin drifted out after them, unnoticed.")
    assert moved == ["mei_lin"]

    moved.clear()
    await routes._dispatch_gm_actions(parser(), "s", "lian", turn_index=2, persona="Vardus",
                                      story_text="Behind them, Mei slipped out of the kitchen to listen.")
    assert sorted(moved) == ["lian", "mei", "user"]   # motivated: the reply names her


# ── first turn: everyone in one room (owner rule, QA 2026-10-17) ─────────────

def test_colocate_first_turn_rules():
    from proxy.core.gm_planner import colocate_first_turn as co

    seed = {"user": "central_nexus", "alba": "central_nexus"}
    # The QA case: the reply sets the scene in a new hut, only the player was moved.
    acts = [{"type": "CREATE_ROOM", "room_id": "derelict_hut", "name": "Derelict Hut"},
            {"type": "MOVE", "entity": "user", "room_id": "derelict_hut"}]
    assert co(acts, seed) == [
        {"type": "CREATE_ROOM", "room_id": "derelict_hut", "name": "Derelict Hut"},
        {"type": "MOVE", "entity": "alba", "room_id": "derelict_hut"},
        {"type": "MOVE", "entity": "user", "room_id": "derelict_hut"}]
    # Only a character moved: the player follows (OPEN-023).
    assert co([{"type": "MOVE", "entity": "alba", "room_id": "garden"}], seed) == [
        {"type": "MOVE", "entity": "alba", "room_id": "garden"},
        {"type": "MOVE", "entity": "user", "room_id": "garden"}]
    # The player and a character move to different rooms: the player's room wins, and a
    # room nobody ends up in is not created.
    acts = [{"type": "CREATE_ROOM", "room_id": "cellar", "name": "Cellar"},
            {"type": "MOVE", "entity": "alba", "room_id": "cellar"},
            {"type": "MOVE", "entity": "user", "room_id": "node_alpha"}]
    assert co(acts, seed) == [
        {"type": "MOVE", "entity": "alba", "room_id": "node_alpha"},
        {"type": "MOVE", "entity": "user", "room_id": "node_alpha"}]
    # No move, already together: nothing to do.
    assert co([], seed) == []
    # No move but a split start (group chat): everyone joins the player.
    assert co([], {"user": "hall", "alba": "hall", "mei": "garden"}) == [
        {"type": "MOVE", "entity": "mei", "room_id": "hall"}]


async def test_first_turn_dispatch_moves_the_character_with_the_player(monkeypatch):
    """Replay of the QA turn 1: the planner created the hut and moved only the player.
    On turn 1 the character is moved too; on turn 2 the same actions move only the player."""
    import proxy.api.routes as routes
    monkeypatch.setattr(routes, "resolve_gm_actions_mode", lambda s: "full")
    acts = [{"type": "CREATE_ROOM", "room_id": "derelict_hut", "name": "Derelict Hut"},
            {"type": "MOVE", "entity": "user", "room_id": "derelict_hut"}]

    async def run(turn_index):
        moved = []

        async def snap(*a, **k):
            return {"rooms": [{"room_id": "central_nexus"}],
                    "occupants": [{"entity_id": "user", "room_id": "central_nexus"},
                                  {"entity_id": "alba", "room_id": "central_nexus"}]}

        async def move(*a, **k):
            moved.append((k.get("character_id"), k.get("room_id") or k.get("target_room_id")))

        async def create(*a, **k):
            return None

        with patch.object(evennia_client, "get_snapshot", side_effect=snap), \
             patch.object(evennia_client, "move_character", side_effect=move), \
             patch.object(evennia_client, "create_room", side_effect=create):
            await _dispatch_gm_actions(MonologueStreamParser(), session_id="s", target_char="alba",
                                       turn_index=turn_index, actions=[dict(a) for a in acts])
        return sorted(c for c, _ in moved)

    assert await run(1) == ["alba", "user"]
    assert await run(2) == ["user"]


def test_placeholder_rooms_are_offered_only_while_occupied():
    # QA 2026-10-17 turn 10: cabin -> sky was mapped onto the placeholder Central Nexus.
    from proxy.core.gm_planner import story_rooms
    rooms = [{"room_id": "central_nexus", "name": "Central Nexus"},
             {"room_id": "node_alpha", "name": "Node Alpha"},
             {"room_id": "derelict_hut", "name": "Derelict Hut"}]
    assert [r["room_id"] for r in story_rooms(rooms, {"user": "derelict_hut", "alba": "derelict_hut"})] \
        == ["derelict_hut"]
    # Turn 1: everyone still in the starting room, which stays on the list.
    assert [r["room_id"] for r in story_rooms(rooms, {"user": "central_nexus"})] \
        == ["central_nexus", "derelict_hut"]


# ── rooms join up as the story moves (owner, QA 2026-10-17) ──────────────────

def test_new_link_rules():
    from proxy.core.gm_planner import linked_pairs, new_link
    snap = {"rooms": [{"room_id": "kitchen", "exits": ["gardens"]}, {"room_id": "gardens", "exits": ["kitchen"]},
                      {"room_id": "derelict_hut", "exits": []}],
            "edges": [{"a": "central_nexus", "b": "node_alpha"}]}
    linked = linked_pairs(snap)
    assert new_link("derelict_hut", "sky", linked) == ("derelict_hut", "sky")
    assert new_link("kitchen", "gardens", linked) is None          # already linked
    assert new_link("gardens", "kitchen", linked) is None          # either direction
    assert new_link("central_nexus", "sky", linked) is None        # placeholder: not a story place
    assert new_link(None, "sky", linked) is None                   # not placed yet
    assert new_link("sky", "sky", linked) is None


async def test_gm_move_links_the_two_rooms_with_a_closed_door(monkeypatch):
    import proxy.api.routes as routes
    monkeypatch.setattr(routes, "resolve_gm_actions_mode", lambda s: "full")
    doors = []

    async def snap(*a, **k):
        return {"rooms": [{"room_id": "derelict_hut", "exits": []}, {"room_id": "central_nexus", "exits": []}],
                "edges": [],
                "occupants": [{"entity_id": "user", "room_id": "derelict_hut"},
                              {"entity_id": "alba", "room_id": "derelict_hut"},
                              {"entity_id": "mei", "room_id": "central_nexus"}]}

    async def ok(*a, **k):
        return {}

    async def barrier(a, b, **k):
        doors.append((a, b, k.get("barrier"), k.get("state")))

    acts = [{"type": "CREATE_ROOM", "room_id": "sky", "name": "Sky"},
            {"type": "MOVE", "entity": "user", "room_id": "sky"},
            {"type": "MOVE", "entity": "alba", "room_id": "sky"}]
    with patch.object(evennia_client, "get_snapshot", side_effect=snap), \
         patch.object(evennia_client, "move_character", side_effect=ok), \
         patch.object(evennia_client, "create_room", side_effect=ok), \
         patch.object(evennia_client, "set_barrier", side_effect=barrier):
        await _dispatch_gm_actions(MonologueStreamParser(), session_id="s", target_char="alba",
                                   turn_index=5, actions=acts, story_text="Alba rises into the sky")
    # One door for the hut -> sky path (two movers, one link), closed.
    assert doors == [("derelict_hut", "sky", "closed_door", "closed")]
