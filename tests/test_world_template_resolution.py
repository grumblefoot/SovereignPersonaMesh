"""Engine template resolution (QA assessment 2026-10-05).

Every endpoint must act on the session's OWN configured world. Before this fix,
/world/action defaulted to the 'dynamic' template; for a seeded session that world
was an empty dict, which is falsy, so the code fell back to the process-global
app_state.current_world — whichever session configured LAST. Session A's speech
could be gated against session B's rooms and occupants. GM moves/rooms likewise
landed in a phantom 'dynamic' world the characters never perceived.
"""
from starlette.testclient import TestClient

import evennia_world.app as world_app
from evennia_world.hybrid_builder import HybridWorldBuilder


def fresh_engine(monkeypatch):
    monkeypatch.setenv("SPM_WORLD_DB", "0")
    st = world_app.app_state
    st._db_pool = None
    for attr in ("session_worlds", "session_ticks", "session_turn_ticks",
                 "idempotency_seen", "session_edges", "mutation_log"):
        setattr(st, attr, {})
    st.current_world = {}
    world_app.world_builder = HybridWorldBuilder()
    return TestClient(world_app.app)


def configure(c, session, placements):
    r = c.post("/api/v1/world/configure", json={
        "template_key": "dungeon_cellar", "session_id": session,
        "placements": [{"character_id": ch, "room_id": room} for ch, room in placements]})
    assert r.status_code == 200, r.text


def recipients(c, session, actor="user"):
    r = c.post("/api/v1/world/action", json={
        "character_id": actor, "action_type": "speak", "raw_text": "hello",
        "session_id": session})
    assert r.status_code == 200, r.text
    return sorted(x["recipient_id"] for x in r.json()["consequences"])


def test_action_never_uses_another_sessions_world(monkeypatch):
    with fresh_engine(monkeypatch) as c:
        configure(c, "sess_A", [("user", "cellar"), ("alice", "cellar")])
        configure(c, "sess_B", [("user", "cellar"), ("bob", "cellar")])   # B configured LAST
        assert recipients(c, "sess_A") == ["alice"]       # was: ['bob'] via current_world
        assert recipients(c, "sess_B") == ["bob"]


def test_gm_move_and_room_land_in_the_live_world(monkeypatch):
    with fresh_engine(monkeypatch) as c:
        configure(c, "sess_G", [("user", "cellar"), ("mira", "cellar")])
        r = c.post("/api/v1/world/rooms", json={
            "room_id": "wine_cellar", "room_name": "Wine Cellar", "description": "x",
            "session_id": "sess_G", "template_key": ""})
        assert r.status_code == 200, r.text
        r = c.post("/api/v1/world/move", json={
            "character_id": "user", "room_id": "wine_cellar",
            "session_id": "sess_G", "template_key": ""})
        assert r.status_code == 200, r.text
        snap = c.get("/api/v1/world/snapshot",
                     params={"session_id": "sess_G", "template_key": ""}).json()
        where = {o["entity_id"]: o["room_id"] for o in snap["occupants"]}
        assert where == {"user": "wine_cellar", "mira": "cellar"}   # one world, both present
        assert set(world_app.app_state.session_worlds["sess_G"]) == {"dungeon_cellar"}


def test_moving_the_player_in_one_chat_leaves_other_chats_untouched(monkeypatch):
    """Every chat's player is 'user'; a move in chat A must not strip them from B."""
    with fresh_engine(monkeypatch) as c:
        configure(c, "sess_A", [("user", "cellar"), ("alice", "cellar")])
        configure(c, "sess_B", [("user", "cellar"), ("bob", "cellar")])
        r = c.post("/api/v1/world/move", json={
            "character_id": "user", "room_id": "tavern_upstairs", "session_id": "sess_A"})
        assert r.status_code == 200, r.text
        snap_b = c.get("/api/v1/world/snapshot", params={"session_id": "sess_B"}).json()
        assert {o["entity_id"]: o["room_id"] for o in snap_b["occupants"]} == {
            "user": "cellar", "bob": "cellar"}


def test_place_endpoint_never_mutates_the_shared_template(monkeypatch):
    """A placement in one session must not appear in a session created later."""
    with fresh_engine(monkeypatch) as c:
        r = c.post("/api/v1/world/characters", json={
            "character_id": "ghost", "room_id": "cellar",
            "template_key": "dungeon_cellar", "session_id": "sess_early"})
        assert r.status_code == 200, r.text
        configure(c, "sess_late", [("user", "cellar")])
        snap = c.get("/api/v1/world/snapshot", params={"session_id": "sess_late"}).json()
        assert "ghost" not in {o["entity_id"] for o in snap["occupants"]}
