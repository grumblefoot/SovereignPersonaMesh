"""Sprint 1 Track A regression tests: per-session tick + spatial gating over HTTP.

Covers (HERMES_TASK.md / SPRINT_PLAN.md §2 Track A, decisions 9/10/11):
  - POST /api/v1/world/tick is idempotent per turn_id (decision 10, gap 3).
  - Two sessions tick independently.
  - /world/action with a repeated turn_id (regeneration) leaves the tick
    unchanged; without turn_id it keeps the legacy advance-per-call.
  - GET /health reports the default_session tick.
  - 5 ft / 15 ft gating edges through the HTTP action path.
  - CLOSED_DOOR and METAL_PARTITION black out speech; no hysteresis carryover
    (decision 9).
  - SHOUT lands one gating level better than speak, but barrier blackouts
    stand (decision 11, gap 8).
  - Adjacency uses the ACTING session's rooms when two sessions hold
    different layouts under the same room ids (gating plan phase 1).
"""

import pytest

from evennia_world.hybrid_builder import HybridWorldBuilder
from evennia_world.models import RoomMetadata


@pytest.fixture(scope="module")
def app_module():
    from evennia_world import app as app_mod
    return app_mod


@pytest.fixture(autouse=True)
def reset_state(app_module):
    """Fresh engine state before every test (tick maps included)."""
    from evennia_world.app import app as fa_app
    fa_app.state.start_time = 0
    app_module.app_state.current_world = {}
    app_module.app_state.room_to_template = {}
    app_module.app_state.session_worlds = {}
    app_module.app_state.idempotency_seen = {}
    app_module.app_state.action_tick_counter = 0
    app_module.app_state.session_ticks = {}
    app_module.app_state.session_turn_ticks = {}
    app_module.lock_manager = type(app_module.lock_manager)(default_ttl=60.0)
    app_module.world_builder = HybridWorldBuilder()
    yield


@pytest.fixture
def client(app_module):
    from starlette.testclient import TestClient
    with TestClient(app=app_module.app, base_url="http://test") as c:
        yield c


def tick(client, session_id, turn_id):
    r = client.post("/api/v1/world/tick",
                    json={"session_id": session_id, "turn_id": turn_id})
    assert r.status_code == 200, r.text
    return r.json()


def action(client, char_id, action_type, raw_text, session_id="default_session",
           template_key="dynamic", turn_id=None, target_id=None):
    body = {
        "character_id": char_id,
        "action_type": action_type,
        "raw_text": raw_text,
        "session_id": session_id,
        "template_key": template_key,
    }
    if turn_id is not None:
        body["turn_id"] = turn_id
    if target_id is not None:
        body["target_id"] = target_id
    r = client.post("/api/v1/world/action", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def consequences_by(json_body):
    return {c["recipient_id"]: c for c in json_body["consequences"]}


def seed_layout(app_module, session_id, template_key, rooms):
    """Install a bespoke topology directly into the session's world.

    rooms: {room_id: (exits, [characters])}. This is the layout the acting
    session's /world/action must consult — never the legacy global world.
    """
    world = {
        rid: RoomMetadata(room_id=rid, room_name=rid, description="test room",
                          lighting="normal", exits=list(exits),
                          present_characters=list(chars))
        for rid, (exits, chars) in rooms.items()
    }
    app_module.app_state.session_worlds.setdefault(session_id, {})[template_key] = world
    return world


# ── Tick endpoint: idempotency per turn_id (decision 10, gap 3) ────────────

def test_tick_advances_once_per_new_turn_id(client):
    a = tick(client, "s1", "turn_a")
    assert a == {"tick": 1, "turn_id": "turn_a", "advanced": True}
    b = tick(client, "s1", "turn_b")
    assert b == {"tick": 2, "turn_id": "turn_b", "advanced": True}


def test_tick_idempotent_per_turn_id(client):
    first = tick(client, "s1", "turn_a")
    for _ in range(3):  # regeneration / swipe retries
        again = tick(client, "s1", "turn_a")
        assert again["tick"] == first["tick"]
        assert again["advanced"] is False


def test_two_sessions_tick_independently(client):
    assert tick(client, "sA", "t1")["tick"] == 1
    assert tick(client, "sB", "t1")["tick"] == 1
    assert tick(client, "sA", "t2")["tick"] == 2
    assert tick(client, "sB", "t2")["tick"] == 2
    # sC has never ticked: its actions start from 1, not from sA/sB's clock.
    res = action(client, "zoe", "speak", "hi", session_id="sC")
    assert res["action_tick"] == 1


# ── /world/action tick semantics ───────────────────────────────────────────

def test_action_with_repeated_turn_id_leaves_tick_unchanged(client):
    r1 = action(client, "rowan", "speak", "hello", turn_id="turn_x")
    r2 = action(client, "rowan", "speak", "hello", turn_id="turn_x")
    assert r1["action_tick"] == r2["action_tick"]
    # A new turn moves the clock exactly once.
    r3 = action(client, "rowan", "speak", "hello", turn_id="turn_y")
    assert r3["action_tick"] == r1["action_tick"] + 1


def test_action_without_turn_id_keeps_legacy_advance_per_call(client):
    t1 = action(client, "rowan", "speak", "one")["action_tick"]
    t2 = action(client, "rowan", "speak", "two")["action_tick"]
    t3 = action(client, "rowan", "speak", "three")["action_tick"]
    assert t1 < t2 < t3


def test_action_turn_id_shares_clock_with_tick_endpoint(client):
    """A turn already opened by /world/tick is not advanced again by /action."""
    t = tick(client, "s1", "turn_shared")
    a = action(client, "rowan", "speak", "hi", turn_id="turn_shared")
    assert a["action_tick"] == t["tick"]
    again = action(client, "rowan", "speak", "hi again", turn_id="turn_shared")
    assert again["action_tick"] == t["tick"]


def test_health_reports_default_session_tick(client):
    tick(client, "default_session", "h1")
    tick(client, "other", "h1")
    tick(client, "other", "h2")
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["tick"] == 1  # default_session tick, not 'other', not 1420


# ── Spatial gating over HTTP: thresholds, barriers, no hysteresis ──────────

def test_same_room_speak_is_direct_over_http(client, app_module):
    seed_layout(app_module, "geo", "layout", {
        "hall_a": ([], ["rowan", "solo"]),
    })
    res = consequences_by(action(client, "rowan", "speak", "the key is cold",
                                 session_id="geo", template_key="layout"))
    assert res["solo"]["gating_level"] == "direct"
    assert "the key is cold" in res["solo"]["sensory_feed"]


def test_closed_door_blacks_out_speech_over_http(client, app_module):
    """Adjacent rooms joined by an exit (15 ft closed-door geometry): silence."""
    seed_layout(app_module, "lay", "layout", {
        "hall_a": (["hall_b"], ["rowan"]),
        "hall_b": (["hall_a"], ["solo"]),
    })
    res = consequences_by(action(client, "rowan", "speak", "psst",
                                 session_id="lay", template_key="layout"))
    # BLACKOUT is filtered out of the consequence payload entirely — the
    # listener receives nothing, not even a murmur. (The actor always sees
    # their own utterance, so only other recipients are checked for leaks.)
    assert "solo" not in res
    feeds = {rid: c["sensory_feed"] for rid, c in res.items() if rid != "rowan"}
    assert all("psst" not in f for f in feeds.values())


def test_metal_partition_blacks_out_speech_over_http(client, app_module):
    """Same-room geometry behind a metal partition: BLACKOUT per SRD 3.3.2."""
    import evennia_world.app as app_mod
    from evennia_world.models import BarrierType

    seed_layout(app_module, "met", "metal", {
        "lab_a": ([], ["rowan", "solo"]),
    })

    orig = app_mod._compute_distance_and_barriers
    app_mod._compute_distance_and_barriers = lambda a, t, at, world=None: (
        (3.0, [BarrierType.METAL_PARTITION]) if a == t else orig(a, t, at, world=world)
    )
    try:
        body = action(client, "rowan", "speak", "hidden text",
                      session_id="met", template_key="metal")
        # Metal partition at 3 ft → BLACKOUT: filtered from consequences,
        # the words never appear in the response.
        assert "hidden text" not in str(body)
    finally:
        app_mod._compute_distance_and_barriers = orig


def test_no_hysteresis_carryover_over_http(client, app_module):
    """Leaving a room blacks the listener out on the NEXT action — no
    2-tick DEGRADED carryover (decision 9)."""
    world = seed_layout(app_module, "hyst", "layout", {
        "hall_a": ([], ["rowan", "solo"]),
        "hall_far": ([], []),  # not connected to hall_a
    })

    # Tick 1: same room — DIRECT.
    res = consequences_by(action(client, "rowan", "speak", "one",
                                 session_id="hyst", template_key="layout"))
    assert res["solo"]["gating_level"] == "direct"

    # Move to a non-adjacent room, then speak immediately on the next tick.
    world["hall_a"].present_characters.remove("rowan")
    world["hall_far"].present_characters.append("rowan")

    res2 = consequences_by(action(client, "rowan", "speak", "two",
                                  session_id="hyst", template_key="layout"))
    # With the old hysteresis, solo would still hear DEGRADED here; now it is
    # instant BLACKOUT (no consequence at all).
    assert "solo" not in res2
    feeds = {rid: c["sensory_feed"] for rid, c in res2.items() if rid != "rowan"}
    assert all("two" not in f for f in feeds.values())


# ── SHOUT over HTTP (decision 11, gap 8) ───────────────────────────────────

def test_shout_is_accepted_action_type(client, app_module):
    seed_layout(app_module, "acc", "layout", {"hall_a": ([], ["rowan"])})
    res = action(client, "rowan", "shout", "GUARDS!",
                 session_id="acc", template_key="layout")
    assert res["success"] is True


def test_shout_lands_direct_where_speak_would(client, app_module):
    """10 ft apart (DEGRADED band for speak): shout is DIRECT with verbatim."""
    import evennia_world.app as app_mod

    seed_layout(app_module, "shouty", "layout", {
        "hall_a": (["hall_b"], ["rowan"]),
        "hall_b": (["hall_a"], ["solo"]),
    })

    # Force the 10 ft open-space geometry (no barrier) for cross-room pairs.
    orig = app_mod._compute_distance_and_barriers
    app_mod._compute_distance_and_barriers = lambda a, t, at, world=None: (
        (10.0, []) if (a != t and a and t) else orig(a, t, at, world=world)
    )
    try:
        speak_res = consequences_by(action(
            client, "rowan", "speak", "can you hear me",
            session_id="shouty", template_key="layout"))
        assert speak_res["solo"]["gating_level"] == "degraded"

        shout_res = consequences_by(action(
            client, "rowan", "shout", "GUARDS INTRUDERS",
            session_id="shouty", template_key="layout"))
        assert shout_res["solo"]["gating_level"] == "direct"
        assert "GUARDS INTRUDERS" in shout_res["solo"]["sensory_feed"]
    finally:
        app_mod._compute_distance_and_barriers = orig


def test_shout_cannot_lift_barrier_blackout_over_http(client, app_module):
    """Closed door between rooms: even a shout produces no consequence."""
    seed_layout(app_module, "doorwall", "layout", {
        "hall_a": (["hall_b"], ["rowan"]),
        "hall_b": (["hall_a"], ["solo"]),
    })
    res = consequences_by(action(client, "rowan", "shout", "CANNOT REACH YOU",
                                 session_id="doorwall", template_key="layout"))
    assert "solo" not in res  # BLACKOUT filtered from consequences
    feeds = {rid: c["sensory_feed"] for rid, c in res.items() if rid != "rowan"}
    assert all("CANNOT REACH YOU" not in f for f in feeds.values())


# ── Session-scoped adjacency (gating plan phase 1) ─────────────────────────

def test_adjacency_uses_acting_sessions_layout(client, app_module):
    """Two sessions share room ids but have different topologies.

    Session 'open' has hall_a <-> hall_b adjacent (open geometry at 10 ft);
    session 'walled' has NO edge between them (45 ft, wall). A speaker in
    hall_a must be heard DEGRADED in session 'open' and not at all in
    session 'walled' — proving adjacency is read from the acting session's
    world, never the legacy global.
    """
    import evennia_world.app as app_mod

    seed_layout(app_module, "open", "twin", {
        "hall_a": (["hall_b"], ["rowan"]),
        "hall_b": (["hall_a"], ["solo"]),
    })
    seed_layout(app_module, "walled", "twin", {
        "hall_a": ([], ["rowan"]),      # no exits: hall_b unreachable
        "hall_b": ([], ["solo"]),
    })

    # Any edge in the acting world resolves to a 10 ft open corridor;
    # missing edges fall through to the real 45 ft + wall computation.
    orig = app_mod._compute_distance_and_barriers
    def patched(actor_room, target_room_id, action_type, world=None):
        if world is not None and actor_room and actor_room in world \
           and target_room_id in world[actor_room].exits:
            return (10.0, [])
        return orig(actor_room, target_room_id, action_type, world=world)
    app_mod._compute_distance_and_barriers = patched
    try:
        open_res = consequences_by(action(
            client, "rowan", "speak", "open session audible",
            session_id="open", template_key="twin"))
        assert open_res["solo"]["gating_level"] == "degraded"

        walled_res = consequences_by(action(
            client, "rowan", "speak", "walled session audible",
            session_id="walled", template_key="twin"))
        assert "solo" not in walled_res  # 45 ft + wall → BLACKOUT
    finally:
        app_mod._compute_distance_and_barriers = orig
