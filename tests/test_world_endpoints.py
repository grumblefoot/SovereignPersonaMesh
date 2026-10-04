"""Sprint 1 Track A2 regression tests: snapshot, stateful barriers, seeded configure,
per-session reset, mutation origins, idempotency conflicts, startup reload gating."""
import pytest
from starlette.testclient import TestClient

from evennia_world.hybrid_builder import HybridWorldBuilder


@pytest.fixture(scope="module")
def app_module():
    from evennia_world import app as app_mod
    return app_mod


@pytest.fixture(autouse=True)
def reset_state(app_module):
    app_module.app_state.current_world = {}
    app_module.app_state.room_to_template = {}
    app_module.app_state.session_worlds = {}
    app_module.app_state.idempotency_seen = {}
    app_module.app_state.session_ticks = {}
    app_module.app_state.session_turn_ticks = {}
    app_module.app_state.session_edges = {}
    app_module.app_state.mutation_log = {}
    app_module.world_builder = HybridWorldBuilder()
    yield


@pytest.fixture
def sid():
    """Unique session id per test: configure_world restores persisted rooms for a
    session from Postgres (by design), so reused ids would leak state between tests."""
    import uuid
    return f"s_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sid2():
    import uuid
    return f"s_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client(app_module):
    with TestClient(app=app_module.app, base_url="http://test") as c:
        yield c


def _place(client, sid, char, room, template="dungeon_cellar"):
    r = client.post("/api/v1/world/characters", json={
        "character_id": char, "room_id": room, "template_key": template, "session_id": sid})
    assert r.status_code == 200, r.text


# ── snapshot ────────────────────────────────────────────────────────────────

def test_snapshot_rooms_edges_occupants_tick(client, sid, sid2):
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid})
    _place(client, sid, "mira", "cellar")
    client.post("/api/v1/world/tick", json={"session_id": sid, "turn_id": "t1"})
    snap = client.get("/api/v1/world/snapshot", params={"session_id": sid, "template_key": "dungeon_cellar"}).json()
    assert snap["session_id"] == sid and snap["tick"] == 1
    rooms = {r["room_id"]: r for r in snap["rooms"]}
    assert "cellar" in rooms and "mira" in rooms["cellar"]["present_characters"]
    assert {"entity_id": "mira", "room_id": "cellar", "kind": "character", "posture": []} in snap["occupants"]
    assert all({"a", "b", "barrier", "state", "distance_ft"} <= set(e) for e in snap["edges"])
    # exit-derived default edge exists between cellar and tavern_upstairs
    assert any({e["a"], e["b"]} == {"cellar", "tavern_upstairs"} for e in snap["edges"])


def test_snapshot_create_on_read(client, sid, sid2):
    snap = client.get("/api/v1/world/snapshot", params={"session_id": sid}).json()
    assert snap["session_id"] == sid and isinstance(snap["rooms"], list)


# ── stateful barriers ───────────────────────────────────────────────────────

def test_barrier_open_door_restores_hearing(client, sid, sid2):
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid})
    _place(client, sid, "listener", "tavern_upstairs")
    _place(client, sid, "user", "cellar")

    def gating():
        res = client.post("/api/v1/world/action", json={
            "character_id": "user", "action_type": "speak", "raw_text": "hello up there",
            "session_id": sid, "template_key": "dungeon_cellar"}).json()
        by = {c["recipient_id"]: c for c in res["consequences"]}
        return by["listener"]["gating_level"], by["listener"]["sensory_feed"]

    # default exit edge: closed door → blackout, silent
    g, feed = gating()
    assert g == "blackout" and feed == ""
    # open the door → degraded (10 ft, no barrier)
    r = client.post("/api/v1/world/barrier", json={
        "session_id": sid, "template_key": "dungeon_cellar", "a": "cellar", "b": "tavern_upstairs",
        "barrier": "open_door", "state": "open", "distance_ft": 10.0})
    assert r.status_code == 200
    g, feed = gating()
    assert g == "degraded" and feed != ""
    # close it as metal → blackout again
    client.post("/api/v1/world/barrier", json={
        "session_id": sid, "template_key": "dungeon_cellar", "a": "cellar", "b": "tavern_upstairs",
        "barrier": "metal_partition", "state": "closed", "distance_ft": 10.0})
    g, feed = gating()
    assert g == "blackout" and feed == ""


def test_barrier_unknown_room_400_and_session_scoped(client, sid, sid2):
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid})
    r = client.post("/api/v1/world/barrier", json={
        "session_id": sid, "template_key": "dungeon_cellar", "a": "cellar", "b": "nowhere"})
    assert r.status_code == 400
    # an edge set in s1 must not exist in s2's snapshot as an explicit open edge
    client.post("/api/v1/world/barrier", json={
        "session_id": sid, "template_key": "dungeon_cellar", "a": "cellar", "b": "tavern_upstairs",
        "barrier": "open_door", "state": "open"})
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid2})
    snap2 = client.get("/api/v1/world/snapshot", params={"session_id": sid2, "template_key": "dungeon_cellar"}).json()
    edge2 = next(e for e in snap2["edges"] if {e["a"], e["b"]} == {"cellar", "tavern_upstairs"})
    assert edge2["state"] == "closed"


# ── idempotency: replay flag + conflict ─────────────────────────────────────

def test_duplicate_replay_flag_and_conflict_409(client, sid, sid2):
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid})
    _place(client, sid, "mira", "cellar")
    body = {"character_id": "mira", "room_id": "tavern_upstairs", "template_key": "dungeon_cellar",
            "session_id": sid, "idempotency_key": "k1"}
    r1 = client.post("/api/v1/world/move", json=body)
    assert r1.status_code == 200 and r1.json()["duplicate"] is False
    r2 = client.post("/api/v1/world/move", json=body)
    assert r2.status_code == 200 and r2.json()["duplicate"] is True
    # same key, different payload → 409, nothing applied
    r3 = client.post("/api/v1/world/move", json={**body, "room_id": "cellar"})
    assert r3.status_code == 409
    snap = client.get("/api/v1/world/snapshot", params={"session_id": sid, "template_key": "dungeon_cellar"}).json()
    rooms = {r["room_id"]: r for r in snap["rooms"]}
    assert "mira" in rooms["tavern_upstairs"]["present_characters"]
    assert "mira" not in rooms["cellar"]["present_characters"]


# ── configure placements ────────────────────────────────────────────────────

def test_configure_placements_applied(client, sid, sid2):
    r = client.post("/api/v1/world/configure", json={
        "template_key": "dungeon_cellar", "session_id": sid,
        "placements": [{"character_id": "mira", "room_id": "cellar"},
                        {"character_id": "tom", "room_id": "tavern_upstairs"}]})
    assert r.status_code == 200
    snap = client.get("/api/v1/world/snapshot", params={"session_id": sid, "template_key": "dungeon_cellar"}).json()
    rooms = {r["room_id"]: r for r in snap["rooms"]}
    assert "mira" in rooms["cellar"]["present_characters"]
    assert "tom" in rooms["tavern_upstairs"]["present_characters"]


def test_configure_bad_placement_places_nobody(client, sid, sid2):
    r = client.post("/api/v1/world/configure", json={
        "template_key": "dungeon_cellar", "session_id": sid,
        "placements": [{"character_id": "mira", "room_id": "cellar"},
                        {"character_id": "tom", "room_id": "no_such_room"}]})
    assert r.status_code == 400 and "no_such_room" in r.json()["detail"]
    snap = client.get("/api/v1/world/snapshot", params={"session_id": sid, "template_key": "dungeon_cellar"}).json()
    assert all(not room["present_characters"] for room in snap["rooms"])


# ── per-session reset + origin audit ────────────────────────────────────────

def test_per_session_reset_leaves_others_intact(client, app_module, sid, sid2):
    for s_ in (sid, sid2):
        client.post("/api/v1/world/configure", json={
            "template_key": "dungeon_cellar", "session_id": s_,
            "placements": [{"character_id": f"char_{s_}", "room_id": "cellar"}]})
        client.post("/api/v1/world/tick", json={"session_id": s_, "turn_id": "t1"})
    r = client.delete("/api/v1/world/admin/reset", params={"session_id": sid})
    assert r.status_code == 200
    assert sid not in app_module.app_state.session_worlds
    assert sid not in app_module.app_state.session_ticks
    snap2 = client.get("/api/v1/world/snapshot", params={"session_id": sid2, "template_key": "dungeon_cellar"}).json()
    assert snap2["tick"] == 1
    assert any(f"char_{sid2}" in room["present_characters"] for room in snap2["rooms"])


def test_origin_echoed_and_audited(client, app_module, sid, sid2):
    client.post("/api/v1/world/configure", json={"template_key": "dungeon_cellar", "session_id": sid})
    _place(client, sid, "mira", "cellar")
    r = client.post("/api/v1/world/move", json={
        "character_id": "mira", "room_id": "tavern_upstairs", "template_key": "dungeon_cellar",
        "session_id": sid, "origin": "gm"})
    assert r.status_code == 200
    log = app_module.app_state.mutation_log[sid]
    assert {"kind": "MOVE", "origin": "gm", "character_id": "mira", "room_id": "tavern_upstairs"} == \
        {k: v for k, v in log[-1].items() if k in ("kind", "origin", "character_id", "room_id")}
