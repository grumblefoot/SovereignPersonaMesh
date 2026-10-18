"""Brief D1: GM rules attached to rooms in the world engine.

Memory-only engine (SPM_WORLD_DB=0, app_state reset like the `rig` fixture in
tests/test_gating_leaks.py). Covers: add -> snapshot, text duplicates,
idempotency-key replay, update, delete, 404/422 cases, old-room-JSON backward
compat, and the proxy client methods over httpx.ASGITransport.
"""
import json
import uuid

import httpx
import pytest
from starlette.testclient import TestClient

import evennia_world.app as world_app
from evennia_world.hybrid_builder import HybridWorldBuilder
from evennia_world.models import RoomMetadata
from proxy.backend_client.evennia_client import EvenniaWorldClient


@pytest.fixture
def engine(monkeypatch):
    # Memory-only: no asyncpg pool, so the engine state lives only in app_state.
    monkeypatch.setenv("SPM_WORLD_DB", "0")
    st = world_app.app_state
    st._db_pool = None
    st.session_worlds = {}
    st.session_ticks = {}
    st.session_turn_ticks = {}
    st.idempotency_seen = {}
    st.session_edges = {}
    st.mutation_log = {}
    world_app.world_builder = HybridWorldBuilder()
    if not hasattr(world_app.app.state, "start_time"):
        world_app.app.state.start_time = 0.0
    with TestClient(world_app.app) as app:
        yield app


def new_session():
    return f"rules_{uuid.uuid4().hex[:10]}"


def make_room(app, room_id, session_id):
    r = app.post("/api/v1/world/rooms", json={
        "room_id": room_id, "room_name": room_id, "description": "test room",
        "template_key": "dynamic", "session_id": session_id})
    assert r.status_code == 200, r.text


def add_rule(app, room_id, session_id, text, **kw):
    r = app.post(f"/api/v1/world/rooms/{room_id}/rules",
                 json={"session_id": session_id, "rule_text": text, **kw})
    return r


def snapshot_rooms(app, session_id):
    r = app.get("/api/v1/world/snapshot", params={"session_id": session_id})
    assert r.status_code == 200, r.text
    return {room["room_id"]: room for room in r.json()["rooms"]}


def test_add_rule_shows_in_snapshot(engine):
    s = new_session()
    make_room(engine, "castle", s)
    r = add_rule(engine, "castle", s, "The castle is always dark and gloomy.",
                 rule_type="invariant", status="active", source="manual")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    rule_id = body["rule_id"]
    assert len(rule_id) == 32  # uuid4().hex

    room = snapshot_rooms(engine, s)["castle"]
    assert len(room["rules"]) == 1
    rule = room["rules"][0]
    assert rule["id"] == rule_id
    assert rule["rule_text"] == "The castle is always dark and gloomy."
    assert rule["rule_type"] == "invariant"
    assert rule["status"] == "active"
    assert rule["source"] == "manual"


def test_defaults_are_invariant_pending_extracted(engine):
    s = new_session()
    make_room(engine, "castle", s)
    r = add_rule(engine, "castle", s, "No guns indoors.")
    assert r.status_code == 200, r.text
    rule = snapshot_rooms(engine, s)["castle"]["rules"][0]
    assert rule["rule_type"] == "invariant"
    assert rule["status"] == "pending"
    assert rule["source"] == "extracted"


def test_duplicate_text_returns_same_id(engine):
    s = new_session()
    make_room(engine, "castle", s)
    r1 = add_rule(engine, "castle", s, "The castle is always dark.")
    rule_id = r1.json()["rule_id"]
    # Same text, different case and surrounding whitespace -> duplicate.
    r2 = add_rule(engine, "castle", s, "  THE CASTLE IS ALWAYS DARK.  ")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["success"] is True
    assert body["duplicate"] is True
    assert body["rule_id"] == rule_id
    # Still exactly one rule on the room, unchanged.
    rules = snapshot_rooms(engine, s)["castle"]["rules"]
    assert len(rules) == 1
    assert rules[0]["rule_text"] == "The castle is always dark."


def test_idempotency_key_replay_returns_same_id(engine):
    s = new_session()
    make_room(engine, "castle", s)
    payload = {"session_id": s, "rule_text": "Lanterns only.",
               "idempotency_key": "k-" + uuid.uuid4().hex}
    r1 = engine.post(f"/api/v1/world/rooms/castle/rules", json=payload)
    assert r1.status_code == 200, r1.text
    rule_id = r1.json()["rule_id"]
    r2 = engine.post(f"/api/v1/world/rooms/castle/rules", json=payload)
    assert r2.status_code == 200, r2.text
    assert r2.json()["success"] is True
    assert r2.json()["rule_id"] == rule_id
    assert len(snapshot_rooms(engine, s)["castle"]["rules"]) == 1


def test_update_rule_text_and_status(engine):
    s = new_session()
    make_room(engine, "castle", s)
    rule_id = add_rule(engine, "castle", s, "Old wording.").json()["rule_id"]

    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rule_id}", json={
        "session_id": s, "rule_text": "New wording.", "status": "active"})
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True

    rule = snapshot_rooms(engine, s)["castle"]["rules"][0]
    assert rule["id"] == rule_id
    assert rule["rule_text"] == "New wording."
    assert rule["status"] == "active"


def test_dashboard_edit_confirms_a_pending_room_rule(engine):
    """Owner decision 2026-10-18 (QA Q8): the dashboard's edit (confirm_edit) of a
    pending rule activates it; a plain edit or an explicit status does not."""
    s = new_session()
    make_room(engine, "castle", s)
    rid = add_rule(engine, "castle", s, "Draft.").json()["rule_id"]

    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rid}",
                   json={"session_id": s, "rule_text": "Reworded draft."})
    assert r.status_code == 200, r.text
    assert snapshot_rooms(engine, s)["castle"]["rules"][0]["status"] == "pending"

    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rid}",
                   json={"session_id": s, "rule_text": "Final.", "confirm_edit": True})
    assert r.status_code == 200, r.text
    rule = snapshot_rooms(engine, s)["castle"]["rules"][0]
    assert (rule["rule_text"], rule["status"]) == ("Final.", "active")

    rid2 = add_rule(engine, "castle", s, "Second draft.").json()["rule_id"]
    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rid2}",
                   json={"session_id": s, "rule_text": "Still draft.", "status": "pending",
                         "confirm_edit": True})
    assert r.status_code == 200, r.text
    assert {x["id"]: x["status"] for x in snapshot_rooms(engine, s)["castle"]["rules"]}[rid2] == "pending"


def test_delete_rule(engine):
    s = new_session()
    make_room(engine, "castle", s)
    rule_id = add_rule(engine, "castle", s, "Temporary rule.").json()["rule_id"]

    r = engine.delete(f"/api/v1/world/rooms/castle/rules/{rule_id}",
                      params={"session_id": s})
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
    assert snapshot_rooms(engine, s)["castle"]["rules"] == []
    # A second DELETE is a 404 (no idempotency key on delete).
    r2 = engine.delete(f"/api/v1/world/rooms/castle/rules/{rule_id}",
                       params={"session_id": s})
    assert r2.status_code == 404


def test_unknown_room_404(engine):
    s = new_session()
    r = add_rule(engine, "nope", s, "Anything at all.")
    assert r.status_code == 404
    r = engine.put("/api/v1/world/rooms/nope/rules/abc123",
                   json={"session_id": s, "rule_text": "x"})
    assert r.status_code == 404
    r = engine.delete("/api/v1/world/rooms/nope/rules/abc123",
                      params={"session_id": s})
    assert r.status_code == 404


def test_unknown_rule_404(engine):
    s = new_session()
    make_room(engine, "castle", s)
    r = engine.put(f"/api/v1/world/rooms/castle/rules/deadbeef",
                   json={"session_id": s, "rule_text": "x"})
    assert r.status_code == 404
    r = engine.delete("/api/v1/world/rooms/castle/rules/deadbeef",
                      params={"session_id": s})
    assert r.status_code == 404


def test_validation_422(engine):
    s = new_session()
    make_room(engine, "castle", s)
    # Bad rule_type on add.
    assert add_rule(engine, "castle", s, "Text.", rule_type="vibe").status_code == 422
    # Bad status on add.
    assert add_rule(engine, "castle", s, "Text.", status="frozen").status_code == 422
    # Empty and whitespace-only rule_text.
    assert add_rule(engine, "castle", s, "").status_code == 422
    assert add_rule(engine, "castle", s, "   \t ").status_code == 422
    assert snapshot_rooms(engine, s)["castle"]["rules"] == []

    # Bad values on update are rejected before any change is written.
    rule_id = add_rule(engine, "castle", s, "Keep me.").json()["rule_id"]
    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rule_id}",
                   json={"session_id": s, "rule_type": "nonsense"})
    assert r.status_code == 422
    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rule_id}",
                   json={"session_id": s, "status": "nonsense"})
    assert r.status_code == 422
    r = engine.put(f"/api/v1/world/rooms/castle/rules/{rule_id}",
                   json={"session_id": s, "rule_text": "   "})
    assert r.status_code == 422                       # an edit cannot blank the rule
    rule = snapshot_rooms(engine, s)["castle"]["rules"][0]
    assert rule["rule_text"] == "Keep me."
    assert rule["rule_type"] == "invariant"
    assert rule["status"] == "pending"


def test_old_room_json_without_rules_still_loads():
    old_room_json = json.dumps({
        "room_id": "throne", "room_name": "Throne Room",
        "description": "A dusty hall.", "lighting": "dim",
        "exits": ["gate"], "present_characters": ["mira"],
        "nearby_objects": ["brazier"], "flavor_text": None,
    })
    room = RoomMetadata(**json.loads(old_room_json))
    assert room.rules == []


def test_room_with_rules_round_trips_through_json():
    room = RoomMetadata(room_id="crypt", room_name="Crypt", description="Cold.")
    room.rules.append(world_app.RoomRule(
        id="a" * 32, rule_text="Undead rise at midnight.",
        rule_type="conditional_trigger", status="active", source="migrated"))
    restored = RoomMetadata.model_validate_json(room.model_dump_json())
    assert len(restored.rules) == 1
    r = restored.rules[0]
    assert r.id == "a" * 32
    assert r.rule_text == "Undead rise at midnight."
    assert r.rule_type == "conditional_trigger"
    assert r.status == "active"
    assert r.source == "migrated"


async def test_client_methods_against_app(engine):
    s = new_session()
    make_room(engine, "castle", s)

    client = EvenniaWorldClient()
    client.base_url = "http://world.test/api/v1"
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=world_app.app), base_url="http://world.test")
    try:
        res = await client.add_room_rule("castle", "No flying.", s,
                                         rule_type="game_over", status="active",
                                         source="manual")
        assert res["success"] is True
        rule_id = res["rule_id"]

        res = await client.update_room_rule("castle", rule_id, s, status="pending")
        assert res["success"] is True
        rule = snapshot_rooms(engine, s)["castle"]["rules"][0]
        assert rule["status"] == "pending"
        assert rule["rule_type"] == "game_over"

        res = await client.delete_room_rule("castle", rule_id, s)
        assert res["success"] is True
        assert snapshot_rooms(engine, s)["castle"]["rules"] == []

        # HTTP errors raise, not silently return.
        with pytest.raises(httpx.HTTPStatusError):
            await client.delete_room_rule("castle", rule_id, s)
        with pytest.raises(httpx.HTTPStatusError):
            await client.add_room_rule("ghost_room", "Nope.", s)
    finally:
        await client._client.aclose()
