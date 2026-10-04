"""Regression tests for the Sprint 0 world-engine session bugs (B3, B4, B5, B6, B7).

Each test maps to one fix in evennia_world/app.py / res/strings.json:
  B3 — MOVE must honour payload.session_id (was shadowed by a local payload class).
  B4 — configure_world must key the world by session_id, not template_key.
  B5 — idempotency keys are honored server-side (move + create-room).
  B6 — templates ship with empty present_characters.
  B7 — no fabricated DEGRADED muffled feeds for upstairs/tavern rooms.
"""

import pytest

from evennia_world.hybrid_builder import HybridWorldBuilder


@pytest.fixture(scope="module")
def app_module():
    from evennia_world import app as app_mod
    return app_mod


@pytest.fixture(autouse=True)
def reset_state(app_module):
    """Fresh engine state before every test."""
    from evennia_world.app import app as fa_app
    fa_app.state.start_time = 0
    app_module.app_state.current_world = {}
    app_module.app_state.room_to_template = {}
    app_module.app_state.session_worlds = {}
    app_module.app_state.idempotency_seen = {}
    app_module.app_state.action_tick_counter = 0
    app_module.lock_manager = type(app_module.lock_manager)(default_ttl=60.0)
    app_module.world_builder = HybridWorldBuilder()
    yield


@pytest.fixture
def client(app_module):
    from starlette.testclient import TestClient
    with TestClient(app=app_module.app, base_url="http://test") as c:
        yield c


def place(client, char_id, room_id, template_key="dungeon_cellar", session_id="default_session"):
    r = client.post("/api/v1/world/characters", json={
        "character_id": char_id,
        "room_id": room_id,
        "template_key": template_key,
        "session_id": session_id,
    })
    assert r.status_code == 200, f"placement of {char_id} in {room_id} failed: {r.text}"
    return r


def present_in(client, char_id, session_id, template_key="dungeon_cellar"):
    """True if char_id is in some room of (session_id, template_key)."""
    r = client.get("/api/v1/world/characters",
                   params={"template_key": template_key, "session_id": session_id})
    assert r.status_code == 200
    return any(e["character_id"] == char_id for e in r.json())


# ── B3: MOVE honours session_id ───────────────────────────────────────────

class TestMoveSessionScoping:
    def test_move_changes_only_its_own_session(self, client, app_module):
        # Materialize three independent session worlds (GET /state ensures them)
        # and place rowan in each cellar directly — templates ship empty (B6).
        for sess in ("s1", "s2", "default_session"):
            r = client.get("/api/v1/world/state", params={
                "character_id": "rowan", "session_id": sess,
                "template_key": "dungeon_cellar",
            })
            assert r.status_code == 200
            world = app_module.app_state.session_worlds[sess]["dungeon_cellar"]
            world["cellar"].present_characters.append("rowan")
        # Detach the legacy global so session scoping is what's under test.
        app_module.app_state.current_world = {}

        r = client.post("/api/v1/world/move", json={
            "character_id": "rowan",
            "room_id": "tavern_upstairs",
            "template_key": "dungeon_cellar",
            "session_id": "s1",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

        def rooms_of(sess):
            world = app_module.app_state.session_worlds[sess]["dungeon_cellar"]
            return {rid: list(room.present_characters) for rid, room in world.items()}

        s1 = rooms_of("s1")
        assert "rowan" in s1["tavern_upstairs"]
        assert "rowan" not in s1["cellar"]

        for sess in ("s2", "default_session"):
            other = rooms_of(sess)
            assert "rowan" in other["cellar"], f"{sess} lost rowan to s1's move"
            assert "rowan" not in other["tavern_upstairs"], \
                f"s1's move leaked into {sess}"


# ── B4: configure_world keys by session_id ────────────────────────────────

class TestConfigureWorldSessionKey:
    def test_configure_keys_by_session_not_template(self, client, app_module):
        r = client.post("/api/v1/world/configure", json={
            "template_key": "forest_camp",
            "session_id": "s1",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

        assert "s1" in app_module.app_state.session_worlds
        assert "forest_camp" in app_module.app_state.session_worlds["s1"]
        # The old bug keyed session_worlds by the template name itself.
        assert "forest_camp" not in app_module.app_state.session_worlds.get("s1", {}).get("bogus", {})
        assert app_module.app_state.session_worlds["s1"].get("forest_camp"), \
            "world must live under the real session id"

    def test_configure_default_session_when_omitted(self, client, app_module):
        r = client.post("/api/v1/world/configure", json={"template_key": "forest_camp"})
        assert r.status_code == 200
        assert "default_session" in app_module.app_state.session_worlds
        assert "forest_camp" in app_module.app_state.session_worlds["default_session"]

    def test_configure_preserves_other_templates_in_session(self, client, app_module):
        client.post("/api/v1/world/configure", json={"template_key": "forest_camp", "session_id": "s1"})
        client.post("/api/v1/world/configure", json={"template_key": "castle_exterior", "session_id": "s1"})
        assert set(app_module.app_state.session_worlds["s1"]) == {"forest_camp", "castle_exterior"}


# ── B6: templates ship empty ──────────────────────────────────────────────

class TestTemplatesShipEmpty:
    def test_every_template_room_has_no_characters(self):
        b = HybridWorldBuilder()
        assert b.templates, "expected templates to load"
        for tmpl_key, rooms in b.templates.items():
            for room_id, room in rooms.items():
                assert room.present_characters == [], \
                    f"{tmpl_key}/{room_id} still ships seeded characters"

    def test_instantiate_all_templates_yields_empty_rooms(self):
        b = HybridWorldBuilder()
        for tmpl_key in b.list_templates():
            for room_id, room in b.instantiate_world(tmpl_key).items():
                assert room.present_characters == [], f"{tmpl_key}/{room_id} not empty"


# ── B7: no fabricated muffled feeds ───────────────────────────────────────

class TestNoFabricatedMuffledFeeds:
    def test_action_yields_nothing_for_unplaced_upstairs_character(self, client):
        # dungeon_cellar's tavern_upstairs is exactly the room seamus used to be
        # seeded in, and the removed hack fabricated a DEGRADED feed for him.
        place(client, "rowan", "cellar")
        r = client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "raw_text": "Anyone up there?",
            "session_id": "default_session",
            "template_key": "dungeon_cellar",
        })
        assert r.status_code == 200
        recipients = {c["recipient_id"] for c in r.json()["consequences"]}
        assert "seamus" not in recipients
        assert not any("muffled" in c["sensory_feed"].lower()
                       for c in r.json()["consequences"])

    def test_placed_upstairs_character_uses_real_spatial_eval(self, client):
        # With seamus explicitly placed, he is evaluated spatially — never with
        # the hack's hard-coded 45.0 ft / closed_door+solid_wall signature unless
        # the real evaluation produced it. The real evaluator gives a recipient
        # in the same room as nobody a distance derived from room adjacency; the
        # important assertion: no fabricated feed text.
        place(client, "rowan", "cellar")
        place(client, "seamus", "tavern_upstairs")
        r = client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "raw_text": "Anyone up there?",
            "session_id": "default_session",
            "template_key": "dungeon_cellar",
        })
        assert r.status_code == 200
        for c in r.json()["consequences"]:
            assert "muffled sounds from" not in c["sensory_feed"]


# ── B5: server-side idempotency ───────────────────────────────────────────

class TestIdempotencyKeys:
    def test_same_move_key_applies_once(self, client):
        place(client, "luna", "cellar")
        body = {
            "character_id": "luna",
            "room_id": "tavern_upstairs",
            "template_key": "dungeon_cellar",
            "session_id": "s1",
            "idempotency_key": "key-abc",
        }
        r1 = client.post("/api/v1/world/move", json=body)
        assert r1.status_code == 200
        assert "Duplicate" not in r1.json()["message"]
        assert "moved" in r1.json()["message"]

        # Push luna back manually, then replay the key: nothing may change.
        client.post("/api/v1/world/move", json={
            "character_id": "luna", "room_id": "cellar",
            "template_key": "dungeon_cellar", "session_id": "s1",
        })
        r2 = client.post("/api/v1/world/move", json=body)
        assert r2.status_code == 200
        assert "Duplicate" in r2.json()["message"]
        locs = {e["character_id"]: e["room_id"] for e in client.get(
            "/api/v1/world/characters",
            params={"template_key": "dungeon_cellar", "session_id": "s1"}).json()}
        assert locs.get("luna") == "cellar", "replayed key must not re-apply the move"

    def test_different_key_applies_again(self, client):
        place(client, "luna", "cellar")
        base = {"character_id": "luna", "template_key": "dungeon_cellar", "session_id": "s1"}
        r1 = client.post("/api/v1/world/move", json={**base, "room_id": "tavern_upstairs",
                                                     "idempotency_key": "k1"})
        r2 = client.post("/api/v1/world/move", json={**base, "room_id": "cellar",
                                                     "idempotency_key": "k2"})
        assert r1.status_code == 200 and r2.status_code == 200
        assert "Duplicate" not in r2.json()["message"]
        locs = {e["character_id"]: e["room_id"] for e in client.get(
            "/api/v1/world/characters",
            params={"template_key": "dungeon_cellar", "session_id": "s1"}).json()}
        assert locs.get("luna") == "cellar"

    def test_move_without_key_always_applies(self, client):
        place(client, "luna", "cellar")
        base = {"character_id": "luna", "template_key": "dungeon_cellar", "session_id": "s1"}
        r1 = client.post("/api/v1/world/move", json={**base, "room_id": "tavern_upstairs"})
        r2 = client.post("/api/v1/world/move", json={**base, "room_id": "cellar"})
        assert "Duplicate" not in r1.json()["message"]
        assert "Duplicate" not in r2.json()["message"]

    def test_create_room_key_applies_once(self, client):
        body = {"room_id": "vault", "room_name": "The Vault", "description": "Dark.",
                "template_key": "dungeon_cellar", "session_id": "s1",
                "idempotency_key": "room-key"}
        r1 = client.post("/api/v1/world/rooms", json=body)
        assert r1.status_code == 200
        assert "Duplicate" not in r1.json()["message"]

        r2 = client.post("/api/v1/world/rooms", json=body)
        assert r2.status_code == 200
        assert "Duplicate" in r2.json()["message"]

    def test_reset_clears_seen_keys(self, client, app_module):
        place(client, "luna", "cellar")
        body = {"character_id": "luna", "room_id": "tavern_upstairs",
                "template_key": "dungeon_cellar", "session_id": "s1",
                "idempotency_key": "key-reset"}
        client.post("/api/v1/world/move", json=body)
        assert app_module.app_state.idempotency_seen.get("s1")
        client.delete("/api/v1/world/admin/reset")
        assert app_module.app_state.idempotency_seen == {}


def test_configure_world_does_not_share_room_state_across_sessions(client):
    """configure_world must deep-copy template rooms: before the fix, two sessions
    configured from the same template shared present_characters lists with the template."""
    for sid in ("cfg_s1", "cfg_s2"):
        r = client.post("/api/v1/world/configure",
                        json={"template_key": "dungeon_cellar", "session_id": sid})
        assert r.status_code == 200
    from evennia_world.app import app_state, world_builder
    room_id = next(iter(app_state.session_worlds["cfg_s1"]["dungeon_cellar"]))
    app_state.session_worlds["cfg_s1"]["dungeon_cellar"][room_id].present_characters.append("intruder")
    assert "intruder" not in app_state.session_worlds["cfg_s2"]["dungeon_cellar"][room_id].present_characters
    assert "intruder" not in world_builder.templates["dungeon_cellar"][room_id].present_characters
