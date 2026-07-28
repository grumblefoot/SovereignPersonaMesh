"""
Unit tests for Evennia World State Liaison Service.
Covers session_lock (expiry, concurrent guards, cleanup), hybrid_builder
(template matching, dynamic character management), and app.py endpoints.
"""

import pytest
import time
import asyncio
from unittest.mock import patch

from evennia_world.session_lock import SessionLockManager, LockError
from evennia_world.hybrid_builder import (
    HybridWorldBuilder, DEFAULT_ROOM_TEMPLATES, _KEYWORD_SCORES,
)
from evennia_world.models import (
    ActionType, GatingLevel, BarrierType, RoomMetadata,
    SensoryConsequence, ActionResponse, CharacterWorldState,
)
from httpx import AsyncClient, ASGITransport


# ─────────────────────────────────────────────────────────────────────────
# Session Lock Manager Tests
# ─────────────────────────────────────────────────────────────────────────

class TestSessionLockManager:
    """Tests for SessionLockManager: acquire, release, expiry, concurrency."""

    @pytest.mark.asyncio
    async def test_acquire_and_verify_token(self):
        mgr = SessionLockManager(default_ttl=60.0)
        token = await mgr.acquire_lock("session-1")
        assert isinstance(token, str) and len(token) > 0
        assert mgr.verify_lock("session-1", token) is True
        assert mgr.verify_lock("session-1", "wrong-token") is False

    @pytest.mark.asyncio
    async def test_acquire_blocks_concurrent(self):
        mgr = SessionLockManager(default_ttl=60.0)
        token1 = await mgr.acquire_lock("s1")
        with pytest.raises(LockError, match="Lock already held"):
            await mgr.acquire_lock("s1")

    @pytest.mark.asyncio
    async def test_release_restores_acquire(self):
        mgr = SessionLockManager(default_ttl=60.0)
        t1 = await mgr.acquire_lock("s2")
        assert mgr.release_lock("s2", t1) is True
        t2 = await mgr.acquire_lock("s2")
        assert t2 != t1
        mgr.release_lock("s2", t2)

    @pytest.mark.asyncio
    async def test_release_wrong_token_fails(self):
        mgr = SessionLockManager(default_ttl=60.0)
        t1 = await mgr.acquire_lock("s3")
        assert mgr.release_lock("s3", "bogus") is False
        with pytest.raises(LockError):
            await mgr.acquire_lock("s3")

    @pytest.mark.asyncio
    async def test_release_without_acquire_fails(self):
        mgr = SessionLockManager(default_ttl=60.0)
        assert mgr.release_lock("no-session", "any-token") is False

    def test_default_ttl(self):
        mgr = SessionLockManager()
        assert mgr._default_ttl == 60.0

    @pytest.mark.asyncio
    async def test_get_lock_info(self):
        mgr = SessionLockManager(default_ttl=120.0)
        await mgr.acquire_lock("info-sess")
        info = mgr.get_lock_info("info-sess")
        assert info is not None
        assert info["session_id"] == "info-sess"
        assert info["expired"] is False
        assert info["remaining_ttl_seconds"] > 0

    @pytest.mark.asyncio
    async def test_is_locked(self):
        mgr = SessionLockManager()
        assert mgr.is_locked("locked-sess") is False
        await mgr.acquire_lock("locked-sess")
        assert mgr.is_locked("locked-sess") is True

    @pytest.mark.asyncio
    async def test_cleanup_expired_locks(self):
        mgr = SessionLockManager(default_ttl=0.1)
        await mgr.acquire_lock("exp-sess")
        now = time.monotonic()
        stale = mgr.cleanup_expired_locks(current_time=now + 0.05)
        assert "exp-sess" not in stale
        stale = mgr.cleanup_expired_locks(current_time=now + 0.5)
        assert "exp-sess" in stale
        assert mgr.is_locked("exp-sess") is False

    @pytest.mark.asyncio
    async def test_acquire_auto_cleanups_stale(self):
        mgr = SessionLockManager(default_ttl=0.1)
        await mgr.acquire_lock("stale-sess")
        mgr._lock_info["stale-sess"]["expiry"] = time.monotonic() - 1.0
        t2 = await mgr.acquire_lock("stale-sess")
        assert mgr.is_locked("stale-sess") is True


# ─────────────────────────────────────────────────────────────────────────
# Hybrid World Builder Tests
# ─────────────────────────────────────────────────────────────────────────

class TestHybridWorldBuilder:
    """Tests for HybridWorldBuilder: matching, instantiation, dynamic management."""

    def test_match_dungeon_keywords(self):
        b = HybridWorldBuilder()
        key = b.match_template(["dungeon", "prison"])
        assert key == "dungeon_cellar"

    def test_match_forest_keywords(self):
        b = HybridWorldBuilder()
        key = b.match_template(["forest", "camp"])
        assert key == "forest_camp"

    def test_match_castle_keywords(self):
        b = HybridWorldBuilder()
        key = b.match_template(["castle", "throne"])
        assert key == "castle_exterior"

    def test_match_tavern_keywords(self):
        b = HybridWorldBuilder()
        key = b.match_template(["tavern", "common"])
        assert key in ["dungeon_cellar", "tavern_common"]

    def test_match_low_score_defaults(self):
        b = HybridWorldBuilder()
        key = b.match_template(["xyzzy", "plugh"], min_score=0.5)
        assert key == "default_meeting_room"

    def test_match_fuzzy_text(self):
        b = HybridWorldBuilder()
        key = b.match_template_fuzzy("We are in the dark dungeon prison")
        assert key == "dungeon_cellar"

    def test_instantiate_dungeon(self):
        b = HybridWorldBuilder()
        rooms = b.instantiate_world("dungeon_cellar")
        assert "cellar" in rooms
        assert "tavern_upstairs" in rooms
        assert rooms["cellar"].room_name == "The Dungeon Cellar"
        assert len(rooms["cellar"].present_characters) == 3

    def test_instantiate_unknown_fallback(self):
        b = HybridWorldBuilder()
        rooms = b.instantiate_world("nonexistent_template")
        assert rooms["meeting_room"].room_id == "meeting_room"

    def test_add_room(self):
        b = HybridWorldBuilder()
        b.add_room("test_world", RoomMetadata(
            room_id="lab", room_name="Alchemist's Lab",
            description="A lab full of potions.", lighting="normal",
            exits=["hallway"], present_characters=[], nearby_objects=[],
        ))
        room = b.get_room("test_world", "lab")
        assert room is not None
        assert room.room_name == "Alchemist's Lab"

    def test_remove_room(self):
        b = HybridWorldBuilder()
        assert b.remove_room("test_noexist", "any") is False
        b.add_room("rem_test", RoomMetadata(
            room_id="r1", room_name="Room 1", description="r1",
            lighting="normal", exits=[], present_characters=[], nearby_objects=[],
        ))
        assert b.remove_room("rem_test", "r1") is True
        assert b.get_room("rem_test", "r1") is None

    def test_add_character_to_room(self):
        b = HybridWorldBuilder()
        rooms = b.instantiate_world("forest_camp")
        b.add_character_to_room("forest_camp", "forest_clearing", "rowan")
        room = b.get_room("forest_camp", "forest_clearing")
        assert room is not None
        assert "rowan" in room.present_characters
        b.add_character_to_room("forest_camp", "forest_clearing", "rowan")
        assert room.present_characters.count("rowan") == 1

    def test_remove_character_from_room(self):
        b = HybridWorldBuilder()
        rooms = b.instantiate_world("dungeon_cellar")
        assert b.remove_character_from_room("dungeon_cellar", "cellar", "rowan") is True
        room = b.get_room("dungeon_cellar", "cellar")
        assert room is not None
        assert "rowan" not in room.present_characters

    def test_get_nearby_characters(self):
        b = HybridWorldBuilder()
        chars = b.get_nearby_characters("dungeon_cellar", "cellar")
        assert "rowan" in chars
        assert "domino" in chars
        assert "luna" in chars

    def test_get_all_characters_in_world(self):
        b = HybridWorldBuilder()
        chars = b.get_all_characters_in_world("dungeon_cellar")
        assert chars == {"rowan", "domino", "luna", "seamus"}

    def test_list_templates(self):
        b = HybridWorldBuilder()
        tpls = b.list_templates()
        assert "dungeon_cellar" in tpls
        assert "forest_camp" in tpls
        assert "castle_exterior" in tpls
        assert "tavern_common" in tpls

    def test_default_templates_count(self):
        assert "dungeon_cellar" in DEFAULT_ROOM_TEMPLATES
        assert "forest_camp" in DEFAULT_ROOM_TEMPLATES
        assert "castle_exterior" in DEFAULT_ROOM_TEMPLATES
        assert "tavern_common" in DEFAULT_ROOM_TEMPLATES

    def test_keyword_scores_exist(self):
        assert "dungeon_cellar" in _KEYWORD_SCORES
        assert "dungeon" in _KEYWORD_SCORES["dungeon_cellar"]


# ─────────────────────────────────────────────────────────────────────────
# App Endpoint Tests (via ASGI transport)
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    from evennia_world import app as app_module
    app_module.current_world = {}
    app_module.room_to_template = {}
    app_module.action_tick_counter = 0
    return app_module


@pytest.fixture(autouse=True)
def reset_world_state(app):
    from evennia_world.app import app as fa_app
    fa_app.state.start_time = 0
    app.current_world = {}
    app.room_to_template = {}
    app.action_tick_counter = 0
    app.lock_manager = type(app.lock_manager)(default_ttl=60.0)
    app.world_builder = HybridWorldBuilder()
    app._ensure_world("dungeon_cellar")
    yield


@pytest.fixture
async def async_client(app):
    transport = ASGITransport(app=app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client):
        r = await async_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "tick" in data
        assert "uptime_seconds" in data


class TestActionEndpoint:
    @pytest.mark.asyncio
    async def test_submit_speak_action(self, async_client):
        r = await async_client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "target_id": "domino",
            "raw_text": "I hear a venomcrawler.",
            "session_id": "test-session",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert isinstance(data["action_tick"], int) and data["action_tick"] > 0
        assert len(data["consequences"]) > 0
        for c in data["consequences"]:
            assert "recipient_id" in c
            assert "sensory_feed" in c
            assert "gating_level" in c
            assert "distance_ft" in c
            assert "barriers" in c

    @pytest.mark.asyncio
    async def test_submit_whisper_action(self, async_client):
        r = await async_client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "whisper",
            "target_id": "domino",
            "raw_text": "The venomcrawler is behind you.",
        })
        assert r.status_code == 200
        data = r.json()
        domino_c = [c for c in data["consequences"] if c["recipient_id"] == "domino"]
        assert len(domino_c) >= 1
        assert domino_c[0]["gating_level"] == "direct"

    @pytest.mark.asyncio
    async def test_submit_move_action(self, async_client):
        r = await async_client.post("/api/v1/world/action", json={
            "character_id": "luna",
            "action_type": "move",
            "raw_text": "Luna steps toward the cellar stairs.",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_action_tick_increments(self, async_client):
        r1 = await async_client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "raw_text": "First tick.",
        })
        r2 = await async_client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "raw_text": "Second tick.",
        })
        tick1 = r1.json()["action_tick"]
        tick2 = r2.json()["action_tick"]
        assert tick2 == tick1 + 1


class TestWorldStateEndpoint:
    @pytest.mark.asyncio
    async def test_query_cellar_character(self, async_client):
        r = await async_client.get("/api/v1/world/state", params={
            "character_id": "rowan",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["character_id"] == "rowan"
        assert data["gating_level"] == "direct"
        assert "cellar" in data["current_room"]["room_id"]

    @pytest.mark.asyncio
    async def test_query_unknown_character(self, async_client):
        r = await async_client.get("/api/v1/world/state", params={
            "character_id": "nobody",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["character_id"] == "nobody"
        assert data["gating_level"] == "blackout"
        assert "Unknown Location" in data["current_room"]["room_name"]

    @pytest.mark.asyncio
    async def test_query_distances(self, async_client):
        r = await async_client.get("/api/v1/world/state", params={
            "character_id": "rowan",
        })
        data = r.json()
        assert isinstance(data["distances"], dict)
        assert "rowan" in data["distances"]
        assert data["distances"]["rowan"] == 0.0


class TestLockEndpoint:
    @pytest.mark.asyncio
    async def test_acquire_lock(self, async_client):
        r = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-1",
            "lock_action": "acquire",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "lock_token" in data

        r2 = await async_client.get("/api/v1/world/lock/lock-test-1")
        assert r2.json()["locked"] is True

    @pytest.mark.asyncio
    async def test_release_lock(self, async_client):
        r1 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-2",
            "lock_action": "acquire",
        })
        token = r1.json()["lock_token"]

        r2 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-2",
            "lock_action": "release",
            "lock_token": token,
        })
        assert r2.status_code == 200
        assert r2.json()["success"] is True

        r3 = await async_client.get("/api/v1/world/lock/lock-test-2")
        assert r3.json()["locked"] is False

    @pytest.mark.asyncio
    async def test_release_invalid_token(self, async_client):
        r1 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-3",
            "lock_action": "acquire",
        })
        token = r1.json()["lock_token"]
        r2 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-3",
            "lock_action": "release",
            "lock_token": "wrong-token",
        })
        assert r2.json()["success"] is False

    @pytest.mark.asyncio
    async def test_double_acquire_returns_409(self, async_client):
        r1 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-4",
            "lock_action": "acquire",
        })
        assert r1.status_code == 200
        r2 = await async_client.post("/api/v1/world/lock", json={
            "session_id": "lock-test-4",
            "lock_action": "acquire",
        })
        assert r2.status_code == 409


class TestCharacterEndpoints:
    @pytest.mark.asyncio
    async def test_list_characters(self, async_client):
        r = await async_client.get("/api/v1/world/characters")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        ids = {entry["character_id"] for entry in data}
        assert "rowan" in ids
        assert "seamus" in ids

    @pytest.mark.asyncio
    async def test_add_character(self, async_client):
        r = await async_client.post("/api/v1/world/characters", json={
            "character_id": "new-char",
            "room_id": "cellar",
            "template_key": "dungeon_cellar",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

    @pytest.mark.asyncio
    async def test_add_character_unknown_room(self, async_client):
        r = await async_client.post("/api/v1/world/characters", json={
            "character_id": "new-char",
            "room_id": "nonexistent_room",
        })
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_add_character_unknown_template(self, async_client):
        r = await async_client.post("/api/v1/world/characters", json={
            "character_id": "new-char",
            "room_id": "room1",
            "template_key": "phantom_template",
        })
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_character(self, async_client):
        r = await async_client.delete("/api/v1/world/characters/rowan")
        assert r.status_code == 200
        assert r.json()["success"] is True
        r2 = await async_client.get("/api/v1/world/state", params={"character_id": "rowan"})
        data = r2.json()
        assert data["character_id"] == "rowan"

    @pytest.mark.asyncio
    async def test_move_character(self, async_client):
        r = await async_client.post("/api/v1/world/move", json={
            "character_id": "luna",
            "room_id": "tavern_upstairs",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

    @pytest.mark.asyncio
    async def test_move_character_unknown_room(self, async_client):
        r = await async_client.post("/api/v1/world/move", json={
            "character_id": "luna",
            "room_id": "nowhere",
        })
        assert r.status_code == 200


class TestWorldConfigureEndpoint:
    @pytest.mark.asyncio
    async def test_configure_forest(self, async_client):
        r = await async_client.post("/api/v1/world/configure", json={
            "template_key": "forest_camp",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True
        # Place rowan in forest clearing
        await async_client.post("/api/v1/world/characters", json={
            "character_id": "rowan",
            "room_id": "forest_clearing",
            "template_key": "forest_camp",
        })
        r2 = await async_client.get("/api/v1/world/state", params={"character_id": "rowan"})
        data = r2.json()
        assert "forest" in data["current_room"]["room_id"]

    @pytest.mark.asyncio
    async def test_configure_unknown_template(self, async_client):
        r = await async_client.post("/api/v1/world/configure", json={
            "template_key": "phantom_world",
        })
        assert r.status_code == 404


class TestListTemplatesEndpoint:
    @pytest.mark.asyncio
    async def test_list_available_templates(self, async_client):
        r = await async_client.get("/api/v1/world/templates")
        assert r.status_code == 200
        data = r.json()
        assert "templates" in data
        assert "dungeon_cellar" in data["templates"]
        assert "forest_camp" in data["templates"]
        assert "castle_exterior" in data["templates"]


class TestModels:
    def test_action_payload_defaults(self):
        from evennia_world.models import ActionPayload
        p = ActionPayload(
            character_id="rowan",
            action_type=ActionType.SPEAK,
            raw_text="Hello",
        )
        assert p.target_id is None
        assert p.session_id == "default_session"

    def test_sensory_consequence_defaults(self):
        c = SensoryConsequence(
            recipient_id="domino",
            sensory_feed="Rowan: Hello",
            gating_level=GatingLevel.DIRECT,
            distance_ft=3.0,
        )
        assert c.barriers == []

    def test_action_response_schema(self):
        resp = ActionResponse(
            success=True,
            action_tick=1,
            consequences=[
                SensoryConsequence(
                    recipient_id="domino",
                    sensory_feed="Rowan: Hello",
                    gating_level=GatingLevel.DIRECT,
                    distance_ft=3.0,
                ),
            ],
        )
        assert len(resp.consequences) == 1

    def test_room_metadata_defaults(self):
        r = RoomMetadata(room_id="r1", room_name="Room 1", description="desc")
        assert r.lighting == "normal"
        assert r.exits == []
        assert r.present_characters == []
        assert r.nearby_objects == []


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_turn_roundtrip(self, async_client):
        lock_r = await async_client.post("/api/v1/world/lock", json={
            "session_id": "turn-1",
            "lock_action": "acquire",
        })
        assert lock_r.status_code == 200
        token = lock_r.json()["lock_token"]

        action_r = await async_client.post("/api/v1/world/action", json={
            "character_id": "rowan",
            "action_type": "speak",
            "target_id": "domino",
            "raw_text": "I spotted the venomcrawler!",
            "session_id": "turn-1",
        })
        assert action_r.status_code == 200
        tick = action_r.json()["action_tick"]

        state_r = await async_client.get("/api/v1/world/state", params={
            "character_id": "rowan",
            "session_id": "turn-1",
        })
        assert state_r.status_code == 200

        release_r = await async_client.post("/api/v1/world/lock", json={
            "session_id": "turn-1",
            "lock_action": "release",
            "lock_token": token,
        })
        assert release_r.json()["success"] is True
        assert tick > 0
