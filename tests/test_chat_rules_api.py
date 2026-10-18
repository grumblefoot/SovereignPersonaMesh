"""Brief C1a: admin API for per-chat rules (/admin/api/v1/chats...).

Real test database (tests/_testdb.py) for spm_chat_rules and spm_perception;
the world engine client methods are patched with fakes. Requests go through
httpx.ASGITransport so the app runs on the test's event loop — the same loop
that owns the asyncpg connection (TestClient's per-request portal cannot share
a real pool).
"""
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import asyncpg
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from proxy.main import app
from proxy.api.admin_routes import set_admin_db_pool
from proxy.api.routes import evennia_client
from proxy.rag import chat_rules
from tests._testdb import TEST_DB_CONFIG


@asynccontextmanager
async def api():
    """Real DB connection wired into the admin pool for the duration of one
    test; yields an AsyncClient that runs the app on this same loop."""
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    suffix = uuid.uuid4().hex[:8]

    @asynccontextmanager
    async def acquire():
        yield conn

    set_admin_db_pool(SimpleNamespace(acquire=acquire))
    try:
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver") as c:
            yield c, conn, suffix
    finally:
        for table in ("spm_chat_rules", "spm_perception"):
            await conn.execute(f"DELETE FROM {table} WHERE session_id LIKE $1",
                               f"c1a_{suffix}%")
        set_admin_db_pool(None)
        await conn.close()


async def add_perception(conn, session_id, recipient, tick=1,
                         created_at=None):
    await conn.execute(
        "INSERT INTO spm_perception (session_id, tick, turn_id, actor_id, "
        "action_type, recipient_id, gating_level, created_at) "
        "VALUES ($1, $2, $3, 'user', 'speak', $4, 'direct', "
        "COALESCE($5::timestamptz, CURRENT_TIMESTAMP))",
        session_id, tick, f"{session_id}:{tick}", recipient, created_at)


# --- GET /chats ---------------------------------------------------------------

async def test_list_chats_union_counts_and_ordering():
    async with api() as (c, conn, sfx):
        perc_only, rules_only, both = (f"c1a_{sfx}_perc", f"c1a_{sfx}_rules",
                                       f"c1a_{sfx}_both")
        # Perception-only chat: characters minus 'user'.
        await add_perception(conn, perc_only, "mei")
        await add_perception(conn, perc_only, "lian")
        await add_perception(conn, perc_only, "user")
        # Rules-only chat: still listed, no characters.
        await chat_rules.add_rule(conn, session_id=rules_only, level="world",
                                  character_id=None, rule_text="Dusk magic.",
                                  rule_type="invariant", status="pending")
        # Both: two pending + one active.
        await add_perception(conn, both, "mei")
        for text, status in (("A", "active"), ("B", "pending"), ("C", "pending")):
            await chat_rules.add_rule(conn, session_id=both, level="world",
                                      character_id=None, rule_text=text,
                                      rule_type="invariant", status=status)

        resp = await c.get("/admin/api/v1/chats")
        assert resp.status_code == 200
        chats = {ch["session_id"]: ch for ch in resp.json()}
        assert perc_only in chats and rules_only in chats and both in chats
        assert chats[perc_only]["characters"] == ["lian", "mei"]  # 'user' excluded
        assert chats[perc_only]["rules"] == {"pending": 0, "active": 0}
        assert chats[perc_only]["last_activity"] is not None
        assert chats[rules_only]["characters"] == []
        assert chats[rules_only]["rules"] == {"pending": 1, "active": 0}
        assert chats[both]["rules"] == {"pending": 2, "active": 1}

        # Newest first: backdate the perception-only chat. (The other two
        # share one timestamp — CURRENT_TIMESTAMP is per-transaction — so
        # only their order against perc_only is deterministic.)
        await conn.execute("UPDATE spm_perception SET created_at = "
                           "TIMESTAMP '2020-01-01T00:00:00+00' WHERE session_id = $1",
                           perc_only)
        resp = await c.get("/admin/api/v1/chats")
        ids = [ch["session_id"] for ch in resp.json()]
        assert ids[-1] == perc_only
        assert ids.index(both) < ids.index(perc_only)
        assert ids.index(rules_only) < ids.index(perc_only)


async def test_list_chats_503_without_pool():
    set_admin_db_pool(None)
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver") as c:
        resp = await c.get("/admin/api/v1/chats")
    assert resp.status_code == 503


# --- GET /chats/{sid}/rules -----------------------------------------------------

async def test_rules_view_grouping_and_snapshot():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_view"
        other = f"c1a_{sfx}_other"
        await chat_rules.add_rule(conn, session_id=sid, level="world",
                                  character_id=None, rule_text="World rule.",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id=sid, level="character_internal",
                                  character_id="mei", rule_text="Mei secret.",
                                  rule_type="invariant", status="pending")
        await chat_rules.add_rule(conn, session_id=sid, level="character_external",
                                  character_id="mei", rule_text="Mei public.",
                                  rule_type="conditional_trigger", status="active")
        await chat_rules.add_rule(conn, session_id=other, level="world",
                                  character_id=None, rule_text="Other chat.",
                                  rule_type="invariant", status="active")

        snap = {"rooms": [
            {"room_id": "castle", "name": "Castle",
             "rules": [{"id": "a" * 32, "rule_text": "Always dark.",
                        "rule_type": "invariant", "status": "active",
                        "source": "manual"}]},
            {"room_id": "gate", "name": "Gate", "rules": []},  # room without rules
        ]}
        with patch.object(evennia_client, "get_snapshot", new_callable=AsyncMock) as m:
            m.return_value = snap
            resp = await c.get(f"/admin/api/v1/chats/{sid}/rules")
        assert resp.status_code == 200
        data = resp.json()
        m.assert_awaited_once_with(session_id=sid, template_key="")

        assert [r["rule_text"] for r in data["world"]] == ["World rule."]
        assert "other chat" not in str(data)
        mei = data["characters"]["mei"]
        assert [r["rule_text"] for r in mei["internal"]] == ["Mei secret."]
        assert [r["rule_text"] for r in mei["external"]] == ["Mei public."]
        rule = mei["internal"][0]
        for key in ("id", "level", "character_id", "rule_text", "rule_type",
                    "status", "source", "created_at"):
            assert key in rule, f"rule dict missing {key}"
        assert rule["created_at"] and "T" in rule["created_at"]

        # All rooms, including the one without rules.
        assert [loc["room_id"] for loc in data["locations"]] == ["castle", "gate"]
        assert data["locations"][0]["name"] == "Castle"
        assert data["locations"][0]["rules"][0]["rule_text"] == "Always dark."
        assert data["locations"][1]["rules"] == []
        assert "engine_error" not in data


async def test_rules_view_engine_error_degrades():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_err"
        await chat_rules.add_rule(conn, session_id=sid, level="world",
                                  character_id=None, rule_text="Still here.",
                                  rule_type="invariant", status="active")
        with patch.object(evennia_client, "get_snapshot",
                          new_callable=AsyncMock) as m:
            m.side_effect = RuntimeError("engine down")
            resp = await c.get(f"/admin/api/v1/chats/{sid}/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert [r["rule_text"] for r in data["world"]] == ["Still here."]
        assert data["locations"] == []
        assert "engine down" in data["engine_error"]


# --- POST /chats/{sid}/rules -----------------------------------------------------

async def test_add_rule_each_level():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_add"
        for level, char in (("world", None), ("character_internal", "mei"),
                            ("character_external", "mei")):
            body = {"level": level, "rule_text": f"Rule for {level}."}
            if char:
                body["character_id"] = char
            resp = await c.post(f"/admin/api/v1/chats/{sid}/rules", json=body)
            assert resp.status_code == 200, resp.text
            out = resp.json()
            assert out["level"] == level
            uuid.UUID(out["id"])  # a real rule id

        rows = await chat_rules.list_rules(conn, sid)
        assert len(rows) == 3
        for row in rows:  # manual rules: active, source manual
            assert row["status"] == "active"
            assert row["source"] == "manual"

        # Location level goes to the engine, active/manual/user.
        fake = AsyncMock(return_value={"success": True, "rule_id": "b" * 32})
        with patch.object(evennia_client, "add_room_rule", fake):
            resp = await c.post(f"/admin/api/v1/chats/{sid}/rules", json={
                "level": "location", "room_id": "castle",
                "rule_text": "No flying.", "rule_type": "game_over"})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"id": "b" * 32, "level": "location"}
        args, kwargs = fake.await_args
        assert args == ("castle", "No flying.", sid)
        assert kwargs["status"] == "active"
        assert kwargs["source"] == "manual"
        assert kwargs["origin"] == "user"
        assert kwargs["rule_type"] == "game_over"

        # The engine reports the same text already on the room: 409 like the other levels.
        dup = AsyncMock(return_value={"success": True, "duplicate": True, "rule_id": "b" * 32})
        with patch.object(evennia_client, "add_room_rule", dup):
            resp = await c.post(f"/admin/api/v1/chats/{sid}/rules", json={
                "level": "location", "room_id": "castle", "rule_text": "No flying."})
        assert resp.status_code == 409


async def test_add_rule_duplicate_409():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_dup"
        body = {"level": "world", "rule_text": "Gravity works here."}
        assert (await c.post(f"/admin/api/v1/chats/{sid}/rules",
                             json=body)).status_code == 200
        resp = await c.post(f"/admin/api/v1/chats/{sid}/rules",
                            json={"level": "world",
                                  "rule_text": "  GRAVITY WORKS HERE. "})
        assert resp.status_code == 409


async def test_add_rule_validation_422():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_valid"
        bad_bodies = [
            {"level": "vibe", "rule_text": "x"},                     # bad level
            {"level": "world", "rule_text": "   "},                  # empty text
            {"level": "world", "rule_text": "ok",
             "rule_type": "nonsense"},                               # bad type
            {"level": "character_internal",
             "rule_text": "no character"},                           # missing char
            {"level": "location", "rule_text": "room rule"},         # missing room_id
        ]
        for body in bad_bodies:
            resp = await c.post(f"/admin/api/v1/chats/{sid}/rules", json=body)
            assert resp.status_code == 422, f"{body} -> {resp.status_code}"
        assert await chat_rules.list_rules(conn, sid) == []


# --- PUT /chats/{sid}/rules/{rule_id} ---------------------------------------------

async def test_update_rule_status_pending_to_active():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_upd"
        rid = await chat_rules.add_rule(conn, session_id=sid, level="world",
                                        character_id=None, rule_text="Pending rule.",
                                        rule_type="invariant")  # default: pending
        resp = await c.put(f"/admin/api/v1/chats/{sid}/rules/{rid}",
                           json={"status": "active"})
        assert resp.status_code == 200, resp.text
        rows = await chat_rules.list_rules(conn, sid)
        assert rows[0]["status"] == "active"


async def test_update_rule_across_sessions_404():
    async with api() as (c, conn, sfx):
        sid_a, sid_b = f"c1a_{sfx}_a", f"c1a_{sfx}_b"
        rid = await chat_rules.add_rule(conn, session_id=sid_a, level="world",
                                        character_id=None, rule_text="Mine.",
                                        rule_type="invariant")
        resp = await c.put(f"/admin/api/v1/chats/{sid_b}/rules/{rid}",
                           json={"status": "active"})
        assert resp.status_code == 404
        rows = await chat_rules.list_rules(conn, sid_a)
        assert rows[0]["status"] == "pending"  # untouched


async def test_update_rule_validation_422():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_updvalid"
        rid = await chat_rules.add_rule(conn, session_id=sid, level="world",
                                        character_id=None, rule_text="Fine.",
                                        rule_type="invariant")
        resp = await c.put(f"/admin/api/v1/chats/{sid}/rules/{rid}",
                           json={"status": "frozen"})
        assert resp.status_code == 422
        resp = await c.put(f"/admin/api/v1/chats/{sid}/rules/{rid}", json={})
        assert resp.status_code == 422  # nothing to update


async def test_update_location_rule_calls_engine():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_locupd"
        fake = AsyncMock(return_value={"success": True})
        with patch.object(evennia_client, "update_room_rule", fake):
            resp = await c.put(f"/admin/api/v1/chats/{sid}/rules/r" + "u" * 31
                               + "?room_id=castle",
                               json={"status": "active", "rule_text": "New."})
        assert resp.status_code == 200, resp.text
        args, kwargs = fake.await_args
        assert args[0] == "castle"
        assert args[2] == sid
        # The dashboard's edit carries confirm_edit (an edit of a pending rule confirms it, QA Q8).
        assert kwargs == {"status": "active", "rule_text": "New.", "confirm_edit": True}

        # Engine 404 maps to 404.
        request = httpx.Request("PUT", "http://x")
        err = httpx.HTTPStatusError("404", request=request,
                                    response=httpx.Response(404, request=request))
        fake404 = AsyncMock(side_effect=err)
        with patch.object(evennia_client, "update_room_rule", fake404):
            resp = await c.put(f"/admin/api/v1/chats/{sid}/rules/nope"
                               "?room_id=castle", json={"status": "active"})
        assert resp.status_code == 404


# --- DELETE /chats/{sid}/rules/{rule_id} ------------------------------------------

async def test_delete_rule_and_404():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_del"
        rid = await chat_rules.add_rule(conn, session_id=sid, level="world",
                                        character_id=None, rule_text="Doomed.",
                                        rule_type="invariant")
        resp = await c.delete(f"/admin/api/v1/chats/{sid}/rules/{rid}")
        assert resp.status_code == 200, resp.text
        assert await chat_rules.list_rules(conn, sid) == []
        resp = await c.delete(f"/admin/api/v1/chats/{sid}/rules/{rid}")
        assert resp.status_code == 404


async def test_delete_location_rule_calls_engine():
    async with api() as (c, conn, sfx):
        sid = f"c1a_{sfx}_locdel"
        fake = AsyncMock(return_value={"success": True})
        with patch.object(evennia_client, "delete_room_rule", fake):
            resp = await c.delete(f"/admin/api/v1/chats/{sid}/rules/r" + "d" * 31
                                  + "?room_id=castle")
        assert resp.status_code == 200, resp.text
        args, kwargs = fake.await_args
        assert args[0] == "castle"
        assert args[2] == sid
