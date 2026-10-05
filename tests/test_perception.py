"""Sprint 2 chunk 3: perception recording + per-character gated history rows."""
import asyncpg
import pytest

from proxy.gating.perception import gated_history, record_turn_perceptions
from tests._testdb import TEST_DB_CONFIG

CONS = [
    {"recipient_id": "mira", "sensory_feed": "Hello there.", "gating_level": "direct",
     "distance_ft": 3.0, "barriers": []},
    {"recipient_id": "tom", "sensory_feed": "(muffled sounds)", "gating_level": "degraded",
     "distance_ft": 12.0, "barriers": ["drywall"]},
    {"recipient_id": "eavesdropper", "sensory_feed": "", "gating_level": "blackout",
     "distance_ft": 45.0, "barriers": ["closed_door", "solid_wall"]},
]


@pytest.mark.asyncio
async def test_record_and_rebuild_history_per_recipient():
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        sid = "percep_s1"
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
        n = await record_turn_perceptions(conn, session_id=sid, tick=1, turn_id=f"{sid}:1",
                                          actor_id="user", action_type="speak",
                                          consequences=CONS)
        assert n == 3
        await record_turn_perceptions(conn, session_id=sid, tick=2, turn_id=f"{sid}:2",
                                      actor_id="user", action_type="speak",
                                      consequences=[{**CONS[0], "sensory_feed": "Second line."}])

        mira = await gated_history(conn, session_id=sid, recipient_id="mira")
        assert [r["perceived_text"] for r in mira] == ["Hello there.", "Second line."]
        assert [r["tick"] for r in mira] == [1, 2]            # oldest first

        dark = await gated_history(conn, session_id=sid, recipient_id="eavesdropper")
        assert len(dark) == 1
        assert dark[0]["gating_level"] == "blackout" and dark[0]["perceived_text"] == ""

        other = await gated_history(conn, session_id="other_session", recipient_id="mira")
        assert other == []                                     # strict session scoping
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_history_limit_returns_most_recent_window():
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        sid = "percep_s2"
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
        for t in range(1, 11):
            await record_turn_perceptions(conn, session_id=sid, tick=t, turn_id=f"{sid}:{t}",
                                          actor_id="user", action_type="speak",
                                          consequences=[{**CONS[0], "sensory_feed": f"line {t}"}])
        got = await gated_history(conn, session_id=sid, recipient_id="mira", limit=4)
        assert [r["perceived_text"] for r in got] == ["line 7", "line 8", "line 9", "line 10"]
    finally:
        await conn.close()


def test_chat_route_records_perceptions_and_sends_turn_id():
    """Route wiring: consequences land in spm_perception and the engine gets a
    user-message-count turn_id (decision 10 on the live path)."""
    import json as _json
    from unittest.mock import AsyncMock, MagicMock, patch
    from fastapi.testclient import TestClient
    from proxy.main import app

    async def fake_stream(*a, **kw):
        yield "<think>plan</think>A reply."

    calls = []

    async def fake_submit(**kw):
        calls.append(kw)
        return {"success": True, "action_tick": 7, "consequences": CONS}

    recorded = {}

    async def fake_record(conn, **kw):
        recorded.update(kw)
        return len(kw["consequences"])

    pool = MagicMock()
    cm = AsyncMock(); cm.__aenter__.return_value = AsyncMock(); cm.__aexit__.return_value = None
    pool.acquire.return_value = cm

    import proxy.api.routes as routes
    with patch.object(routes, "_db_pool", pool), \
         patch("proxy.api.routes.record_turn_perceptions", fake_record), \
         patch("proxy.api.routes.lemonade_client.generate_stream", fake_stream), \
         patch("proxy.api.routes.evennia_client.submit_action", fake_submit), \
         patch("proxy.api.routes._dispatch_lore_extraction"), \
         patch("proxy.api.routes._dispatch_gm_actions", new_callable=AsyncMock), \
         TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "spm-sovereign-mesh", "stream": True, "session_id": "wire_s1",
            "messages": [{"role": "system", "content": "[Character: Mira]"},
                          {"role": "user", "content": "First."},
                          {"role": "assistant", "content": "Hi."},
                          {"role": "user", "content": "Hello?"}]})
    assert r.status_code == 200
    assert calls[0]["turn_id"] == "wire_s1:2"      # two user messages
    assert recorded["tick"] == 7 and len(recorded["consequences"]) >= 3
    # chunk 4: the character's reply is itself submitted as a world action
    import time as _t
    deadline = _t.time() + 2
    while len(calls) < 2 and _t.time() < deadline:
        _t.sleep(0.05)
    assert any(c.get("turn_id") == "wire_s1:2#reply" and c["character_id"] == "mira"
               for c in calls[1:])


def test_first_turn_seeds_world_and_fallback_redacts_thoughts():
    """Chunk 2 wiring: turn 1 configures a seeded world; the raw-history fallback
    never contains the user's 'quoted thoughts'."""
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from proxy.main import app

    seen = {}

    async def fake_stream(*a, **kw):
        seen["messages"] = kw.get("messages", [])
        yield "Noted."

    configured = {}

    async def fake_configure(**kw):
        configured.update(kw)
        return {"success": True}

    with patch("proxy.api.routes.lemonade_client.generate_stream", fake_stream), \
         patch("proxy.api.routes.evennia_client.configure_world", fake_configure), \
         patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock,
               return_value={"success": True, "action_tick": 1, "consequences": []}), \
         patch("proxy.api.routes._dispatch_lore_extraction"), \
         TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "spm-sovereign-mesh", "stream": True, "session_id": "seedwire_s1",
            "messages": [
                {"role": "system", "content": "[scene:dungeon_cellar] [Character: Mira]"},
                {"role": "user", "content": "\"Hello.\" 'They must never find the amulet.'"}]})
    assert r.status_code == 200
    assert configured["template_key"] == "dungeon_cellar"
    assert {p["character_id"] for p in configured["placements"]} == {"user", "mira"}
    joined = " ".join(m["content"] for m in seen["messages"])
    assert "amulet" not in joined          # thought redacted from feed AND fallback history
    assert "Hello." in joined


@pytest.mark.asyncio
async def test_tainted_reply_is_not_recorded_as_world_state():
    from unittest.mock import AsyncMock, patch
    import proxy.api.routes as routes

    with patch.object(routes, "_db_pool", object()), \
         patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as sub, \
         patch("proxy.api.routes.record_turn_perceptions", new_callable=AsyncMock) as rec:
        await routes._record_reply_action("s1", "mira", "leaked planning text", "s1:3", tainted=True)
    sub.assert_not_called()
    rec.assert_not_called()
