"""QA session 2026-10-12 (My Hero Academia scenario, tester): N1, N4, N5, N7, N14.

Real spm_test database (rebuilt from scripts/init_db.sql each run); each test opens its
own pool on its own event loop and points routes._db_pool at it.
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import asyncpg
import pytest

from proxy.core.gm_actions import validate_batch
from proxy.gating.action_parser import parse_message
from proxy.gating.perception import gated_history, record_turn_perceptions, render_history_rows
from proxy.rag import budget
from tests._testdb import TEST_DB_CONFIG

SHEET_MSG = ("Name: Amy\nGender: Female\nAge: 15\nBirth date: June 9\n"
             "Appearance: Mixed skin color, hazel eyes, brown hair, \n"
             "traits: smart, charming, funny, pretty, \n"
             "quirk: Stealing quirks( i can steal anyones quirk by seeing them do it)\n"
             "path: Villain")


@asynccontextmanager
async def routes_pool(monkeypatch, sid):
    import proxy.api.routes as routes
    pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
    monkeypatch.setattr(routes, "_db_pool", pool)
    try:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
            await conn.execute("DELETE FROM spm_chat_settings WHERE session_id = $1", sid)
        yield routes, pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
            await conn.execute("DELETE FROM spm_chat_settings WHERE session_id = $1", sid)
        await pool.close()


# --- N7: a character sheet is OOC setup, not speech ---------------------------------

def test_n7_real_character_sheet_is_not_speech():
    r = parse_message(SHEET_MSG)
    assert r.sheet_name == "Amy"
    assert r.actions == []                              # nothing perceived as said aloud
    assert r.ooc and "character sheet" in r.ooc_texts[0]
    assert r.sheet.count("\n") + 1 == 8


def test_n7_sheet_with_a_following_action_keeps_the_action():
    r = parse_message("Name: Amy\nAge: 15\nQuirk: Steal\n\n*I step out of the darkness*")
    assert r.sheet_name == "Amy"
    assert [a.action_type for a in r.actions] in (["emote"], ["move"])


def test_n7_two_key_lines_or_normal_speech_are_not_a_sheet():
    assert parse_message("Name: Amy\nAge: 15").sheet is None
    assert parse_message('"Hello there officer" I say chuckling').sheet is None


# --- N4: the sheet's Name is a player alias for GM moves ----------------------------

def test_n4_in_story_name_resolves_to_user():
    world = dict(known_rooms={"alley"}, known_entities={"user", "narr"}, target_char="narr")
    out = validate_batch([{"type": "MOVE", "entity": "Amy", "room_id": "alley"}],
                         user_aliases={"Amanda", "Amy"}, **world)
    assert not out.rejections and out.moves[0].entity == "user"


@pytest.mark.asyncio
async def test_n4_n7_sheet_is_stored_per_chat(monkeypatch):
    sid = "qa1012_sheet"
    async with routes_pool(monkeypatch, sid) as (routes, _):
        assert await routes._chat_player(sid) == (None, None)
        r = parse_message(SHEET_MSG)
        await routes._save_player_sheet(sid, r.sheet, r.sheet_name)
        sheet, name = await routes._chat_player(sid)
        assert name == "Amy" and "quirk: Stealing quirks" in sheet
        # a later sheet without a Name keeps the learned name
        await routes._save_player_sheet(sid, "Age: 16\nQuirk: x\nPath: Hero", None)
        assert (await routes._chat_player(sid))[1] == "Amy"


# --- N1: the greeting is recorded as turn 0 ------------------------------------------

@pytest.mark.asyncio
async def test_n1_greeting_recorded_and_replaced_on_swipe(monkeypatch):
    sid = "qa1012_greet"
    msgs = lambda g: [SimpleNamespace(role="system", content="card"),
                      SimpleNamespace(role="assistant", content=g),
                      SimpleNamespace(role="user", content="Name: Amy")]
    async with routes_pool(monkeypatch, sid) as (routes, pool):
        await routes._record_greeting(sid, "narr", msgs("You have arrived in the world..."))
        await routes._record_greeting(sid, "narr", msgs("Alternate opening: a rooftop."))
        async with pool.acquire() as conn:
            rows = await gated_history(conn, session_id=sid, recipient_id="narr")
            user_rows = await gated_history(conn, session_id=sid, recipient_id="user")
        assert [r["perceived_text"] for r in rows] == ["Alternate opening: a rooftop."]
        assert render_history_rows(rows, "narr") == [
            {"role": "assistant", "content": "Alternate opening: a rooftop."}]
        assert user_rows[0]["gating_level"] == "direct"


# --- N14: budget-sized history with the opening pinned -------------------------------

@pytest.mark.asyncio
async def test_n14_gated_history_returns_pinned_head_outside_the_window():
    sid = "qa1012_head"
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
        for t in range(0, 12):
            await record_turn_perceptions(conn, session_id=sid, tick=t, turn_id=f"{sid}:{t}",
                                          actor_id="user", action_type="speak",
                                          consequences=[{"recipient_id": "narr",
                                                         "sensory_feed": f"line {t}",
                                                         "gating_level": "direct"}])
        rows = await gated_history(conn, session_id=sid, recipient_id="narr", limit=4, head=2)
        assert [r["perceived_text"] for r in rows] == ["line 0", "line 1", "line 8", "line 9",
                                                       "line 10", "line 11"]
        # head rows already inside the window are not duplicated
        rows = await gated_history(conn, session_id=sid, recipient_id="narr", limit=50, head=2)
        assert len(rows) == 12
    finally:
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", sid)
        await conn.close()


def test_n14_allocator_never_drops_pinned_opening():
    import proxy.api.routes as routes
    hist = [{"role": "assistant", "content": "GREETING " + "x " * 40},
            {"role": "user", "content": "CHARACTER SHEET " + "y " * 40}]
    hist += [{"role": "user" if i % 2 else "assistant", "content": f"turn {i} " + "z " * 400}
             for i in range(40)]
    hist.append({"role": "user", "content": "latest"})
    settings = {"context_window_override": 4096}
    _, _, kept, report = budget.allocate(
        settings=settings, hw_config=routes.prompt_builder.config, model="m",
        system_text="card", memories=[], lore={"invariants": [], "triggers": []},
        history=hist, pinned_head=2)
    assert report.trimmed["history"] > 0
    assert kept[0]["content"].startswith("GREETING") and kept[1]["content"].startswith("CHARACTER SHEET")
    assert kept[-1]["content"] == "latest"


# --- N5: initial lore extraction is tracked per chat ---------------------------------

@pytest.mark.asyncio
async def test_n5_initial_lore_flag(monkeypatch):
    sid = "qa1012_lore"
    async with routes_pool(monkeypatch, sid) as (routes, _):
        assert await routes._initial_lore_done(sid) is False
        await routes._mark_initial_lore_done(sid)
        assert await routes._initial_lore_done(sid) is True


# --- N11: warn once when ST's Summarize extension is active --------------------------

def test_n11_summarize_prompt_warns_once(caplog):
    import proxy.api.routes as routes
    routes._summarize_warned.discard("qa1012_sum")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "Ignore previous instructions. Summarize the most important "
             "facts and events in the story so far. Limit the summary to 200 words or less."}]
    with caplog.at_level("INFO"):
        routes._warn_if_summarize(msgs, "qa1012_sum")
        routes._warn_if_summarize(msgs, "qa1012_sum")
    assert sum("Summarize extension is active" in r.message for r in caplog.records) == 1
    caplog.clear()
    with caplog.at_level("INFO"):
        routes._warn_if_summarize([{"role": "user", "content": "Write Amanda's next reply."}], "qa1012_imp")
    assert not caplog.records
