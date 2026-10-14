"""Perception backfill for history SPM never saw (live test 2026-10-13: an imported chat's
first reply was written from the card and the last message alone)."""
import asyncpg
import pytest

from proxy.gating import backfill
from proxy.gating.perception import gated_history, prune_later_turns, render_history_rows
from tests._testdb import TEST_DB_CONFIG

SID = "bf_s1"


def _solo():
    return [("assistant", "Welcome, traveller."),
            ("user", '"Hello." \'She looks tired.\''), ("assistant", "Arvenia smiles."),
            ("user", '"Shall we?"'), ("assistant", "Arvenia leads the way."),
            ("user", '"Now?"')]                                   # the current turn (3)


def test_solo_import_plans_every_turn_before_the_first_live_one():
    rows = backfill.plan_rows(SID, _solo(), target_char="arvenia", current_turn=3,
                              first_live_turn=3, has_greeting=False,
                              redact=lambda t: t.replace("'She looks tired.'", "").strip())
    mine = render_history_rows(backfill.for_recipient(rows, "arvenia"), "arvenia")
    assert mine == [{"role": "assistant", "content": "Welcome, traveller."},
                    {"role": "user", "content": '"Hello."'},
                    {"role": "assistant", "content": "Arvenia smiles."},
                    {"role": "user", "content": '"Shall we?"'},
                    {"role": "assistant", "content": "Arvenia leads the way."}]
    assert {r["turn_id"].split("#")[0] for r in rows} == {f"{SID}:0:bf", f"{SID}:1:bf", f"{SID}:2:bf"}
    assert all(r["tick"] == 0 for r in rows)
    assert {r["recipient_id"] for r in rows} == {"arvenia", "user"}


def test_recorded_greeting_and_a_chat_spm_saw_from_the_start_need_nothing():
    assert backfill.plan_rows(SID, _solo(), target_char="arvenia", current_turn=3,
                              first_live_turn=1, has_greeting=True) == []
    rows = backfill.plan_rows(SID, _solo(), target_char="arvenia", current_turn=3,
                              first_live_turn=3, has_greeting=True)
    assert not any(r["turn_id"].startswith(f"{SID}:0") for r in rows)


def test_group_lines_go_to_their_speaker_and_spm_notices_are_skipped():
    msgs = [("user", '"Hello both."'),
            ("assistant", "美 Mei: Mei bows."), ("assistant", "莲 Lian: Lian pours tea."),
            ("user", '"Thank you."'),
            ("assistant", "美 Mei: *Mei hears only muffled sounds from elsewhere...*"),
            ("assistant", "莲 Lian: Lian nods."),
            ("user", '"Now?"')]
    rows = backfill.plan_rows(SID, msgs, target_char="mei", current_turn=3,
                              first_live_turn=3, has_greeting=True)
    lian = render_history_rows(backfill.for_recipient(rows, "lian"), "lian")
    assert lian == [{"role": "user", "content": '"Hello both."'},
                    {"role": "user", "content": "[mei] Mei bows."},
                    {"role": "assistant", "content": "Lian pours tea."},
                    {"role": "user", "content": '"Thank you."'},
                    {"role": "assistant", "content": "Lian nods."}]
    assert not any("muffled" in r["perceived_text"] for r in rows)


def test_trimmed_history_keeps_the_chats_turn_numbers():
    msgs = [("user", '"Late line."'), ("assistant", "A reply."), ("user", '"Now?"')]
    rows = backfill.plan_rows(SID, msgs, target_char="mira", current_turn=20,
                              first_live_turn=20, has_greeting=True)
    assert {r["turn_id"].split("#")[0] for r in rows} == {f"{SID}:19:bf"}


def test_coverage_reads_turn_ids():
    ids = [f"{SID}:0", f"{SID}:15", f"{SID}:15#reply:mira", f"{SID}:16"]
    assert backfill.coverage(SID, ids) == (False, 15, True)
    assert backfill.coverage(SID, ids + [f"{SID}:3:bf"])[0] is True


@pytest.mark.asyncio
async def test_persist_once_orders_before_live_rows_and_rewinds_prune_it():
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", SID)
        await conn.execute(
            "INSERT INTO spm_perception (session_id, tick, turn_id, actor_id, action_type, "
            "recipient_id, gating_level, perceived_text) VALUES ($1, 1, $2, 'user', 'speak', "
            "'arvenia', 'direct', '\"Now?\"')", SID, f"{SID}:3")
        rows = backfill.plan_rows(SID, _solo(), target_char="arvenia", current_turn=3,
                                  first_live_turn=3, has_greeting=False)
        assert await backfill.persist(conn, SID, rows) == len(rows)
        assert await backfill.persist(conn, SID, rows) == 0          # written once
        hist = await gated_history(conn, session_id=SID, recipient_id="arvenia")
        assert hist[0]["perceived_text"] == "Welcome, traveller."
        assert hist[-1]["perceived_text"] == '"Now?"'                 # live turn last
        await prune_later_turns(conn, session_id=SID, turn_no=1)      # rewind to turn 1
        left = {r["turn_id"].split("#")[0] for r in await gated_history(conn, session_id=SID,
                                                                         recipient_id="arvenia")}
        assert left == {f"{SID}:0:bf", f"{SID}:1:bf"}
    finally:
        await conn.execute("DELETE FROM spm_perception WHERE session_id = $1", SID)
        await conn.close()


@pytest.mark.asyncio
async def test_scheduler_reports_idle_only_when_nothing_runs_or_waits():
    from proxy.core.llm_scheduler import LLMScheduler
    s = LLMScheduler()
    assert s.idle()
    async with s.slot("chat", session_id="x"):
        assert not s.idle()
    assert s.idle()
