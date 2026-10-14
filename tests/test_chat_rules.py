"""Brief A1a: the spm_chat_rules store (proxy/rag/chat_rules.py) and migration 007.

Real test database (tests/_testdb.py); each test cleans the rows it inserts.
"""
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import pytest

from proxy.rag import chat_rules
from tests._testdb import TEST_DB_CONFIG

MIGRATION_007 = (Path(__file__).resolve().parent.parent
                 / "scripts" / "migrations" / "007_chat_rules.sql")


@asynccontextmanager
async def conn_ctx():
    """Per-test connection on THIS test's event loop; spm_chat_rules cleaned
    in and out (the session db_pool fixture lives on another loop)."""
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        await conn.execute("DELETE FROM spm_chat_rules;")
        yield conn
        await conn.execute("DELETE FROM spm_chat_rules;")
    finally:
        await conn.close()


async def _test_space(conn) -> int:
    return await conn.fetchval(
        "INSERT INTO spm_embedding_spaces (provider, model, dim) "
        "VALUES ('test', 'chat-rules-test-' || gen_random_uuid()::text, 3) "
        "RETURNING id")


# --- add_rule: duplicates -----------------------------------------------------

async def test_add_rule_exact_duplicate_rejected():
    async with conn_ctx() as conn:
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="The dragon fears mirrors.", rule_type="invariant")
        assert rid is not None
        dup = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="The dragon fears mirrors.", rule_type="invariant")
        assert dup is None
        # Another chat may hold the same rule text (OPEN-007).
        other = await chat_rules.add_rule(
            conn, session_id="T", level="character_internal", character_id="mei",
            rule_text="The dragon fears mirrors.", rule_type="invariant")
        assert other is not None


async def test_add_rule_duplicate_ignores_case_and_whitespace():
    async with conn_ctx() as conn:
        await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id=None,
            rule_text="Magic hums at dusk.", rule_type="invariant")
        dup = await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id=None,
            rule_text="  MAGIC hums at dusk.  ", rule_type="invariant")
        assert dup is None


async def test_add_rule_near_duplicate_by_embedding():
    """A reworded restatement in the same space within NEAR_DUPLICATE_DISTANCE is
    a duplicate; a distinct vector, another space, or no embedding is not.
    The test DB's vector column is dimension-free, so 3-d vectors are fine."""
    async with conn_ctx() as conn:
        space = await _test_space(conn)
        near, far = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Bursts of power at dawn.", rule_type="invariant",
            embedding=near, space_id=space)
        assert rid is not None
        # 0.98,0.15,0 has cosine distance ~0.011 from 1,0,0 (< 0.20).
        dup = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Sudden power surges at daybreak.", rule_type="invariant",
            embedding=[0.98, 0.15, 0.0], space_id=space)
        assert dup is None
        # Far vector: not a duplicate.
        kept = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Rivers run backwards in March.", rule_type="invariant",
            embedding=far, space_id=space)
        assert kept is not None
        # Same near vector in ANOTHER space: spaces are never compared.
        other_space = await _test_space(conn)
        cross = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Dawn brings surging strength.", rule_type="invariant",
            embedding=near, space_id=other_space + 1000)
        assert cross is not None


async def test_add_rule_duplicate_scope_world_versus_character():
    """A world rule already covers every character, so a character restating it is
    a duplicate; restating a single character's rule at world level covers new
    characters, so it is allowed."""
    async with conn_ctx() as conn:
        await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id=None,
            rule_text="Gravity works here.", rule_type="invariant")
        dup = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Gravity works here.", rule_type="invariant")
        assert dup is None

        await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="lian",
            rule_text="Lian counts stars.", rule_type="invariant")
        promoted = await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id=None,
            rule_text="Lian counts stars.", rule_type="invariant")
        assert promoted is not None


# --- add_rule: validation -----------------------------------------------------

async def test_add_rule_validation_errors():
    async with conn_ctx() as conn:
        kw = dict(session_id="S", rule_text="x", rule_type="invariant")
        with pytest.raises(ValueError):
            await chat_rules.add_rule(conn, level="galaxy", character_id="mei", **kw)
        with pytest.raises(ValueError):
            await chat_rules.add_rule(conn, session_id="S",
                                      level="character_internal",
                                      character_id="mei", rule_text="x",
                                      rule_type="vibe")
        with pytest.raises(ValueError):
            await chat_rules.add_rule(conn, session_id="S",
                                      level="character_internal",
                                      character_id="mei", rule_text="x",
                                      rule_type="invariant", status="approved")
        with pytest.raises(ValueError):   # character level needs a character_id
            await chat_rules.add_rule(conn, level="character_internal",
                                      character_id=None, **kw)
        with pytest.raises(ValueError):   # empty text
            await chat_rules.add_rule(conn, level="world", character_id=None,
                                      rule_text="   ", rule_type="invariant",
                                      session_id="S")
        # world level silently drops any character_id that is passed in.
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id="mei",
            rule_text="The sky is green.", rule_type="invariant")
        row = await conn.fetchrow("SELECT character_id FROM spm_chat_rules WHERE id = $1",
                                  uuid.UUID(rid))
        assert row["character_id"] is None


# --- list_rules ---------------------------------------------------------------

async def test_list_rules_filters():
    async with conn_ctx() as conn:
        await chat_rules.add_rule(conn, session_id="S", level="world",
                                  character_id=None, rule_text="W1",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id="S", level="character_internal",
                                  character_id="mei", rule_text="M1",
                                  rule_type="invariant")
        await chat_rules.add_rule(conn, session_id="S", level="character_external",
                                  character_id="lian", rule_text="L1",
                                  rule_type="conditional_trigger", status="active")
        await chat_rules.add_rule(conn, session_id="T", level="world",
                                  character_id=None, rule_text="Other chat",
                                  rule_type="invariant")

        all_s = await chat_rules.list_rules(conn, "S")
        assert {r["rule_text"] for r in all_s} == {"W1", "M1", "L1"}
        for r in all_s:
            assert r["status"] and r["source"] and r["id"]

        by_level = await chat_rules.list_rules(conn, "S", level="world")
        assert [r["rule_text"] for r in by_level] == ["W1"]
        by_char = await chat_rules.list_rules(conn, "S", character_id="mei")
        assert [r["rule_text"] for r in by_char] == ["M1"]
        by_status = await chat_rules.list_rules(conn, "S", status="pending")
        assert [r["rule_text"] for r in by_status] == ["M1"]
        combo = await chat_rules.list_rules(
            conn, "S", level="character_external", status="active")
        assert [r["rule_text"] for r in combo] == ["L1"]
        assert await chat_rules.list_rules(conn, "no_such_chat") == []


# --- update_rule / delete_rule --------------------------------------------------

async def test_update_rule_fields_and_session_scope():
    async with conn_ctx() as conn:
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Draft rule.", rule_type="invariant")
        assert await chat_rules.update_rule(conn, "S", rid, status="active",
                                            rule_text="Final rule.",
                                            rule_type="conditional_trigger") is True
        row = await conn.fetchrow("SELECT * FROM spm_chat_rules WHERE id = $1",
                                  uuid.UUID(rid))
        assert row["status"] == "active"
        assert row["rule_text"] == "Final rule."
        assert row["rule_type"] == "conditional_trigger"
        assert row["updated_at"] >= row["created_at"]

        # Another chat's id returns False and changes nothing.
        assert await chat_rules.update_rule(conn, "T", rid, status="pending") is False
        row = await conn.fetchrow("SELECT status FROM spm_chat_rules WHERE id = $1",
                                  uuid.UUID(rid))
        assert row["status"] == "active"
        assert await chat_rules.update_rule(conn, "S", str(uuid.uuid4())) is False


async def test_update_rule_validation():
    async with conn_ctx() as conn:
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="Keep me.", rule_type="invariant")
        with pytest.raises(ValueError):        # unknown field
            await chat_rules.update_rule(conn, "S", rid, source="manual")
        with pytest.raises(ValueError):        # bad status
            await chat_rules.update_rule(conn, "S", rid, status="retired")
        with pytest.raises(ValueError):        # bad rule_type
            await chat_rules.update_rule(conn, "S", rid, rule_type="omens")
        with pytest.raises(ValueError):        # bad level
            await chat_rules.update_rule(conn, "S", rid, level="plane")
        with pytest.raises(ValueError):        # character level without a character
            await chat_rules.update_rule(conn, "S", rid, level="character_external",
                                         character_id=None)
        # world level drops the character; internal -> external keeps it.
        assert await chat_rules.update_rule(conn, "S", rid,
                                            level="character_external") is True
        assert await chat_rules.update_rule(conn, "S", rid, level="world") is True
        row = await conn.fetchrow("SELECT level, character_id FROM spm_chat_rules "
                                  "WHERE id = $1", uuid.UUID(rid))
        assert row["level"] == "world" and row["character_id"] is None
        # garbage (non-UUID) id returns False instead of raising
        assert await chat_rules.update_rule(conn, "S", "not-a-uuid", status="active") is False


async def test_delete_rule():
    async with conn_ctx() as conn:
        rid = await chat_rules.add_rule(
            conn, session_id="S", level="world", character_id=None,
            rule_text="Delete me.", rule_type="invariant")
        other = await chat_rules.add_rule(
            conn, session_id="T", level="world", character_id=None,
            rule_text="Other chat rule.", rule_type="invariant")
        assert await chat_rules.delete_rule(conn, "T", rid) is False   # wrong chat
        assert await chat_rules.delete_rule(conn, "S", rid) is True
        assert await chat_rules.delete_rule(conn, "S", rid) is False   # already gone
        assert await chat_rules.delete_rule(conn, "S", "not-a-uuid") is False
        assert await chat_rules.list_rules(conn, "S") == []
        assert [r["rule_text"] for r in await chat_rules.list_rules(conn, "T")] \
            == ["Other chat rule."]
        await chat_rules.delete_rule(conn, "T", other)


# --- rules_for_turn -------------------------------------------------------------

async def _turn(conn, char, co_present=(), query_embedding=None, space_id=None,
                max_cosine_distance=0.35, session_id="S"):
    return await chat_rules.rules_for_turn(
        conn, session_id=session_id, character_id=char, co_present=co_present,
        query_embedding=query_embedding, space_id=space_id,
        max_cosine_distance=max_cosine_distance)


async def test_rules_for_turn_audience():
    async with conn_ctx() as conn:
        await chat_rules.add_rule(conn, session_id="S", level="world",
                                  character_id=None, rule_text="Shared world truth.",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id="S", level="character_internal",
                                  character_id="mei", rule_text="Mei's secret.",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id="S", level="character_external",
                                  character_id="mei", rule_text="Mei glows.",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id="S", level="character_internal",
                                  character_id="lian", rule_text="Lian's secret.",
                                  rule_type="invariant", status="active")
        await chat_rules.add_rule(conn, session_id="S", level="world",
                                  character_id=None, rule_text="Not approved yet.",
                                  rule_type="invariant")   # pending
        await chat_rules.add_rule(conn, session_id="T", level="world",
                                  character_id=None, rule_text="Other chat world rule.",
                                  rule_type="invariant", status="active")

        def texts(res):
            return {r["rule_text"] for r in res["invariants"]}

        mei = await _turn(conn, "mei")
        assert texts(mei) == {"Shared world truth.", "Mei's secret.", "Mei glows."}

        lian_alone = await _turn(conn, "lian")
        assert texts(lian_alone) == {"Shared world truth.", "Lian's secret."}

        lian_with_mei = await _turn(conn, "lian", co_present=["mei"])
        assert texts(lian_with_mei) == {"Shared world truth.", "Lian's secret.",
                                        "Mei glows."}
        # co_present exposes external rules only, never another's internal.
        assert "Mei's secret." not in texts(lian_with_mei)

        # pending reaches nobody; another chat never bleeds in.
        assert "Not approved yet." not in texts(mei)
        in_t = await _turn(conn, "mei", session_id="T")
        assert texts(in_t) == {"Other chat world rule."}

        # rule dicts carry the spec'd keys
        r = (await _turn(conn, "mei"))["invariants"][0]
        assert set(r) == {"id", "rule_text", "rule_type", "level", "character_id"}


async def test_rules_for_turn_triggers_need_matching_embedding():
    async with conn_ctx() as conn:
        space = await _test_space(conn)
        await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="When fire appears, water rises.", rule_type="conditional_trigger",
            status="active", embedding=[1.0, 0.0, 0.0], space_id=space)
        await chat_rules.add_rule(
            conn, session_id="S", level="character_internal", character_id="mei",
            rule_text="The world ends at midnight.", rule_type="game_over",
            status="active", embedding=[0.0, 1.0, 0.0], space_id=space)

        # No query embedding: triggers are skipped entirely.
        none = await _turn(conn, "mei")
        assert none["triggers"] == []

        near = await _turn(conn, "mei", query_embedding=[0.99, 0.01, 0.0],
                           space_id=space)
        assert [r["rule_text"] for r in near["triggers"]] \
            == ["When fire appears, water rises."]
        assert near["triggers"][0]["cosine_distance"] < 0.35
        assert set(near["triggers"][0]) == {"id", "rule_text", "rule_type", "level",
                                            "character_id", "cosine_distance"}

        far = await _turn(conn, "mei", query_embedding=[0.0, 0.99, 0.01],
                          space_id=space)
        assert [r["rule_text"] for r in far["triggers"]] == ["The world ends at midnight."]

        # A rule stored in another space is never compared.
        wrong_space = await _turn(conn, "mei", query_embedding=[0.99, 0.01, 0.0],
                                  space_id=space + 1000)
        assert wrong_space["triggers"] == []


# --- migration 007 ---------------------------------------------------------------

async def test_migration_007_copies_chat_rows_skips_canon_and_legacy():
    char = "migtest"
    table = f"csa_lore_rules_{char}"
    async with conn_ctx() as conn:
        try:
            await conn.execute("SELECT create_csa_lore_rules_table($1);", char)
            await conn.execute(
                f"INSERT INTO {table} (rule_text, rule_type, status, session_id, scope) "
                "VALUES "
                "('Chat rule one.', 'invariant', 'active', 'st_mig', 'chat'), "
                "('Trigger restated.', 'trigger', 'pending', 'st_mig', 'chat'), "
                "('Weird type.', 'omens', 'active', 'st_mig', 'chat'), "
                "('Canon truth.', 'invariant', 'active', 'canon', 'global'), "
                "('Legacy row.', 'invariant', 'active', 'legacy_global', 'chat');")

            await conn.execute(MIGRATION_007.read_text())

            rows = await conn.fetch(
                "SELECT * FROM spm_chat_rules WHERE character_id = $1", char)
            assert len(rows) == 3
            for r in rows:
                assert r["level"] == "character_internal"
                assert r["source"] == "migrated"
                assert r["session_id"] == "st_mig"
            by_text = {r["rule_text"]: r for r in rows}
            assert by_text["Chat rule one."]["rule_type"] == "invariant"
            assert by_text["Chat rule one."]["status"] == "active"
            assert by_text["Trigger restated."]["rule_type"] == "conditional_trigger"
            assert by_text["Weird type."]["rule_type"] == "invariant"
            assert "Canon truth." not in by_text       # scope='global' skipped
            assert "Legacy row." not in by_text        # legacy_global skipped

            # Re-running the migration adds nothing.
            await conn.execute(MIGRATION_007.read_text())
            rows2 = await conn.fetch(
                "SELECT * FROM spm_chat_rules WHERE character_id = $1", char)
            assert len(rows2) == 3
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {table};")
