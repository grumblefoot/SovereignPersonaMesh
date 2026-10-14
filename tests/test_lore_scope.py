"""OPEN-007 / Part B2 acceptance: lore rules are scoped per chat.

Real spm_test database: a rule approved in chat A is never injected in chat B;
a world rule covers every character in its chat; deleting a session removes its
chat-scoped rules and keeps canon.

Brief A1b: the RETRIEVAL tests read spm_chat_rules (retrieve_lore_rules no
longer touches csa_lore_rules_*). The old-table tests around the promote/delete
endpoints stay — those endpoints move to the new table in brief B1.
"""
import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest

from proxy.rag import chat_rules
from proxy.rag.retriever import EpisodicRAGRetriever
from tests._testdb import TEST_DB_CONFIG

CHAR = "scopetester"
TABLE = f"csa_lore_rules_{CHAR}"
LORE_SESSIONS = ("st_chat_A", "st_chat_B", "st_chat_OTHER")


@asynccontextmanager
async def scoped_pool():
    """Per-test pool on THIS test's event loop (the session db_pool fixture lives
    on another loop); the tables are cleaned on the way in and out."""
    pool = await asyncpg.create_pool(**TEST_DB_CONFIG)

    async def _clean(conn):
        await conn.execute(f"DELETE FROM {TABLE};")
        await conn.execute(
            "DELETE FROM spm_chat_rules WHERE session_id = ANY($1::text[]);",
            list(LORE_SESSIONS))

    try:
        async with pool.acquire() as conn:
            await conn.execute("SELECT create_csa_lore_rules_table($1);", CHAR)
            await _clean(conn)
        yield pool
        async with pool.acquire() as conn:
            await _clean(conn)
    finally:
        await pool.close()


async def _seed_rule(conn, text, session_id, scope="chat", status="active"):
    rid = uuid.uuid4()
    await conn.execute(
        f"INSERT INTO {TABLE} (id, rule_text, rule_type, status, session_id, scope) "
        f"VALUES ($1, $2, 'invariant', $3, $4, $5)",
        rid, text, status, session_id, scope)
    return str(rid)


@pytest.mark.asyncio
async def test_chat_scoped_rule_stays_in_its_chat():
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            await chat_rules.add_rule(conn, session_id="st_chat_A", level="world",
                                      character_id=None,
                                      rule_text="In chat A the dragon is friendly.",
                                      rule_type="invariant", status="active")
        retriever = EpisodicRAGRetriever(pool)
        in_a = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_A")
        in_b = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_B")
        assert [r["rule_text"] for r in in_a["invariants"]] == ["In chat A the dragon is friendly."]
        assert in_b["invariants"] == []      # OPEN-007: no bleed into other chats


@pytest.mark.asyncio
async def test_world_rule_covers_every_character_in_its_chat():
    """Old brief: a promoted (canon) rule appeared in every chat. Brief A1 moved
    rules to spm_chat_rules, which is chat-scoped: the level that covers everyone
    is 'world', and it covers every character of ITS chat — never another chat."""
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            await chat_rules.add_rule(conn, session_id="st_chat_A", level="world",
                                      character_id=None,
                                      rule_text="The dragon hoards memories, not gold.",
                                      rule_type="invariant", status="active")
        retriever = EpisodicRAGRetriever(pool)
        for who in (CHAR, "lian"):
            res = await retriever.retrieve_lore_rules(who, None, session_id="st_chat_A")
            assert [r["rule_text"] for r in res["invariants"]] == [
                "The dragon hoards memories, not gold."]
        other = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_B")
        assert other["invariants"] == []


@pytest.mark.asyncio
async def test_promote_endpoint_makes_rule_canon():
    import proxy.api.admin_routes as admin
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            rid = await _seed_rule(conn, "Promoted truth.", "st_chat_A")
        admin.set_admin_db_pool(pool)
        try:
            # Called directly: TestClient would run the endpoint on another event
            # loop than this test's pool (the leak rig has its own loop for that).
            resp = await admin.promote_lore(CHAR, rid)
            assert resp.status_code == 200, resp.body
            async with pool.acquire() as conn:
                scope = await conn.fetchval(
                    f"SELECT scope FROM {TABLE} WHERE id = $1", uuid.UUID(rid))
            assert scope == "global"
            # Retrieval no longer reads this table (brief A1b): retrieval from
            # spm_chat_rules is covered in tests/test_chat_rules.py, and this
            # endpoint moves to the new table in brief B1.
        finally:
            admin.set_admin_db_pool(None)


@pytest.mark.asyncio
async def test_session_delete_removes_chat_rules_keeps_canon():
    import proxy.api.admin_routes as admin
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            await _seed_rule(conn, "Chat-only detail.", "st_chat_DEL")
            await _seed_rule(conn, "Eternal canon.", "canon", scope="global")
        admin.set_admin_db_pool(pool)
        try:
            resp = await admin.delete_session("st_chat_DEL")
            assert resp.status_code == 200, resp.body
            async with pool.acquire() as conn:
                left = [r["rule_text"] for r in await conn.fetch(f"SELECT rule_text FROM {TABLE};")]
            assert left == ["Eternal canon."]
        finally:
            admin.set_admin_db_pool(None)


@pytest.mark.asyncio
async def test_same_rule_text_allowed_in_two_chats_but_deduped_within_one():
    """The extractor's dedupe check is per (chat ∪ canon), not global (B.4)."""
    from proxy.rag.lore_extractor import strings as lore_strings
    sql = lore_strings.get("sql.check_lore_rule_exists").format(table_name=TABLE)
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            await _seed_rule(conn, "The moon is a lie.", "st_chat_A")
            assert await conn.fetchval(sql, "The moon is a lie.", "st_chat_A") is not None
            assert await conn.fetchval(sql, "The moon is a lie.", "st_chat_B") is None
            await _seed_rule(conn, "The moon is a lie.", "canon", scope="global")
            assert await conn.fetchval(sql, "The moon is a lie.", "st_chat_B") is not None


@pytest.mark.asyncio
async def test_near_duplicate_rule_detected_by_embedding_distance():
    """QA O4: a reworded restatement of an existing rule is a duplicate (same chat or
    canon, same embedding space); a distinct rule, another chat or another space is not."""
    from proxy.rag.lore_extractor import NEAR_DUPLICATE_DISTANCE, strings as lore_strings
    sql = lore_strings.get("sql.check_lore_rule_near_duplicate").format(table_name=TABLE)
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            space = await conn.fetchval("SELECT id FROM spm_embedding_spaces ORDER BY id LIMIT 1")
            if space is None:
                space = await conn.fetchval(
                    "INSERT INTO spm_embedding_spaces (provider, model, dim) "
                    "VALUES ('test', 'near-dup-test', 3) RETURNING id")
            await conn.execute(
                f"INSERT INTO {TABLE} (id, rule_text, rule_type, status, session_id, scope, "
                f"rule_embedding, embedding_space_id) VALUES ($1, 'Bursts of power.', 'invariant', "
                f"'pending', 'st_chat_A', 'chat', '[1,0,0]'::vector, $2)", uuid.uuid4(), space)
            near, far = "[0.98,0.15,0]", "[0.6,0.8,0]"          # cos dist ~0.011 / 0.4
            assert await conn.fetchval(sql, near, "st_chat_A", space, NEAR_DUPLICATE_DISTANCE)
            assert await conn.fetchval(sql, far, "st_chat_A", space, NEAR_DUPLICATE_DISTANCE) is None
            assert await conn.fetchval(sql, near, "st_chat_B", space, NEAR_DUPLICATE_DISTANCE) is None
            assert await conn.fetchval(sql, near, "st_chat_A", space + 1000, NEAR_DUPLICATE_DISTANCE) is None
