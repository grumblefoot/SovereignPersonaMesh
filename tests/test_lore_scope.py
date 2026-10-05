"""OPEN-007 / Part B2 acceptance: lore rules are scoped per chat.

Real spm_test database: a rule approved in chat A is never injected in chat B;
a promoted (canon) rule appears in both; deleting a session removes its
chat-scoped rules and keeps canon.
"""
import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest

from proxy.rag.retriever import EpisodicRAGRetriever
from tests._testdb import TEST_DB_CONFIG

CHAR = "scopetester"
TABLE = f"csa_lore_rules_{CHAR}"


@asynccontextmanager
async def scoped_pool():
    """Per-test pool on THIS test's event loop (the session db_pool fixture lives
    on another loop); the table is cleaned on the way in and out."""
    pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
    try:
        async with pool.acquire() as conn:
            await conn.execute("SELECT create_csa_lore_rules_table($1);", CHAR)
            await conn.execute(f"DELETE FROM {TABLE};")
        yield pool
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {TABLE};")
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
            await _seed_rule(conn, "In chat A the dragon is friendly.", "st_chat_A")
        retriever = EpisodicRAGRetriever(pool)
        in_a = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_A")
        in_b = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_B")
        assert [r["rule_text"] for r in in_a["invariants"]] == ["In chat A the dragon is friendly."]
        assert in_b["invariants"] == []      # OPEN-007: no bleed into other chats


@pytest.mark.asyncio
async def test_canon_rule_appears_in_every_chat():
    async with scoped_pool() as pool:
        async with pool.acquire() as conn:
            await _seed_rule(conn, "The dragon hoards memories, not gold.", "canon", scope="global")
        retriever = EpisodicRAGRetriever(pool)
        for sid in ("st_chat_A", "st_chat_B"):
            res = await retriever.retrieve_lore_rules(CHAR, None, session_id=sid)
            assert [r["rule_text"] for r in res["invariants"]] == [
                "The dragon hoards memories, not gold."]


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
            retriever = EpisodicRAGRetriever(pool)
            res = await retriever.retrieve_lore_rules(CHAR, None, session_id="st_chat_OTHER")
            assert [r["rule_text"] for r in res["invariants"]] == ["Promoted truth."]
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
