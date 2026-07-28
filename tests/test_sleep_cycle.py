"""
Unit & integration tests for Nightly Memory Consolidation (Phase 6).
Tests MemoryConsolidationWorker against a real PostgreSQL/pgvector instance
and validates core memory node insertion, volatile log pruning, and
Gemma 9B HTTP consolidation stub.
"""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import asyncpg
import pytest

from scripts.sleep_cycle import MemoryConsolidationWorker, SLEEP_CYCLE_PROMPT_TEMPLATE


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


# ── Prompt Template Tests ─────────────────────────────────────────────────

def test_prompt_template_contains_required_sections():
    """Verify the template has all required sections for Gemma 9B."""
    rendered = SLEEP_CYCLE_PROMPT_TEMPLATE.format(
        character_id="testchar",
        daily_logs="[2026-01-01] Sensory: walked into room",
    )
    assert "testchar" in rendered
    assert "walked into room" in rendered
    assert "Memory Consolidation Engine" in rendered
    assert "first-person perspective" in rendered
    assert "<boss>" in rendered
    assert "<idle>" in rendered


# ── MemoryConsolidationWorker Initialization ──────────────────────────────

def test_worker_initialization():
    """Verify worker stores db_config and model_url."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    assert worker.db_config == DB_CONFIG
    assert worker.consolidation_model_url == "http://localhost:13305/v1"


# ── Unit: _call_consolidation_model (Mocked) ──────────────────────────────

@pytest.mark.asyncio
async def test_call_consolidation_model_success():
    """Verify _call_consolidation_model parses the Gemma 9B response correctly."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")

    mock_response = AsyncMock()
    # response.raise_for_status() is sync in httpx, so use Mock
    mock_response.raise_for_status = Mock()
    # response.json() is sync in httpx, so use Mock
    mock_response.json = Mock(return_value={
        "choices": [
            {
                "message": {
                    "content": "<boss>I was there.</boss> I walked through the dark corridor.\n<idle>"
                }
            }
        ]
    })

    with patch("httpx.AsyncClient") as MockClient:
        instance = Mock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await worker._call_consolidation_model("test prompt")

    # Should strip tags and return first sentence
    assert result == "I was there."


@pytest.mark.asyncio
async def test_call_consolidation_model_no_period():
    """Verify fallback when Gemma 9B returns text without a period."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")

    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={
        "choices": [{"message": {"content": "no period here"}}]
    })

    with patch("httpx.AsyncClient") as MockClient:
        instance = Mock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await worker._call_consolidation_model("test prompt")

    assert result == "no period here"


# ── Integration: get_active_character_tables ──────────────────────────────

@pytest.mark.asyncio
async def test_get_active_character_tables():
    """Verify the worker can discover all csa_memory_* tables."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        tables = await worker.get_active_character_tables(conn)
        table_names = [t for t in tables if t.startswith("csa_memory_")]
        expected_demo = [
            "csa_memory_domino",
            "csa_memory_luna",
            "csa_memory_rowan",
            "csa_memory_seamus",
        ]
        for name in expected_demo:
            assert name in table_names, f"Missing expected table: {name}"
    finally:
        await conn.close()


# ── Integration: process_character_sleep_cycle (empty table) ──────────────

@pytest.mark.asyncio
async def test_process_character_empty_table():
    """Verify no crash when table has no un-consolidated logs."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await conn.execute("SELECT create_csa_memory_table($1);", "test_empty")
        # No records inserted — should return gracefully
        await worker.process_character_sleep_cycle(conn, "csa_memory_test_empty")
    finally:
        await conn.close()


# ── Integration: core memory node insertion (is_core_memory=TRUE) ────────

@pytest.mark.asyncio
async def test_core_memory_node_insertion():
    """Verify that consolidation inserts a node with is_core_memory=TRUE."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await conn.execute("SELECT create_csa_memory_table($1);", "test_core_insert")

        # Insert a volatile record within 24h
        await conn.execute(
            """INSERT INTO csa_memory_test_core_insert
               (session_id, sensory_input, inner_monologue, is_core_memory, timestamp)
               VALUES ($1, $2, $3, FALSE, NOW())""",
            "test", "Walked into the tavern", "Felt nervous",
        )

        # Mock the model call to return a real consolidation sentence
        with patch.object(worker, "_call_consolidation_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "I entered the tavern and felt nervous."
            await worker.process_character_sleep_cycle(conn, "csa_memory_test_core_insert")

        # Verify core memory node was inserted
        core_nodes = await conn.fetch(
            "SELECT sensory_input, is_core_memory FROM csa_memory_test_core_insert WHERE is_core_memory = TRUE"
        )
        assert len(core_nodes) >= 1
        assert core_nodes[0]["is_core_memory"] is True
        # Verify the sensory_input is the consolidation summary
        assert core_nodes[0]["sensory_input"] == "I entered the tavern and felt nervous."
        # Verify session_id is set correctly
        session_check = await conn.fetchval(
            "SELECT session_id FROM csa_memory_test_core_insert WHERE is_core_memory = TRUE LIMIT 1"
        )
        assert session_check == "sleep_cycle_consolidated"
    finally:
        await conn.close()


# ── Integration: volatile log pruning ────────────────────────────────────

@pytest.mark.asyncio
async def test_volatile_log_pruning():
    """Verify old volatile logs are pruned while core memory nodes are retained."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await conn.execute("SELECT create_csa_memory_table($1);", "test_prune")
        # Clean up stale data from previous test runs
        await conn.execute("DELETE FROM csa_memory_test_prune WHERE session_id = 'test';")

        # Insert old volatile log (48h ago — should be pruned)
        await conn.execute(
            """INSERT INTO csa_memory_test_prune
               (session_id, sensory_input, is_core_memory, timestamp)
               VALUES ($1, $2, FALSE, NOW() - INTERVAL '48 hours')""",
            "test", "Old volatile log",
        )

        # Insert recent volatile log (1h ago — should survive pruning)
        await conn.execute(
            """INSERT INTO csa_memory_test_prune
               (session_id, sensory_input, is_core_memory, timestamp)
               VALUES ($1, $2, FALSE, NOW() - INTERVAL '1 hour')""",
            "test", "Recent volatile log",
        )

        # Insert core memory node (should NEVER be pruned)
        await conn.execute(
            """INSERT INTO csa_memory_test_prune
               (session_id, sensory_input, is_core_memory, timestamp)
               VALUES ($1, $2, TRUE, NOW() - INTERVAL '48 hours')""",
            "test", "Core memory old",
        )

        # Mock the model call
        with patch.object(worker, "_call_consolidation_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Consolidated memory."
            await worker.process_character_sleep_cycle(conn, "csa_memory_test_prune")

        # Verify old volatile log was pruned
        old_volatile = await conn.fetchval(
            "SELECT COUNT(*) FROM csa_memory_test_prune "
            "WHERE is_core_memory = FALSE AND sensory_input = 'Old volatile log'"
        )
        assert old_volatile == 0, "Old volatile log should have been pruned"

        # Verify recent volatile log survived
        recent_volatile = await conn.fetchval(
            "SELECT COUNT(*) FROM csa_memory_test_prune "
            "WHERE is_core_memory = FALSE AND sensory_input = 'Recent volatile log'"
        )
        assert recent_volatile == 1, "Recent volatile log should survive pruning"

        # Verify core memory node was NEVER pruned
        core_retained = await conn.fetchval(
            "SELECT COUNT(*) FROM csa_memory_test_prune "
            "WHERE is_core_memory = TRUE AND sensory_input = 'Core memory old'"
        )
        assert core_retained == 1, "Core memory node should never be pruned"
    finally:
        await conn.close()


# ── Integration: run() discovers and processes all tables ─────────────────

@pytest.mark.asyncio
async def test_run_processes_all_tables():
    """Verify run() iterates over all discovered csa_memory_* tables."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")

    # Insert a test record that will trigger consolidation
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await conn.execute("SELECT create_csa_memory_table($1);", "test_run")
        await conn.execute(
            """INSERT INTO csa_memory_test_run
               (session_id, sensory_input, is_core_memory, timestamp)
               VALUES ($1, $2, FALSE, NOW())""",
            "test", "Run test record",
        )

        with patch.object(worker, "_call_consolidation_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "I processed a record."
            await worker.run()

        # Verify consolidation node was inserted
        core_count = await conn.fetchval(
            "SELECT COUNT(*) FROM csa_memory_test_run WHERE is_core_memory = TRUE"
        )
        assert core_count >= 1
    finally:
        await conn.close()


# ── Integration: first-person single-sentence synthesis ──────────────────

@pytest.mark.asyncio
async def test_synthesis_is_first_person_sentence():
    """Verify the consolidation output is a first-person single sentence."""
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")

    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={
        "choices": [{"message": {"content": "<boss>I saw the dragon at dawn.</boss>"}}]
    })

    with patch("httpx.AsyncClient") as MockClient:
        instance = Mock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await worker._call_consolidation_model("test")

    assert result.startswith("I ")
    assert result.endswith(".")
    # Should be a single sentence (one period, no extra periods)
    assert result.count(".") == 1
