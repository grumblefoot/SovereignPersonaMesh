"""
Unit tests for Nightly Memory Consolidation Worker ("The Sleep Cycle", Phase 6).
Tests core memory synthesis (is_core_memory=TRUE) and volatile log pruning.
"""

import pytest
import asyncpg
from scripts.sleep_cycle import MemoryConsolidationWorker

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


@pytest.mark.asyncio
async def test_get_active_character_tables():
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        tables = await worker.get_active_character_tables(conn)
        assert isinstance(tables, list)
        assert "csa_memory_luna" in tables
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sleep_cycle_consolidation_and_pruning():
    worker = MemoryConsolidationWorker(DB_CONFIG, "http://localhost:13305/v1")
    conn = await asyncpg.connect(**DB_CONFIG)
    char_id = "test_sleep_char"
    table_name = f"csa_memory_{char_id}"

    try:
        await conn.execute("SELECT create_csa_memory_table($1);", char_id)

        # Insert old volatile log (>24h ago)
        await conn.execute(f"""
            INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, timestamp, is_core_memory)
            VALUES ('old_sess', 'Old sensory experience from yesterday', 'Old thought', NOW() - INTERVAL '25 hours', FALSE);
        """)

        # Insert recent volatile log (<24h ago)
        await conn.execute(f"""
            INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, timestamp, is_core_memory)
            VALUES ('recent_sess', 'Recent sensory experience from today', 'Recent thought', NOW() - INTERVAL '1 hour', FALSE);
        """)

        # Insert core memory (is_core_memory = TRUE)
        await conn.execute(f"""
            INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, timestamp, is_core_memory)
            VALUES ('core_sess', 'Permanent core identity memory', 'Core thought', NOW() - INTERVAL '48 hours', TRUE);
        """)

        # Run sleep cycle consolidation
        await worker.process_character_sleep_cycle(conn, table_name)

        # Verify core memory summary node was inserted
        consolidated_nodes = await conn.fetch(f"""
            SELECT sensory_input, is_core_memory, importance_score 
            FROM {table_name} 
            WHERE session_id = 'sleep_cycle_consolidated';
        """)
        assert len(consolidated_nodes) == 1
        assert consolidated_nodes[0]["is_core_memory"] is True
        assert consolidated_nodes[0]["importance_score"] == 8

        # Verify old volatile log was pruned (>24h ago)
        old_volatile = await conn.fetch(f"""
            SELECT id FROM {table_name} WHERE session_id = 'old_sess';
        """)
        assert len(old_volatile) == 0

        # Verify core memory was NOT pruned (Data Integrity SLA-3 protection)
        core_nodes = await conn.fetch(f"""
            SELECT id FROM {table_name} WHERE session_id = 'core_sess';
        """)
        assert len(core_nodes) == 1

    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        await conn.close()
