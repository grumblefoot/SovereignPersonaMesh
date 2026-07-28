"""
Unit tests for Observer Inference Gating & Bypass Protocol (Phase 4).
Tests zero-inference ambient log commits to litellm_postgres.
"""

import pytest
import asyncpg
from proxy.core.sensory_filter import ObserverInferenceGatingFilter

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


@pytest.mark.asyncio
async def test_evaluate_and_bypass_blackout():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        gating_filter = ObserverInferenceGatingFilter(pool)
        char_id = "test_bypass_blackout"
        session_id = "sess_bypass_1"

        bypassed = await gating_filter.evaluate_and_bypass(
            character_id=char_id,
            session_id=session_id,
            gating_level="blackout",
            sensory_feed="Muffled thuds heard from downstairs."
        )

        assert bypassed is True

        # Verify ambient log was committed directly to DB with null monologue
        async with pool.acquire() as conn:
            records = await conn.fetch(
                f"SELECT sensory_input, inner_monologue, importance_score FROM csa_memory_{char_id} WHERE session_id = $1",
                session_id
            )
            assert len(records) == 1
            assert records[0]["sensory_input"] == "Muffled thuds heard from downstairs."
            assert records[0]["inner_monologue"] is None
            assert records[0]["importance_score"] == 1

            await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_evaluate_and_bypass_null():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        gating_filter = ObserverInferenceGatingFilter(pool)
        char_id = "test_bypass_null"

        bypassed = await gating_filter.evaluate_and_bypass(
            character_id=char_id,
            session_id="sess_null",
            gating_level="null",
            sensory_feed=""
        )

        assert bypassed is True
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_evaluate_and_bypass_direct_requires_llm():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        gating_filter = ObserverInferenceGatingFilter(pool)
        bypassed = await gating_filter.evaluate_and_bypass(
            character_id="luna",
            session_id="sess_direct",
            gating_level="direct",
            sensory_feed="Hello Luna!"
        )
        assert bypassed is False
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_evaluate_and_bypass_degraded_requires_llm():
    pool = await asyncpg.create_pool(**DB_CONFIG)
    try:
        gating_filter = ObserverInferenceGatingFilter(pool)
        bypassed = await gating_filter.evaluate_and_bypass(
            character_id="luna",
            session_id="sess_degraded",
            gating_level="degraded",
            sensory_feed="You hear whispering."
        )
        assert bypassed is False
    finally:
        await pool.close()
