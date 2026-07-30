"""
FR-003: Tiered Data Lifecycle & Cold Storage Reconstitution Tests.

Verifies:
  1. Core memory immunity (is_core_memory = TRUE never archived).
  2. Gzip compression and archive schema tracking.
  3. Full reconstitution pipeline (archive -> reconstitute -> verify data).
  4. Stats endpoint accuracy.
  5. API endpoints (archive, reconstitute, stats) via TestClient.
"""

import asyncio
import gzip
import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest
from fastapi.testclient import TestClient

from proxy.main import app as proxy_app
from proxy.api.routes import set_db_pool
from proxy.rag.tier_manager import MemoryTierManager, COLD_ARCHIVE_DIR

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _embedding(size: int = 3584, seed: int = 42) -> list:
    """Generate a deterministic embedding vector."""
    rng_state = 0
    vals = []
    for _ in range(size):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        vals.append((rng_state / 0x7FFFFFFF) - 0.5)
    return vals


def _make_msg(role: str, content: str, importance: int = 5) -> dict:
    return {
        "role": role,
        "content": content,
        "importance_score": importance,
        "inner_monologue": None,
        "is_core_memory": False,
        "is_subjective": True,
        "access_count": 1,
    }


async def _insert_memories(
    pool: asyncpg.Pool,
    char_id: str,
    session_id: str,
    records: list,
):
    """Insert memory records into csa_memory_{char_id}."""
    table = f"csa_memory_{char_id.lower()}"
    async with pool.acquire() as conn:
        await conn.execute(f"SELECT create_csa_memory_table($1);", char_id.lower())
        for rec in records:
            ts = rec.get("timestamp", datetime.utcnow() - timedelta(days=60))
            emb = rec.get("episodic_embedding")
            emb_str = f"[{','.join(map(str, emb))}]" if emb else None
            await conn.execute(
                f"""
                INSERT INTO {table}
                    (id, session_id, timestamp, sensory_input, inner_monologue,
                     episodic_embedding, importance_score, is_core_memory,
                     is_subjective, access_count, last_accessed_at)
                VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, NOW());
                """,
                session_id,
                ts,
                rec.get("sensory_input", ""),
                rec.get("inner_monologue"),
                emb_str,
                rec.get("importance_score", 5),
                rec.get("is_core_memory", False),
                rec.get("is_subjective", True),
                rec.get("access_count", 1),
            )


async def _cleanup(char_id: str, session_id: str):
    """Clean up test data from DB and filesystem."""
    table = f"csa_memory_{char_id.lower()}"
    async with asyncpg.create_pool(**DB_CONFIG) as pool:
        try:
            async with pool.acquire() as conn:
                await conn.execute(f"DELETE FROM {table} WHERE session_id = $1;", session_id)
        except Exception:
            pass  # Table may not exist yet
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_cold_archives WHERE session_id = $1;", session_id
                )
                await conn.execute(f"DROP TABLE IF EXISTS {table};")
        except Exception:
            pass  # Cold archives table may not exist
    # Remove any archive files for this session
    base = os.path.join(COLD_ARCHIVE_DIR, char_id.lower())
    if os.path.isdir(base):
        for f in os.listdir(base):
            if session_id in f:
                os.remove(os.path.join(base, f))


# ======================================================================
# 1. Core Memory Immunity
# ======================================================================

class TestCoreMemoryImmunity:
    """Core memories (is_core_memory = TRUE) must never be archived or deleted."""

    async def test_core_memories_not_archived(self):
        """Archive must skip all rows where is_core_memory = TRUE."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_core_immunity"
            session_id = f"fr003_ci_sess_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            now = datetime.utcnow()
            # 6 volatile records (old)
            volatile = [
                {
                    "sensory_input": f"volatile msg {i}",
                    "is_core_memory": False,
                    "timestamp": now - timedelta(days=60),
                    "episodic_embedding": _embedding(),
                    "importance_score": 3,
                }
                for i in range(6)
            ]
            # 3 core memories (old — should survive)
            core = [
                {
                    "sensory_input": f"core memory {i}",
                    "is_core_memory": True,
                    "timestamp": now - timedelta(days=60),
                    "episodic_embedding": _embedding(),
                    "importance_score": 10,
                }
                for i in range(3)
            ]
            await _insert_memories(pool, char_id, session_id, volatile + core)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )

            assert result["deleted_count"] == 6
            assert result["record_count"] == 6

            # Verify core memories still exist
            async with pool.acquire() as conn:
                core_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} "
                    f"WHERE session_id = $1 AND is_core_memory = TRUE;",
                    session_id,
                )
                volatile_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} "
                    f"WHERE session_id = $1 AND is_core_memory = FALSE;",
                    session_id,
                )
                assert core_count == 3, f"Expected 3 core memories, got {core_count}"
                assert volatile_count == 0, f"Expected 0 volatile memories, got {volatile_count}"

            await _cleanup(char_id, session_id)

    async def test_no_archives_when_no_eligible_records(self):
        """If no volatile records older than cutoff, archive does nothing."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_no_elig"
            session_id = f"fr003_ne_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            # Only core memories exist
            core = [
                {
                    "sensory_input": "core only",
                    "is_core_memory": True,
                    "timestamp": datetime.utcnow() - timedelta(days=60),
                    "episodic_embedding": _embedding(),
                }
            ]
            await _insert_memories(pool, char_id, session_id, core)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )

            assert result["record_count"] == 0
            assert result["archive_id"] is None
            await _cleanup(char_id, session_id)

    async def test_recent_volatile_not_archived(self):
        """Volatile records within the age threshold must not be archived."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_recent"
            session_id = f"fr003_rc_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            recent = [
                {
                    "sensory_input": f"recent msg {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                    "episodic_embedding": _embedding(),
                }
                for i in range(10)
            ]
            await _insert_memories(pool, char_id, session_id, recent)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )

            assert result["record_count"] == 0

            # All records still present
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1;",
                    session_id,
                )
                assert count == 10

            await _cleanup(char_id, session_id)


# ======================================================================
# 2. Gzip Compression & Archive Schema Tracking
# ======================================================================

class TestGzipCompressionAndSchema:
    """Test that archives are gzip-compressed and DB schema is tracked."""

    async def test_archive_is_valid_gzip(self):
        """Archived file must be a valid gzip .jsonl.gz file."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_gzip"
            session_id = f"fr003_gz_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            records = [
                {
                    "sensory_input": f"gzip test msg {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(8)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )

            assert result["archive_path"] is not None
            assert os.path.isfile(result["archive_path"])
            assert result["archive_path"].endswith(".jsonl.gz")

            # Verify it's valid gzip + jsonl
            with gzip.open(result["archive_path"], "rt") as f:
                lines = f.readlines()
            assert len(lines) == 8
            for line in lines:
                rec = json.loads(line)
                assert "sensory_input" in rec
                assert rec["sensory_input"].startswith("gzip test msg")

            await _cleanup(char_id, session_id)

    async def test_archive_record_in_db(self):
        """spm_cold_archives must have a row for each archive."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_db_schema"
            session_id = f"fr003_dbs_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            records = [
                {
                    "sensory_input": f"schema test {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(5)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )

            assert result["archive_id"] is not None

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT archive_id, session_id, character_id,
                           archive_path, record_count
                    FROM spm_cold_archives
                    WHERE archive_id = $1;
                    """,
                    result["archive_id"],
                )

            assert row is not None
            assert str(row["archive_id"]) == result["archive_id"]
            assert row["session_id"] == session_id
            assert row["character_id"] == char_id
            assert row["record_count"] == 5
            assert row["archive_path"] == result["archive_path"]

            await _cleanup(char_id, session_id)

    async def test_max_records_respected(self):
        """Only up to max_records should be archived."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_maxrec"
            session_id = f"fr003_mr_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            # 20 old volatile records
            records = [
                {
                    "sensory_input": f"max rec msg {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=60),
                    "episodic_embedding": _embedding(),
                }
                for i in range(20)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            manager = MemoryTierManager(pool)
            result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=10, max_age_days=30
            )

            assert result["record_count"] == 10

            # 10 should remain
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1;",
                    session_id,
                )
                assert count == 10

            await _cleanup(char_id, session_id)


# ======================================================================
# 3. Full Reconstitution Pipeline
# ======================================================================

class TestReconstitutionPipeline:
    """Test the full archive -> reconstitute -> verify data lifecycle."""

    async def test_full_archive_reconstitute_cycle(self):
        """Archive then reconstitute should restore all original records."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_recon"
            session_id = f"fr003_re_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            original_count = 12
            records = [
                {
                    "sensory_input": f"recon msg {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                    "importance_score": i % 10 + 1,
                }
                for i in range(original_count)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            # Step 1: Archive
            manager = MemoryTierManager(pool)
            archive_result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )
            assert archive_result["record_count"] == original_count
            archive_id = archive_result["archive_id"]

            # Step 2: Verify memory table is empty
            async with pool.acquire() as conn:
                count_after_archive = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1;",
                    session_id,
                )
                assert count_after_archive == 0

            # Step 3: Reconstitute
            reconst_result = await manager.reconstitute_cold_archive(
                archive_id=archive_id, character_id=char_id
            )
            assert reconst_result["reconstituted_count"] == original_count

            # Step 4: Verify all records restored (count only; order doesn't matter)
            async with pool.acquire() as conn:
                count_after_reconst = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1;",
                    session_id,
                )
                assert count_after_reconst == original_count

            # Verify each original record exists by sensory_input
            for i in range(original_count):
                async with pool.acquire() as conn:
                    found = await conn.fetchval(
                        f"SELECT 1 FROM csa_memory_{char_id} "
                        f"WHERE session_id = $1 AND sensory_input = $2;",
                        session_id, f"recon msg {i}",
                    )
                assert found is not None, f"Missing record: recon msg {i}"

            # Step 5: Archive file and DB row removed
            assert not os.path.isfile(archive_result["archive_path"])
            async with pool.acquire() as conn:
                arc_row = await conn.fetchrow(
                    "SELECT archive_id FROM spm_cold_archives WHERE archive_id = $1;",
                    archive_id,
                )
                assert arc_row is None

            await _cleanup(char_id, session_id)

    async def test_reconstitute_nonexistent_archive_raises(self):
        """Reconstituting a non-existent archive_id must raise ValueError."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            manager = MemoryTierManager(pool)
            with pytest.raises(ValueError, match="not found"):
                await manager.reconstitute_cold_archive(
                    archive_id="00000000-0000-0000-0000-000000000000",
                    character_id="nonexistent_char",
                )

    async def test_core_memory_survives_reconstitution(self):
        """Core memories must remain untouched during archive + reconstitute."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_recon_core"
            session_id = f"fr003_rc_core_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            # Mix of volatile and core
            volatile_records = [
                {
                    "sensory_input": f"vol {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(10)
            ]
            core_records = [
                {
                    "sensory_input": f"core {i}",
                    "is_core_memory": True,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(5)
            ]
            await _insert_memories(pool, char_id, session_id, volatile_records + core_records)

            manager = MemoryTierManager(pool)
            archive_result = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id, max_records=500, max_age_days=30
            )
            assert archive_result["record_count"] == 10  # only volatile

            # Reconstitute
            await manager.reconstitute_cold_archive(
                archive_id=archive_result["archive_id"], character_id=char_id
            )

            # Verify both sets restored
            async with pool.acquire() as conn:
                vol_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} "
                    f"WHERE session_id = $1 AND is_core_memory = FALSE;",
                    session_id,
                )
                core_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} "
                    f"WHERE session_id = $1 AND is_core_memory = TRUE;",
                    session_id,
                )
                assert vol_count == 10
                assert core_count == 5

            await _cleanup(char_id, session_id)


# ======================================================================
# 4. Stats Endpoint Accuracy
# ======================================================================

class TestStatsEndpoint:
    """get_tier_stats should accurately report hot / warm / cold counts."""

    async def test_stats_hot_warm_cold_counts(self):
        """Stats must reflect correct counts for each tier."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_stats"
            session_id = f"fr003_st_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            # 7 volatile (hot)
            volatile = [
                {
                    "sensory_input": f"hot {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(7)
            ]
            # 4 core (warm)
            core = [
                {
                    "sensory_input": f"warm {i}",
                    "is_core_memory": True,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(4)
            ]
            await _insert_memories(pool, char_id, session_id, volatile + core)

            manager = MemoryTierManager(pool)
            stats = await manager.get_tier_stats(character_id=char_id, session_id=session_id)

            assert stats["hot"] == 7
            assert stats["warm"] == 4
            assert stats["cold"] == 0
            assert stats["total"] == 11

            await _cleanup(char_id, session_id)

    async def test_stats_with_cold_archives(self):
        """Cold count should reflect record_count from spm_cold_archives."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_stats_cold"
            session_id = f"fr003_sc_{time.time_ns()}"
            await _cleanup(char_id, session_id)

            # Create a cold archive record directly (simulate prior archive)
            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT create_csa_memory_table($1);", char_id
                )
                await conn.execute(
                    """
                    INSERT INTO spm_cold_archives
                        (session_id, character_id, archive_path, record_count)
                    VALUES ($1, $2, '/fake/path.jsonl.gz', 15);
                    """,
                    session_id, char_id,
                )

            # 3 volatile hot records
            volatile = [
                {
                    "sensory_input": f"hot {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(3)
            ]
            await _insert_memories(pool, char_id, session_id, volatile)

            manager = MemoryTierManager(pool)
            stats = await manager.get_tier_stats(character_id=char_id, session_id=session_id)

            assert stats["hot"] == 3
            assert stats["warm"] == 0
            assert stats["cold"] == 15
            assert stats["total"] == 18

            await _cleanup(char_id, session_id)

    async def test_stats_no_session_filter(self):
        """Without session_id, stats should aggregate across all sessions."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_stats_agg"
            s1 = f"fr003_sa_1_{time.time_ns()}"
            s2 = f"fr003_sa_2_{time.time_ns()}"
            await _cleanup(char_id, s1)
            await _cleanup(char_id, s2)

            # Session 1: 2 volatile
            vol1 = [
                {
                    "sensory_input": f"s1 vol {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(2)
            ]
            # Session 2: 3 volatile
            vol2 = [
                {
                    "sensory_input": f"s2 vol {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(3)
            ]
            await _insert_memories(pool, char_id, s1, vol1)
            await _insert_memories(pool, char_id, s2, vol2)

            manager = MemoryTierManager(pool)

            # Session-specific: s1 has 2
            stats1 = await manager.get_tier_stats(character_id=char_id, session_id=s1)
            assert stats1["hot"] == 2

            # No session filter: should have 5 total
            stats_all = await manager.get_tier_stats(character_id=char_id)
            assert stats_all["hot"] == 5

            await _cleanup(char_id, s1)
            await _cleanup(char_id, s2)


# ======================================================================
# 5. API Endpoints via TestClient
# ======================================================================

class TestAPIEndpoints:
    """Test /v1/memories/archive, /v1/memories/reconstitute, /v1/memories/stats."""

    async def test_archive_endpoint_success(self):
        """POST /v1/memories/archive should archive and return results."""
        manager_mock = AsyncMock()
        manager_mock.archive_old_memories = AsyncMock(return_value={
            "archive_id": "a1b2c3d4-0000-0000-0000-000000000001",
            "archive_path": "/tmp/test_archive/test_123.jsonl.gz",
            "record_count": 5,
            "deleted_count": 5,
            "char_id": "fr003_api_arc",
        })
        with patch("proxy.api.routes.MemoryTierManager", return_value=manager_mock), \
             patch("proxy.api.routes._db_pool", None):
            pass  # pool check skipped since we're testing the manager path

        # Actually use a real pool but patch the manager for loop safety
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_api_arc"
            session_id = f"fr003_aa_{time.time_ns()}"
            set_db_pool(pool)
            await _cleanup(char_id, session_id)

            records = [
                {
                    "sensory_input": f"api arc {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(5)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            # Patch MemoryTierManager to use the pool-created manager directly
            real_manager = MemoryTierManager(pool)
            with patch("proxy.api.routes.MemoryTierManager", return_value=real_manager):
                with TestClient(proxy_app) as client:
                    resp = client.post(
                        "/v1/memories/archive",
                        json={
                            "character_id": char_id,
                            "session_id": session_id,
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["record_count"] == 5
                    assert data["archive_id"] is not None
                    assert data["archive_path"].endswith(".jsonl.gz")

            await _cleanup(char_id, session_id)

    async def test_archive_endpoint_missing_char(self):
        """POST /v1/memories/archive without character_id returns 400."""
        with TestClient(proxy_app) as client:
            resp = client.post(
                "/v1/memories/archive",
                json={"session_id": "test"},
            )
            assert resp.status_code == 400
            assert "character_id is required" in resp.json()["error"]

    async def test_reconstitute_endpoint_success(self):
        """POST /v1/memories/reconstitute should restore archived data."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_api_recon"
            session_id = f"fr003_ar_{time.time_ns()}"
            set_db_pool(pool)
            await _cleanup(char_id, session_id)

            records = [
                {
                    "sensory_input": f"api recon {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=45),
                    "episodic_embedding": _embedding(),
                }
                for i in range(7)
            ]
            await _insert_memories(pool, char_id, session_id, records)

            # Archive first (direct call, not via API)
            manager = MemoryTierManager(pool)
            arc = await manager.archive_old_memories(
                character_id=char_id, session_id=session_id,
                max_records=500, max_age_days=30,
            )

            # Now reconstitute via API (patch manager to use same pool)
            real_manager = MemoryTierManager(pool)
            with patch("proxy.api.routes.MemoryTierManager", return_value=real_manager):
                with TestClient(proxy_app) as client:
                    resp = client.post(
                        "/v1/memories/reconstitute",
                        json={
                            "archive_id": arc["archive_id"],
                            "character_id": char_id,
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["reconstituted_count"] == 7
                    assert data["char_id"] == char_id

            await _cleanup(char_id, session_id)

    async def test_reconstitute_endpoint_missing_fields(self):
        """POST /v1/memories/reconstitute without required fields returns 400."""
        with TestClient(proxy_app) as client:
            resp = client.post(
                "/v1/memories/reconstitute",
                json={"archive_id": "some-id"},  # missing character_id
            )
            assert resp.status_code == 400

    async def test_stats_endpoint_success(self):
        """GET /v1/memories/stats should return tier counts."""
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            char_id = "fr003_api_stats"
            session_id = f"fr003_as_{time.time_ns()}"
            set_db_pool(pool)
            await _cleanup(char_id, session_id)

            vol = [
                {
                    "sensory_input": f"stat {i}",
                    "is_core_memory": False,
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "episodic_embedding": _embedding(),
                }
                for i in range(3)
            ]
            await _insert_memories(pool, char_id, session_id, vol)

            real_manager = MemoryTierManager(pool)
            with patch("proxy.api.routes.MemoryTierManager", return_value=real_manager):
                with TestClient(proxy_app) as client:
                    resp = client.get(
                        "/v1/memories/stats",
                        params={"character_id": char_id, "session_id": session_id},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["hot"] == 3
                    assert data["warm"] == 0
                    assert data["cold"] == 0
                    assert data["total"] == 3

            await _cleanup(char_id, session_id)

    async def test_stats_endpoint_missing_char(self):
        """GET /v1/memories/stats without character_id returns 400."""
        with TestClient(proxy_app) as client:
            resp = client.get("/v1/memories/stats")
            assert resp.status_code == 400

    async def test_db_pool_not_configured(self):
        """All endpoints return 503 when _db_pool is None."""
        set_db_pool(None)

        with TestClient(proxy_app) as client:
            # Archive
            resp = client.post(
                "/v1/memories/archive",
                json={"character_id": "test", "session_id": "test"},
            )
            assert resp.status_code == 503

            # Reconstitute
            resp = client.post(
                "/v1/memories/reconstitute",
                json={"archive_id": "test", "character_id": "test"},
            )
            assert resp.status_code == 503

            # Stats
            resp = client.get("/v1/memories/stats")
            assert resp.status_code == 503
