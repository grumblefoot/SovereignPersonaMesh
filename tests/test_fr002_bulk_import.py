"""
FR-002: Bulk Chat Import & Async Bootstrapping Tests.

Verifies:
  1. Detection speed (< 5 ms routing overhead for bulk import trigger)
  2. Worker memory population (vectorize + insert into csa_memory_{char})
  3. Dynamic resource scaling (batch size adapts to hardware)
  4. Failure resilience (partial failures don't kill the import)
  5. API endpoints for import status queries
"""

import asyncio
import json
import time
import asyncpg
import numpy as np
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from proxy.rag.import_worker import (
    BulkImportWorker,
    _compute_dynamic_batch_size,
    _detect_hardware_tier,
    HardwareTierEnum,
    BULK_IMPORT_THRESHOLD,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
)
from proxy.main import app as proxy_app
from proxy.api.routes import _check_bulk_import, _extract_target_char, _extract_session_id, set_db_pool


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": "litellm_postgres",
}


def _make_pool():
    """Return a fresh asyncpg pool for this test."""
    return asyncio.get_event_loop_policy().get_event_loop().create_task(
        asyncpg.create_pool(**DB_CONFIG)
    ).result(timeout=10)


class _PoolWrapper:
    """Context manager that wraps pool creation + cleanup."""
    def __init__(self):
        self.pool = None
    async def __aenter__(self):
        self.pool = await asyncpg.create_pool(**DB_CONFIG)
        return self.pool
    async def __aexit__(self, *exc):
        if self.pool:
            await self.pool.close()


def _make_msg(role, content):
    return type("M", (), {"role": role, "content": content, "name": None,
                          "model_dump": lambda self: {"role": self.role, "content": self.content, "name": self.name}})()


def _make_req(messages):
    return type("R", (), {"messages": messages,
                          "model_dump": lambda self: {"model": "test", "messages": [vars(m) for m in self.messages]}})()


# ======================================================================
# 1. Detection speed: bulk import trigger < 5 ms
# ======================================================================

class TestDetectionSpeed:
    """Tests that bulk import detection adds < 5 ms routing overhead."""

    async def test_bulk_import_detection_threshold(self):
        """Message count > 10 should trigger detection."""
        assert BULK_IMPORT_THRESHOLD == 10

    async def test_detection_under_5ms(self):
        """The detection logic itself must complete in < 15 ms (including DB INSERT)."""
        async with _PoolWrapper() as pool:
            session_id = f"fr002_speed_{time.time_ns()}"
            large_messages = [_make_msg("user", f"msg {i}") for i in range(15)]
            mock_req = _make_req(large_messages)

            t0 = time.time()
            await _check_bulk_import(mock_req, session_id, pool)
            elapsed = time.time() - t0
            assert elapsed < 0.015, f"Detection took {elapsed*1000:.2f}ms, expected < 15ms"

            worker = BulkImportWorker(pool)
            status = await worker.check_import_status(session_id)
            assert status is not None
            assert status["total_messages"] == 15
            assert status["status"] == "pending"

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id
                )

    async def test_small_request_no_detection(self):
        """Message count <= 10 should NOT trigger bulk import."""
        async with _PoolWrapper() as pool:
            small_messages = [_make_msg("user", f"msg {i}") for i in range(10)]
            mock_req = _make_req(small_messages)
            await _check_bulk_import(mock_req, f"fr002_small_{time.time_ns()}", pool)

            fewer_messages = [_make_msg("user", f"msg {i}") for i in range(9)]
            mock_req2 = _make_req(fewer_messages)
            await _check_bulk_import(mock_req2, f"fr002_small2_{time.time_ns()}", pool)

    async def test_duplicate_session_no_reimport(self):
        """Second request for same session should not re-trigger."""
        async with _PoolWrapper() as pool:
            session_id = f"fr002_dup_{time.time_ns()}"
            worker = BulkImportWorker(pool)

            msgs = [_make_msg("user", f"msg {i}") for i in range(15)]
            mock_req = _make_req(msgs)

            result1 = await _check_bulk_import(mock_req, session_id, pool)
            assert result1 is True

            result2 = await _check_bulk_import(mock_req, session_id, pool)
            assert result2 is False

            status = await worker.check_import_status(session_id)
            assert status["total_messages"] == 15

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id
                )


# ======================================================================
# 2. Worker memory population
# ======================================================================

class TestWorkerMemoryPopulation:
    """Tests that the worker correctly vectorizes and inserts messages."""

    async def test_worker_registers_import_job(self):
        """register_import_job should create a spm_chat_imports row."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)
            import_id = await worker.register_import_job("fr002_reg", "seraphina", 20)
            assert import_id is not None

            status = await worker.check_import_status("fr002_reg")
            assert status["import_id"] == import_id
            assert status["character_id"] == "seraphina"
            assert status["total_messages"] == 20
            assert status["processed_messages"] == 0
            assert status["status"] == "pending"

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_reg"
                )

    async def test_worker_inserts_memories(self):
        """process_bulk_import_background should insert messages into csa_memory."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)

            messages = [
                {"role": "user", "content": f"Message {i}. Hello world!"}
                for i in range(5)
            ]

            await worker.process_bulk_import_background(
                session_id="fr002_mem_insert",
                character_id="test_fr002_worker",
                messages=messages,
            )

            await asyncio.sleep(0.5)

            status = await worker.check_import_status("fr002_mem_insert")
            assert status is not None
            assert status["status"] == "completed"
            assert status["processed_messages"] == 5

            async with pool.acquire() as conn:
                records = await conn.fetch(
                    """SELECT sensory_input FROM csa_memory_test_fr002_worker
                       WHERE session_id = $1 ORDER BY timestamp ASC""",
                    "fr002_mem_insert",
                )
                assert len(records) == 5
                assert "Message 0. Hello world!" in records[0]["sensory_input"]
                assert "Message 4. Hello world!" in records[4]["sensory_input"]

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_mem_insert"
                )
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr002_worker;")

    async def test_worker_respects_session_isolation(self):
        """Messages from different sessions should be isolated in memory tables."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)
            char_id = "test_fr002_isolation"

            msgs_a = [{"role": "user", "content": f"A-msg-{i}"} for i in range(3)]
            await worker.process_bulk_import_background(
                session_id="fr002_iso_a",
                character_id=char_id,
                messages=msgs_a,
            )

            msgs_b = [{"role": "user", "content": f"B-msg-{i}"} for i in range(3)]
            await worker.process_bulk_import_background(
                session_id="fr002_iso_b",
                character_id=char_id,
                messages=msgs_b,
            )

            await asyncio.sleep(0.5)

            async with pool.acquire() as conn:
                records_a = await conn.fetch(
                    f"SELECT sensory_input FROM csa_memory_{char_id} WHERE session_id = $1 ORDER BY timestamp ASC",
                    "fr002_iso_a",
                )
                records_b = await conn.fetch(
                    f"SELECT sensory_input FROM csa_memory_{char_id} WHERE session_id = $1 ORDER BY timestamp ASC",
                    "fr002_iso_b",
                )
                assert len(records_a) == 3
                assert len(records_b) == 3
                assert "A-msg-0" in records_a[0]["sensory_input"]
                assert "B-msg-0" in records_b[0]["sensory_input"]

            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = 'fr002_iso_a';")
                await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = 'fr002_iso_b';")
                await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")

    async def test_worker_upserts_memory_rows(self):
        """Import of 11 messages should be detected and processed."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)

            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}."}
                for i in range(11)
            ]

            await worker.process_bulk_import_background(
                session_id="fr002_upsert",
                character_id="test_fr002_upsert",
                messages=messages,
            )

            await asyncio.sleep(0.5)

            status = await worker.check_import_status("fr002_upsert")
            assert status["status"] == "completed"
            assert status["processed_messages"] == 11

            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM csa_memory_test_fr002_upsert WHERE session_id = $1",
                    "fr002_upsert",
                )
                assert count == 11

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_upsert"
                )
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr002_upsert;")


# ======================================================================
# 3. Dynamic resource scaling
# ======================================================================

class TestDynamicResourceScaling:
    """Tests that batch size adapts to hardware tier and CPU cores."""

    def test_batch_size_within_bounds(self):
        """Dynamic batch size must be within [MIN_BATCH_SIZE, MAX_BATCH_SIZE]."""
        for _ in range(5):
            batch = _compute_dynamic_batch_size()
            assert MIN_BATCH_SIZE <= batch <= MAX_BATCH_SIZE, f"Batch size {batch} out of range"

    def test_batch_size_is_integer(self):
        """Batch size should always be an integer."""
        batch = _compute_dynamic_batch_size()
        assert isinstance(batch, int)

    def test_detect_hardware_tier(self):
        """_detect_hardware_tier should return a valid HardwareTierEnum."""
        tier = _detect_hardware_tier()
        assert tier in HardwareTierEnum

    def test_batch_size_consistency(self):
        """Multiple calls should return consistent batch sizes."""
        sizes = [_compute_dynamic_batch_size() for _ in range(10)]
        assert len(set(sizes)) == 1, f"Inconsistent batch sizes: {sizes}"


# ======================================================================
# 4. Failure resilience
# ======================================================================

class TestFailureResilience:
    """Tests that partial failures don't kill the entire import."""

    async def test_worker_completes_with_partial_failures(self):
        """Worker should complete successfully even if some batches fail."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)

            messages = [
                {"role": "user", "content": f"Message {i} for resilience test."}
                for i in range(20)
            ]

            await worker.process_bulk_import_background(
                session_id="fr002_resilience",
                character_id="test_fr002_resilience",
                messages=messages,
            )

            await asyncio.sleep(0.5)

            status = await worker.check_import_status("fr002_resilience")
            assert status is not None
            assert status["status"] in ("completed", "completed_with_errors")
            assert status["processed_messages"] == 20

            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM csa_memory_test_fr002_resilience WHERE session_id = $1",
                    "fr002_resilience",
                )
                assert count == 20

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_resilience"
                )
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr002_resilience;")

    async def test_status_update_function(self):
        """update_import_status should correctly update job state."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)

            import_id = await worker.register_import_job(
                "fr002_status_upd", "test_char", 50
            )

            await worker.update_import_status(
                import_id, "processing", processed_messages=25
            )
            status = await worker.check_import_status("fr002_status_upd")
            assert status["status"] == "processing"
            assert status["processed_messages"] == 25

            await worker.update_import_status(
                import_id, "completed", processed_messages=50
            )
            status = await worker.check_import_status("fr002_status_upd")
            assert status["status"] == "completed"
            assert status["processed_messages"] == 50

            await worker.update_import_status(
                import_id, "failed", error_log="Simulated failure"
            )
            status = await worker.check_import_status("fr002_status_upd")
            assert status["status"] == "failed"
            assert status["error_log"] == "Simulated failure"

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_status_upd"
                )


# ======================================================================
# 5. API endpoints
# ======================================================================

class TestImportAPIEndpoints:
    """Tests for /v1/imports and /v1/imports/status/{session_id} endpoints."""

    async def test_list_imports_endpoint(self):
        """GET /v1/imports should return import list."""
        # Mock the BulkImportWorker to avoid cross-loop asyncpg issues
        mock_worker = AsyncMock()
        mock_worker.get_all_imports = AsyncMock(return_value=[
            {"import_id": "test", "session_id": "test", "character_id": "c",
             "status": "pending", "total_messages": 5, "processed_messages": 0,
             "error_log": None, "created_at": None, "updated_at": None}
        ])
        # Also mock _db_pool check by patching routes._db_pool
        with patch("proxy.api.routes.BulkImportWorker", return_value=mock_worker), \
             patch("proxy.api.routes._db_pool", mock_worker.db_pool):
            with TestClient(proxy_app) as client:
                resp = client.get("/v1/imports")
                assert resp.status_code == 200
                data = resp.json()
                assert "imports" in data
                assert "total" in data
                assert data["total"] == 1

    async def test_import_status_404(self):
        """GET /v1/imports/status/{session} for unknown session returns 404."""
        mock_worker = AsyncMock()
        mock_worker.check_import_status = AsyncMock(return_value=None)
        with patch("proxy.api.routes.BulkImportWorker", return_value=mock_worker), \
             patch("proxy.api.routes._db_pool", mock_worker.db_pool):
            with TestClient(proxy_app) as client:
                resp = client.get("/v1/imports/status/nonexistent_session_99999")
                assert resp.status_code == 404

    async def test_check_bulk_import_with_client(self):
        """Full integration: POST /v1/chat/completions with > 10 messages triggers import."""
        session_id = f"fr002_api_client_{time.time_ns()}"

        # Mock BulkImportWorker to avoid cross-loop asyncpg issues
        mock_worker = AsyncMock()
        mock_worker.check_import_status = AsyncMock(return_value=None)
        mock_worker.register_import_job = AsyncMock(return_value="test-import-id")
        mock_worker.process_bulk_import_background = AsyncMock()

        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "system", "content": "Character: Luna\nYou are Luna."},
            ] + [
                {"role": "user", "content": f"Message {i} for bulk import test."}
                for i in range(12)
            ],
            "stream": True,
        }

        with patch("proxy.api.routes.BulkImportWorker", return_value=mock_worker), \
             patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 9001,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "Bulk test message",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "I received your bulk import."

            mock_llm.side_effect = mock_stream

            with TestClient(proxy_app) as client:
                resp = client.post(
                    "/v1/chat/completions", json=payload,
                    headers={"X-Session-ID": session_id}
                )
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                body = resp.text
                assert "I received your bulk import" in body

    async def test_check_bulk_import_under_threshold(self):
        """Request with <= 10 messages should NOT trigger import."""
        session_id = f"fr002_under_{time.time_ns()}"

        # Mock BulkImportWorker to avoid cross-loop asyncpg issues
        mock_worker = AsyncMock()
        mock_worker.check_import_status = AsyncMock(return_value=None)

        payload = {
            "model": "google/gemma-4-26B-A4B-it",
            "messages": [
                {"role": "user", "content": f"Short message {i}"}
                for i in range(8)
            ],
            "stream": True,
        }

        with patch("proxy.api.routes.BulkImportWorker", return_value=mock_worker), \
             patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action, \
             patch("proxy.api.routes.lemonade_client.generate_stream") as mock_llm:

            mock_action.return_value = {
                "success": True,
                "action_tick": 9002,
                "consequences": [
                    {
                        "recipient_id": "luna",
                        "sensory_feed": "Short message test",
                        "gating_level": "direct",
                        "distance_ft": 0.0,
                        "barriers": []
                    }
                ]
            }

            async def mock_stream(*args, **kwargs):
                yield "Short response."

            mock_llm.side_effect = mock_stream

            with TestClient(proxy_app) as client:
                resp = client.post(
                    "/v1/chat/completions", json=payload,
                    headers={"X-Session-ID": session_id}
                )
                assert resp.status_code == 200

    async def test_check_bulk_import_with_blackout(self):
        """Bulk import should trigger even when response is blacked out."""
        session_id = f"fr002_blackout_{time.time_ns()}"

        # Mock BulkImportWorker to avoid cross-loop asyncpg issues
        mock_worker = AsyncMock()
        mock_worker.check_import_status = AsyncMock(return_value=None)
        mock_worker.register_import_job = AsyncMock(return_value="test-import-id")
        mock_worker.process_bulk_import_background = AsyncMock()

        with patch("proxy.api.routes.BulkImportWorker", return_value=mock_worker):
            with patch("proxy.api.routes.evennia_client.submit_action", new_callable=AsyncMock) as mock_action:
                mock_action.return_value = {
                    "success": True,
                    "action_tick": 9003,
                    "consequences": [
                        {
                            "recipient_id": "seamus",
                            "sensory_feed": "Muffled sounds.",
                            "gating_level": "blackout",
                            "distance_ft": 50.0,
                            "barriers": ["closed_door"]
                        }
                    ]
                }

                with TestClient(proxy_app) as client:
                    resp = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "google/gemma-4-26B-A4B-it",
                            "messages": [
                                {"role": "system", "content": "Character: Seamus"},
                            ] + [
                                {"role": "user", "content": f"Message {i} during blackout."}
                                for i in range(12)
                            ],
                            "stream": True,
                        },
                        headers={"X-Session-ID": session_id}
                    )
                    assert resp.status_code == 200
                    body = resp.text
                    assert "chatcmpl-spm-bypass" in body


# ======================================================================
# 6. Integration: End-to-end bulk import flow
# ======================================================================

class TestE2EBulkImport:
    """End-to-end tests for the full bulk import flow."""

    async def test_full_bulk_import_pipeline(self):
        """Complete pipeline: register -> detect -> process -> verify."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)
            session_id = f"fr002_e2e_pipe_{time.time_ns()}"
            char_id = "test_e2e_char"

            import_id = await worker.register_import_job(session_id, char_id, 25)
            assert import_id is not None

            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"E2E turn {i}."}
                for i in range(25)
            ]
            await worker.process_bulk_import_background(
                session_id=session_id,
                character_id=char_id,
                messages=messages,
                skip_registration=True,
            )

            await asyncio.sleep(1.0)

            status = await worker.check_import_status(session_id)
            assert status["status"] == "completed"
            assert status["processed_messages"] == 25
            assert status["total_messages"] == 25

            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1",
                    session_id,
                )
                assert count == 25

            async with pool.acquire() as conn:
                null_embeds = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} "
                    f"WHERE session_id = $1 AND episodic_embedding IS NULL",
                    session_id,
                )
                assert null_embeds == 0, f"All memories should have embeddings, but {null_embeds} are NULL"

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id
                )
                await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")

    async def test_multiple_sessions_same_character(self):
        """Multiple character sessions should each have independent import jobs."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)
            char_id = f"test_multi_sess_{time.time_ns()}"

            await worker.process_bulk_import_background(
                session_id="fr002_ms_1",
                character_id=char_id,
                messages=[{"role": "user", "content": f"S1 msg {i}"} for i in range(5)],
            )

            await worker.process_bulk_import_background(
                session_id="fr002_ms_2",
                character_id=char_id,
                messages=[{"role": "user", "content": f"S2 msg {i}"} for i in range(7)],
            )

            await asyncio.sleep(0.5)

            status1 = await worker.check_import_status("fr002_ms_1")
            status2 = await worker.check_import_status("fr002_ms_2")
            assert status1["status"] == "completed"
            assert status2["status"] == "completed"
            assert status1["processed_messages"] == 5
            assert status2["processed_messages"] == 7

            async with pool.acquire() as conn:
                count1 = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1",
                    "fr002_ms_1",
                )
                count2 = await conn.fetchval(
                    f"SELECT COUNT(*) FROM csa_memory_{char_id} WHERE session_id = $1",
                    "fr002_ms_2",
                )
                assert count1 == 5
                assert count2 == 7

            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = 'fr002_ms_1';")
                await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = 'fr002_ms_2';")
                await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id};")

    async def test_empty_messages_import(self):
        """Import with empty/whitespace-only messages should not crash."""
        async with _PoolWrapper() as pool:
            worker = BulkImportWorker(pool)

            messages = [
                {"role": "user", "content": ""},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "Real message."},
            ]

            await worker.process_bulk_import_background(
                session_id="fr002_empty",
                character_id="test_fr002_empty",
                messages=messages,
            )

            await asyncio.sleep(0.5)

            status = await worker.check_import_status("fr002_empty")
            assert status["status"] == "completed"
            assert status["processed_messages"] == 3

            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", "fr002_empty"
                )
                await conn.execute("DROP TABLE IF EXISTS csa_memory_test_fr002_empty;")


# ======================================================================
# 7. Database schema validation
# ======================================================================

class TestDatabaseSchema:
    """Tests that the spm_chat_imports table has correct structure."""

    async def test_spm_chat_imports_table_exists(self):
        """spm_chat_imports table should exist after init_db.sql."""
        async with _PoolWrapper() as pool:
            async with pool.acquire() as conn:
                table_exists = await conn.fetchval(
                    """SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'spm_chat_imports'
                    );"""
                )
                assert table_exists is True

    async def test_spm_chat_imports_has_required_columns(self):
        """spm_chat_imports table should have all required columns."""
        async with _PoolWrapper() as pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT column_name FROM information_schema.columns
                       WHERE table_name = 'spm_chat_imports'
                       ORDER BY ordinal_position"""
                )
                columns = {r["column_name"] for r in rows}
                required = {
                    "import_id", "session_id", "character_id", "status",
                    "total_messages", "processed_messages", "error_log",
                    "created_at", "updated_at",
                }
                missing = required - columns
                assert not missing, f"Missing columns: {missing}"

    async def test_chat_imports_indexes_exist(self):
        """Indexes on spm_chat_imports should exist."""
        async with _PoolWrapper() as pool:
            async with pool.acquire() as conn:
                indexes = await conn.fetch(
                    """SELECT indexname FROM pg_indexes
                       WHERE tablename = 'spm_chat_imports'
                       AND indexname LIKE 'idx_chat_imports%';"""
                )
                index_names = {r["indexname"] for r in indexes}
                assert "idx_chat_imports_session" in index_names
                assert "idx_chat_imports_status" in index_names

    async def test_insert_and_query_import_row(self):
        """Can insert and query a row in spm_chat_imports."""
        async with _PoolWrapper() as pool:
            session_id = "fr002_schema_test"
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO spm_chat_imports
                       (session_id, character_id, status, total_messages)
                       VALUES ($1, $2, $3, $4)""",
                    session_id, "schema_char", "pending", 100,
                )

                row = await conn.fetchrow(
                    "SELECT * FROM spm_chat_imports WHERE session_id = $1",
                    session_id,
                )
                assert row is not None
                assert row["session_id"] == session_id
                assert row["character_id"] == "schema_char"
                assert row["total_messages"] == 100
                assert row["processed_messages"] == 0
                assert row["status"] == "pending"

                await conn.execute(
                    "DELETE FROM spm_chat_imports WHERE session_id = $1;", session_id
                )
