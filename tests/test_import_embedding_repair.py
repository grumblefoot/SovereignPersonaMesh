"""QA 2026-10-17 T1: imported memories stored without vectors.

Bulk embedding calls get their own timeout, import batches run as low-priority
scheduler jobs, and rows still without a vector are embedded again before the
import reports its status. Real test database; no network (MockTransport / fakes).
"""

import asyncpg
import httpx
import pytest

from proxy.embeddings.service import EmbeddingService, OpenAICompatProvider
from proxy.rag import import_worker as iw
from tests._testdb import TEST_DB_CONFIG


async def test_bulk_timeout_is_passed_per_call_and_live_calls_keep_the_default():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout", {}).get("read"))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

    provider = OpenAICompatProvider("http://embed.test/v1", "m", timeout_s=3.0,
                                    transport=httpx.MockTransport(handler))
    service = EmbeddingService(provider, "openai_compat", "m", 3)
    await service.batch_generate_embeddings(["live query"])
    await service.batch_generate_embeddings(["bulk import text"], timeout_s=60.0)
    assert seen == [3.0, 60.0]


class _FlakyEmbedder:
    """First batch fails (timeout under GPU load); later calls succeed."""
    available = True

    def __init__(self, fail_calls=1):
        self.calls = 0
        self.fail_calls = fail_calls
        self.timeouts = []

    async def batch_generate_embeddings(self, texts, timeout_s=None):
        self.calls += 1
        self.timeouts.append(timeout_s)
        if self.calls <= self.fail_calls:
            return [None] * len(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def ensure_space(self, conn):
        return None


async def _run_import(embedder, session_id, char_id, n=12):
    pool = await asyncpg.create_pool(**TEST_DB_CONFIG)
    try:
        worker = iw.BulkImportWorker(pool)
        worker.embedder = embedder
        msgs = [{"role": "user", "content": f"Imported message number {i} about the plot."}
                for i in range(n)]
        await worker.process_bulk_import_background(session_id=session_id, character_id=char_id,
                                                    messages=msgs)
        status = await worker.check_import_status(session_id)
        async with pool.acquire() as conn:
            total, missing = await conn.fetchrow(
                f"SELECT count(*), count(*) FILTER (WHERE episodic_embedding IS NULL) "
                f"FROM csa_memory_{char_id} WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM spm_chat_imports WHERE session_id = $1", session_id)
            await conn.execute(f"DROP TABLE IF EXISTS csa_memory_{char_id}")
        return status, int(total), int(missing)
    finally:
        await pool.close()


async def test_rows_without_vectors_are_embedded_again_before_the_import_completes():
    emb = _FlakyEmbedder(fail_calls=1)
    status, total, missing = await _run_import(emb, "t1_repair_ok", "t1_repair_ok")
    assert total == 12 and missing == 0
    assert status["status"] == "completed"
    assert all(t == iw.IMPORT_EMBED_TIMEOUT_S for t in emb.timeouts)   # bulk timeout, every call


async def test_rows_still_missing_are_reported_not_silently_completed():
    emb = _FlakyEmbedder(fail_calls=99)
    status, total, missing = await _run_import(emb, "t1_repair_fail", "t1_repair_fail")
    assert total == 12 and missing == 12
    assert status["status"] == "completed_with_errors"
    assert "without embeddings" in (status.get("error_log") or "") and "reembed" in status["error_log"]


class _FakeJob:
    def __init__(self):
        self.evicted = False


class _EvictingScheduler:
    """Stand-in scheduler: hands out a job per slot and remembers the latest one, so the
    embedder can mark it evicted the way _preempt_background does before cancelling."""

    def __init__(self):
        self.slots = 0
        self.last_job = None

    def slot(self, kind, model="", session_id="", coalesce_key=None):
        sched = self

        class _Slot:
            async def __aenter__(self):
                sched.slots += 1
                sched.last_job = _FakeJob()
                return sched.last_job

            async def __aexit__(self, *exc):
                return False
        return _Slot()


class _PreemptedEmbedder:
    """Raises CancelledError on the first call, as a cancelled slot holder would."""
    available = True

    def __init__(self, scheduler, mark_evicted=True):
        self.calls = 0
        self.sched = scheduler
        self.mark_evicted = mark_evicted

    async def batch_generate_embeddings(self, texts, timeout_s=None):
        import asyncio
        self.calls += 1
        if self.calls == 1:
            if self.mark_evicted:
                self.sched.last_job.evicted = True   # preempted for a chat turn
            raise asyncio.CancelledError()
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_a_preempted_batch_is_retried_but_a_real_cancel_propagates(monkeypatch):
    import asyncio
    from proxy.core import llm_scheduler
    monkeypatch.setattr(iw, "IMPORT_RETRY_WAIT_S", 0.01)
    sched = _EvictingScheduler()
    monkeypatch.setattr(llm_scheduler, "get_scheduler", lambda: sched)
    worker = iw.BulkImportWorker(db_pool=None)

    emb = _PreemptedEmbedder(sched)
    worker.embedder = emb
    out = await worker._embed_batch(["a", "b"], "s1")
    assert out == [[0.1, 0.2, 0.3]] * 2 and emb.calls == 2 and sched.slots == 2

    # Not evicted: a real cancellation (shutdown) is not swallowed.
    worker.embedder = _PreemptedEmbedder(sched, mark_evicted=False)
    with pytest.raises(asyncio.CancelledError):
        await worker._embed_batch(["a"], "s1")
