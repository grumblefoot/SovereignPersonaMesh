"""
One in-process scheduler for every LLM call SPM makes (OPEN-004; docs/plans/fifo_queue.md).

Replaces the never-wired InferenceFIFOQueue. Phase P1+P2 scope:
- lanes with max_concurrency (1 for local single-slot backends),
- strict priority (P0 chat > P1 embed > P2 lore > P3 sleep/batch), FIFO within a priority,
- coalescing for background jobs (newest pending job per key wins),
- caps and queue-wait deadlines, with typed errors the chat route can surface,
- slot held for the WHOLE stream and always released exactly once,
- a snapshot for telemetry, and sequence ids (SRD 5.1 "sequence ID tracking").

Phase P3 (decision 13 approved both):
- chat PREEMPTS running background work: a P0 arrival with no free slot cancels one
  running P2/P3 job (P3 first). P1 embed is never preempted — an embed may be serving
  the very chat turn that is asking for the slot. Coalescing makes the loss cheap: the
  next turn re-enqueues the same background key.
- regenerate SUPERSEDES the in-flight turn: a new P0 for a session cancels any other
  P0 for that session, queued or running. SillyTavern has already abandoned the old
  stream by the time it re-sends the turn.

Still deferred: idle grace, model-affinity batching, retries and the circuit breaker.
"""

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Priorities (lower number dispatches first)
P0_CHAT = 0
P1_EMBED = 1
P2_LORE = 2
P3_BATCH = 3

KIND_PRIORITY = {"chat": P0_CHAT, "embed": P1_EMBED, "lore": P2_LORE, "sleep": P3_BATCH, "batch": P3_BATCH}

DEFAULT_MAX_INTERACTIVE_WAITING = 4
DEFAULT_BACKGROUND_CAP = 50
DEFAULT_P0_WAIT_DEADLINE_S = 120.0


class QueueFull(RuntimeError):
    """Too many interactive turns are already waiting (HTTP 503 / SSE notice)."""


class QueueWaitTimeout(RuntimeError):
    """An interactive turn waited longer than the deadline for a slot."""


@dataclass
class _Job:
    seq: int
    kind: str
    priority: int
    model: str
    session_id: str
    coalesce_key: Optional[str]
    enqueued_at: float
    deadline_at: Optional[float]
    ready: asyncio.Future = field(repr=False, default=None)
    task: Optional["asyncio.Task"] = field(repr=False, default=None)  # holder, for preempt/supersede
    evicted: bool = False               # preempted/superseded while running


class LLMScheduler:
    """Priority FIFO over one backend lane. All waits are event-driven; the clock is
    injectable (``time_fn``) so tests never sleep."""

    def __init__(self, max_concurrency: int = 1, *,
                 max_interactive_waiting: int = DEFAULT_MAX_INTERACTIVE_WAITING,
                 background_cap: int = DEFAULT_BACKGROUND_CAP,
                 p0_wait_deadline_s: float = DEFAULT_P0_WAIT_DEADLINE_S,
                 time_fn: Callable[[], float] = time.monotonic):
        self.max_concurrency = max_concurrency
        self.max_interactive_waiting = max_interactive_waiting
        self.background_cap = background_cap
        self.p0_wait_deadline_s = p0_wait_deadline_s
        self._now = time_fn
        self._seq = itertools.count(1)
        self._queues: Dict[int, deque] = {p: deque() for p in (P0_CHAT, P1_EMBED, P2_LORE, P3_BATCH)}
        self._running: List[_Job] = []
        self.loaded_model: Optional[str] = None
        self.counters = {"dispatched": 0, "completed": 0, "rejected": 0, "timed_out": 0,
                         "cancelled": 0, "coalesced": 0, "model_swaps": 0,
                         "preempted": 0, "superseded": 0}
        self._wait_times: deque = deque(maxlen=200)

    # ── Introspection ────────────────────────────────────────────────

    def depth(self, priority: Optional[int] = None) -> int:
        if priority is None:
            return sum(len(q) for q in self._queues.values())
        return len(self._queues[priority])

    def snapshot(self) -> Dict:
        waits = sorted(self._wait_times)
        pct = lambda p: round(waits[min(len(waits) - 1, int(len(waits) * p))], 3) if waits else 0.0
        return {
            "max_concurrency": self.max_concurrency,
            "depth": {f"p{p}": len(q) for p, q in self._queues.items()},
            "running": [{"seq": j.seq, "kind": j.kind, "model": j.model,
                         "session_id": j.session_id,
                         "elapsed_s": round(self._now() - j.enqueued_at, 3)}
                        for j in self._running],
            "loaded_model": self.loaded_model,
            "wait_p50_s": pct(0.50),
            "wait_p95_s": pct(0.95),
            **self.counters,
        }

    # ── Core ─────────────────────────────────────────────────────────

    def _background_size(self) -> int:
        return sum(len(self._queues[p]) for p in (P1_EMBED, P2_LORE, P3_BATCH))

    def _expire_overdue(self) -> None:
        now = self._now()
        for q in self._queues.values():
            for job in list(q):
                if job.deadline_at is not None and now >= job.deadline_at and not job.ready.done():
                    q.remove(job)
                    self.counters["timed_out"] += 1
                    job.ready.set_exception(QueueWaitTimeout(
                        f"waited {now - job.enqueued_at:.0f}s for a backend slot"))

    def _dispatch(self) -> None:
        """Hand slots to the highest-priority waiters. Called on every enqueue/release."""
        self._expire_overdue()
        while len(self._running) < self.max_concurrency:
            job = None
            for p in (P0_CHAT, P1_EMBED, P2_LORE, P3_BATCH):
                while self._queues[p]:
                    head = self._queues[p][0]
                    if head.ready.done():          # cancelled/expired while waiting
                        self._queues[p].popleft()
                        continue
                    job = self._queues[p].popleft()
                    break
                if job:
                    break
            if job is None:
                return
            if self.loaded_model and job.model and job.model != self.loaded_model:
                self.counters["model_swaps"] += 1
            if job.model:
                self.loaded_model = job.model
            self._running.append(job)
            self.counters["dispatched"] += 1
            self._wait_times.append(self._now() - job.enqueued_at)
            logger.info(f"[LLMScheduler] dispatch seq={job.seq} kind={job.kind} "
                        f"model={job.model} session={job.session_id} "
                        f"waited={self._now() - job.enqueued_at:.2f}s depth={self.depth()}")
            job.ready.set_result(None)

    def _release(self, job: _Job) -> None:
        if job in self._running:
            self._running.remove(job)
            if not job.evicted:
                self.counters["completed"] += 1
            logger.info(f"[LLMScheduler] release seq={job.seq} kind={job.kind}"
                        + (" (evicted)" if job.evicted else ""))
        self._dispatch()

    def _supersede(self, session_id: str) -> None:
        """P3: a new interactive turn for a session replaces any other P0 for that
        session — queued or running. The frontend only re-sends a turn (regenerate,
        retry after a stall) once it has abandoned the previous stream."""
        me = asyncio.current_task()
        for job in list(self._queues[P0_CHAT]):
            if job.session_id == session_id and not job.ready.done() and job.task is not me:
                self._queues[P0_CHAT].remove(job)
                self.counters["superseded"] += 1
                logger.info(f"[LLMScheduler] supersede queued seq={job.seq} session={session_id}")
                job.ready.cancel()
        for job in list(self._running):
            if (job.priority == P0_CHAT and job.session_id == session_id
                    and job.task is not None and job.task is not me and not job.task.done()):
                self.counters["superseded"] += 1
                job.evicted = True
                logger.info(f"[LLMScheduler] supersede running seq={job.seq} session={session_id}")
                job.task.cancel()  # its __aexit__/finally releases the slot

    def _preempt_background(self) -> None:
        """P3: free a slot for an interactive turn by cancelling one running P2/P3 job
        (P3 first). P1 embed is exempt: it may be serving the chat turn itself."""
        me = asyncio.current_task()
        victims = [j for j in self._running
                   if j.priority >= P2_LORE and j.task is not None
                   and j.task is not me and not j.task.done()]
        if not victims:
            return
        victim = max(victims, key=lambda j: (j.priority, -j.seq))  # P3 first, newest first
        self.counters["preempted"] += 1
        victim.evicted = True
        logger.info(f"[LLMScheduler] preempt seq={victim.seq} kind={victim.kind} for chat")
        victim.task.cancel()  # its __aexit__/finally releases the slot

    def _enqueue(self, kind: str, model: str, session_id: str,
                 coalesce_key: Optional[str]) -> _Job:
        priority = KIND_PRIORITY.get(kind, P3_BATCH)
        loop = asyncio.get_running_loop()
        now = self._now()

        if priority == P0_CHAT:
            if session_id:
                self._supersede(session_id)  # before the cap: a regenerate frees its own seat
            if len(self._queues[P0_CHAT]) >= self.max_interactive_waiting:
                self.counters["rejected"] += 1
                raise QueueFull(f"{len(self._queues[P0_CHAT])} interactive turns already waiting")
            if len(self._running) >= self.max_concurrency:
                self._preempt_background()

        if priority != P0_CHAT:
            if coalesce_key:
                for job in list(self._queues[priority]):
                    if job.coalesce_key == coalesce_key and not job.ready.done():
                        self._queues[priority].remove(job)
                        self.counters["coalesced"] += 1
                        job.ready.cancel()
            if self._background_size() >= self.background_cap:
                # Drop the oldest coalescible background job to make room.
                for p in (P3_BATCH, P2_LORE, P1_EMBED):
                    dropped = next((j for j in self._queues[p] if j.coalesce_key), None)
                    if dropped is not None:
                        self._queues[p].remove(dropped)
                        self.counters["rejected"] += 1
                        dropped.ready.cancel()
                        logger.warning(f"[LLMScheduler] background cap: dropped seq={dropped.seq}")
                        break
                else:
                    self.counters["rejected"] += 1
                    raise QueueFull("background queue full and nothing coalescible to drop")

        job = _Job(
            seq=next(self._seq), kind=kind, priority=priority, model=model or "",
            session_id=session_id, coalesce_key=coalesce_key, enqueued_at=now,
            deadline_at=(now + self.p0_wait_deadline_s) if priority == P0_CHAT else None,
            ready=loop.create_future(),
            task=asyncio.current_task(),
        )
        self._queues[priority].append(job)
        self._dispatch()
        return job

    def slot(self, kind: str, model: str = "", session_id: str = "",
             coalesce_key: Optional[str] = None):
        """``async with scheduler.slot("chat", model, session):`` — the only way in."""
        scheduler = self

        class _Slot:
            async def __aenter__(self):
                self._job = scheduler._enqueue(kind, model, session_id, coalesce_key)
                try:
                    await self._job.ready
                except asyncio.CancelledError:
                    scheduler.counters["cancelled"] += 1
                    raise
                return self._job

            async def __aexit__(self, exc_type, exc, tb):
                scheduler._release(self._job)
                return False

        return _Slot()

    def tick(self) -> None:
        """Re-evaluate deadlines; tests drive this with an injected clock."""
        self._expire_overdue()
        self._dispatch()


class ScheduledLLMClient:
    """Drop-in wrapper over LemonadeLLMClient: same generate_stream signature, but every
    call holds a scheduler slot for the life of the stream. Metadata calls
    (_resolve_model, _chat_model_ids) pass through without a slot."""

    def __init__(self, inner, scheduler: LLMScheduler):
        self.inner = inner
        self.scheduler = scheduler

    # Metadata / plumbing passthroughs (slot-exempt by design)
    @property
    def base_url(self):
        return self.inner.base_url

    @base_url.setter
    def base_url(self, value):
        self.inner.base_url = value

    @property
    def _client(self):  # tests/conftest reset_clients pokes this
        return self.inner._client

    @_client.setter
    def _client(self, value):
        self.inner._client = value

    @property
    def client(self):
        return self.inner.client

    async def close(self):
        await self.inner.close()

    async def _chat_model_ids(self):
        return await self.inner._chat_model_ids()

    async def _resolve_model(self, requested_model: str) -> str:
        return await self.inner._resolve_model(requested_model)

    async def generate_stream(self, *args, job_kind: str = "chat",
                              session_id: str = "", coalesce_key: Optional[str] = None,
                              **kwargs) -> AsyncGenerator[str, None]:
        model = kwargs.get("model") or (args[1] if len(args) > 1 else "")
        async with self.scheduler.slot(job_kind, model=str(model), session_id=session_id,
                                       coalesce_key=coalesce_key):
            inner = self.inner.generate_stream(*args, **kwargs)
            try:
                async for chunk in inner:
                    yield chunk
            finally:
                # On client disconnect, GeneratorExit lands at our yield; `async for` does
                # NOT close the inner generator, so without this the backend keeps decoding
                # to a dead connection while the slot is already free.
                await inner.aclose()


_scheduler: Optional[LLMScheduler] = None
_scheduled_client: Optional[ScheduledLLMClient] = None


def get_scheduler() -> LLMScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = LLMScheduler(max_concurrency=1)
    return _scheduler


def get_scheduled_client() -> ScheduledLLMClient:
    """The one shared, scheduled LLM client (chat, lore, future embed/sleep callers)."""
    global _scheduled_client
    if _scheduled_client is None:
        from proxy.backend_client.lemonade_client import LemonadeLLMClient
        _scheduled_client = ScheduledLLMClient(LemonadeLLMClient(), get_scheduler())
    return _scheduled_client


def reset_scheduler() -> None:
    """Test hook: drop the singletons."""
    global _scheduler, _scheduled_client
    _scheduler = None
    _scheduled_client = None
