"""Scheduler core acceptance (docs/plans/fifo_queue.md P1) + wiring semantics (P2)."""
import asyncio

import pytest

from proxy.core.llm_scheduler import (
    LLMScheduler, ScheduledLLMClient, QueueFull, QueueWaitTimeout,
)
from tests.fake_llm_backend import FakeLLMBackend


def make(scheduler=None, **kw):
    backend = FakeLLMBackend()
    sched = scheduler or LLMScheduler(max_concurrency=kw.pop("max_concurrency", 1), **kw)
    return backend, sched, ScheduledLLMClient(backend, sched)


async def run_job(client, *, tag, kind="chat", model="m", session=None):
    # Each job is its own client session unless the test says otherwise: since P3, two
    # chat turns on ONE session are a regenerate and the newer supersedes the older.
    session = session if session is not None else f"sess-{tag}"
    out = []
    async for c in client.generate_stream(model=model, tag=tag, job_kind=kind,
                                          session_id=session, coalesce_key=(
                                              f"{kind}:{session}:{tag}" if kind != "chat" else None)):
        out.append(c)
    return out


@pytest.mark.asyncio
async def test_single_slot_high_water_is_one_under_mixed_load():
    backend, sched, client = make()
    tags = [f"j{i}" for i in range(20)]
    kinds = ["chat", "lore", "sleep", "embed"] * 5
    tasks = [asyncio.create_task(run_job(client, tag=t, kind=k, model=f"m{i % 3}", session=t))
             for i, (t, k) in enumerate(zip(tags, kinds))]
    for t in tags:
        backend.gate(t).set()
    await asyncio.gather(*tasks)
    assert backend.high_water == 1
    assert sched.counters["completed"] == 20


@pytest.mark.asyncio
async def test_chat_jumps_queue_ahead_of_background():
    backend, sched, client = make()
    # occupy the slot
    runner = asyncio.create_task(run_job(client, tag="hold", kind="chat"))
    await asyncio.sleep(0)
    # five lore jobs queue up, then one chat job arrives last
    lore = [asyncio.create_task(run_job(client, tag=f"lore{i}", kind="lore", session=f"s{i}"))
            for i in range(5)]
    await asyncio.sleep(0)
    chat = asyncio.create_task(run_job(client, tag="chat_late", kind="chat"))
    await asyncio.sleep(0)
    for t in ("hold", "chat_late", "lore0", "lore1", "lore2", "lore3", "lore4"):
        backend.gate(t).set()
    await asyncio.gather(runner, chat, *lore)
    assert backend.started.index("chat_late") == 1  # right after the holder, before every lore job


@pytest.mark.asyncio
async def test_background_coalescing_keeps_newest_per_key():
    backend, sched, client = make()
    holder = asyncio.create_task(run_job(client, tag="hold"))
    await asyncio.sleep(0)
    first = asyncio.create_task(run_job(client, tag="l_old", kind="lore", session="sX"))
    await asyncio.sleep(0)

    async def same_key_job(tag):
        out = []
        async for c in client.generate_stream(model="m", tag=tag, job_kind="lore",
                                              session_id="sX", coalesce_key="lore:sX:fixed"):
            out.append(c)
        return out

    a = asyncio.create_task(same_key_job("l_a"))
    await asyncio.sleep(0)
    b = asyncio.create_task(same_key_job("l_b"))   # replaces l_a while it waits
    await asyncio.sleep(0)
    for t in ("hold", "l_old", "l_b"):
        backend.gate(t).set()
    await asyncio.gather(holder, first, b)
    with pytest.raises(asyncio.CancelledError):
        await a
    assert sched.counters["coalesced"] == 1
    assert "l_a" not in backend.started


@pytest.mark.asyncio
async def test_interactive_cap_rejects_with_queue_full():
    backend, sched, client = make(max_interactive_waiting=2)
    holder = asyncio.create_task(run_job(client, tag="hold"))
    await asyncio.sleep(0)
    waiting = [asyncio.create_task(run_job(client, tag=f"w{i}")) for i in range(2)]
    await asyncio.sleep(0)
    with pytest.raises(QueueFull):
        await run_job(client, tag="overflow")
    for t in ("hold", "w0", "w1"):
        backend.gate(t).set()
    await asyncio.gather(holder, *waiting)
    assert sched.counters["rejected"] == 1


@pytest.mark.asyncio
async def test_wait_deadline_expires_with_injected_clock():
    now = {"t": 100.0}
    backend, sched, client = make(p0_wait_deadline_s=30.0, time_fn=lambda: now["t"])
    holder = asyncio.create_task(run_job(client, tag="hold"))
    await asyncio.sleep(0)
    late = asyncio.create_task(run_job(client, tag="late"))
    await asyncio.sleep(0)
    now["t"] = 131.0            # beyond the 30 s deadline
    sched.tick()                # deadlines are event-driven; tests drive the clock
    with pytest.raises(QueueWaitTimeout):
        await late
    backend.gate("hold").set()
    await holder
    assert sched.counters["timed_out"] == 1


@pytest.mark.asyncio
async def test_three_slots_reach_exactly_three():
    backend, sched, client = make(max_concurrency=3, max_interactive_waiting=10)
    tags = [f"n{i}" for i in range(9)]
    tasks = [asyncio.create_task(run_job(client, tag=t)) for t in tags]
    await asyncio.sleep(0)
    assert backend.active == 3  # exactly the lane width, no more
    for t in tags:
        backend.gate(t).set()
    await asyncio.gather(*tasks)
    assert backend.high_water == 3


@pytest.mark.asyncio
async def test_closing_the_stream_midway_frees_the_slot():
    backend, sched, client = make()
    gen = client.generate_stream(model="m", tag="g1", job_kind="chat", session_id="s")
    backend.gate("g1").set()
    assert await gen.__anext__() == "a"
    await gen.aclose()           # client disconnect: Starlette closes the generator
    # slot must be free: a second job runs immediately
    backend.gate("g2").set()
    out = await run_job(client, tag="g2")
    assert out == ["a", "b"]
    assert backend.high_water == 1


@pytest.mark.asyncio
async def test_model_swaps_counted():
    backend, sched, client = make()
    for tag, model in (("a", "mA"), ("b", "mB"), ("c", "mB")):
        backend.gate(tag).set()
        await run_job(client, tag=tag, model=model)
    assert sched.counters["model_swaps"] == 1
    assert sched.loaded_model == "mB"


@pytest.mark.asyncio
async def test_snapshot_shape():
    backend, sched, client = make()
    backend.gate("x").set()
    await run_job(client, tag="x")
    snap = sched.snapshot()
    assert snap["completed"] == 1 and snap["depth"]["p0"] == 0
    assert set(snap) >= {"max_concurrency", "running", "loaded_model", "wait_p95_s", "model_swaps"}


# ── P3: preemption and supersede (decision 13) ──────────────────────────────

async def _hold_slot(client, *, kind, session, tag):
    """Hold the slot on a gated stream (the gate never opens); report how it ended."""
    try:
        async for _ in client.generate_stream(model="m", tag=tag, job_kind=kind,
                                              session_id=session,
                                              coalesce_key=(f"{kind}:{tag}" if kind != "chat" else None)):
            pass
        return "finished"
    except asyncio.CancelledError:
        return "cancelled"


async def _slot_taken(backend, tag):
    while tag not in backend.started:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_chat_preempts_running_lore():
    backend, sched, client = make()
    lore = asyncio.create_task(_hold_slot(client, kind="lore", session="s1", tag="lore1"))
    await _slot_taken(backend, "lore1")          # lore holds the single slot
    backend.gate("turn").set()
    out = await run_job(client, tag="turn", kind="chat", session="s2")
    assert out == ["a", "b"]                     # the chat turn ran to completion
    assert await lore == "cancelled"
    assert sched.counters["preempted"] == 1
    assert sched.counters["completed"] == 1      # the evicted job is not 'completed'
    assert not sched._running


@pytest.mark.asyncio
async def test_chat_never_preempts_chat_or_embed():
    for kind in ("chat", "embed"):
        backend, sched, client = make()
        holder = asyncio.create_task(_hold_slot(client, kind=kind, session="sA", tag="holdA"))
        await _slot_taken(backend, "holdA")
        backend.gate("turn").set()
        waiter = asyncio.create_task(run_job(client, tag="turn", kind="chat", session="sB"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert sched.counters["preempted"] == 0, kind
        assert not waiter.done(), kind           # queued behind the holder, not served by eviction
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)
        assert await waiter == ["a", "b"]


@pytest.mark.asyncio
async def test_regenerate_supersedes_running_turn_same_session():
    backend, sched, client = make()
    old = asyncio.create_task(_hold_slot(client, kind="chat", session="sX", tag="v1"))
    await _slot_taken(backend, "v1")
    backend.gate("v2").set()
    out = await run_job(client, tag="v2", kind="chat", session="sX")
    assert out == ["a", "b"]
    assert await old == "cancelled"
    assert sched.counters["superseded"] == 1
    assert sched.counters["preempted"] == 0      # chat-on-chat is supersede, never preempt


@pytest.mark.asyncio
async def test_different_session_waits_instead_of_superseding():
    backend, sched, client = make()
    holder = asyncio.create_task(_hold_slot(client, kind="chat", session="sX", tag="v1"))
    await _slot_taken(backend, "v1")
    backend.gate("turn").set()
    waiter = asyncio.create_task(run_job(client, tag="turn", kind="chat", session="sY"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert sched.counters["superseded"] == 0
    assert not waiter.done()
    holder.cancel()
    await asyncio.gather(holder, return_exceptions=True)
    assert await waiter == ["a", "b"]


@pytest.mark.asyncio
async def test_regenerate_supersedes_queued_turn_same_session():
    backend, sched, client = make()
    holder = asyncio.create_task(_hold_slot(client, kind="chat", session="other", tag="hold"))
    await _slot_taken(backend, "hold")
    queued = asyncio.create_task(_hold_slot(client, kind="chat", session="sX", tag="v1"))
    for _ in range(20):
        await asyncio.sleep(0)                   # v1 is waiting behind the holder
    backend.gate("v2").set()
    regen = asyncio.create_task(run_job(client, tag="v2", kind="chat", session="sX"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert sched.counters["superseded"] == 1
    assert await queued == "cancelled"           # evicted from the queue, never started
    assert "v1" not in backend.started
    holder.cancel()
    await asyncio.gather(holder, return_exceptions=True)
    assert await regen == ["a", "b"]
