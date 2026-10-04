# Plan: Wire the LLM queue (OPEN-004)

Status: planning only, 2026-10-03. Nothing here is implemented yet.

## Goal

Every LLM call SPM makes to the backend goes through one in-process scheduler. That covers chat, lore extraction, the sleep cycle, and future backend embeddings. The scheduler must:

- Never run more generations at once than the selected backend can serve: 1 for Lemonade, KoboldCpp or single-slot llama.cpp; N for OpenAI or multi-slot servers. This meets PRD FR-1.4 and PRD 6.1 ("Concurrent GPU execution is strictly forbidden"), PRD 4.3 step 1 and SRD 5.1 ("Asyncio-based FIFO queue for concurrency control and sequence ID tracking").
- Put interactive chat first. Background work waits, is merged where possible, and avoids extra model swaps.
- Keep streaming working, cancel cleanly, and show its state on the admin dashboard.

## Current state (evidence)

| Fact | Evidence |
|---|---|
| The queue holds a lock but is never called. `_queue` is never filled, so `size()` is always 0. | `proxy/core/fifo_queue.py:15-27`; created at `proxy/api/routes.py:53` (`fifo_queue = InferenceFIFOQueue()`). There are no other references. |
| Streaming chat calls the backend directly and has no slot. | `routes.py:478-500` (`sse_event_generator`, `lemonade_client.generate_stream`) |
| Non-streaming chat calls it directly as well. | `routes.py:410-423` |
| Lore extraction starts as soon as the stream finishes, from a fire-and-forget task, with its own client instance. | `routes.py:533` and `710-742` (`asyncio.create_task`); `proxy/rag/lore_extractor.py:25` (`self.llm_client = LemonadeLLMClient()`) and `:54-56` (`enable_thinking: False`) |
| Lore extraction currently runs on **every** turn. | `config/config.json` `"periodic_review_cadence": 1`; cadence check at `routes.py:735` |
| The extraction model can differ from the chat model, which forces a swap on Lemonade. | `routes.py:715-717` (`use_alternate_extraction_model`, `alternate_extraction_model_name` = `Gemma-4-E4B-it-GGUF`) |
| The sleep cycle runs as a separate process with its own httpx client and model (E4B). | `scripts/sleep_cycle.py:28-47`; systemd timer at 03:00 (README:12, playbook:7) |
| The import worker makes no LLM calls today; it only uses the CPU ONNX embedder. | `proxy/rag/import_worker.py:226` |
| Model listing is metadata only and cached for 30 s. | `proxy/backend_client/lemonade_client.py:27,51-68` |
| Backend failures are typed, and the chat route already turns them into a user-visible message without saving the turn. | `lemonade_client.py:30,187-223`; `routes.py:420-423,500-508` |
| Telemetry has no queue fields, even though the `/stats` docstring mentions "queue depth". | `proxy/core/telemetry.py:240-266`; `proxy/api/admin_routes.py:5,83-102` |
| The runtime is Starlette 1.6 / FastAPI 0.141 / httpx 0.28. Starlette cancels a `StreamingResponse` generator when the client disconnects. | `.venv` site-packages |
| On Lemonade only one LLM is loaded at a time, and a request for another model evicts it (about 6-15 s). | `lemonade_playbook.md:39-51,279-287` |

## Design

### 1. One scheduler, wrapping the client rather than the routes

- **New `proxy/core/llm_scheduler.py`.** It replaces `InferenceFIFOQueue`; delete `fifo_queue.py` and the playbook line that calls it wired.
- **`ScheduledLLMClient`.** It has the same `generate_stream(...)` signature as `LemonadeLLMClient`. It acquires a slot, then yields from the inner client, and releases the slot in `finally`. Routes and `LoreExtractionWorker` receive the one shared instance:
  - `lore_extractor.py:25` stops building its own client.
  - Callers cannot bypass the scheduler by accident.
- **Job metadata:** `kind` (chat / lore / sleep / embed), `priority`, `model`, `session_id`, `coalesce_key`, `enqueued_at`, `deadline`, and a monotonically increasing `seq`. `seq` is SRD 5.1's "sequence ID tracking".
- **Lanes.** A lane is one backend resource with its own `max_concurrency`:
  - `llm`: one lane per configured backend.
  - `embed`: a separate lane. Lemonade loads embedding models in their own class, while KoboldCpp shares the GPU with the LLM. This is configured per backend profile.
- **Exempt from slots:** `/models` and `_resolve_model`. They are cheap metadata calls, are already cached, and never load a model.

### 2. Priorities and policy (non-preemptive by default)

| Prio | Kind | Policy |
|---|---|---|
| P0 | Interactive chat (stream and non-stream) | Strict FIFO among P0, so a group chat's turns run in order. Always dispatched before any waiting background job. |
| P1 | Embeddings on the chat path (sibling) | Uses the embed lane. Only queues behind P0 when the lane shares the GPU. |
| P2 | Lore extraction | Coalesced per `(session_id, character)`: a newer pending job replaces an older one that hasn't started. Waits for an **idle grace** period (default 3 s with no P0 running or queued) before starting. |
| P3 | Sleep cycle, future import or LLM batch work | Same rules as P2, with a longer grace (default 30 s), and yields between batches. |

**Model-swap awareness.** The scheduler remembers `loaded_model`, the model of the last dispatched job. When it picks a background job:

1. It prefers jobs whose model matches `loaded_model`.
2. It runs a different-model job only once the idle grace has passed. It then drains **all** queued jobs for that model before switching back.
3. It counts swaps in telemetry.

This stops chat (26B) and extraction (E4B) from swapping back and forth. With today's default (`use_alternate_extraction_model: false`) there is no swap; the rule matters once the alternate model is on, and for the sleep cycle.

**Optional background preemption** (an owner decision). If a P0 job arrives while a P2 or P3 job is running on a single-slot lane, the scheduler cancels the background job and requeues it once. A lore call takes about 7 s, so without preemption a user can wait up to roughly 7 s plus a swap. Lore extraction can be redone safely: its results are written only after the call completes.

### 3. Streaming semantics and cancellation

- **Slot timing.** The slot is acquired before the backend request and held until the generator finishes, raises, or is closed. Release is always in `finally`, and is guarded so a slot is never released twice.
- **Disconnect.** When SillyTavern disconnects (the user presses Stop, or closes the tab), Starlette cancels `sse_event_generator`. `CancelledError` then propagates into `generate_stream`, the httpx stream context closes, and the slot is freed. Phase 2 must verify that llama-server/Lemonade actually stops decoding when the connection closes, rather than finishing the generation unseen. If it doesn't, the slot is effectively still busy, and the scheduler should add a short "cool-down" before the next dispatch.
- **Regenerate.** SillyTavern normally aborts the old request, then sends a new one for the same session. Option to add: a new P0 job with the same `session_id` **supersedes** the waiting or running P0 job for that session, which is cancelled. This is an owner decision.
- **Time spent queued while streaming.** Send SSE comment keep-alives (`: queued position=2\n\n`) every 10 s so proxies and SillyTavern don't time out. PRD 4.3.1 status text ("*Luna is planning…*") would appear inside the chat text, so it should stay off by default (an owner decision).
- **Non-streaming path.** Acquire around the full consumption loop at `routes.py:416-419`.
- **Turn finalisation.** Saving the turn, GM actions and lore dispatch run **after** the slot is released, so the DB and Evennia work never holds the GPU.

### 4. Limits, timeouts, failure

- **Queue caps.**
  - P0: `max_interactive_waiting` (default 4). When exceeded, return HTTP 503 with `Retry-After`, or an SSE error chunk in the existing outage style.
  - Background: cap of 50; when full, drop the oldest coalescible job and log it.
- **Timeouts.**
  - P0 queue-wait deadline: default 120 s. Then reply with "*[SPM: backend busy…]*" and don't save the turn.
  - Background run timeout: default 120 s total, separate from httpx's per-read 120 s.
  - Waiting background jobs expire after 1 h (sleep-cycle work: 6 h).
- **Failure.** `LLMBackendError` frees the slot and is passed to the caller unchanged, so the existing chat outage handling still applies.
  - Background jobs retry twice with backoff, then are dropped and logged.
  - **Circuit breaker:** after 3 consecutive backend errors the lane is marked `down` for 30 s. P0 fails fast with the outage message, and background work pauses.
- **OpenAI profile.**
  - `max_concurrency` N.
  - An optional RPM token bucket.
  - On 429, honour `Retry-After` by pausing the lane rather than failing the job.
  - A `background_llm_enabled` switch, because each lore call costs money on a paid backend.

### 5. Cross-process: the sleep cycle

| Option | Pros | Cons |
|---|---|---|
| **A. Run the sleep cycle inside the proxy (recommended).** The proxy has an internal 03:00 trigger, or the systemd timer just calls a new `POST /admin/maintenance/sleep-cycle` endpoint. The worker submits each consolidation call as a P3 job. | One scheduler, one source of truth; swap batching and telemetry apply to it as well | Needs the proxy running at 03:00; the timer unit changes |
| B. Postgres advisory lock (`pg_advisory_lock(<SPM_LLM_KEY>)`). The sleep cycle holds it for each call; the proxy's scheduler holds it while any job runs, on a dedicated connection. | Keeps the process separate | Chat would then depend on the DB being up, and the sleep cycle waits without priority. A crashed holder is released only when its connection dies. |
| C. Separate process with try-lock. The sleep cycle runs only if `pg_try_advisory_lock` succeeds and the proxy is not "busy" (a flag row). | Small change | Racy, and has no priority |

Recommendation: A. Keep `python -m scripts.sleep_cycle` as a standalone fallback for when the proxy is down. It should take a session-level advisory lock so the two can never run together.

### 6. Telemetry

`scheduler.snapshot()` will be merged into `get_stats()` as `llm_queue`. It reports:

- depth per priority;
- running jobs (kind, model, session, elapsed);
- wait-time p50/p95 per kind over the last 200 jobs;
- `loaded_model` and swap count;
- rejected, timed-out, cancelled, superseded and preempted counts;
- circuit state.

Add a dashboard tile in `admin_routes.py`, and a `GET /admin/queue` endpoint for the detailed list. Log one line per job at dispatch and completion, including `seq`.

## Phases

### P1: Scheduler core (about 1 day)

`llm_scheduler.py` with lanes, priorities, FIFO-within-priority, slot context manager, coalescing, caps, deadlines and snapshot. No wiring yet.

Tests use `tests/fake_llm_backend.py`, a fake with the `generate_stream` interface. Each call blocks on an `asyncio.Event` per token or step that the test controls. It records the concurrency high-water mark and the sequence of models dispatched. There are no sleeps or wall-clock timing; the clock is injected.

Acceptance:

- With `max_concurrency=1`, the high-water mark is 1 under 20 mixed jobs.
- A P0 job queued behind 5 P2 jobs runs next.
- Coalescing keeps only the newest job per key.
- Caps reject the overflow and deadlines expire.
- `N=3` reaches exactly 3.

### P2: Wire chat and lore extraction (about 1-1.5 days)

- Shared `ScheduledLLMClient` in `routes.py` and `lore_extractor.py`.
- Hold the slot across the stream; post-turn work runs after release.
- Disconnect and cancel handling, plus keep-alives.
- Delete `fifo_queue.py`; update the playbook and known_issues.

Tests:

- Route-level, via an `httpx.ASGITransport` client against the fake backend.
- `aclose()` on the SSE generator mid-stream frees the slot.
- A lore task created after a turn starts only after that turn's slot is released, and never overlaps the next chat turn.
- The existing `test_backend_outage.py` still passes.

Acceptance:

- Full suite green on `spm_test`, with no live services.
- In a manual check on Lemonade, the owner sees chat and lore requests serialised in the proxy log.

### P3: Background policy (about 1 day)

Idle grace, model-affinity batching, retries and backoff, the circuit breaker, and optional preemption and supersede, each behind a setting.

Tests:

- Alternating chat (model A) and lore (model B) produces one A→B→A swap per idle window, not one per turn.
- Preemption requeues once.
- The circuit opens after 3 failures and closes after the cool-down (injected clock).

### P4: Telemetry and admin (about 0.5 day)

`llm_queue` in `/stats`, `GET /admin/queue`, and the dashboard tile.

Test: the snapshot shape, and that the counts match the jobs the fake backend ran.

### P5: Sleep cycle coordination (about 0.5-1 day, after the owner picks an option)

- With option A: the admin endpoint, the worker submitting P3 jobs, the updated timer unit, and an advisory lock for the standalone run.
- Tests:
  - Run the sleep-cycle worker against the fake backend inside the scheduler and check it never overlaps P0.
  - Check the advisory lock on `spm_test`.

### P6: Backend profiles (about 0.5-1 day; can wait until a second backend is in use)

`backend_profile` settings: `lemonade` (1, swap-aware), `koboldcpp` (1, embed lane shared), `openai` (N, RPM, 429 pause, background switch).

Tests: a fake backend returning 429 with `Retry-After`.

**Total: about 4.5-6 developer-days.** P1+P2 alone close the P1 risk in OPEN-004.

## Risks and open questions for the owner

1. **Preempt background work for chat?** Recommended: yes on single-slot lanes.
2. **Should regenerate supersede the in-flight turn for the same session?** Recommended: yes.
3. **Sleep cycle:** option A (in-proxy, timer calls the endpoint) or B/C? Recommended: A.
4. **Queue-wait UX:** invisible keep-alives only (recommended), or PRD 4.3.1 status text that appears in the chat?
5. **Defaults:** P0 cap 4, wait deadline 120 s, lore idle grace 3 s. Are these acceptable?
6. **Lore cadence:** `periodic_review_cadence: 1` doubles the backend calls per turn. Consider 3, which is the code's default.
7. **Not yet verified:** whether Lemonade/llama-server stops decoding when the client disconnects, and what Lemonade does with a different-model request arriving mid-stream (it may wait or fail). P2 checks this manually on the dev box; no automated test touches it.
8. **OpenAI cost:** should background LLM work be on by default on paid backends?

## Dependencies on sibling workstreams

- **Embeddings:** any backend embedding call must go through `ScheduledLLMClient` on the `embed` lane, with priority P1 on the chat path and P3 for bulk import or backfill. The embeddings plan decides per backend profile whether the lane is shared with the LLM.
- **Gating:** fewer active turns means less P0 load, and bypassed characters never enter the queue (PRD 4.1.1's O(N)→O(K)). No interface change.
- **GM_ACTION toggle and session-scoped lore:** lore jobs keep the coalesce key `(session_id, character)`. If extraction gets a toggle, the dispatch site just doesn't submit. GM actions don't use the LLM.
- **Token budget:** the scheduler can use the budgeted `max_tokens` to estimate job cost and set run timeouts, and for OpenAI TPM limits. The two plans share no code.
- **Evennia vs FastAPI:** unaffected. Evennia calls stay outside the LLM slot, and finalisation runs after release.
