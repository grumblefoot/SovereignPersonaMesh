# SPM next sprints: combined plan (2026-10-03)

This plan combines six workstream plans written today, checks them against each other, and orders the work. Read the individual plans for designs, evidence (file:line), tests and full decision lists:

| Workstream | Plan | Owner priority | Estimate |
|---|---|---|---|
| Real embeddings (OPEN-002) | [embeddings.md](embeddings.md) | **P0** | ~9 d |
| Real sensory gating (OPEN-003/006) | [gating.md](gating.md) | **P0** | ~18–25 d |
| World engine: Evennia vs FastAPI (OPEN-013) | [world_engine.md](world_engine.md) | P1, foundation for gating | ~12–17 d (overlaps gating) |
| FIFO / LLM scheduler (OPEN-004) | [fifo_queue.md](fifo_queue.md) | P1 | ~4.5–6 d |
| GM_ACTION toggle (OPEN-005) + chat-scoped lore (OPEN-007) | [gm_actions_and_lore_scope.md](gm_actions_and_lore_scope.md) | P1 / P2 | ~3.5 d + ~4.5 d |
| Configurable token budget (OPEN-008) | [token_budget.md](token_budget.md) | P2 (after main features) | ~5–8 d (+2–3 optional summaries) |
| Authentication and binding (OPEN-001) | (none yet) | **Deferred** by owner: dev-only on the homelab network | (none) |

The raw total is about 57–73 dev-days. After removing the gating/engine overlap and the shared foundation work, it comes to roughly **50–60 dev-days**.

---

## 1. Where the plans had to be reconciled

1. **Engine interface (gating vs world_engine).** The world-engine plan puts `perceive()` in the engine. The gating plan keeps the engine as a state store and computes perception in the proxy as a pure function.
   - **Resolution (recommended):** one contract built from the world-engine `Protocol` (`ensure_world`, `advance_tick`, `submit_action`, `get_graph`/snapshot, `get_entity_state`, `apply_mutation`, `reset`, `health`), **without `perceive`**.
   - Perception lives in `core/gating` and works on the graph snapshot. That keeps it deterministic, unit-testable and identical whichever engine adapter is used.
   - It also means one snapshot call per request instead of one call per message.
2. **Session and chat identity (gating vs lore).** Both plans need a stable per-chat key that does not include the target character; today's key breaks group chats.
   - **Resolution:** one shared workstream, done first. Resolution order: `X-SPM-Chat-ID` (from a small SillyTavern extension, using `chat_metadata.integrity`) → `X-Session-ID` → body `session_id` → message-matching fingerprint → legacy `st_user_<char>`.
   - A related current bug: `routes.py:251` passes `request.model_dump()`, which drops body `session_id`/`user`, so that fallback is dead today.
3. **Ticks.** The gating and world-engine plans agree: a per-session tick that counts user messages, does not advance on regenerations or swipes, and is idempotent per turn. It is built once, in the engine phase.
4. **GM_ACTION validation.** The lore/GM plan adds `proxy/core/gm_actions.py` (schema, whitelist, caps, deterministic idempotency key). The engine plan adds whitelisted `apply_mutation(origin="gm")`. Both are kept: the proxy validates, the engine enforces, and gating only ever reads engine state.
5. **Schema migrations.** The embeddings plan (dimension-free vectors, `embedding_space_id`, full-text column) and the lore-scope plan (`session_id`, `scope` on lore tables) both change `csa_lore_rules_*`. They should ship as **one migration**.
6. **LLM traffic.** Every LLM call goes through the scheduler (FIFO plan): chat first, then embeddings on the chat path, then lore, then the sleep cycle. Gating, the blackout skip path and tokenizer calls never go through the queue.
7. **Embedding defaults vs the "one backend" goal.** The embeddings plan uses the backend's own `/v1/embeddings` when it has one, including Lemonade, KoboldCpp with `--embeddingsmodel`, and llama-server with `--embedding`. Otherwise it uses a local CPU model (EmbeddingGemma-300m ONNX), and otherwise full-text search.
   - OpenAI users default to the **local CPU** model because PRD SLA-2 says memories stay local. Cloud embeddings are opt-in only.
   - This keeps lower tiers working with a single chat backend, and the Sovereign box can use the NPU.

---

## 2. Sprint order

### Sprint 0: foundations and quick fixes (~3–4 d)
These are small, high-value fixes that the planning found, plus the shared foundations everything else depends on:
- **Bugs found during planning** (each is small and needs a regression test):
  - `prompt_builder.py:152` drops every SillyTavern system message after the first (world info, author's note, persona).
  - `_extract_session_id(request.model_dump())` loses body `session_id`.
  - Every GM `MOVE` lands in `default_session`: the local `CharacterMovePayload` in `evennia_world/app.py` has no `session_id`.
  - `configure_world` uses the template key as the session id.
  - GM dispatch uses a random `uuid4` idempotency key, so a regenerate re-applies actions.
  - World templates come pre-populated with demo characters (rowan, domino, luna, seamus).
  - The hard-coded "upstairs/tavern" muffled-feed hack (`app.py:189–203`).
  - The prompt builder always uses the SOVEREIGN tier, ignoring `SPM_HARDWARE_TIER` (`routes.py:54`).
  - The request `max_tokens` defaults to 128000.
  - `periodic_review_cadence` is 1, so a lore LLM call follows every turn; the plan suggests 3.
- **Chat identity** (§1.2), including the ST extension and header recipe.
- **Engine interface contract** (§1.1) with `HttpWorldEngine` and `InProcessWorldEngine` adapters, plus the contract test suite.
- **Embeddings phase 0:** stop storing random vectors.
- **Check by hand on the dev box:**
  - Lemonade `max_loaded_models` (config says 2, playbook says 1).
  - Whether the NPU embed model evicts the GPU LLM.
  - Whether Lemonade stops generating when the client disconnects.
  - What SillyTavern actually sends (log one raw request).

### Sprint 1: engine hardening and the embedding layer (~8–10 d, two parallel tracks)
- **Track A, world engine** (world_engine phases A–C together with gating phase 1):
  - per-session ticks;
  - session-world adjacency;
  - 15 ft degraded limit;
  - closed doors and metal partitions black out;
  - reload from Postgres on restart;
  - doors and exits as stateful objects;
  - a clean fallback when the engine is down (OPEN-010).
- **Track B, embeddings phases 1–3:**
  - the provider layer and config;
  - the combined schema migration with the lore columns (§1.5);
  - the resumable re-embed job;
  - the full-text fallback.

  After Track B, recall works on any OpenAI-compatible backend.

### Sprint 2: gating core (~10–12 d) alongside the scheduler (~3 d)
- **Gating phases 2–4:**
  - the deterministic action parser (speak, whisper, shout, move, emote, plus optional `[whisper:X]` / `[move:Y]` tags);
  - world seeding when GM_ACTION is off (tag, then template keywords, then a single room);
  - the perception table;
  - per-character gated history.
- **FIFO phases 1–2:** scheduler core, then wiring chat and lore into it. This alone closes OPEN-004.
- **Exit criterion:** the 14-scenario gating suite passes with 0 leaks of the planted phrases (including the captured Vardus fixture), and gating adds under 30 ms (p95) at 300 messages.

### Sprint 3: complete the main features (~8–10 d)
- **Gating phases 5–6:**
  - the blackout skip path and ambient rows;
  - closing the side channels (lore extraction and bulk import still see the raw transcript);
  - a "what each character perceived" view in the dashboard.
- **GM_ACTION toggle (A1–A5):** `gm_actions_mode` (`off` / `move_only` / `full`) with caps; validation in `core/gm_actions.py`. Document it as spec deviation SD-01.
- **Embeddings phases 4–6:** local ONNX model, threshold calibration, NPU measurement on Sovereign, admin UI.

### Sprint 4: after the main features (~10–14 d)
- **Chat-scoped lore (B1–B5):** schema already migrated in Sprint 1; add retrieval by session or global, promote-to-canon, and session delete.
- **Token budget P0–P3:** the `token_budget` config block, context-window auto-detection, token counting, trim order, per-request `max_tokens`, and a cost dashboard. Summaries (P4) are optional.
- **FIFO phases 3–6:** model-swap batching, sleep cycle inside the proxy, limits and circuit breaker, telemetry.
- **Spec amendments** (PRD/SRD errata) for everything marked as a deliberate deviation.

### Deferred
- **OPEN-001 auth and binding.** Return to this before SPM leaves the homelab, or with the Lemonade gateway project. The FIFO, embeddings and world-engine plans each flag admin endpoints that will need it.

---

## 3. Owner decisions

Each plan lists its full set. These are the ones that block the start of a sprint; the recommended answer is given first.

**Before Sprint 0/1**
1. **Retire Evennia as the runtime** and keep the FastAPI engine as SPM's own deterministic engine. Record this in the specs. Recommended. Optional: a 5-day Evennia trial (world_engine phase G).
2. **Engine interface split:** the proxy computes perception and the engine only stores state (§1.1). Recommended.
3. **Chat identity:** ship the small ST extension that sends `X-SPM-Chat-ID`, with the fingerprint as fallback. Recommended.
4. **Embedding default model:** EmbeddingGemma-300m (768 dims) on every tier, or Qwen3-Embedding-0.6B on Lemonade.
5. **Cloud embeddings for OpenAI users:** opt-in only (recommended), or never.
6. **Spec amendments:** a per-model similarity threshold instead of the fixed 0.35; the native embedding dimension instead of 3584; HNSW optional.
7. **New dependencies:** `tokenizers`, plus a model download of a few hundred MB.

**Before Sprint 2**

8. **When the target can't perceive the turn:** a short status line with no LLM call (recommended), or an LLM "room-bound" reaction.
9. **Remove the 2-tick hysteresis** (recommended).
10. **Tick rules:** per session; regenerations don't advance it; editing old messages doesn't rewind it.
11. **Gating conventions:**
    - `'single quotes'` mean thoughts.
    - Whisper bystanders hear a "murmur".
    - A shout carries one level further.
    - Emotes need line of sight.
    - The explicit tag syntax, possibly with ST Quick Replies.
12. **ST Summary and Author's Note blocks:** pass them through or strip them. They can't be redacted per character.
13. **Scheduler:** should chat pre-empt background work, and should a regenerate cancel the in-flight turn?

**Before Sprint 3/4**

14. **GM_ACTION default per backend:** recommended `full` for local backends and `off` for paid APIs.
15. **Sleep cycle:** run it inside the proxy scheduler (recommended), or keep systemd with an advisory lock.
16. **Token budget:**
    - the default split (PRD 5.3; the SRD split leaves no room for output);
    - what to do when the character card alone overflows;
    - summaries on or off;
    - whether to keep the hardware tiers as preset aliases.
17. **Cost caps for paid backends:** warn or block.

---

## 4. Already done (2026-10-03)
- The assessment fixes are on `V0.4` and pushed (`9d7dd04`).
- The live database was reset, with owner approval. Backups are in `~/Desktop/Experiments/SillyTavern/db-backups/`.
- The existing random vectors are gone with the reset, which settles embeddings decision 6 (discard and re-embed).

---

## 5. Verification pass (2026-10-03, Fable)

**Verdict: GO, with two conditions** (§5.2). Every Sprint 0 bug claim was re-verified against the code before this verdict; none failed.

### 5.1 What was checked
- **All 10 Sprint 0 bug claims confirmed in code:** system messages after the first dropped (`prompt_builder.py:147-153` appends only user/assistant roles); body `session_id` unreachable (`ChatCompletionRequest` declares no such field, so `model_dump()` can't contain it); `CharacterMovePayload` in `app.py:332` shadows the session-aware one in `models.py:86`; `configure_world` uses the template key as session id (`app.py:516`); `uuid.uuid4()` idempotency keys (`routes.py:757,765`); demo characters seeded in `evennia_world/res/strings.json` `present_characters` (the gating plan cited hybrid_builder.py; the data actually lives in the strings file — same fix, different file); upstairs/tavern hack (`app.py:189-203`); SOVEREIGN tier frozen as a default argument (`prompt_builder.py:17`); `max_tokens: Optional[int] = 128000` (`routes.py:71`); `periodic_review_cadence: 1` in config.json.
- **External claims spot-checked:** SillyTavern really does persist `chat_metadata.integrity = uuidv4()` per chat (`script.js:7665`) and branches get a fresh one that records the parent (`bookmarks.js:201,284`), so the chat-identity design rests on something real. Lemonade's live config says `max_loaded_models: 2` (playbook says 1) — discrepancy confirmed, keep the manual check. The lore extractor does build its own LLM client (`lore_extractor.py:25`), so the FIFO plan's client-level wrapping is the right choice.
- **Cross-plan consistency:** the §1 reconciliations hold; no plan contradicts another after them. The queue-bypass rules (gating, tokenize calls, local CPU embeddings) agree across the FIFO, gating, embeddings and token-budget plans.

### 5.2 Conditions on the GO
1. **Answer decisions 1–7 (§3) before starting Sprint 0.** Decision 3 (ST extension) gates Sprint 0's chat-identity work directly.
2. **Re-baseline Sprint 0.** 3–4 days is optimistic for: ten bug fixes with regression tests + chat identity with an ST extension + the engine interface contract with two adapters and a contract suite + embeddings phase 0. Either call it ~6–9 days, or move the engine interface contract into Sprint 1 Track A (it is natural there and nothing else in Sprint 0 depends on it).

### 5.3 Notes (non-blocking)
- The headline "50–60 dev-days" doesn't match the sprint sums (42–53 as written; ~45–58 with the re-baselined Sprint 0). Treat the per-sprint numbers as the estimate.
- "Two parallel tracks" in Sprints 1–2 assumes two workers. Solo, wall-clock is the sum.
- In Sprint 1, backend embedding calls exist before the scheduler does (Sprint 2). Acceptable interim; the provider layer should make its call sites queue-ready so wiring in Sprint 2 is mechanical.
- FR-001 session-isolation tests assume the `st_user_<char>` key shape; the chat-identity fix must update them deliberately, not incidentally.

---

## 6. Decision record (2026-10-04)

All 17 decisions in §3 are resolved (owner, via the review doc). Sprint 0 is unblocked.

- **Approved as recommended:** 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17.
- **4 — modified:** EmbeddingGemma-300m is a **provisional** default. Run a bake-off with Sprint 1/3 calibration: 30–50 real chat/lore queries across EmbeddingGemma-300m, Qwen3-Embedding-0.6B and the backend's own `/v1/embeddings`, scored on recall@5, latency and memory (incl. NPU slot behaviour). The per-row `embedding_space_id` makes a later swap a re-embed job, not a migration.
- **11 — modified:** gating syntax markers are **fail-open hints, never requirements**. Unclosed/mismatched/absent markers degrade to plain `speak` (a line heard normally, never a leak); every gate is opt-in per message; add malformed-marker fixtures (missing closer, swapped delimiter, mixed styles between turns) to the 14-scenario suite.
- **12 — resolved:** **strip ST Summary and Author's Note by default** (they are shared, un-redactable text and would leak hidden intent to every character), with a per-chat opt-in pass-through for secret-free chats. The canon event log is the shared story truth; private monologues live only in per-character memory; the admin thoughts tab is a read-only dev window, gated by the deferred OPEN-001 auth.
- **16 — partially open:** PRD 5.3 split approved; the sub-items (character-card overflow behaviour, summaries on/off, keep hardware tiers as preset aliases) stay open until Sprint 4.

Sprint 0 manual-check updates: `max_loaded_models` is **resolved** — the 2-slot experiment caused an OOM and was reverted to 1 on 2026-10-03 (verified: loading a second LLM evicts the first; backup `~/.config/lemonade/config.json.bak-2026-10-03`). Hermes is verified working as a local agent (first headless turn ≈ 1.5–2 min: ~26 s model swap + ~59 s prefill). Still to run: NPU embed-model slot behaviour under 1-slot config; log one raw SillyTavern request.

---

## 7. Sprint 0: complete (2026-10-04)

All Sprint 0 items landed on `V0.4` (`02f3ac5`..`b7babfe`); full suite 478 passed, 1 skipped (431 at sprint start). Verified live end to end with one SillyTavern message: `X-SPM-Chat-ID` arrived resolved, the session became `st_chat_<uuid>`, the world room and turn memory persisted under that session, embeddings stored NULL (phase 0), lore extracted 8 pending character/world rules and none about the user, and no monologue or GM_ACTION text leaked to the frontend. The raw-request manual check is closed: ST's Custom source natively sends only accept/authorization/content-type/user-agent — no chat id — confirming the extension was required. Engine-side bonus fix: sessions no longer share room objects (found by Hermes, extended to configure_world in review). Sprint 1 starts with the engine hardening list in proxy/engine/http_adapter.py's docstring (ten verified gaps) and the embedding provider layer.

## 8. Sprint 1: complete and verified (2026-10-04)

Landed on `V0.4` (`bec341c`..`7003e3e`), suite **576 passed, 2 skipped** (431 at Sprint 0 start). Track B: embedding provider layer, combined migration (applied live, backup first), space-aware recall + FTS fallback, re-embed job — live-verified (768-dim NPU embeddings; all rows backfilled; proxy boots `provider=openai_compat`). Track A: per-session idempotent ticks, SRD spatial rules (15 ft, doors/metal occlude, hysteresis gone), shout, snapshot, stateful barriers, seeded configure, per-session reset, mutation origins, Postgres reload; the HTTP adapter reached full contract parity — all 20 contract behaviors run against BOTH engines. OPEN-010 closed. Live smoke: tick idempotency, seeded snapshot, per-session reset all correct on the wire. Notable finds en route: idempotency keys were sent as headers the server never read; the actor received their own action as a consequence; moves silently 200'd to nonexistent rooms; `configure_world`'s DB restore makes shared test session-ids leak state (per-test unique ids now). Division of labor: Hermes landed A1 and the Sprint 0 engine package; A2 and the adapter were done by the session after a goldfish context-thrash stall (see workflow assessment).

## 9. Sprint 2: complete — exit gate met (2026-10-05)

Landed on `V0.4` (`..57c39f5`), suite **687 passed, 2 skipped**. Gating core: deterministic action parser (speak/whisper/shout/move/emote + fail-open `[whisper:X]`/`[move:Y]`/`[shout]` tags, private 'quoted thoughts'), deterministic world seeding (`[scene:KEY]` tag → keyword template → generic_void) wired to first-turn configure, perception table (`spm_perception`, idempotent per turn_id so regenerates rewrite their turn), and per-character gated history — each character's prompt history is rebuilt from what THEY perceived, never the raw transcript. The omniscience fix was verified live (stolen-amulet scenario: before, the character alluded to a thought she couldn't hear; after, it is absent from her prompt, memory and world state). Scheduler: phases 1–2 (lanes, strict priority, coalescing, caps/deadlines, slot-held-for-stream) plus P3 preempt/supersede per decision 13 — chat cancels one running P2/P3 background job when no slot is free (never P1 embed), and a new turn for a session supersedes that session's queued or in-flight turn.

**Exit gate:** the 14-scenario leak suite (`tests/test_gating_leaks.py`) passes with 0 leaks of planted phrases, including the captured Vardus fixture. It runs the real proxy against the real engine (ASGI, LLM faked and prompt-captured) and resends full history each turn like SillyTavern. Building it caught three real leaks, all fixed in `5d1bf18`: (1) the raw-transcript fallback triggered on the character's own row count, handing the full history to exactly the characters who had perceived nothing — it now triggers only when the whole session has no perception rows; (2) `/world/configure` was destructive on re-configure, so the proxy's first-turn seeding silently un-placed every character it didn't know about — re-configure now keeps rooms/occupants/edges and placements are spawn points, never teleports; (3) whisper target matching was case-sensitive (`[whisper:Mira]` vs room id `mira`), so the addressee heard nothing.

Still queued from decision 11: malformed-marker fixtures join the suite (Sprint 3, with the bleed-hardening work). Sprint 3 starts with the blackout side channels: lore extraction and bulk import still see the raw transcript.

## 10. Sprint 3: complete except ONNX fallback (2026-10-05)

Landed on `V0.4` (`..02a7caa`), suite **730 passed, 2 skipped**.

- **Side channels closed (gating phase 5):** lore extraction and bulk import no longer
  see the user's private 'quoted thoughts'; leak suite grew to s15/s16 to prove it.
  Admin perception view: `GET /admin/api/v1/sessions/{id}/perception`.
- **Bleed hardening:** the >8192-token runaway-scratchpad failsafe no longer dumps the
  monologue to the user (notice + discard + taint instead). Malformed-marker fixtures
  (decision 11) added: markers degrade to speak, never grant whisper/shout/move.
- **GM_ACTION Part A complete (OPEN-005 closed, SD-01):** config keys + admin UI (A1,
  Hermes), mode-gated directive through the prompt builder (A2), mode-gated dispatch
  (A3), deterministic validator with caps/reason codes/stable idempotency keys (A4,
  `proxy/core/gm_actions.py`), SPEC_DEVIATIONS.md (A5). New `auto` mode is decision
  14's per-backend preset (full local / off cloud).
- **Embedding bake-off (decision 4 RESOLVED):** the NPU/FLM embed-gemma build is
  semantically collapsed (recall@5 0.125; production recall was top-k noise).
  **Qwen3-Embedding-0.6B-GGUF Q8** wins decisively (recall@5 0.953, MRR 0.918, 32 ms
  p50, coexists with the chat LLM). Default + live config switched; per-model
  threshold setting added (decision 6), calibrated 0.45. Report:
  `docs/plans/BAKEOFF_2026-10-05.md`; harness `scripts/embed_bakeoff.py` reusable on
  real-chat corpora.
- **Hermes cross-review of Sprint 2** produced 9 findings; 7 accepted and fixed
  (case-insensitive placements, flavor_text on re-configure, memory-only startup,
  perception transaction, supersede-vs-cap ordering, typed TurnSuperseded, eviction
  counters), 2 rejected with documented reasoning.

Deferred from Sprint 3: the `onnx_local` CPU provider (embeddings phase 4; serves
backend-less users — `none`+FTS remains their fallback) and NPU re-measurement, since
the NPU embedding path itself is what the bake-off disqualified.

Live verification: services restarted on this code; end-to-end turn through :5050 with
gated history, validated GM actions and Qwen3 embeddings confirmed in the logs.

## 11. Sprint 4: complete (2026-10-05)

Landed on `V0.4` (`..c61393b`), suite **753 passed, 2 skipped**.

- **Chat-scoped lore (OPEN-007 closed):** extraction stamps `session_id`/`scope`,
  retrieval filters to (this chat OR canon), promote-to-canon endpoint, session
  delete clears the chat's rules and keeps canon, per-chat dedupe. `test_lore_scope.py`.
- **Token budget (OPEN-008 closed, P0-P2+UI):** max_tokens clamped to the window;
  PRD-split allocator (card + latest user turn untrimmable, oversized card refuses
  visibly per SD-03; memories/lore/history trim in plan order, monologue stripped
  before turns drop); per-model chars/token EMA calibrated from streamed usage;
  admin UI fields (Hermes). Deferred: cost caps for paid APIs, rolling summaries.
- **Scheduler (OPEN-004 closed):** sleep cycle moved inside the proxy through the
  P3 lane (decision 15) — chat preempts it, coalescing re-queues it; snapshot
  telemetry at `GET /admin/api/v1/scheduler`. Deferred: model-affinity batching,
  circuit breaker.
- **B5:** `X-SPM-Gen-Type` honoured — quiet/impersonate generations answer with
  gated context but persist nothing (no tick, rows, memories, lore, GM actions).
- Spec deviations recorded: SD-02 (per-model threshold, native dims), SD-03
  (budget split + refusal). Division of labor: Hermes built A1 and both admin UI
  sections; elephant side did the allocator, scoping, scheduler and reviews.

**All five planned sprints (0-4) are complete.** Remaining backlog, all optional:
ONNX CPU embedding fallback, B4 fingerprint fallback, cost accounting, rolling
summaries, model-affinity batching, circuit breaker, OPEN-001 auth (deferred by
decision until SPM leaves the homelab).

## 12. Path to the v0.4 release (decided 2026-10-05)

The release proceeds in this order, owner-approved:

1. **QA verification** — the owner runs the 28-step plan in the
   [SPM v0.4 QA Test Plan](https://claude.ai/code/artifact/c5632042-b000-4741-9017-2269471ceec1)
   (Claude Doc; results and comments recorded there). The live DB was backed up
   (`db-backups/litellm_postgres_pre_qa_wipe_2026-10-05.sql.gz`) and wiped of all
   dev/test data first, so everything in it during QA comes from the QA session.
   The re-embed job was exercised before the wipe: 84 rows moved to the Qwen3 space
   in under 3 s, a re-run was a no-op, and a paraphrase probe recalled a migrated
   memory (distance 0.384) while unrelated text did not (0.822).
2. **Bugfixes** — each QA failure fixed with a regression test, then re-run.
3. **Packaging (final v0.4 task)** — make SPM buildable and runnable on another
   machine. Full analysis in `docs/ASSESSMENT_2026-10-05.md` §4. Work list:
   - [ ] Safe network defaults: bind `127.0.0.1`; make the world engine's host and
         port configurable (both currently `0.0.0.0`, and the admin UI has no auth).
   - [ ] One config source: regenerate `.env.example` from `config/manager.py`
         (it names variables the code never reads); make
         `config/docker-compose.yml` agree with the code and run SPM itself.
   - [ ] Cross-platform `spm` launcher replacing the Linux-only, out-of-repo
         `start_spm.sh`: `spm init` (create DB, apply `init_db.sql` + migrations),
         `start`, `stop`, `status`.
   - [ ] First-run setup: backend URL, chat model, embedding model (or `none`),
         plus the SillyTavern connection and extension-install steps.
   - [ ] Remove the dead `current_world` alias from the world engine.
   - [ ] Windows smoke test (nothing has run there yet).
   - [ ] Route A: Docker Compose bundle (recommended first, ≈1–2 days after the
         items above). Route B (pip + embedded Postgres) and route C (native
         installer) are follow-ups, not v0.4 blockers.
4. **Tag `v0.4`.**

Already done toward packaging: runtime dependencies cut to six packages, dev
dependencies split out, Python 3.13+ verified (`d761b82`).

## 13. v0.5 backlog (decided 2026-10-05)

- **Group scenes and user-directed scene transitions** — headline v0.5 feature, found in
  v0.4 group-chat QA. Spec: `docs/plans/group_scenes_v0.5.md` (current behaviour, intent,
  proposed `[focus:room]` / `[at:Character:room]` tags, narrator room events, a
  group-aware skip rule, placement by room name). v0.4 ships with the interim F10
  behaviour: a character joining a group chat is placed in the user's room.
