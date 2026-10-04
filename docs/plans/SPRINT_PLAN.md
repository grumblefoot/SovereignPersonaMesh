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
