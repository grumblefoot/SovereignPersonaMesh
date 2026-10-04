# SPM state assessment: 2026-10-03

The project was paused from about 2026-09-09 until now. This report covers where it stands, what was broken and is now fixed, where the code has drifted from the PRD/SRD, and what still needs an owner decision.

Scope: code on branch `V0.4` (commits `5623ba8`…`f6b6a44`), the live stack on the Strix Halo box (Lemonade 11.9, Postgres `spm-postgres`, SillyTavern), and the two spec PDFs.

---

## 1. Bottom line

- **SPM runs end to end again.** SillyTavern → SPM (5050) → Lemonade (13305) works with `Qwen3.8-27B-GGUF`, the narrator model from RPG-HaloTales-V2. The full suite passes: **431 passed, 1 skipped (opt-in live-LLM test)**. It now runs against a throwaway database, and **0 rows are written to the live one** during a run.
- **The test suite had been damaging the live system.** Before today, every `pytest` run:
  - truncated every character's memory table in the live database (`test_factory_reset_wipes_all_state`)
  - planted a fake core memory ("I processed a record.") in each real character's table
  - rewrote `config/config.json` to point at a dead port
  - created rooms in the live world engine
  - sent dozens of real requests to Lemonade

  Any real roleplay memories from before today are gone (`arvenia`, `seraphina` and `mira` were empty). This is fixed.
- **Long-term memory never worked.**
  - The nightly sleep cycle failed every night since at least 2026-09-25.
  - Its logic would have deleted the backlog of memories on its first successful run.
  - The core memories it writes could never be retrieved.

  It is fixed and verified against the live model.
- **The central spec promises are mostly not delivered.** These are design gaps, not regressions:
  - Sensory gating is inert by default.
  - The embedder is a random-vector stub, so retrieval-augmented recall (RAG) returns nothing useful.
  - The GPU FIFO queue is never used.
  - There is no authentication.

  Details in §4; they need owner decisions (§5).

---

## 2. Verified working (live, 2026-10-03)

| Check | Result |
|---|---|
| Stack health | Proxy, Evennia stand-in, Postgres, SillyTavern and Lemonade all healthy. Proxy and engine run on SPM's own `.venv` (no longer the `spm-demo-mvp` venv). |
| Roleplay turn via SPM | Clean in-character reply. No `<think>` or `[GM_ACTION]` text leaked. Each GM action dispatched once. |
| Two-word character ("Mira Vale") | Turn persisted with an embedding (previously impossible; see §3). |
| Lore extraction | 7 pending rules about the character and the world, none about the user's persona (previously 0 rules from any reasoning model). |
| Sleep cycle | `systemctl --user start spm-sleep-cycle` exits 0. The live consolidation model returns clean one-sentence memories (4/4). |
| Test isolation | 0 tuples written to `litellm_postgres`, 0 Lemonade requests, `config.json` unchanged across a full run. |
| Logs | `logs/spm_proxy.log` is now written (rotating, 10 MB × 5). |

---

## 3. Fixed today

Ranked by impact. Each fix has regression tests.

| # | Problem | Impact | Fix | Commit |
|---|---|---|---|---|
| 1 | Test suite used the live DB, Evennia, Lemonade and config | Wiped real character memories on every run, planted fake core memories, repointed the live config to `:9999` (committed to git), created rooms in the live world, ran real GPU inference | Tests use a fresh `spm_test` DB rebuilt each session; backends are pointed at a closed port; settings use a temp file; there is a guard against rebuilding the live DB | `7c4cc4c`, `9959e28` |
| 2 | `init_db.sql` could not initialise a fresh database | `create_csa_memory_table` nested a `format()` placeholder, so it failed on any new install | Separate `EXECUTE` | `9959e28` |
| 3 | Sleep cycle: failure path lost data | On a failed or empty model call it stored a canned sentence as a permanent core memory, then deleted all volatile logs older than 24 h, summarised or not | Consolidate per (session, day) and delete only the summarised rows in the same transaction; on failure write and delete nothing | `86e635e` |
| 4 | Sleep cycle: unreachable memories | Core nodes were saved under session `sleep_cycle_consolidated` with no embedding; the retriever requires both | Saved in the source session, with an embedding | `86e635e` |
| 5 | Sleep cycle service dead (OPS-001) | System `python3` had no `asyncpg`, then `No module named 'core'`; the model id no longer existed | `.venv/bin/python -m scripts.sleep_cycle`; model `Gemma-4-E4B-it-GGUF`; reasoning off (it had used the whole token budget); removed the `<boss>/<idle>` markers; echoed instructions are rejected | `d36cfe4` |
| 6 | SQL injection through the character name | A chat message's `name` was formatted into `ALTER/DELETE/INSERT` table names | `core.identifiers.safe_char_id` (`[a-z0-9_]`, ≤48 chars) at every table-name site | `9fa3a70` |
| 7 | Multi-word characters never stored memories | "Mira Vale" became `csa_memory_mira vale`, a syntax error that was swallowed | Same fix as #6 | `9fa3a70` |
| 8 | GM_ACTION and planning lines leaked to SillyTavern | Seen live with Qwen3.8: raw `[GM_ACTION: {...}]` JSON, "Then the narrative.", "Done." in chat and in stored memory; duplicate dispatch gave `409 Conflict` | Parser holds back these lines, still dispatches the actions, and removes duplicates | `93acb5b` |
| 9 | Fabricated replies when Lemonade was down | Client invented "…I am ready." (or "Error from LLM Backend: 500"), streamed it as the character's turn, saved it as memory, and it fed the playbook's ~1.9 ms "TTFT" | `LLMBackendError`. Streaming: visible notice, nothing saved. Non-streaming: HTTP 502 | `c03bb13` |
| 10 | Model resolution could pick any model | Any unknown name went to `available[0]` (the Flux image model here, or `hermes-coder`); retired `google/gemma-4-*` ids were the defaults | Explicit chain: virtual id → default, exact, legacy map, case-insensitive, unique substring; chat models only; `hermes-coder` excluded; otherwise passed through. List cached 30 s. `/v1/models` lists the real Lemonade chat models | `d72f523` |
| 11 | Lore extraction broken with reasoning models | Reasoning prose was kept in front of the JSON, so 0 rules were extracted; each call took ~2.5 min | Reasoning chunks skipped; reasoning turned off for extraction (~7 s per call) | `15537b4` |
| 12 | BUG-008: user persona treated as NPC lore | Rules about the player | Persona system message removed before extraction (user name read from ST's main prompt) | `62254e4` |
| 13 | Blackout bypass always said "*Luna hears…*" | Wrong character named | Uses the target character | `c03bb13` |
| 14 | Proxy ran with `reload=True` in production | An edit restarted it and the restart hung while the dashboard SSE was open (happened twice) | Off unless `SPM_RELOAD=1` | `e04955d` |
| 15 | Rotating log never attached | `logs/` was empty; `/tmp/spm_proxy.log` is overwritten on every start | Root logger gets the rotating file handler | `e04955d` |
| 16 | Hardware tier misdetected | The 128 GB Strix Halo reports 124.4 GiB in MemTotal, so it was treated as PERFORMANCE | Threshold 120 GiB | `33a38ff` |
| 17 | Deprecated `on_event` hooks | 8 of 11 test warnings | `lifespan` | `f6b6a44` |

Outside the repo:
- `~/.local/bin/start_spm.sh` now uses `.venv`.
- `spm-sleep-cycle.service` reinstalled.
- SillyTavern's model is set to `Qwen3.8-27B-GGUF`.

---

## 4. Spec drift: PRD/SRD versus code

Source: a full read of both PDFs against the code. The key claims were checked by hand.

### Not delivered (core intent)

| Spec intent | State | Evidence |
|---|---|---|
| **Sensory gating** (PRD FR-1.1/1.2, SRD 3.1/3.3): characters perceive only what reaches them | **Inert by default** | The world starts empty (`dynamic` template), so gating defaults to `direct`. Every action is sent as actor `"user"` / `"speak"` (`proxy/api/routes.py:263-264`), so there is no whisper or move detection. The last 15 raw SillyTavern messages go to the LLM ungated (`proxy/rag/prompt_builder.py:67`). Blackout recipients are dropped by the engine, so the proxy's blackout bypass is rarely reachable. |
| **Episodic RAG** (PRD 5.4, SRD 3.4): cosine < 0.35, decay scoring | **Non-functional** | `scripts/onnx_embedder.py:34` returns random vectors seeded by Python's per-process salted `hash()`. Different texts sit about 1.0 apart, so nothing passes the 0.35 cut-off. Vectors also change on every restart. Lore trigger rules depend on the same stub. |
| **FIFO GPU queue** (FR-1.4, SRD 5.1) | **Built, never used** | `fifo_queue` is created in `routes.py` and never called. Chat streams, lore extraction and the sleep cycle can hit the GPU at the same time, which matters with Lemonade swapping models on demand. |
| **Inference bypass with ambient log** (PRD 4.1.1) | Partial | `ObserverInferenceGatingFilter` is imported (`routes.py:22`) and never called, so there is no ambient memory commit. The activation hooks are missing. |
| **Spatial matrix** (SRD 3.3.2): 5/15 ft; closed door or metal partition → blackout | Drift | Degraded runs to 20 ft, and only `SOLID_WALL` blacks out (`evennia_world/spatial_matrix.py:43-50`). Adjacency reads the legacy global world, not the session world. |
| **Temporal anchor** (playbook): ticks = user messages | Contradicted | The tick is a global in-memory counter (seeded at 1420) that increments on every action call, including regenerations, and is shared across sessions. It is not reloaded at startup. |
| **Token budget** (32K partitioned) | Declared only | Tier budgets exist (`config/hardware_tiers.py`) but nothing enforces them. `backend_max_tokens` is 128000. |
| **Security** (SRD 4.3, PRD "secure internal API") | Absent | No auth on any admin route: factory reset, shutdown (which runs `docker stop`/`pkill`), config, lore approve/reject. No auth on Evennia's reset either. `CORS *` with credentials. Everything binds `0.0.0.0`. The thought SSE exposes every character's private monologue to the network. |

### Deliberate or accepted deviations (should be documented, not reverted)

- **The world engine is a FastAPI in-memory stand-in, not Evennia/Django.** Nothing imports `evennia`.
- **Monologue tags:** `<ctrl94>` became `<think>`/`<thinking>` (v0.3, `migrate_tags.py`). The fail-safe cap was raised from 500 to 8192 tokens.
- **No HNSW index:** pgvector caps HNSW at 2000 dimensions and the schema uses 3584. `halfvec` (up to 4000 dimensions) was not evaluated.
- **Session-scoped retrieval (FR-001)** replaces the PRD's persistent cross-session character memory.
- **LiteLLM / port 8000 / WSD Gemma 9B** were replaced by Lemonade on 13305 with Python-only gating.
- **Sleep cycle (today's change):** logs stay raw for 24 h, then are consolidated per (session, day). This is one day later than "summarise the last 24 h", and it is what makes a failed run lossless.

### Scope beyond the spec that strains its intent

- **`[GM_ACTION]` MOVE / CREATE_ROOM:** the LLM writes the world state that gating reads. This erodes the Zero-LLM core rule and PRD 11.1.1, which says templates only and no raw room scripts.
- **Lore extraction:** these are LLM-written rules that steer later prompts, run outside any queue, and are stored per character without a session id. Lore therefore bleeds across chats until a factory reset.
- **Admin dashboard and thought stream:** useful for observability, but unauthenticated, and they expose private monologues.
- **Dead code:** the vibe profiler style card and `parse_sillytavern_context` are never used. `_gather_public_response` is unused.

### Playbook claims that were wrong

- **Embeddings "verified":** the embedder is a stub.
- **"FIFO queue dispatch":** never called.
- **"All 5 SRD SLAs met" / TTFT ~1.9 ms:** this measured the fabricated fallback reply with no backend running. `e2e_sla_metrics.json` itself says `total_tests: 0`.
- **The sleep-cycle timer step is checked off,** but the service failed every night.
- **Test counts don't add up** (108 → 80 → 131/144 …). The per-file counts in the directory map are stale. "CERTIFIED" entries are self-certifications by the authoring agent.

---

## 5. Open work, ranked: needs owner decisions

| P | Item | Why it matters | Decision needed |
|---|---|---|---|
| P0 | **Authentication and binding** | Anyone on the LAN or Tailscale can factory-reset, shut down, reconfigure, or read every private monologue | Bind to 127.0.0.1 or the Tailscale IP now, or add an API key? This fits the planned Lemonade gateway work. |
| P0 | **Real embedder** | Without it RAG, lore triggers and imported memories don't work | Which model and dimension? Lemonade has `embed-gemma-300m-FLM` (NPU, a candidate for the NPU "use case"); llama.cpp embedding GGUFs are another option. Changing from 3584 needs a schema migration, but allows HNSW. |
| P1 | **Make gating real** | It is the product's reason to exist | Detect actor and action type (speak/whisper/move) in Python as the SRD says; gate the history sent to the LLM; seed worlds from templates; fix closed-door/metal → blackout and the 15 ft threshold, or document 20 ft. |
| P1 | **Wire the FIFO queue** | Background LLM calls race the chat stream and force model swaps | Route all LLM calls (chat, lore, sleep cycle) through one queue. |
| P1 | **Decide GM_ACTION's future** | It conflicts with the Zero-LLM rule | Keep it (and amend the spec), or constrain it to validated template operations. |
| P2 | Lore rules per session | Lore currently bleeds across chats | Add `session_id` to `csa_lore_rules_*`. |
| P2 | Token budget enforcement; `max_tokens` 128000 | Context overflow risk | Use a tokenizer and enforce the tier budget. |
| P2 | Evennia unreachable → raw 500 from the chat route | Poor failure mode | Same treatment as the LLM outage fix. |
| P2 | Leading blank lines in streamed replies; tick semantics; DESIGN-001 telepathy scrubber; Omni images/voice via `lemonade_omni`; DESIGN-003 strings (partly done) | Quality and features | Schedule as wanted. |

---

## 6. Actions waiting on the owner

1. ~~**Clean the live database.**~~ **Done 2026-10-03 with the owner's approval:** `litellm_postgres` was dropped and rebuilt from `scripts/init_db.sql` (0 rows). Backups: `~/Desktop/Experiments/SillyTavern/db-backups/litellm_postgres_pre-cleanup_2026-10-03.sql.gz` and `…_pre-reset_2026-10-03.sql.gz`.
2. **Refresh any open SillyTavern tab** before chatting, so it doesn't save the old Gemma model setting back.
3. ~~**Push `V0.4`**~~ Pushed 2026-10-03 (`9d7dd04`).

---

## 7. How to verify

```bash
cd SovereignPersistanceMesh
.venv/bin/python -m pytest tests/ -q          # 431 passed, 1 skipped; rebuilds spm_test, never touches litellm_postgres
RUN_LIVE_LLM_TESTS=1 .venv/bin/python -m pytest tests/test_st_parser.py   # opt-in live Lemonade test
systemctl --user start spm-sleep-cycle && journalctl --user -u spm-sleep-cycle -n 5
curl -s localhost:5050/v1/models               # spm-sovereign-mesh + Lemonade chat models
```
