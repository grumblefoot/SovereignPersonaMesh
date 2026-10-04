# Plan: token budget enforcement (OPEN-008)

Status: planning only. Scheduled after the main features (embeddings, gating, FIFO, GM_ACTION/lore). Written 2026-10-03.

## Goal

Every request SPM sends to the backend should fit that backend's real context window. Each part of the prompt gets a share of the window that the user can configure. `max_tokens` is computed per request rather than fixed at 128000. Token use, and cost on paid backends, shows up on the admin dashboard. SPM runs on one backend at a time, and all three must work:

- Lemonade: local, contexts up to 262k, with ctx set per model.
- KoboldCpp on a 16 GB GPU: 8k to 16k.
- OpenAI: paid per token.

## Current state (evidence)

| Fact | Where |
|---|---|
| The tier budgets copy the SRD split (4096/8192/4096/16384 for SOVEREIGN) and nothing reads them except `inner_monologue_enabled` and `/` | `config/hardware_tiers.py:30-58`, `proxy/main.py:79` |
| The prompt builder is always created with the SOVEREIGN default. It ignores `SPM_HARDWARE_TIER` | `proxy/api/routes.py:54`, `proxy/rag/prompt_builder.py:17` |
| History is cut by message count, not tokens | `prompt_builder.py:67`, `:145` (`chat_history[-15:]`) |
| Only the first ST system message is kept and other system messages are dropped (ST world info, author's note, persona) | `routes.py:313-316`, `prompt_builder.py:152` |
| Memories are hard-coded to `top_k=3`. Lore rules are appended without limit | `routes.py:324-331`, `:353-363` |
| Backend `max_tokens` = `backend_max_tokens` (config 128000). The code fallback is 2048 | `config/config.json`, `config/manager.py:28`, `routes.py:311`, `:414`, `:484` |
| If ST omits `max_tokens`, the public cap defaults to 128000 | `routes.py:71`, `:309` |
| Public and monologue "tokens" are counted with `str.split()` (words). The monologue fail-safe is 8192 "tokens" | `proxy/core/stream_parser.py:30`, `:136`, `:411` |
| Lore extraction uses a fixed `max_tokens=8192` | `proxy/rag/lore_extractor.py:55` |
| No usage is captured. Streaming does not request `stream_options`. The model cache keeps only ids and drops `context_length` | `proxy/backend_client/lemonade_client.py:51-67`, `:129-135` |
| The admin config UI exposes only the backend URL, tier and cadence | `proxy/ui/index.html:171-209` |

**The two specs disagree.**

- PRD 5.3 splits 32,768 tokens as 4096 system/card, 2048 sensory feed, 6144 RAG, 12288 history, 2048 monologue instruction and 6144 KV/generation margin.
- SRD 3.4.3 splits it as 4096 system, 8192 RAG, 4096 spatial and 16384 history. That adds up to exactly 32,768 and leaves **zero for output**, so it cannot be used as written.

The PRD split is the only consistent one. I recommend basing the default preset's percentages on it.

## Design

### 1. Configuration model (`config.json` → `token_budget`)

```json
"token_budget": {
  "enabled": true,
  "preset": "auto",
  "context_window": "auto",
  "context_window_override": null,
  "safety_margin_pct": 5,
  "mode": "percent",
  "partitions": {
    "system_card": 12, "lore": 5, "memories": 15, "sensory_spatial": 6,
    "history": "rest", "monologue_reserve": 8, "output_reserve": 12
  },
  "floors": { "history": 1024, "output_reserve": 256, "monologue_reserve": 0 },
  "public_output_default": 400,
  "summarize_trimmed_history": false,
  "cost": { "enabled": false, "input_per_1m": null, "output_per_1m": null, "session_cap_usd": null, "cap_action": "warn" }
}
```

- **Values.** `mode: absolute` takes token counts instead of percentages. `history: "rest"` makes history elastic: any partition that does not use its full share passes the remainder to history.
- **Precedence:** values in the request (ST `max_tokens`) > explicit user config > preset > auto-detect > SOVEREIGN-tier fallback (32,768).
- **Presets.**
  - `sovereign-32k`: the PRD split.
  - `lemonade-large`: window from detection. Memories and history are capped in absolute tokens, because a 262k prompt hurts time to first token (TTFT) and quality.
  - `kobold-small`: 8k–16k. Memories top-k 2, no summaries, small monologue reserve.
  - `openai-cost`: a deliberately small window (e.g. 16k) whatever the model allows, with cost tracking on.
- **Hardware tiers.** The tiers in `hardware_tiers.py` become preset aliases and no longer get their own code path.

### 2. Context window detection (`BackendProfile`)

A new `proxy/backend_client/backend_profile.py` probes once per model and caches the result for each model id. It is refreshed when the model changes. The probe is chosen by `backend_kind` (`auto|lemonade|llamacpp|koboldcpp|openai`):

| Backend | Window source | Tokenizer source | Status |
|---|---|---|---|
| Lemonade | `GET /v1/models` → `context_length`: the value loaded, else the configured `ctx_size`. `max_context_window` is the architecture maximum, so use it only for display | No tokenize endpoint is documented | Verified (lemonade-server.ai/docs/api/openai) |
| llama.cpp server | `GET /props` → `default_generation_settings.n_ctx` | `POST /tokenize {"content": ...}` → `{"tokens": [...]}`. `POST /apply-template` gives the exact templated prompt | Verified (llama.cpp server README) |
| KoboldCpp | `GET /api/extra/true_max_context_length` | `POST /api/extra/tokencount` (returns a count plus ids) | Endpoints verified. **The JSON key names are unverified**: probably `{"value": n}` / `{"value": n, "ids": [...]}`. Confirm against `/api` on the box |
| OpenAI | A static, user-editable table of model → window. The API does not report it | `tiktoken` (`o200k_base` for current models) | tiktoken itself is verified. The window table must be maintained by the user |

The window actually used is `min(detected, override, preset)` × (1 − `safety_margin_pct`). The ST-facing `/v1/models` listing (`routes.py:74`) can also report `context_length` so ST shows it.

### 3. Token counting

`TokenCounter` interface: `count(text) -> int` and `count_messages(msgs) -> int`. Backends, in order of preference:

1. A remote tokenize endpoint (llama.cpp, KoboldCpp). Its calls go **outside** the generation FIFO, but their concurrency is limited.
2. tiktoken for OpenAI.
3. A calibrated heuristic: chars ÷ ratio. The ratio starts at a conservative 3.2 chars/token and is updated per model with an exponential moving average (EMA) from the backend's reported `usage.prompt_tokens`. llama-server reports usage in streaming; Lemonade's docs do not mention it, so whether it passes usage through is **unverified**. Lemonade will probably need this path unless HF `tokenizers` is added (an owner decision).

Additional rules:

- Every block also gets a fixed per-message template overhead, about 6 tokens, which is calibrated.
- Counts are cached in an LRU keyed by `(model, sha1(text))`. History messages almost never change, so after the first turn each turn costs about one new count.
- Counting must stay within the SRD 4.1 proxy overhead of under 150 ms. If the tokenize call fails or times out after 50 ms, fall back to the heuristic.

### 4. Allocation and trimming (`proxy/rag/budget.py`)

The allocator is a pure function. Its input is the candidate blocks, each with a priority and a partition. Its output is the kept blocks plus a `BudgetReport`.

**Never trimmed:**

- the character card/system prompt
- the SPM directive and prefill
- the latest user turn

If these alone exceed the window, return a visible notice through the same path as the LLM-outage fix and send nothing. Do not silently truncate the card. That choice is an owner decision.

**Trim order, first to last:**

1. Retrieved memories: lowest score first.
2. Lore trigger rules: lowest relevance first. Invariants go last.
3. History: oldest first, removed as whole user/assistant pairs. Monologue in older assistant turns is stripped before whole turns are dropped.
4. Sensory/spatial detail, cut down to the room name only.
5. Optionally, trimmed history becomes a rolling "Earlier:" summary. It is generated **in the background**, never in the critical path (SRD 4.1 prohibits LLMs in the routing path), and cached per session. Until a summary exists, the dropped turns are simply omitted.

**Turn cap.** The 15-message cap is removed, or kept only as an optional `max_history_turns`.

**Double-trimming with SillyTavern.** ST trims history to its own Context Size before sending. SPM then adds memories, lore and the directive, so its prompt can be larger than the one ST budgeted. Rules:

- SPM is authoritative.
- SPM trims only by the amount that is actually over budget, never to a fixed count.
- The report logs when ST already trimmed (history starts mid-conversation).
- Docs and the admin UI tell users to set ST's Context Size to at least SPM's window, or to the same value. A smaller ST size only means less history.
- ST's "Response Length" maps to `max_tokens` and stays the public output cap.
- Whether ST sends its context size in the request for the Custom OpenAI source is **unverified**. If it does, cap the window by it.

### 5. Generation `max_tokens`

`max_tokens = min(window − prompt_tokens − margin, monologue_reserve + public_cap)`, with a floor of `floors.output_reserve`.

- `public_cap` = ST `max_tokens`, or `public_output_default` if ST did not send one. The `routes.py:71` default changes to `None`.
- `backend_max_tokens` becomes an optional **hard ceiling** instead of a value that is always sent.
- **Reasoning models.** `<think>` text and `reasoning_content` both count as output, so the monologue reserve has to cover them.
  - Tie `MAX_MONOLOGUE_TOKENS` to `monologue_reserve` and count with the real counter instead of words.
  - On OpenAI reasoning models, send `max_completion_tokens`. It includes reasoning tokens; verified from the OpenAI docs. Optionally also send `reasoning_effort`.
  - Read `usage.completion_tokens_details.reasoning_tokens` for telemetry.
  - Lore extraction and the sleep cycle already disable thinking. Their `max_tokens` is also budgeted, at small fixed values.

### 6. Telemetry and cost

- **Per turn:** estimated and actual prompt tokens, completion tokens, reasoning tokens and cached tokens, plus tokens per partition, trimmed counts and window source.
- **Usage source:** OpenAI needs `stream_options: {"include_usage": true}` (verified). llama.cpp returns usage and timings in the stream.
- **Storage:** `TelemetryCollector.record_turn_trace`, so it shows on the existing session trace.
- **Cost:** price × tokens, accumulated per session. Prices are user-entered and never hard-coded. A session cap either warns or blocks.
- **Dashboard:** a "Context budget" panel with a stacked bar of the last turn by partition and the cost total. Also add `GET /admin/api/v1/budget/preview`, which takes a session id and shows what would be trimmed.

## Phases

**P0: safety fixes (S, about 0.5 day).**
- Default request `max_tokens` → `None`.
- Compute `max_tokens` as `min(backend_max_tokens, window − estimate)`, using the tier window and the heuristic counter.
- Make `CognitivePromptBuilder` honour `SPM_HARDWARE_TIER`.
- *Acceptance:* no request carries `max_tokens` above the window. Changing the tier changes the budget.
- *Tests:* a mocked client captures the payload. Tier env var → config. Missing ST `max_tokens` → `public_output_default`.

**P1: allocator and counter (M, 2–3 days).**
- `token_budget` config with migration from the old keys.
- `BackendProfile` detection, `TokenCounter` and the LRU.
- `budget.py` allocator wired into `build_csa_messages`, covering memories, lore and history.
- *Acceptance:* with a fake 1-token-per-word counter, a 200-turn ST history fits inside the window. The latest user turn and the card are always present. An oversized card produces a visible notice and no backend call.
- *Tests:*
  - Allocator unit tests: percent and absolute modes, the elastic "rest", floors, trim order, pair-wise history drop.
  - Detection parsing against canned JSON for each backend through a mocked httpx transport.
  - Tokenize timeout → fallback.
  - All with no live services (`tests/conftest.py`). DB-touching paths use `spm_test` via `tests/_testdb.py`.

**P2: usage, calibration, cost (S–M, 1.5–2 days).**
- Parse usage from the stream. EMA calibration.
- Cost accounting and caps. Telemetry fields.
- *Acceptance:* the estimate is within ±10% of the reported `prompt_tokens` after 5 turns, in a simulated stream. The cap action fires.
- *Tests:* synthetic SSE with a usage chunk, the reasoning-token field, cap warn/block.

**P3: admin UI and presets (S–M, 1–2 days).**
- Config form, preset picker, detected window display, preview endpoint, budget panel.
- *Acceptance:* a preset or override saved in the UI takes effect on the next turn without a restart.
- *Tests:* admin route tests on a temp settings file.

**P4 (optional): rolling summaries (M, 2–3 days).**
- Background summarisation through the FIFO queue, cached per session.
- *Acceptance:* the trimmed span is replaced by a summary within N turns. A summary failure omits the span and stores nothing.

Total: about 5–8 days for P0–P3, plus P4.

## Risks / open questions

- **Tokenizer accuracy on Lemonade.** Without a tokenize endpoint, the margin carries the risk. The 5% default may need to be 8–10% for Qwen/Gemma chat templates.
- **Prompt cache churn.** Trimming at the front of history invalidates the KV prefix cache on every turn. Mitigation: trim in chunks (drop 20% at a time, with hysteresis) so the prefix stays stable for several turns. This matters most for the SRD 4.1 prefill SLA and for OpenAI cached-token pricing.
- **System messages.** Dropped ST system messages (world info etc.) are a separate correctness bug. If they are restored, they need their own partition.
- **Large windows on this box.** A 262k context is technically possible but slow. Absolute caps in `lemonade-large` are a judgement call.

**Owner decisions:**

1. Default split: PRD (recommended) or SRD with an output reserve added.
2. Oversized card: notice and refuse (recommended), or truncate the card.
3. Rolling summaries (P4): yes or no.
4. Add HF `tokenizers` for exact Lemonade/GGUF counts, or rely on the calibrated heuristic.
5. Cost cap behaviour: warn or block.
6. Keep hardware tiers as named presets, or retire them.

## Dependencies / interface points

- **Gating:** gated history replaces the raw ST history as the allocator's `history` input. Gating should hand over per-character messages newest-first, already filtered, and must not cut by count itself. Each message should carry a stable id for count caching.
- **Embeddings/RAG:** the retriever returns scored memories. The allocator takes them in score order and sets the effective top-k. The SRD's "Top 5" becomes a maximum.
- **Session-scoped lore and the GM_ACTION toggle:** lore rules arrive tagged invariant or trigger with a relevance score. The GM directive's size changes with the toggle and is counted as non-trimmable.
- **FIFO queue:** generation calls carry the computed `max_tokens`, which can feed queue time estimates. Tokenize calls bypass the queue. P4 summaries go through it.
- **Evennia vs FastAPI:** the size of the spatial/sensory text depends on which engine is used. The allocator only needs a string plus a "minimal" fallback (the room name).
