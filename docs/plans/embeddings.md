# Plan: real embedder (OPEN-002)

Status: planning only. Written 2026-10-03. Nothing in this plan has been implemented.

## Goal

Replace the random-vector stub with real embeddings so that episodic recall, lore triggers, imported chats and sleep-cycle core memories can actually be retrieved. It has to work on whichever single backend the user picked at setup (Lemonade on the Strix Halo, KoboldCpp on a 16 GB GPU, llama.cpp, or an OpenAI subscription) and degrade cleanly when no embedder is available. A random vector must never be stored again.

## Current state

| Fact | Evidence |
|---|---|
| The embedder is a stub. It seeds numpy with Python's salted `hash()`, so vectors are random and change on every restart. Empty text returns a zero vector, and a zero vector makes cosine distance NaN. | `scripts/onnx_embedder.py:24-40` |
| There are five separate `CPUEmbeddingEngine()` instances. | `proxy/api/routes.py:57`, `proxy/rag/lore_extractor.py:26`, `proxy/rag/import_worker.py:119`, `scripts/sleep_cycle.py:26` (injectable) |
| The chat path embeds only `user_text`, both for the query and when it stores the turn. The PRD (5.4) says the query should be the "last 3 turns". | `routes.py:324`, `routes.py:452`, `routes.py:560` |
| The import worker embeds `"role: content"` but stores only `content`. | `import_worker.py:221-238` |
| Lore triggers reuse the chat query vector with the same 0.35 cut-off. | `retriever.py:96-126`, `routes.py:339` |
| The schema is `VECTOR(3584)` on memory and lore tables. There is no vector index, and no record of which model produced a vector. | `scripts/init_db.sql:36`, `:110`, `:47-49` |
| Retrieval uses a hard `<=> < 0.35`, then decay scoring in Python. The 0.35 value has never been validated against any model. | `retriever.py:25`, `:48`, `:63-69` |
| Restoring a cold archive re-inserts the archived vector as-is. | `proxy/rag/tier_manager.py:390-409` |
| Embedding failures are swallowed. The turn then either isn't stored at all or is stored with the random vector. | `routes.py:466`, `lore_extractor.py:121` |
| The spec says 3584 dims with HNSW (impossible on `vector`), CPU ONNX offload, and "no memories to external cloud APIs" (SLA-2). | SRD 3.4.1, 3.4.5, 5.2; PRD 5.4, SLA-2 |

What is already in place: `onnxruntime` 1.29 and `pgvector` 0.5 (the Python package) are in `.venv`, and the embedding columns are nullable. `tokenizers` is not installed. Its abi3 wheel installs on Python 3.14; I checked by downloading `tokenizers-0.23.2-cp310-abi3`.

## Options considered

| Option | Who it serves | Verified facts | Concerns |
|---|---|---|---|
| **A. Backend `/v1/embeddings`** (OpenAI-compatible HTTP) | Lemonade, KoboldCpp, llama.cpp, Ollama, OpenAI | **Lemonade** has `/v1/embeddings` for `llamacpp` and `flm` recipes, and embedding models get their own LRU slot type separate from LLMs ([multi-model docs](https://lemonade-server.ai/docs/guide/configuration/multi-model/)). Its registry ships `nomic-embed-text-v1/v2`, `Qwen3-Embedding-0.6B/4B/8B-GGUF` (`/opt/share/lemonade-server/resources/server_models.json`). FLM lists `embed-gemma:300m` with label `embeddings` and a 2048-token context (`~/.cache/lemonade/bin/flm/npu/model_list.json`). **KoboldCpp** serves `/v1/embeddings` when started with `--embeddingsmodel <gguf>` ([wiki](https://github.com/LostRuins/koboldcpp/wiki)). **llama-server** needs `--embedding`, and its `/v1/embeddings` requires pooling other than `none` ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **OpenAI** `text-embedding-3-small` produces 1536 dims, reducible with `dimensions`. It costs about $0.02 per 1M tokens and accepts up to 2048 inputs per request (third-party pricing pages, **not checked on openai.com**). | One llama-server instance usually serves one model, and mean-pooled embeddings from a chat LLM are poor, so llama.cpp users need a second instance. KoboldCpp shares the GPU: issue #2069 reports extra VRAM being allocated. On Lemonade, user-pulled embedding models sometimes lack the `embeddings` label and return 501 (issue #1745, seen on 10.2; **unverified on 11.9**). OpenAI sends memories to a cloud service, which conflicts with SLA-2. |
| **B. Local CPU ONNX** | Everyone, including OpenAI-only users and machines with no GPU headroom | `onnx-community/embeddinggemma-300m-ONNX` comes in fp32, q8 and q4 (fp16 is unsupported). It outputs `sentence_embedding` with 768 dims (Matryoshka truncation to 512, 256 or 128 is possible) and accepts up to 2048 tokens. Query prompt: `task: search result \| query: …`. Document prompt: `title: none \| text: …`. Needs `onnxruntime` and `tokenizers`. | Adds a dependency and a model download of a few hundred MB (**size unverified**). CPU latency on Strix Halo and on a typical 16 GB-GPU desktop is **unmeasured**. Gemma licence terms apply. |
| **C. Sovereign NPU** (`embed-gemma-300m-FLM` via Lemonade) | Strix Halo | This is option A pointed at an FLM model. FLM can hold one LLM, one ASR and one embedding model on the NPU at once. whispercpp, FLM and ryzenai-llm are mutually exclusive on the NPU (Lemonade docs). | The local `npu_playbook.md` says NPU models share the `standard/llm` slot, which contradicts the per-type slots in the docs. `~/.config/lemonade/config.json` has `max_loaded_models: 2`, but the playbook says 1. **Whether the FLM embedder evicts the GPU LLM is unverified** and must be measured. Whether the model is downloaded and whether its output is 768 dims are also unverified. |
| **D. Non-vector fallback** (Postgres full-text search) | No embedder configured, or the embedder is down | Built into Postgres: a `tsvector` generated column with a GIN index and `ts_rank_cd`. | Lexical matching only. Use the `simple` config so it works for any language. Its scores are not comparable with cosine scores. |

**Recommendation.** Build a provider abstraction with three real implementations (`openai_compat`, `onnx_local`, `none`/FTS) and a deterministic `fake` for tests. Make **EmbeddingGemma-300m (768 dims)** the default model on every tier, because the same model is available on all of them: FLM on the NPU, GGUF for llama.cpp, KoboldCpp and Lemonade (`ggml-org/embeddinggemma-300M-GGUF`), and ONNX on the CPU. 768 dims is well under the HNSW limit. The default for each deployment:

- **SOVEREIGN** (Lemonade): `embed-gemma-300m-FLM` on the NPU, if the measurement shows it doesn't evict the chat LLM. Otherwise the same model via `onnx_local`.
- **PERFORMANCE** (KoboldCpp or llama.cpp, 16 GB): the backend's embeddings endpoint if one is configured, otherwise `onnx_local`.
- **OpenAI subscriber:** `onnx_local`, which is free, keeps memories local and has no per-request cost. `text-embedding-3-small` only as an explicit opt-in (`EMBEDDING_ALLOW_REMOTE`), because of SLA-2.
- **EXPERIMENTAL:** `onnx_local` with the q4 model, or `none` (FTS).

## Design

### Interfaces (`proxy/rag/embeddings/`)

```python
@dataclass(frozen=True)
class EmbeddingSpace:      # identity of a vector space
    id: int; provider: str; model: str; dim: int
    query_prefix: str; doc_prefix: str
    max_distance_memory: float; max_distance_lore: float; calibrated: bool

class Embedder(Protocol):
    space: EmbeddingSpace | None          # None => provider "none"
    async def embed_queries(self, texts: list[str]) -> list[list[float]]
    async def embed_documents(self, texts: list[str]) -> list[list[float]]
    async def health(self) -> EmbedderHealth
```

- `EmbeddingService` is a single process-wide instance, injected into the routes, lore extractor, import worker and sleep cycle. It replaces the five stub instances. It handles:
  - adding the query or document prefix
  - truncation to the model's maximum input
  - L2 normalisation
  - a dimension check: a wrong dimension raises, and nothing is stored
  - an LRU cache keyed by `(space_id, sha256(prefixed_text))`. Regenerations re-embed the same user text, so they hit the cache.
  - batching: 32 inputs locally, up to 256 over HTTP
  - a timeout: `EMBEDDING_TIMEOUT_S`, default 3 s on the chat path
  - a circuit breaker: after 3 failures the embedder is down for 30 s. This matches the FIFO plan.
- On failure it raises `EmbedderUnavailable`. It never returns a fallback vector, and empty text returns `None`.
- `scripts/onnx_embedder.py` is deleted. Leaving it in place would invite reuse.

### What gets embedded

| Use | Text embedded |
|---|---|
| Document (stored turn) | `sensory_input` plus `public_response`, truncated |
| Query | Last user message plus the previous assistant turn. This is narrower than the PRD's "last 3 turns", so the plan offers it as an owner choice. |
| Lore rule | `rule_text` |
| Import | `content`. The role prefix is dropped so the embedded text matches what is stored. |

### Config keys (`config/config.json`, via `SettingsManager`)

| Key | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `auto` | `auto`, `openai_compat`, `onnx_local` or `none`. `auto` is resolved once at setup or when the admin saves, never on a per-request basis. |
| `EMBEDDING_URL` / `EMBEDDING_API_KEY` | `BACKEND_LLM_URL` / `BACKEND_API_KEY` | Lets a llama.cpp user point at a second instance. |
| `EMBEDDING_MODEL` | from the tier profile | For example `embed-gemma-300m-FLM`, `embeddinggemma-300m`, or `text-embedding-3-small`. |
| `EMBEDDING_DIM` | model native | Sent as `dimensions` to OpenAI. Truncates Matryoshka models. |
| `EMBEDDING_ONNX_DIR` | `models/embeddinggemma-300m-onnx-q8` | Filled by a setup download step. |
| `EMBEDDING_ALLOW_REMOTE` | `false` | Refuses a non-loopback, non-LAN `EMBEDDING_URL` unless set (SLA-2). |
| `EMBEDDING_TIMEOUT_S` | `3` | |
| `RAG_MAX_DISTANCE_OVERRIDE` / `LORE_MAX_DISTANCE_OVERRIDE` | unset | Otherwise the calibrated per-space values are used. |

`auto` resolution:
1. Look for a model labelled for embeddings in the backend's `/v1/models`.
2. Run a one-text probe against `/v1/embeddings` and check the dimension.
3. Otherwise use `onnx_local`, if the model files are present.
4. Otherwise use `none`.

The admin UI gets an "Embeddings" card with the provider, model, dimension, health, calibration status, re-embed progress and a "Re-embed now" button. The button must sit behind the OPEN-001 auth.

### Schema and migration

- **New table `spm_embedding_spaces`:** `id SMALLSERIAL`, `provider`, `model`, `dim`, prefixes, thresholds, `calibrated BOOL`, `active BOOL` (with a partial unique index on active), `created_at`.
- **Memory and lore tables:**
  - The column type changes to an **untyped** `vector`. pgvector allows mixed dimensions in one column and indexes them with an expression plus a partial index ([pgvector FAQ](https://github.com/pgvector/pgvector)).
  - New column `embedding_space_id SMALLINT NULL`.
  - Memory tables get `fts tsvector GENERATED ALWAYS AS (to_tsvector('simple', sensory_input || ' ' || coalesce(public_response,''))) STORED` with a GIN index.
  - All of this is added in `create_csa_memory_table()` and `create_csa_lore_rules_table()` with `IF NOT EXISTS`. These run on every access, so old tables upgrade lazily, and the test DB gets it from `init_db.sql`.
- **Queries** filter on `embedding_space_id = $active`, so vectors from different models are never compared. Matching dimensions does not make two models' spaces compatible.
- **Index:** exact scan by default. These tables hold thousands of rows per character, and an exact scan is deterministic, as the SRD wants. Add `CREATE INDEX … USING hnsw ((episodic_embedding::halfvec(768)) halfvec_cosine_ops) WHERE embedding_space_id = N` only when a table passes `HNSW_MIN_ROWS` (default 20k). pgvector caps HNSW at 2000 dims for `vector` and 4000 for `halfvec`. The `session_id` filter needs `hnsw.iterative_scan`, which requires pgvector 0.8 or later. **The container's extension version is unverified:** check it with `SELECT extversion FROM pg_extension WHERE extname='vector'`.
- **Migration of existing rows:** every existing vector is random, so `scripts/migrate_embeddings.py` (with `--dry-run`, and refusing to run unless a recent backup exists) sets `episodic_embedding` and `rule_embedding` to NULL and `embedding_space_id` to NULL. A **backfill worker** then re-embeds every row whose `embedding_space_id IS DISTINCT FROM active` from its stored text, in batches, resumably, at background priority. The same worker handles later model switches: switching creates a new space row and backfills while the old space keeps serving, then flips `active`. Cold-archive restore drops archived vectors whose space doesn't match and lets the backfill re-embed them. Do OPEN-012 (the live DB cleanup) first.

### Failure behaviour

| Case | Behaviour |
|---|---|
| Query embedding fails | Retrieve through FTS for this turn. Log it, and emit a telemetry flag so the dashboard shows "degraded recall". Never block the chat. |
| Storing a turn, a lore rule, an import or a sleep summary when the embedder is down | Store the row with a NULL embedding; the backfill fills it later. This also fixes the sleep cycle, which currently aborts the transaction if embedding raises. |
| Provider returns the wrong dimension, or NaN | Hard error. The row is stored un-embedded. |
| Provider is `none` | FTS only, and lore triggers fall back to FTS matching. Invariants are unaffected. |

## Phases

| # | Phase | Size | Acceptance criteria and tests |
|---|---|---|---|
| 0 | **Stop storing garbage.** The stub is replaced by provider `none`. NULL embeddings are written, and vector retrieval is skipped while no space is active. | 0.5 d | A test asserts that no module imports `numpy.random` for embeddings. Turns persist with NULL vectors, and retrieval returns `[]` without error. The full suite stays green. |
| 1 | **Abstraction, `openai_compat` and `fake`.** Single injected service, cache, batching, timeouts, breaker, config keys, `auto` probe. | 2 d | Contract tests with an `httpx.MockTransport` cover OpenAI response shape, 4xx/5xx/timeouts, the dimension mismatch and the remote-URL guard. An autouse fixture in `conftest.py` forces `fake`; network providers raise under pytest unless `RUN_LIVE_EMBED_TESTS=1`. |
| 2 | **Schema, spaces, migration and backfill.** | 2 d | Run against `spm_test`: a table with the old `VECTOR(3584)` upgrades in place; the migration is idempotent and `--dry-run` writes nothing; the backfill fills NULL rows, survives an interruption and never mixes spaces; the archive round-trip covers a space mismatch. |
| 3 | **FTS fallback and retrieval routing.** Vector, FTS when degraded or `none`, and FTS lore triggers. | 1 d | FTS returns lexical matches, and the decay score uses a normalised rank. A test with the embedder down still gets memories. |
| 4 | **`onnx_local` (EmbeddingGemma ONNX q8).** Setup download step and model profiles. | 1.5 d | Unit tests run on a tiny ONNX fixture or are skipped without model files. The live opt-in test checks that "the sword is cursed" is closer to "my blade carries a curse" than to "the tavern is warm". |
| 5 | **Calibration and the Sovereign NPU.** `scripts/calibrate_embeddings.py` runs about 60 labelled roleplay query/memory pairs (fixtures) and reports distance distributions and a suggested threshold per space. On the Strix Halo, measure FLM latency and whether loading it evicts the chat LLM. | 1 d plus owner review | Thresholds are stored per space with `calibrated=true`. Lore triggers get their own threshold. The results are written into this plan. |
| 6 | **Admin UI card, setup selection, optional HNSW.** | 1 d | The card shows health and backfill progress. HNSW is created above `HNSW_MIN_ROWS`, and `EXPLAIN` shows it in use with the space filter. |

Total: about 9 developer days. Phases 0 to 3 already give working, honest recall with any OpenAI-compatible embedder.

The `fake` embedder hashes word tokens (sha256, not `hash()`) into buckets and normalises the result. It is deterministic and gives a meaningful similarity, so texts that share words end up close, and retrieval tests can assert ranking.

## Risks and open questions for the owner

1. **SLA-2 vs the OpenAI tier.** Should remote embeddings be allowed as an opt-in (`EMBEDDING_ALLOW_REMOTE`), or never? Recommended: opt-in, with `onnx_local` as the default for OpenAI users.
2. **Canonical model.** EmbeddingGemma-300m (768) on every tier is recommended. The alternative is Qwen3-Embedding-0.6B (1024) on Lemonade, which needs instruction prompts and ships with Lemonade. Note that Qwen3-Embedding-8B outputs 4096 dims, above even the halfvec HNSW limit. The `ggml-org` GGUF, the FLM build and the ONNX build of EmbeddingGemma are quantised differently and **are not guaranteed to give interchangeable vectors**. By default each is its own space; merging them needs a calibration check (mean cross-runtime cosine above 0.98).
3. **NPU for embeddings.** Approve the NPU as the embedding device (it is the open "NPU use case"), pending the eviction measurement. Also resolve whether `max_loaded_models` should be 1 or 2: the playbook says 1, the config file says 2.
4. **Spec amendments.**
   - The fixed `< 0.35` becomes a per-model calibrated threshold. Expect it to be looser for most modern models (**unmeasured**).
   - `VECTOR(3584)` becomes the model's native dimension.
   - HNSW becomes optional above a row count.
   - "Decay SQL" stays in Python, unchanged from today.
5. **Text composition.** Should stored turns embed the user text plus the reply, and should the query use the last user message plus one turn, or the PRD's last 3 turns?
6. **Existing live vectors.** Discard them and re-embed from text, as recommended. This depends on the OPEN-012 cleanup and a backup.
7. **New dependencies.** `tokenizers`, an optional `huggingface_hub` for the download step, and a model download of a few hundred MB.

## Dependencies on sibling workstreams

- **FIFO queue** (`docs/plans/fifo_queue.md`):
  - Backend embedding calls go through `ScheduledLLMClient` on the `embed` lane: P1 on the chat path, P3 for backfill and import.
  - This plan supplies the per-`backend_profile` fact that decides whether the lane is shared with the LLM. Lemonade: not shared, since it has its own slot type (NPU to be confirmed). KoboldCpp and single-instance llama.cpp: shared.
  - `onnx_local` bypasses the scheduler and runs in a thread pool.
  - The breaker settings are aligned with that plan (3 failures, 30 s).
- **Gating (OPEN-003):** gating decides which text is "perceived". Documents should embed the gated sensory text, not the raw ST message. The ambient-log commit will need `embed_documents`.
- **GM_ACTION and session-scoped lore (OPEN-005/007):** both plans alter `csa_lore_rules_*`, one adding `session_id` and the other `embedding_space_id`. Do it in one combined `create_csa_lore_rules_table()` change and one migration.
- **Token budget** (`docs/plans/token_budget.md`): the allocator consumes retriever output in score order. Share the tokenizer or counting utility for truncating to the embedder's 2048-token maximum input.
- **Evennia vs FastAPI:** no direct coupling, since the world log is not embedded.
- **Auth (OPEN-001):** the re-embed, provider-switch and calibration admin endpoints must be authenticated.
