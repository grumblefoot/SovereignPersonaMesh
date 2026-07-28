# Sovereign Persona Mesh (SPM) Implementation Playbook

> **Author**: Antigravity (Senior Code Architect & Auditor)  
> **Executor**: Hermes Agent (Autonomous Software Engineering Agent)  
> **Target Environment**: AMD Ryzen AI Max 395 (Strix Halo APU), Fedora 44 Desktop, Rootless Podman  
> **Primary Models**: `google/gemma-4-26B-A4B-it` (Inference on Port 13305) & `google/gemma-4-9B-it` (WSD & Consolidation)  

---

## 1. Executive Overview & Architecture Alignment

The **Sovereign Persona Mesh (SPM)** is an edge-computing, multi-agent roleplay orchestration system designed to eliminate "omniscient character" syndrome. Traditional group chat frontends share the entire context window across all characters. SPM introduces:

1. **Sensory Gating (The Objective Frame)**: Characters only perceive events within physical proximity and line of sight/hearing.
2. **Episodic Decoupling (The Memory Barrier)**: Character memories are committed to isolated PostgreSQL vector spaces (`csa_memory_{character_id}`).
3. **Private Inner Monologue**: Two-state token parsing state machine (`<ctrl94>thoughts` vs `</ctrl94>` public response).
4. **Zero-LLM Core Rule**: Proxy routing, state evaluation, tag parsing, and spatial gating are strictly executed in Python code.

---

## 2. Directory Structure & Workspace Map

```
/home/osmon/Desktop/Experiments/SillyTavern/
├── SillyTavern/                  # Cloned SillyTavern frontend repository
├── evennia/                      # Cloned Evennia text-game engine repository
└── SovereignPersistanceMesh/     # Core SPM project workspace
    ├── config/
    │   ├── Containerfile             # Rootless Podman container manifest
    │   ├── docker-compose.yml        # PostgreSQL + pgvector (litellm_postgres) service
    │   └── hardware_tiers.py         # Sovereign (96GB GTT), Performance, & Experimental configs
    ├── proxy/
    │   ├── main.py                   # FastAPI Proxy entry point (Port 5050)
    │   ├── api/
    │   │   └── routes.py             # OpenAI-compatible /v1/chat/completions endpoint
    │   ├── core/
    │   │   ├── fifo_queue.py         # Asyncio FIFO request queue (100% GPU safety margin)
    │   │   ├── stream_parser.py      # Two-state monologue parser with fail-safe passthrough
    │   │   └── sensory_filter.py     # Observer Inference Gating & Bypass Protocol
    │   ├── rag/
    │   │   ├── prompt_builder.py     # 32,768 Token Budget Partitioning Matrix
    │   │   └── retriever.py          # pgvector search & Game AI decay scoring engine
    │   └── backend_client/
    │       ├── lemonade_client.py    # Async SSE client for Lemonade Server (Port 13305)
    │       └── evennia_client.py     # Async REST client for Evennia (Port 4005)
    ├── evennia_world/
    │   ├── app.py                    # Evennia REST API service (Port 4005)
    │   ├── models.py                 # Pydantic & Django ORM data models
    │   ├── spatial_matrix.py         # Deterministic Spatial & Acoustic Constraints Matrix
    │   ├── session_lock.py           # Tick & Session Lock manager
    │   └── hybrid_builder.py         # Hybrid Semantic-Template World Builder
    ├── scripts/
    │   ├── init_db.sql               # Database schema & pgvector initialization DDL
    │   ├── sleep_cycle.py            # Daily 3:00 AM memory consolidation worker
    │   ├── onnx_embedder.py          # CPU offloaded embedding worker (AVX-512)
    │   └── setup_systemd_timer.sh    # Systemd timer installer script
    ├── tests/
    │   ├── test_spatial_matrix.py    # Unit tests for spatial gating rules
    │   └── test_stream_parser.py     # Unit tests for monologue token stream parser
    ├── README.md
    ├── playbook.md                   # Master engineering playbook
    ├── requirements.txt
    └── .env.example
```

---

## 3. Implementation Phasing & Task Status Checklist

- [x] **Phase 0: Workspace Architecture & Stubbing** *(Completed by Antigravity)*
  - [x] Git repository initialization on `main` branch.
  - [x] Directory structure setup with typed stub classes.
  - [x] Configuration manifests, `.env.example`, and `requirements.txt`.
  - [x] Master `playbook.md` creation.

- [x] **Phase 1: Database & Container Infrastructure** *(Completed by Hermes)*
  - [x] Spin up `litellm_postgres` container (reused existing `spm-postgres` container, created `litellm_postgres` DB).
  - [x] Execute `scripts/init_db.sql` with pgvector extension and `create_csa_memory_table()` helper function.
  - [x] Fix HNSW index incompatibility: pgvector HNSW/IVFFlat capped at 2000 dims; switched to B-tree index on `(is_core_memory, timestamp)` with brute-force cosine search for 3584-dim vectors.
  - [x] Fix asyncpg authentication: set user password, granted DB privileges, HBA trust on loopback.
  - [x] Verify `asyncpg` connection pool initialization in `proxy/rag/retriever.py` (100% of retriever tests pass).
  - [x] All 4 demo character tables created: `csa_memory_rowan`, `csa_memory_domino`, `csa_memory_luna`, `csa_memory_seamus`.
  - [x] 9 new Phase 1-specific tests written and passing + 4 pre-existing tests passing = **13/13 passed**.
  - [x] Created missing `__init__.py` package files across all Python packages.

- [x] **Phase 2: Evennia World State Liaison Service (Port 4005)** *(Completed by Hermes)*
  - [x] Completed `evennia_world/app.py` REST service integration: health endpoint, action evaluation, world state query, lock management, character management, world configuration, template listing.
  - [x] Verified `POST /api/v1/world/action` response structure against SBD payload schema (consequences with gating levels, sensory feeds, distances).
  - [x] Validated session tick locking (`evennia_world/session_lock.py`) under concurrent requests: TTL expiry, stale lock auto-cleanup, concurrent request guards (409 on double-acquire), token verification.
  - [x] Expanded room templates in `evennia_world/hybrid_builder.py`: dungeon_cellar (2 rooms), forest_camp (4 rooms), castle_exterior (6 rooms), tavern_common (4 rooms). Score-based keyword matching, dynamic character/room management, deep-copy isolation.
  - [x] 54 new unit tests in `tests/test_world_liaison.py`: session lock (10), hybrid builder (23), app endpoints (21), models (4), integration (1).
  - [x] **67/67 tests passing** (13 pre-existing + 54 new Phase 2 tests).
  - [x] Fixed stale asyncio.Lock issue in session_lock._cleanup_session to prevent deadlock on TTL expiry.

- [x] **Phase 3: SPM FastAPI Proxy & Stream Parser (Port 5050)** *(Completed by Hermes & Antigravity)*
  - [x] Implement full OpenAI-compatible `/v1/chat/completions` request pipeline in `proxy/api/routes.py`.
  - [x] Validate `MonologueStreamParser` token state machine under real streaming responses from Lemonade Server.
  - [x] Verify Monologue Parser Fail-Safe Passthrough (>500 tokens / malformed tags / unexpected EOS auto-close).
  - [x] Write comprehensive unit & integration tests in `tests/test_proxy_routes.py` (74/74 total unit tests passing).

- [x] **Phase 4: Observer Inference Gating & Bypass Protocol** *(Completed by Hermes & Antigravity)*
  - [x] Test `ObserverInferenceGatingFilter.evaluate_and_bypass()` with asyncpg pool.
  - [x] Verify zero-inference ambient log commits for characters in `Blackout` or `Null` sensory state (78/78 total unit tests passing).

- [ ] **Phase 5: RAG Search & Game AI Decay Engine** *(Assigned to Hermes)*
  - [ ] Verify cosine distance search using pgvector `<=>` operator (< 0.35 limit).
  - [ ] Implement mathematical scoring algorithm:
    $$\text{RAG Score} = (1 - \text{cosine\_distance}) \times e^{-\lambda \Delta t} \times \left(1 + \frac{\text{importance}}{10}\right) \times \text{access\_multiplier}$$
  - [ ] Verify CPU-offloaded embedding generation via `scripts/onnx_embedder.py`.

- [ ] **Phase 6: Nightly Sleep Cycle Consolidation (3:00 AM)** *(Assigned to Hermes)*
  - [ ] Connect `scripts/sleep_cycle.py` to `google/gemma-4-9B-it` backend.
  - [ ] Validate first-person single-sentence core memory synthesis.
  - [ ] Verify volatile log pruning while enforcing `is_core_memory = TRUE` retention rules.
  - [ ] Execute `scripts/setup_systemd_timer.sh` and verify user timer status (`systemctl --user status spm-sleep-cycle.timer`).

- [ ] **Phase 7: End-to-End Verification & SLA Benchmarking** *(Joint Audit: Hermes & Antigravity)*
  - [ ] Run full `pytest` suite across all components.
  - [ ] Test SillyTavern integration pointing to `http://localhost:5050/v1`.
  - [ ] Benchmark proxy processing overhead (< 150ms) and TTFT (< 1.8s at 32K context).

---

## 4. Guidelines for Hermes Agent Execution

1. **Adhere to the Zero-LLM Core Rule**: Do not insert LLM calls inside proxy routing, state evaluation, tag parsing, or spatial gating. Keep core routing 100% Python regex/heuristic driven.
2. **Follow Existing Component Contracts**: Ensure function signatures and Pydantic schemas in `proxy/` and `evennia_world/` remain consistent with PRD/SRD specifications.
3. **Execute Test Verification**: Run `pytest tests/` after completing each phase.
4. **Checkpoint Triggers**: Pause and request Antigravity audit upon completing Phase 3 (Proxy & Stream Parser) and Phase 6 (Sleep Cycle & Systemd Timer).

---

## 5. Senior Auditor Checkpoint Log

| Date | Phase | Status | Auditor Notes |
|---|---|---|---|
| 2026-07-28 | Phase 0 | APPROVED | Initialized workspace, git repo, stubs, tests, and playbook. Prepared for Hermes Agent buildout. |
| 2026-07-28 | Phase 1 | COMPLETE | Hermes built out DB infra: pgvector 3584-dim vectors require brute-force cosine search (HNSW capped at 2000 dims). B-tree index on (is_core_memory, timestamp). Reused existing spm-postgres container. 13/13 tests pass. |
| 2026-07-28 | Phase 2 | COMPLETE | Hermes & Antigravity built out Evennia Liaison API: REST endpoints (Port 4005), 60s TTL lock manager with LockError guards & auto-cleanup, score-based template matcher with 4 room networks, dynamic character placement. 67/67 unit tests pass. |
| 2026-07-28 | Phase 3 | COMPLETE | Hermes & Antigravity built out SPM Proxy: OpenAI /v1/chat/completions endpoint (Port 5050), MonologueStreamParser two-state machine with unexpected EOS & >500 token fail-safe passthrough, FIFO queue, ObserverInferenceGatingFilter blackout bypass. 74/74 unit tests pass. |
| 2026-07-28 | Phase 4 | COMPLETE | Hermes & Antigravity tested ObserverInferenceGatingFilter: zero-inference ambient log commits to litellm_postgres for blackout and null states. 78/78 unit tests pass. |
