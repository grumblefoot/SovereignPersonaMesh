# Phase 2 Handoff: Evennia World State Liaison Service

## Current State
- **Phase 0** ✅: Workspace, git repo, stubs, config, tests, playbook by Antigravity
- **Phase 1** ✅: DB & Container Infrastructure by Hermes
  - `litellm_postgres` container running on port 5432 with pgvector 0.5.0
  - 4 character tables created: `csa_memory_rowan`, `csa_memory_domino`, `csa_memory_luna`, `csa_memory_seamus`
  - HNSW index incompatible with 3584-dim vectors → B-tree index on `(is_core_memory, timestamp)`, brute-force cosine search
  - `litellm_postgres` DB created on existing `spm-postgres` container
  - User `spm_user` password: `spm_secure_password`
  - HBA trust configured for loopback
  - All `__init__.py` files created across packages
  - `pytest.ini` with `asyncio_mode = auto`
  - **13/13 tests passing** (9 new Phase 1 retriever tests + 4 pre-existing spatial/parser tests)
  - Commit: `ec4e46c`

## Phase 2: Evennia World State Liaison Service (Port 4005)
Playbook checklist:
- [ ] Complete `evennia_world/app.py` REST service integration
- [ ] Verify `POST /api/v1/world/action` response structure against SRD payload schema
- [ ] Validate session tick locking (`evennia_world/session_lock.py`) under concurrent requests
- [ ] Expand room templates in `evennia_world/hybrid_builder.py`

### Existing Code (already written, needs completion/verification)

**`evennia_world/app.py`** — FastAPI server (Port 4005):
- `POST /api/v1/world/action` — Evaluates physical intentions (speak/whisper/move/manipulate), returns consequences with gating levels and sensory feeds
- `GET /api/v1/world/state?character_id=&session_id=` — Returns character room metadata, distances, gating
- `POST /api/v1/world/lock` — Session/tick lock management (acquire/release)
- Uses `HybridWorldBuilder` to instantiate world templates, `SpatialConstraintsMatrix` for gating evaluation
- Currently hardcodes "cellar" world with cellar_chars and upstairs tavern for Seamus

**`evennia_world/session_lock.py`** — `SessionLockManager` class:
- `acquire_lock(session_id)` → returns UUID token
- `release_lock(session_id, lock_token)` → boolean success
- Uses `asyncio.Lock` per session, stores tokens for verification
- Missing: concurrent request validation, lock expiry/ttl

**`evennia_world/hybrid_builder.py`** — `HybridWorldBuilder` class:
- `PREDEFINED_ROOM_TEMPLATES` dict with `dungeon_cellar` (cellar + tavern_upstairs) and `default_meeting_room`
- `match_template(keywords)` → template key string
- `instantiate_world(template_key)` → dict of RoomMetadata objects
- Needs: more room templates, better matching logic, dynamic character management

**`evennia_world/models.py`** — Pydantic models (Enum types + data classes):
- `ActionType` (SPEAK, WHISPER, MOVE, MANIPULATE)
- `GatingLevel` (DIRECT, DEGRADED, BLACKOUT)
- `BarrierType` (NONE, OPEN_DOOR, CLOSED_DOOR, DRYWALL, SOLID_WALL, METAL_PARTITION)
- `ActionPayload`, `SensoryConsequence`, `ActionResponse`, `WorldStateQuery`, `RoomMetadata`, `CharacterWorldState`, `SessionLockPayload`

**`evennia_world/spatial_matrix.py`** — `SpatialConstraintsMatrix.evaluate_sensory_feed()`:
- Distance rules: 0-5ft DIRECT, 5-15ft DEGRADED, >15ft BLACKOUT
- Whisper rules: ≤5ft DIRECT if target, ≤15ft DEGRADED, otherwise BLACKOUT
- Solid wall / metal partition = always BLACKOUT
- Already tested (3 tests pass)

### Key Architecture Notes
- **Port 4005**: Evennia REST service, standalone FastAPI
- **Port 5050**: SPM Proxy (consumes Evennia for spatial gating)
- **Port 13305**: Lemonade/llama-server (LLM backend)
- Zero-LLM Core Rule: Proxy routing, state evaluation, spatial gating are ALL Python code (no LLM calls)
- Spatial matrix is deterministic — no ML/LLM involved
- `proxy/backend_client/evennia_client.py` calls `http://localhost:4005/api/v1/world/action`

### Files to Modify/Create for Phase 2
1. `evennia_world/app.py` — Add tests, verify response schema, possibly add endpoints
2. `evennia_world/session_lock.py` — Add lock expiry/ttl, concurrent request guards
3. `evennia_world/hybrid_builder.py` — Expand room templates, improve matching
4. `tests/test_spatial_matrix.py` — Already exists (3 tests pass)
5. **New test file**: `tests/test_world_liaison.py` — Integration tests for app.py endpoints

### Running Tests
```bash
cd /home/osmon/Desktop/Experiments/SillyTavern/SovereignPersistanceMesh
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Playbook Location
`/home/osmon/Desktop/Experiments/SillyTavern/SovereignPersistanceMesh/playbook.md`
(Update with Phase 2 progress after completion)

### Git Status
- Branch: `main`
- Latest commit: `ec4e46c` (Phase 1 complete)
- Pushed to GitHub

---
**Next step**: Complete Phase 2 tasks, write tests, run pytest, update playbook, commit and push.
**Checkpoint phase**: Phase 3 (Proxy & Stream Parser) and Phase 6 (Sleep Cycle) require explicit antigravity review.
