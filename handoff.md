# Phase 3 Handoff: SPM FastAPI Proxy & Stream Parser (Port 5050)

## Current Progress & Status
- **Phase 0** ✅: Workspace, git repo, stubs, config, tests, playbook by Antigravity.
- **Phase 1** ✅: DB & Container Infrastructure by Hermes (`13/13` tests passing, commit `ec4e46c`).
- **Phase 2** ✅: Evennia World State Liaison Service by Hermes & Antigravity (`67/67` total unit tests passing, commit `2d92f03`).
  - Implemented REST API on Port 4005 (`/api/v1/world/action`, `/api/v1/world/state`, `/api/v1/world/lock`, `/api/v1/world/characters`, `/api/v1/world/configure`, `/api/v1/world/templates`).
  - Added 60s TTL lock manager with auto-stale cleanup and `LockError` guards in `evennia_world/session_lock.py`.
  - Expanded `evennia_world/hybrid_builder.py` with 4 room network templates (`dungeon_cellar`, `forest_camp`, `castle_exterior`, `tavern_common`), keyword scoring matcher, and dynamic character APIs.
  - Created 647 lines of unit tests in `tests/test_world_liaison.py`.

---

## Phase 3 Goals: SPM FastAPI Proxy & Stream Parser (Port 5050)
Playbook checklist:
- [ ] Complete `proxy/api/routes.py` OpenAI-compatible `/v1/chat/completions` pipeline.
- [ ] Validate `MonologueStreamParser` token state machine under streaming responses.
- [ ] Verify Monologue Parser Fail-Safe Passthrough (>500 tokens / malformed tags / unexpected EOS auto-close).
- [ ] Write comprehensive unit tests in `tests/test_proxy_routes.py`.
- [ ] Run full `pytest` suite.

### Existing Code Inventory & Target Files
- **`proxy/main.py`**: FastAPI entry point on Port 5050.
- **`proxy/api/routes.py`**: OpenAI emulation routes (`/v1/models`, `/v1/chat/completions`).
- **`proxy/core/fifo_queue.py`**: `InferenceFIFOQueue` for sequential turn execution.
- **`proxy/core/stream_parser.py`**: `MonologueStreamParser` for two-state monologue token stripping.
- **`proxy/core/sensory_filter.py`**: `ObserverInferenceGatingFilter` for Null/Blackout bypass.
- **`proxy/rag/prompt_builder.py`**: `CognitivePromptBuilder` for 32K token budget assembly.
- **`proxy/backend_client/lemonade_client.py`**: `LemonadeLLMClient` (Port 13305).
- **`proxy/backend_client/evennia_client.py`**: `EvenniaWorldClient` (Port 4005).

### Test Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Status
- Branch: `main`
- Latest commit: `65f767d` (Phase 2 complete)
- Tracking `origin/main` on GitHub
