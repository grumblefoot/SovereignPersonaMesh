# FR-001 Handoff: Session-Bound Context & Memory Isolation

## Objectives
Implement strict, zero-bleed session-bound memory and spatial isolation (`session_id`) across PostgreSQL database tables, RAG retrieval queries, Evennia spatial room state, and SPM Proxy routing. Every chat session must start 100% fresh for a character.

### Requirements
1. **Database & RAG**:
   - Ensure DDL in `scripts/init_db.sql` includes `session_id VARCHAR(255) NOT NULL DEFAULT 'default_session'` and index `idx_csa_memory_{char_id}_session_timestamp`.
   - Update `proxy/rag/retriever.py` `retrieve_memories()` to filter strictly by `WHERE session_id = $1`.
   - Update `proxy/core/sensory_filter.py` `evaluate_and_bypass()` to persist `session_id`.
2. **Evennia Liaison (Port 4005)**:
   - Update `evennia_world/app.py` state maps to session-keyed dictionary: `session_worlds: Dict[str, Dict[str, RoomMetadata]]`.
   - Update `/api/v1/world/action`, `/api/v1/world/state`, `/api/v1/world/characters`, `/api/v1/world/move`, `/api/v1/world/configure` to support `session_id`.
3. **SPM Proxy (Port 5050)**:
   - Update `proxy/api/routes.py` with `_extract_session_id()` precedence: `X-Session-ID` header → body `session_id` → `"default_session"`.
4. **Unit Tests**:
   - Create `tests/test_fr001_session_isolation.py` and run full pytest suite (`pytest tests/ -v`).

### Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Status
- Target branch: `main`
