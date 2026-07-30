# FR-003 Handoff: Tiered Data Lifecycle & Cold Storage Reconstitution

## Objectives
Implement a 3-tiered data lifecycle (Hot vector storage, Warm core memories, and Cold `.jsonl.gz` archives) with 100% Core Memory Immunity (`is_core_memory = TRUE`), seamless reconstitution pipelines, and lifecycle REST endpoints.

### Requirements
1. **Database Schema (`scripts/init_db.sql`)**:
   - Table `spm_cold_archives`:
     `archive_id UUID PRIMARY KEY`, `session_id VARCHAR(255)`, `character_id VARCHAR(255)`, `archive_path TEXT`, `record_count INT`, `created_at TIMESTAMP`.
2. **Memory Tier Manager (`proxy/rag/tier_manager.py`)**:
   - Class `MemoryTierManager`:
     - `archive_old_memories(character_id, session_id, max_records=500, max_age_days=30)`: Export volatile records to `storage/cold_archives/{character_id}/{session_id}_{timestamp}.jsonl.gz`, register archive, delete volatile rows (`is_core_memory = FALSE`).
     - `reconstitute_cold_archive(session_id, character_id)`: Decompress `.jsonl.gz`, re-insert/index vectors into `csa_memory_{character_id}`, remove archive record and file.
     - `get_tier_stats(character_id, session_id)`: Return hot, warm, cold record counts.
3. **SPM Proxy Endpoints (`proxy/api/routes.py`)**:
   - `POST /v1/memories/archive`
   - `POST /v1/memories/reconstitute`
   - `GET /v1/memories/stats`
4. **Unit Tests (`tests/test_fr003_tiered_lifecycle.py`)**:
   - Test core memory immunity (`is_core_memory = TRUE` never archived).
   - Test gzip compression & schema tracking.
   - Test full reconstitution pipeline.
   - Test stats endpoint.
   - Run `pytest tests/ -v` (100% passing).

### Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Target
- Branch: `main`
