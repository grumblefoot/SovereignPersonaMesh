# FR-002 Handoff: Bulk Chat Import & Async Bootstrapping

## Objectives
Implement automatic bulk chat import detection (> 10 historical messages for a new `session_id`), non-blocking live chat generation (< 5 ms routing overhead), and an asynchronous background worker (`proxy/rag/import_worker.py`) with dynamic hardware resource allocation.

### Requirements
1. **Database Schema (`scripts/init_db.sql`)**:
   - Create table `spm_chat_imports`:
     `import_id UUID PRIMARY KEY`, `session_id VARCHAR(255) UNIQUE`, `character_id VARCHAR(255)`, `status VARCHAR(50)`, `total_messages INT`, `processed_messages INT`, `error_log TEXT`, `created_at TIMESTAMP`, `updated_at TIMESTAMP`.
2. **Background Import Worker (`proxy/rag/import_worker.py`)**:
   - Dynamic batch size allocator inspecting `config/hardware_tiers.py` and CPU cores/memory.
   - `process_bulk_import_background(session_id, character_id, messages, db_pool)` function:
     - Record `status = 'processing'`.
     - Process message pairs in dynamic batches.
     - Vectorize embeddings via ONNX CPU embedder and insert into `csa_memory_{character_id}`.
     - Update status to `'completed'` or `'failed'` with error logs.
3. **SPM Proxy Interceptor (`proxy/api/routes.py`)**:
   - Check `len(request.messages) > 10` for new `session_id`.
   - Register job in `spm_chat_imports` and spawn `asyncio.create_task()`.
   - Complete routing setup in < 5 ms for live response.
4. **Unit Tests (`tests/test_fr002_bulk_import.py`)**:
   - Create tests for detection speed (< 5 ms), worker memory population, dynamic resource scaling, and failure resilience.
   - Run `pytest tests/ -v` (all tests passing).

### Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Target
- Branch: `main`
