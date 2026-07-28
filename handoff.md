# Phase 8 Handoff: End-to-End SPM Integration Test Harness (Seraphina & SillyTavern)

## Goal & Objectives
Build a complete end-to-end integration test suite that tests **SPM Proxy (Port 5050)**, **Evennia Liaison (Port 4005)**, **Lemonade LLM Server (Port 13305)**, **PostgreSQL (`litellm_postgres`)**, and **SillyTavern's default character `Seraphina`**.

### Required Files
1. `tests/test_e2e_integration.py`
   - Test 1: Service readiness probes for Port 4005 (Evennia), Port 5050 (Proxy), Port 13305 (Lemonade), and Port 5432 (Postgres).
   - Test 2: Character registration (`Seraphina`) in Evennia `dungeon_cellar` room.
   - Test 3: OpenAI `/v1/chat/completions` request simulating SillyTavern payload for Seraphina.
   - Test 4: Real-time monologue stripping (`<ctrl94>`) verification in SSE stream.
   - Test 5: PostgreSQL verification: query `csa_memory_seraphina` to verify turn persistence.
   - Test 6: RAG memory retrieval injection on follow-up turn.
   - Test 7: Zero-inference blackout bypass verification when character is distant.
2. `scripts/run_e2e_test.py`
   - Script that ensures background microservices (Evennia app on 4005, SPM proxy on 5050) are started, runs `pytest tests/test_e2e_integration.py`, captures SLA metrics, and outputs formatted findings.

### Virtual Environment & Test Execution
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/test_e2e_integration.py -v
python scripts/run_e2e_test.py
```

### Git Status
- Branch: `main`
- Target remote: `origin/main`
