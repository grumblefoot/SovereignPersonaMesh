# Phase 4 & Phase 5 Handoff: Observer Gating & RAG Decay Engine

## Current Progress & Status
- **Phase 0** ✅: Workspace, git repo, stubs, config, tests, playbook by Antigravity.
- **Phase 1** ✅: DB & Container Infrastructure by Hermes (`13/13` tests passing, commit `ec4e46c`).
- **Phase 2** ✅: Evennia World State Liaison Service by Hermes & Antigravity (`67/67` total unit tests passing, commit `2d92f03`).
- **Phase 3** ✅: SPM FastAPI Proxy & Stream Parser by Hermes & Antigravity (`74/74` total unit tests passing, commit `2ae76a8`).
  - Implemented `/v1/models` and `/v1/chat/completions` OpenAI emulation endpoints in `proxy/api/routes.py`.
  - Validated `MonologueStreamParser` token state machine with Fail-Safe Passthrough (>500 tokens / unexpected EOS).
  - Validated `InferenceFIFOQueue` sequential turn dispatch.
  - Validated `ObserverInferenceGatingFilter` zero-inference blackout bypass.
  - Created 7 integration tests in `tests/test_proxy_routes.py`.

---

## Phase 4 & 5 Goals: Observer Gating, RAG Search & Game AI Decay Engine
Playbook checklist:
- [ ] Test `ObserverInferenceGatingFilter.evaluate_and_bypass()` with asyncpg pool.
- [ ] Verify zero-inference ambient log commits to `csa_memory_{id}` for characters in `Blackout` or `Null` sensory state.
- [ ] Verify cosine distance search using pgvector `<=>` operator (< 0.35 threshold).
- [ ] Verify mathematical decay scoring algorithm in `proxy/rag/retriever.py`:
  $$\text{RAG Score} = (1 - \text{cosine\_distance}) \times e^{-\lambda \Delta t} \times \left(1 + \frac{\text{importance}}{10}\right) \times \text{access\_multiplier}$$
- [ ] Connect `scripts/onnx_embedder.py` for CPU-offloaded AVX-512 vectorization.
- [ ] Write tests in `tests/test_sensory_filter.py` and run full `pytest` suite.

### Target Files
- `proxy/core/sensory_filter.py`
- `proxy/rag/retriever.py`
- `scripts/onnx_embedder.py`
- `tests/test_sensory_filter.py`

### Test Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Status
- Branch: `main`
- Latest commit: `4393911` (Phase 3 complete)
- Tracking `origin/main` on GitHub
