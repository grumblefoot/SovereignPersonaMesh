"""Embedding provider layer (embeddings plan phases 1-3, OPEN-002).

One process-wide EmbeddingService replaces the five CPUEmbeddingEngine stub
instances. Providers:

- ``openai_compat``: POST {base}/v1/embeddings over httpx. Works for Lemonade
  (embed-gemma-300m-FLM, 768 dims, own 'embedding' slot), KoboldCpp
  (--embeddingsmodel) and llama-server (--embedding).
- ``none``: no vectors; retrieval falls back to Postgres full-text search.
- ``fake``: deterministic sha256 word-bucket vectors for tests.
- ``onnx_local``: phase 4, not implemented yet (stub raises).

Resolution is config-driven (EMBEDDING_* keys in config/manager.py) and never
chooses a cloud endpoint implicitly (SPRINT_PLAN decision 5 / PRD SLA-2).
"""

from proxy.embeddings.service import (
    DeterministicEmbedder,
    EmbeddingService,
    EmbeddingSpace,
    FakeEmbeddingProvider,
    OnnxLocalProvider,
    OpenAICompatProvider,
    create_embedding_service,
    get_embedding_service,
    reset_embedding_service,
    resolve_provider_name,
    space_id_for,
)

__all__ = [
    "DeterministicEmbedder",
    "EmbeddingService",
    "EmbeddingSpace",
    "FakeEmbeddingProvider",
    "OnnxLocalProvider",
    "OpenAICompatProvider",
    "create_embedding_service",
    "get_embedding_service",
    "reset_embedding_service",
    "resolve_provider_name",
    "space_id_for",
]
