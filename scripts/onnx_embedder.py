"""DEPRECATED shim over proxy.embeddings (kept for old import sites).

The provider layer (proxy/embeddings/) replaced this module in Sprint 1
Track B. ``CPUEmbeddingEngine`` now delegates every call to the process-wide
EmbeddingService; it fabricates nothing. Phase-0 behaviour is preserved: when
no embedding provider is available (provider 'none', misconfiguration, or an
outage), every call returns None and callers store NULL / skip vector search.

New code should use ``proxy.embeddings.get_embedding_service()`` directly.
The real local ONNX runtime ('onnx_local', EmbeddingGemma-300m) is phase 4 of
docs/plans/embeddings.md and lives in proxy/embeddings when it lands.
"""

import logging
from typing import List, Optional

from proxy.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class CPUEmbeddingEngine:
    """Deprecated: thin delegate to the EmbeddingService singleton."""

    def __init__(self, dimension: Optional[int] = None, num_threads: int = 4):
        # Both arguments are legacy no-ops: the service owns model and dimension.
        self.dimension = dimension
        self.num_threads = num_threads
        self.model = None
        logger.debug(
            "[CPUEmbeddingEngine] deprecated shim constructed; calls delegate "
            "to proxy.embeddings.get_embedding_service()."
        )

    @property
    def available(self) -> bool:
        """True when a real embedding provider is configured."""
        try:
            return get_embedding_service().available
        except Exception:
            return False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Embed `text`, or None when no provider is available (store NULL)."""
        try:
            return await get_embedding_service().generate_embedding(text)
        except Exception as e:
            logger.warning("[CPUEmbeddingEngine] embedding failed: %s", e)
            return None

    async def batch_generate_embeddings(self, texts: List[str]):
        """Batch form of generate_embedding; failed entries are None."""
        try:
            return await get_embedding_service().batch_generate_embeddings(texts)
        except Exception as e:
            logger.warning("[CPUEmbeddingEngine] batch embedding failed: %s", e)
            return [None] * len(texts)

    async def ensure_space(self, conn) -> Optional[int]:
        """Active embedding-space id for stamping rows, or None (provider 'none')."""
        try:
            return await get_embedding_service().ensure_space(conn)
        except Exception as e:
            logger.warning("[CPUEmbeddingEngine] ensure_space failed: %s", e)
            return None
