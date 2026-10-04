"""
CPU Offloading Embedding Engine (ONNX Runtime).
Executes vectorization on host CPU threads using AVX-512 extensions to prevent GPU resource contention.
"""

import os
import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class CPUEmbeddingEngine:
    #: True once a real model is loaded. The stub must never fake vectors: random embeddings
    #: poisoned RAG recall and changed on every restart (embeddings plan, phase 0).
    available: bool = False

    def __init__(self, dimension: int = 3584, num_threads: int = 4):
        self.dimension = dimension
        self.num_threads = num_threads
        self.model = None
        self._initialize_onnx()

    def _initialize_onnx(self):
        """Initialize ONNX runtime session configured for multi-threaded CPU inference."""
        logger.warning(f"CPU Embedding Engine is a stub (dim={self.dimension}): no model loaded, embeddings disabled until Sprint 1.")
        # Stub: Hermes will integrate actual ONNX model weights (e.g. Gemma/bge-large-en)
        pass

    async def generate_embedding(self, text: str):
        """Return the embedding for `text`, or None while no real model is loaded.

        Phase 0 of the embeddings plan: the old stub returned a random unit vector seeded by
        Python's salted hash(), so stored vectors were noise and differed per process. Callers
        must treat None as "store NULL / skip vector search".
        """
        if not self.available or self.model is None:
            return None
        raise NotImplementedError("real ONNX inference lands with the provider layer (Sprint 1)")

    async def batch_generate_embeddings(self, texts: List[str]):
        """Batch form of generate_embedding; entries are None while no model is loaded."""
        return [await self.generate_embedding(t) for t in texts]
