"""Deterministic test doubles shared across the suite."""
import hashlib
from typing import List, Optional


class DeterministicEmbedder:
    """Stable, salt-free embeddings for tests (the real stub returns None by design)."""

    available = True

    def __init__(self, dimension: int = 3584):
        self.dimension = dimension

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        digest = hashlib.sha256((text or "").encode()).digest()
        return [(digest[i % len(digest)] - 128) / 128.0 for i in range(self.dimension)]

    async def batch_generate_embeddings(self, texts: List[str]):
        return [await self.generate_embedding(t) for t in texts]
