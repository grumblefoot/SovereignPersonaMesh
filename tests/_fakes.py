"""Deterministic test doubles shared across the suite.

DeterministicEmbedder moved into the provider layer (proxy/embeddings) as the
'fake' provider's compatibility wrapper; this re-export keeps old imports
working.
"""

from proxy.embeddings import DeterministicEmbedder  # noqa: F401
