"""EmbeddingService and providers (embeddings plan phases 1-3).

Contract for callers (routes, lore extractor, import worker, sleep cycle,
re-embed job): embedding failures NEVER raise into the caller. An outage, a
misconfiguration, an empty text or a dimension mismatch all come back as
``None`` and the row is stored un-embedded (the re-embed job fills it later).
"""

import asyncio
import hashlib
import logging
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Circuit breaker: after N consecutive failures the provider is considered
# down for COOLDOWN seconds (aligned with the FIFO plan: 3 failures, 30 s).
_BREAKER_FAILURES = 3
_BREAKER_COOLDOWN_S = 30.0

_HTTP_MAX_BATCH = 256
_CACHE_SIZE = 512

_TRUTHY = {"1", "true", "yes", "on"}

# Hostname suffixes that count as LAN, not cloud.
_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal", ".localdomain")


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


def _is_local_url(url: str) -> bool:
    """True when the URL points at loopback, a private/LAN address, or a
    LAN-style hostname. api.openai.com and friends return False."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.lower()
    try:
        ip = ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        pass
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return True
    # Single-label hostname (e.g. "strix") resolves on the LAN only.
    return "." not in host


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity of a vector space. Vectors from different spaces are never
    compared, even when the dimensions happen to match."""

    provider: str
    model: str
    dim: int


class EmbedderUnavailable(RuntimeError):
    """Internal provider failure. Callers of EmbeddingService never see it."""


# ──────────────────────────────────────────────────────────────────────────
# Providers
# ──────────────────────────────────────────────────────────────────────────

class OpenAICompatProvider:
    """POST {base}/v1/embeddings — Lemonade, KoboldCpp, llama-server, OpenAI."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 3.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        base = (base_url or "").rstrip("/")
        if base.endswith("/v1"):
            self.endpoint = f"{base}/embeddings"
        elif base.endswith("/embeddings"):
            self.endpoint = base
        else:
            self.endpoint = f"{base}/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    max_batch = _HTTP_MAX_BATCH

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Guard rail: the suite must never reach a live embedding endpoint.
        # Contract tests inject an httpx.MockTransport, which is exempt.
        if (
            self._transport is None
            and "PYTEST_CURRENT_TEST" in os.environ
            and os.environ.get("RUN_LIVE_EMBED_TESTS") != "1"
            and os.environ.get("RUN_LIVE_LLM_TESTS") != "1"
        ):
            raise EmbedderUnavailable(
                "network embedding call blocked under pytest "
                "(set RUN_LIVE_LLM_TESTS=1 for the opt-in live test)"
            )
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = await self.client.post(
                self.endpoint,
                json={"model": self.model, "input": texts},
                headers=headers,
            )
        except httpx.HTTPError as e:
            raise EmbedderUnavailable(f"embeddings endpoint unreachable: {e}") from e
        if resp.status_code != 200:
            raise EmbedderUnavailable(
                f"embeddings endpoint returned {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()["data"]
            rows = sorted(data, key=lambda d: d["index"])
            vectors = [list(map(float, r["embedding"])) for r in rows]
        except (KeyError, TypeError, ValueError) as e:
            raise EmbedderUnavailable(f"malformed embeddings response: {e}") from e
        if len(vectors) != len(texts):
            raise EmbedderUnavailable(
                f"embeddings response has {len(vectors)} rows for {len(texts)} inputs"
            )
        return vectors


class FakeEmbeddingProvider:
    """Deterministic, salt-free embeddings for tests (plan: sha256 word-bucket).

    Each word token is hashed with sha256 into a bucket, so texts that share
    words land close in cosine space and retrieval tests can assert ranking.
    No per-process salt (unlike Python's hash()), so vectors are stable
    across runs.
    """

    max_batch = 1024

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.model = "fake-wordhash"

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", (text or "").lower())
        if not tokens:
            # Non-empty but wordless text ("...") still gets a stable vector.
            tokens = [text]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8", "replace")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return vec

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class OnnxLocalProvider:
    """Phase 4 of the embeddings plan (EmbeddingGemma-300m ONNX). Not built yet."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "The 'onnx_local' embedding provider is phase 4 of "
            "docs/plans/embeddings.md and is not implemented yet. "
            "Set EMBEDDING_PROVIDER to auto, openai_compat, none or fake."
        )


# ──────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────

class EmbeddingService:
    """Process-wide embedding front door.

    - never raises into callers: outages, bad dimensions and empty texts all
      return None entries;
    - L2-normalises every vector;
    - locks the dimension after the first successful embed (or from
      EMBEDDING_DIM) and refuses vectors of any other size;
    - LRU cache keyed by sha256(text), so regenerations hit the cache;
    - circuit breaker: 3 consecutive failures => down for 30 s;
    - registers its (provider, model, dim) in spm_embedding_spaces and caches
      the space id (ensure_space).
    """

    def __init__(
        self,
        provider: Optional[Any],
        provider_name: str,
        model: str = "",
        dim: int = 0,
    ):
        self._provider = provider
        self.provider_name = provider_name
        self.model = model
        self._dim = int(dim or 0)
        self._space_id: Optional[int] = None
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._failures = 0
        self._down_until = 0.0

    # -- introspection ----------------------------------------------------

    @property
    def available(self) -> bool:
        """True when a real provider is configured (independent of outages)."""
        return self._provider is not None

    @property
    def space(self) -> Optional[EmbeddingSpace]:
        if self._provider is None:
            return None
        return EmbeddingSpace(self.provider_name, self.model, self._dim)

    # -- breaker ----------------------------------------------------------

    def _breaker_open(self) -> bool:
        return time.monotonic() < self._down_until

    def _record_failure(self, err: Exception) -> None:
        self._failures += 1
        logger.warning(
            "[EmbeddingService] %s/%s failed (%d/%d): %s",
            self.provider_name, self.model, self._failures, _BREAKER_FAILURES, err,
        )
        if self._failures >= _BREAKER_FAILURES:
            self._down_until = time.monotonic() + _BREAKER_COOLDOWN_S
            self._failures = 0
            logger.error(
                "[EmbeddingService] circuit open for %.0fs: %s/%s is down; "
                "retrieval degrades to full-text search.",
                _BREAKER_COOLDOWN_S, self.provider_name, self.model,
            )

    def _record_success(self) -> None:
        self._failures = 0

    # -- vector hygiene ---------------------------------------------------

    def _validate_and_normalise(self, vec: List[float]) -> Optional[List[float]]:
        if not vec:
            return None
        if any(math.isnan(x) or math.isinf(x) for x in vec):
            logger.error("[EmbeddingService] provider returned NaN/Inf; vector dropped.")
            return None
        if self._dim == 0:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            logger.error(
                "[EmbeddingService] dimension mismatch: expected %d, got %d; "
                "vector dropped (nothing is stored with a wrong dimension).",
                self._dim, len(vec),
            )
            return None
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return None
        return [x / norm for x in vec]

    def _cache_get(self, key: str) -> Optional[List[float]]:
        vec = self._cache.get(key)
        if vec is not None:
            self._cache.move_to_end(key)
        return vec

    def _cache_put(self, key: str, vec: List[float]) -> None:
        self._cache[key] = vec
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)

    # -- embedding --------------------------------------------------------

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Embed one text. None on provider 'none', empty text, or any failure."""
        result = await self.batch_generate_embeddings([text])
        return result[0]

    async def batch_generate_embeddings(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        if self._provider is None or not texts:
            return out

        # Pre-filter: empty texts and cache hits never reach the provider.
        pending: List[int] = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            cached = self._cache_get(key)
            if cached is not None:
                out[i] = cached
            else:
                pending.append(i)

        if not pending or self._breaker_open():
            return out

        batch_size = getattr(self._provider, "max_batch", _HTTP_MAX_BATCH)
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            try:
                vectors = await self._provider.embed([texts[i] for i in chunk])
            except asyncio.CancelledError:
                raise
            except Exception as e:  # outage => None, never an exception upward
                self._record_failure(e)
                if self._breaker_open():
                    break
                continue
            self._record_success()
            for i, vec in zip(chunk, vectors):
                clean = self._validate_and_normalise(vec)
                if clean is not None:
                    key = hashlib.sha256(
                        texts[i].encode("utf-8", "replace")
                    ).hexdigest()
                    self._cache_put(key, clean)
                out[i] = clean
        return out

    # -- space registry ---------------------------------------------------

    async def ensure_space(self, conn: Any) -> Optional[int]:
        """Register (provider, model, dim) in spm_embedding_spaces and return
        its id. None for provider 'none' or while the dimension is unknown
        and unprobeable. `conn` is an asyncpg connection."""
        if self._provider is None:
            return None
        if self._space_id is not None:
            return self._space_id
        if self._dim == 0:
            probe = await self.generate_embedding("dimension probe")
            if probe is None:
                return None
        await conn.execute(
            """
            INSERT INTO spm_embedding_spaces (provider, model, dim)
            VALUES ($1, $2, $3)
            ON CONFLICT (provider, model, dim) DO NOTHING;
            """,
            self.provider_name, self.model, self._dim,
        )
        self._space_id = await conn.fetchval(
            "SELECT id FROM spm_embedding_spaces WHERE provider = $1 AND model = $2 AND dim = $3;",
            self.provider_name, self.model, self._dim,
        )
        return self._space_id


# ──────────────────────────────────────────────────────────────────────────
# Resolution (config -> provider)
# ──────────────────────────────────────────────────────────────────────────

def _effective_base_url(settings: Dict[str, Any]) -> str:
    return (
        str(settings.get("EMBEDDING_URL") or "").strip()
        or str(settings.get("BACKEND_LLM_URL") or "").strip()
    )


def resolve_provider_name(settings: Dict[str, Any]) -> str:
    """Resolve EMBEDDING_PROVIDER to a concrete provider name.

    - explicit openai_compat / none / fake are honoured as-is (an explicit
      openai_compat is the owner's documented cloud opt-in, decision 5);
    - onnx_local raises: it is phase 4 and not implemented;
    - auto: openai_compat when the effective base URL is loopback/LAN;
      for a remote URL (e.g. api.openai.com) auto resolves to 'none' unless
      EMBEDDING_ALLOW_REMOTE is set — the cloud is never chosen implicitly.
    """
    raw = str(settings.get("EMBEDDING_PROVIDER") or "auto").strip().lower()
    if raw == "onnx_local":
        OnnxLocalProvider()  # raises NotImplementedError with the full message
    if raw in ("openai_compat", "none", "fake"):
        return raw
    if raw != "auto":
        logger.warning(
            "[EmbeddingService] unknown EMBEDDING_PROVIDER=%r; using 'none'.", raw
        )
        return "none"

    base_url = _effective_base_url(settings)
    if not base_url:
        return "none"
    if _is_local_url(base_url):
        return "openai_compat"
    if _is_truthy(settings.get("EMBEDDING_ALLOW_REMOTE", False)):
        logger.warning(
            "[EmbeddingService] EMBEDDING_ALLOW_REMOTE is set: embedding via "
            "remote endpoint %s (memories leave the machine, PRD SLA-2).",
            base_url,
        )
        return "openai_compat"
    logger.info(
        "[EmbeddingService] auto: %s is not a local/LAN endpoint; cloud "
        "embeddings are opt-in only (decision 5). Resolved to 'none' "
        "(full-text retrieval). Set EMBEDDING_PROVIDER=openai_compat to opt in.",
        base_url,
    )
    return "none"


def create_embedding_service(
    settings: Optional[Dict[str, Any]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> EmbeddingService:
    """Build an EmbeddingService from the (given or live) settings."""
    if settings is None:
        from config.manager import get_settings_manager
        settings = get_settings_manager().get_settings()

    name = resolve_provider_name(settings)
    model = str(settings.get("EMBEDDING_MODEL") or "").strip()
    try:
        dim = int(settings.get("EMBEDDING_DIM") or 0)
    except (TypeError, ValueError):
        dim = 0

    if name == "none":
        return EmbeddingService(None, "none")
    if name == "fake":
        provider = FakeEmbeddingProvider(dim=dim or 768)
        return EmbeddingService(provider, "fake", provider.model, provider.dim)

    # openai_compat
    base_url = _effective_base_url(settings)
    try:
        timeout_s = float(settings.get("EMBEDDING_TIMEOUT_S") or 3)
    except (TypeError, ValueError):
        timeout_s = 3.0
    api_key = (
        str(settings.get("EMBEDDING_API_KEY") or "").strip()
        or str(settings.get("BACKEND_API_KEY") or "").strip()
    )
    provider = OpenAICompatProvider(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        transport=transport,
    )
    logger.info(
        "[EmbeddingService] provider=openai_compat model=%s endpoint=%s dim=%s",
        model, provider.endpoint, dim or "native",
    )
    return EmbeddingService(provider, "openai_compat", model, dim)


async def space_id_for(embedder: Any, conn: Any) -> Optional[int]:
    """Best-effort active-space id from any embedder-shaped object (the
    service, the deprecated shim, or a test double). Missing method,
    non-int results (mocks) and errors all become None, so a row is simply
    stored without a space stamp and the re-embed job fixes it later."""
    ensure = getattr(embedder, "ensure_space", None)
    if ensure is None:
        return None
    try:
        sid = await ensure(conn)
    except Exception as e:
        logger.warning("[EmbeddingService] space_id_for failed: %s", e)
        return None
    return sid if isinstance(sid, int) else None


_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Process-wide singleton. Construction failures degrade to provider 'none'
    so a misconfiguration can never take the chat path down."""
    global _service_instance
    if _service_instance is None:
        try:
            _service_instance = create_embedding_service()
        except Exception as e:
            logger.error(
                "[EmbeddingService] failed to build the configured provider; "
                "falling back to 'none' (full-text retrieval): %s", e,
            )
            _service_instance = EmbeddingService(None, "none")
    return _service_instance


def reset_embedding_service() -> None:
    """Drop the singleton (tests, and admin provider switches)."""
    global _service_instance
    _service_instance = None


# ──────────────────────────────────────────────────────────────────────────
# Test compatibility shim (was tests/_fakes.DeterministicEmbedder)
# ──────────────────────────────────────────────────────────────────────────

class DeterministicEmbedder:
    """Old-interface wrapper around the fake provider for existing tests.

    Stable, salt-free vectors; shared words => close in cosine space.
    """

    available = True

    def __init__(self, dimension: int = 3584):
        self.dimension = dimension
        self._service = EmbeddingService(
            FakeEmbeddingProvider(dim=dimension), "fake", "fake-wordhash", dimension
        )

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        return await self._service.generate_embedding(text)

    async def batch_generate_embeddings(self, texts: List[str]):
        return await self._service.batch_generate_embeddings(texts)

    async def ensure_space(self, conn: Any) -> Optional[int]:
        return await self._service.ensure_space(conn)
