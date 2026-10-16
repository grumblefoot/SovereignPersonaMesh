"""Detect the model server's context window (OPEN-024, docs/plans/window_detection.md).

SPM used to size every prompt from a fixed hardware tier (32,768) or a manual override;
LM Studio just-in-time loaded Gemma at its 8,192 default and SPM overfilled it. The window
must come from the model server itself.

``detect_window`` probes each known server shape in turn (plain GETs, outside the LLM
scheduler); the first that reports a positive integer for this model wins. ``window_for``
adds virtual-id resolution, caching, and the latency/backoff rules of plan review item 3:
one overall 2 s cap, cached values served immediately with a background refresh when
stale, and after 3 unknowns in a row a 600 s pause before probing again. Never raises.
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from proxy.backend_client.lemonade_client import (
    DEFAULT_CHAT_MODEL,
    best_match,
    model_matches,
)
from proxy.core.reasoning import _is_local

logger = logging.getLogger(__name__)

# Sources that report the server's real configured window; a manual override may cap
# these (plan review item 4). 'lmstudio-jit-default' and 'unknown' are guesses, so the
# override wins over them.
REPORTED_SOURCES = {"lmstudio-loaded", "models-list", "lemonade-launch", "llamacpp-props"}

LMSTUDIO_JIT_DEFAULT = 8192          # LM Studio's window for a just-in-time load

OVERALL_CAP_S = 2.0                  # all probes together (review item 3)
REQUEST_TIMEOUT_S = 1.5              # per request
FOUND_TTL_S = 300.0                  # a found value may be stale for 5 min
UNKNOWN_TTL_S = 60.0                 # failures must not stick
BACKOFF_AFTER_UNKNOWNS = 3           # after this many unknowns in a row,
BACKOFF_S = 600.0                    # wait 10 min before probing again

# Injected clock (tests monkeypatch this; same value style as reasoning's probes).
_now = time.monotonic


@dataclass
class WindowInfo:
    tokens: Optional[int]   # None = unknown
    source: str             # lmstudio-loaded | lmstudio-jit-default | models-list |
                            # lemonade-launch | llamacpp-props | unknown
    detail: str = ""


def _unknown(detail: str = "") -> WindowInfo:
    return WindowInfo(None, "unknown", detail)


# (base_url, model) -> (when_seen, WindowInfo, ttl)
_cache: Dict[Tuple[str, str], Tuple[float, WindowInfo, float]] = {}
# (base_url, model) -> consecutive unknown results
_unknown_streak: Dict[Tuple[str, str], int] = {}
# (base_url, resolved_model) -> the id the caller asked for (virtual-id resolution cache)
_resolved: Dict[Tuple[str, str], Tuple[float, str]] = {}
_RESOLVE_TTL_S = 300.0
# One probe in flight per key (same pattern as proxy/core/reasoning.py resolve_mode).
_inflight: Dict[Tuple[str, str], "asyncio.Future"] = {}


def reset_cache() -> None:
    _cache.clear()
    _unknown_streak.clear()
    _resolved.clear()
    _inflight.clear()


# ── parsing helpers (review item 7: guard shape, never trust the server) ────

def _json_dict(resp: httpx.Response) -> Optional[Any]:
    """Parsed JSON object, or None for anything that is not a 200 JSON dict."""
    if resp.status_code != 200:
        return None
    if "json" not in resp.headers.get("content-type", ""):
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _positive_int(value: Any) -> Optional[int]:
    """A positive integer, or None for 'auto', floats, bools, negatives, strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


_CTX_FLAG = re.compile(r"--ctx-size(?:\s+|=)(\d+)")


def _ctx_from_launch_command(cmd: Any) -> Optional[int]:
    """'lemonade-server serve --model X --ctx-size 262144' -> 262144."""
    if not isinstance(cmd, str):
        return None
    m = _CTX_FLAG.search(cmd)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    return n if n > 0 else None


async def _get_json(http: httpx.AsyncClient, url: str) -> Optional[Any]:
    try:
        resp = await http.get(url)
    except Exception:
        return None
    return _json_dict(resp)


# ── the probes (plan table order) ───────────────────────────────────────────

async def _probes(http: httpx.AsyncClient, root: str, base: str, model: str) -> Optional[WindowInfo]:
    # 1. LM Studio native API: the entry for this model, with its load state.
    lm = await _get_json(http, f"{root}/api/v0/models")
    lm_entry_found = False
    if isinstance(lm, dict):
        entries = {str(m["id"]): m for m in lm.get("data") or [] if isinstance(m, dict) and m.get("id")}
        pick = best_match(model, entries)
        if pick is not None:
            m = entries[pick]
            lm_entry_found = True
            # A model list says nothing until the model is loaded here.
            if str(m.get("state", "")).lower() == "loaded":
                n = _positive_int(m.get("loaded_context_length"))
                if n:
                    return WindowInfo(n, "lmstudio-loaded", f"{m['id']} loaded at {n}")
    # 2/3. Lemonade health (real server on :13305 — probe before the generic /models
    # so --ctx-size is preferred; llama-server-style/vLLM/OpenRouter-style servers have no such health endpoint).
    health = await _get_json(http, f"{root}/api/v1/health")
    if isinstance(health, dict):
        by_name = {}
        for m in health.get("all_models_loaded") or []:
            if isinstance(m, dict):
                for v in (m.get("model_name"), m.get("checkpoint")):
                    if v:
                        by_name.setdefault(str(v), m)
        pick = best_match(model, by_name)
        if pick is not None:
            n = _ctx_from_launch_command(by_name[pick].get("launch_command"))
            if n:
                return WindowInfo(n, "lemonade-launch", f"{pick} ctx-size {n}")
    # 4. OpenAI-style /models: vLLM, llama-server-style servers, LM Studio (same data without state).
    models = await _get_json(http, f"{base}/models")
    if isinstance(models, dict):
        entries = {str(m["id"]): m for m in models.get("data") or [] if isinstance(m, dict) and m.get("id")}
        pick = best_match(model, entries)
        if pick is not None:
            m = entries[pick]
            n = _positive_int(m.get("context_length")) or _positive_int(m.get("max_model_len"))
            if n:
                return WindowInfo(n, "models-list", f"{m['id']} context_length {n}")
    # 5. LM Studio, model listed but never loaded: a request will just-in-time load it
    # at LM Studio's 8,192 default. Conservative; an override or a manual load wins.
    if lm_entry_found:
        return WindowInfo(LMSTUDIO_JIT_DEFAULT, "lmstudio-jit-default",
                          f"{model} not loaded; LM Studio would JIT-load it at {LMSTUDIO_JIT_DEFAULT}")
    # 6. llama.cpp server.
    props = await _get_json(http, f"{root}/props")
    if isinstance(props, dict):
        gen = props.get("default_generation_settings")
        n = None
        if isinstance(gen, dict):
            n = _positive_int(gen.get("n_ctx"))
        if n is None:
            n = _positive_int(props.get("n_ctx"))
        if n:
            return WindowInfo(n, "llamacpp-props", f"llama.cpp n_ctx {n}")
    return None


def _root_of(base_url: str) -> str:
    """base = BACKEND_LLM_URL (ends in /v1); root = base without a trailing /v1."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3].rstrip("/")
    return base


async def detect_window(base_url: str, model: str, *, http: httpx.AsyncClient) -> WindowInfo:
    """Probe every known server shape for this model's window. Never raises; unknown on
    any failure. All probes share one overall 2 s cap (review item 3)."""
    root, base = _root_of(base_url), (base_url or "").rstrip("/")
    try:
        found = await asyncio.wait_for(_probes(http, root, base, model), timeout=OVERALL_CAP_S)
    except Exception:
        found = None
    return found or _unknown()


# ── cached entry point ──────────────────────────────────────────────────────

async def _resolve_for(base_url: str, model: str, client: Any) -> str:
    """Resolve SPM's virtual ids the way generation does (review item 1)."""
    key = (base_url, str(model))
    hit = _resolved.get(key)
    if hit and _now() - hit[0] < _RESOLVE_TTL_S:
        return hit[1]
    resolved = str(model).strip()
    resolver = getattr(client, "_resolve_model", None)
    if resolver is not None:
        try:
            resolved = str(await resolver(str(model))).strip() or resolved
        except Exception as e:
            logger.debug(f"[BackendProfile] model resolution failed for '{model}': {e}")
    resolved = resolved or DEFAULT_CHAT_MODEL
    _resolved[key] = (_now(), resolved)
    return resolved


def _in_backoff(key: Tuple[str, str]) -> bool:
    streak = _unknown_streak.get(key, 0)
    if streak < BACKOFF_AFTER_UNKNOWNS:
        return False
    seen = _cache.get(key)
    return bool(seen and _now() - seen[0] < BACKOFF_S)


async def _probe_and_cache(base_url: str, resolved: str, http: httpx.AsyncClient) -> WindowInfo:
    key = (base_url, resolved)
    info = await detect_window(base_url, resolved, http=http)
    prev = _cache.get(key)
    _cache[key] = (_now(), info, FOUND_TTL_S if info.tokens else UNKNOWN_TTL_S)
    _unknown_streak[key] = _unknown_streak.get(key, 0) + 1 if not info.tokens else 0
    if prev is None or (prev[1].tokens, prev[1].source) != (info.tokens, info.source):
        logger.info(f"[BackendProfile] {base_url} ({resolved}): window "
                    f"{info.tokens if info.tokens is not None else 'unknown'} ({info.source})")
    return info


async def window_for(client: Any, model: str) -> WindowInfo:
    """The window for this client + model, cached per (base_url, resolved model).

    Cloud URLs never probe. A fresh cache entry is served without a request; a stale
    one is served immediately while a background refresh runs; after
    BACKOFF_AFTER_UNKNOWNS consecutive unknowns probing pauses for BACKOFF_S. Never
    raises; the caller falls back to override/tier on 'unknown'.
    """
    base_url = (getattr(client, "base_url", "") or "").strip()
    if not base_url:
        return _unknown("no backend url")
    resolved = await _resolve_for(base_url, model, client)
    if not _is_local(base_url):
        return _unknown("cloud backend")
    key = (base_url, resolved)
    if not _in_backoff(key):
        hit = _cache.get(key)
        if hit is not None and _now() - hit[0] < hit[2]:
            return hit[1]
        task = _inflight.get(key)
        if task is not None and task.done():
            _inflight.pop(key, None)      # done_callback may not have run yet
            task = None
        if task is None:
            http = getattr(client, "_window_http", None)
            if not isinstance(http, httpx.AsyncClient) or http.is_closed:
                http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
            task = asyncio.ensure_future(_probe_and_cache(base_url, resolved, http))
            _inflight[key] = task
            task.add_done_callback(lambda _t, k=key: _inflight.pop(k, None))
        if hit is not None and hit[1].tokens:
            # Stale-while-revalidate (review item 3) for found values only: serve the
            # last known window now, refresh in the background. An expired *unknown*
            # is re-probed synchronously (capped at 2 s) — a failure must not stick.
            return hit[1]
        return await asyncio.shield(task)
    hit = _cache.get(key)
    return hit[1] if hit is not None else _unknown("backoff after repeated probe failures")
