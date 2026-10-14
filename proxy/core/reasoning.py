"""How SPM handles the model's private reasoning (native-reasoning spike, 2026-10-13).

Modes:
- native:  the server separates the model's own thinking into a reasoning field (vLLM,
           llama.cpp, LM Studio, gufo, ...). SPM sends no text-tag instructions; the client
           already turns the field into SPM's private block. No pattern-guessing needed.
- none:    no private block at all. For models without native thinking, and for cloud
           backends where hidden thinking costs money.
- prefill: the legacy text scratchpad (assistant prefill "<think>\\n", GM duties and inline
           GM_ACTION tags), parsed with the salvage patterns. Kept as an explicit option.

`auto` probes the backend once per (URL, model): a tiny request with thinking switched on;
if a reasoning chunk comes back, the backend is native-capable. A cloud URL never probes and
resolves to none (decision 5/14 spirit: nothing costly or private happens implicitly).

Thinking switches differ by server, so native/none send BOTH (harmless where unused):
`reasoning_effort` (LM Studio honours only this one for Gemma 4; OpenAI-style) and
`chat_template_kwargs.enable_thinking` (vLLM, llama.cpp, gufo).
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MODES = ("native", "none", "prefill")
EFFORTS = ("low", "medium", "high")
PROBE_TTL_S = 600.0
PROBE_TIMEOUT_S = 45.0

_probe_cache: Dict[Tuple[str, str], Tuple[float, str]] = {}


def _is_local(url: str) -> bool:
    try:
        from proxy.embeddings.service import _is_local_url
        return _is_local_url(url)
    except Exception:
        return False


def configured_mode(settings: Dict[str, Any]) -> str:
    raw = str(settings.get("reasoning_mode") or "auto").strip().lower()
    return raw if raw in MODES + ("auto",) else "auto"


def effort(settings: Dict[str, Any]) -> str:
    raw = str(settings.get("reasoning_effort") or "low").strip().lower()
    return raw if raw in EFFORTS else "low"


def extra_body(mode: str, settings: Dict[str, Any], base_url: str) -> Optional[Dict[str, Any]]:
    """Request fields that switch the server's thinking on or off. Only for local
    backends: strict cloud APIs reject unknown fields such as chat_template_kwargs."""
    if mode == "prefill" or not _is_local(base_url):
        return None
    if mode == "native":
        return {"reasoning_effort": effort(settings),
                "chat_template_kwargs": {"enable_thinking": True}}
    return {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def gm_channel(mode: str, settings: Dict[str, Any]) -> str:
    """inline (GM_ACTION tags in the prefill scratchpad) or separate (structured call)."""
    raw = str(settings.get("gm_actions_channel") or "auto").strip().lower()
    if raw in ("inline", "separate"):
        return raw
    return "inline" if mode == "prefill" else "separate"


async def _probe(client, model: str) -> str:
    """'native' if the backend streams a reasoning chunk with thinking switched on."""
    body = {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": True}}
    kwargs = {"job_kind": "utility"} if hasattr(client, "scheduler") else {}
    gen = client.generate_stream(
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        model=model, temperature=0.0, max_tokens=64, extra_body=body, **kwargs)
    try:
        async for chunk in gen:
            if chunk == "<thinking>":
                return "native"
            if chunk and chunk.strip():
                return "none"
        return "none"
    finally:
        await gen.aclose()


async def resolve_mode(settings: Dict[str, Any], client, model: str) -> str:
    """The mode for this turn. Never raises: a failed probe means 'none' for a while."""
    mode = configured_mode(settings)
    if mode != "auto":
        return mode
    base_url = getattr(client, "base_url", "") or ""
    if not _is_local(base_url):
        return "none"
    key = (base_url, str(model))
    hit = _probe_cache.get(key)
    if hit and time.monotonic() - hit[0] < PROBE_TTL_S:
        return hit[1]
    try:
        found = await asyncio.wait_for(_probe(client, model), timeout=PROBE_TIMEOUT_S)
    except Exception as e:
        logger.warning(f"[Reasoning] probe of {base_url} ({model}) failed, using 'none': {e}")
        found = "none"
    _probe_cache[key] = (time.monotonic(), found)
    logger.info(f"[Reasoning] {base_url} ({model}): reasoning_mode auto -> {found}")
    return found


def reset_cache() -> None:
    _probe_cache.clear()
