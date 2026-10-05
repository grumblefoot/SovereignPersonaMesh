"""
Async HTTP Client for Lemonade Server (LLM Backend on Port 13305).
Supports streaming Server-Sent Events (SSE) and continuous batching / prompt caching configs.
"""

import json
import logging
import os
import time
import httpx
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:13305/v1"

# SPM is user-facing roleplay chat: default to Lemonade's Gemma build, never the coding alias (lemonade_playbook.md).
DEFAULT_CHAT_MODEL = os.getenv("SPM_DEFAULT_MODEL", "Gemma-4-26B-A4B-it-GGUF")
SPM_VIRTUAL_MODEL_ID = "spm-sovereign-mesh"
EXCLUDED_MODEL_IDS = {"hermes-coder"}
# The vLLM google/gemma-4-* checkpoints were removed with the Lemonade 11.9 upgrade (2026-09-26).
LEGACY_MODEL_IDS = {
    "google/gemma-4-26b-a4b-it": "Gemma-4-26B-A4B-it-GGUF",
    "google/gemma-4-e4b-it": "Gemma-4-E4B-it-GGUF",
    "google/gemma-4-12b-it": "Gemma-4-12B-it-GGUF",
}
MODELS_CACHE_SECONDS = 30.0


class LLMBackendError(RuntimeError):
    """The LLM backend could not produce a reply (unreachable or non-200)."""


class LemonadeLLMClient:
    def __init__(self, base_url: Optional[str] = None):
        # Resolve the default at call time (not definition time) so tests can redirect DEFAULT_BASE_URL.
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = None
        self._models_cache: Optional[Tuple[float, List[str]]] = None


    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        await self.client.aclose()

    async def _chat_model_ids(self) -> Optional[List[str]]:
        """Chat-capable model ids from Lemonade, cached briefly (it was fetched on every turn)."""
        now = time.monotonic()
        if self._models_cache is not None and now - self._models_cache[0] < MODELS_CACHE_SECONDS:
            return self._models_cache[1]
        resp = await self.client.get(f"{self.base_url}/models")
        if resp.status_code != 200:
            return None
        ids = []
        for m in resp.json().get("data", []):
            model_id = m.get("id")
            labels = m.get("labels")
            # Skip non-chat models (image, TTS, ...) and the coding-agent alias; models without labels are kept.
            if not model_id or model_id.lower() in EXCLUDED_MODEL_IDS or (labels is not None and "chat" not in labels):
                continue
            ids.append(model_id)
        self._models_cache = (now, ids)
        return ids

    async def _resolve_model(self, requested_model: str) -> str:
        """Map the requested model to a Lemonade model id.

        Order: SPM's virtual id -> default model; exact match; retired google/gemma-4-* ids -> their Lemonade
        GGUF builds; case-insensitive match; a *unique* substring match. Anything else is returned unchanged
        so Lemonade reports the error, instead of silently switching to whichever model happens to be listed first.
        """
        requested = (requested_model or "").strip()
        if not requested or requested.lower() == SPM_VIRTUAL_MODEL_ID:
            requested = DEFAULT_CHAT_MODEL
        try:
            available = await self._chat_model_ids()
        except Exception as e:
            logger.warning(f"[LemonadeClient] Model resolution failed: {e}")
            available = None
        if available is not None and requested in available:
            return requested
        requested = LEGACY_MODEL_IDS.get(requested.lower(), requested)
        if available is None or requested in available:
            return requested
        req_lower = requested.lower()
        for av in available:
            if av.lower() == req_lower:
                return av
        matches = [av for av in available if req_lower in av.lower()]
        if len(matches) == 1:
            return matches[0]
        logger.warning(f"[LemonadeClient] Model '{requested_model}' not found among chat models; passing it through unchanged.")
        return requested

    def _extract_reasoning_and_content(self, choices: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Helper to unify reasoning extraction from both chat and completions legacy endpoints."""
        if not choices:
            return None, None
        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        content = delta.get("content") or choices[0].get("text")
        return reasoning, content

    async def generate_stream(
        self,
        prompt: Optional[str] = None,
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 128000,
        stop: Optional[list] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams completion tokens asynchronously from Lemonade server over SSE.
        """
        if stop is None:
            stop = ["\nUser:", "\nHuman:"]

        target_model = await self._resolve_model(model)
        req_messages = messages or [{"role": "user", "content": prompt or ""}]

        payload = {
            "model": target_model,
            "messages": req_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "stream": True,
            # Token budget P2: Lemonade (verified 2026-10-05) and OpenAI only put
            # `usage` in a stream when asked; without this the calibration EMA
            # never received a single sample live.
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            payload.update(extra_body)  # backend-specific options, e.g. chat_template_kwargs

        endpoint = f"{self.base_url}/chat/completions"
        logger.info(f"[LemonadeClient] Dispatching completion request (model={target_model}) to {endpoint}...")

        try:
            async with self.client.stream("POST", endpoint, json=payload) as response:
                if response.status_code == 404:
                    # Fallback to legacy /completions prompt endpoint
                    fallback_endpoint = f"{self.base_url}/completions"
                    fallback_payload = {
                        "model": model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "stream": True
                    }
                    logger.info(f"[LemonadeClient] 404 on chat/completions, retrying {fallback_endpoint}...")
                    async with self.client.stream("POST", fallback_endpoint, json=fallback_payload) as fb_resp:
                        if fb_resp.status_code != 200:
                            logger.error(f"[LemonadeClient] LLM Backend error {fb_resp.status_code}")
                            raise LLMBackendError(f"LLM backend returned HTTP {fb_resp.status_code}")
                        
                        in_reasoning = False
                        async for line in fb_resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    choices = data.get("choices", [])
                                    reasoning, content = self._extract_reasoning_and_content(choices)
                                    if reasoning:
                                        if not in_reasoning:
                                            in_reasoning = True
                                            yield "<thinking>"
                                        yield reasoning
                                    if content:
                                        if in_reasoning:
                                            in_reasoning = False
                                            yield "</thinking>"
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                        if in_reasoning:
                            yield "</thinking>"
                    return

                if response.status_code != 200:
                    logger.error(f"[LemonadeClient] LLM Backend error {response.status_code}")
                    raise LLMBackendError(f"LLM backend returned HTTP {response.status_code}")

                in_reasoning = False
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            # Token budget P2: llama.cpp-family backends attach usage to
                            # the final chunk. Feed prompt_tokens back into the per-model
                            # chars/token EMA so estimates converge on real counts.
                            usage = data.get("usage")
                            if usage and usage.get("prompt_tokens"):
                                self.last_usage = usage
                                try:
                                    from proxy.rag import budget as _budget
                                    prompt_chars = sum(len(str(m.get("content", "")))
                                                       for m in req_messages)
                                    _budget.calibrate(target_model, prompt_chars,
                                                      int(usage["prompt_tokens"]))
                                    logger.info(
                                        f"[LemonadeClient] usage: prompt={usage.get('prompt_tokens')} "
                                        f"completion={usage.get('completion_tokens')} "
                                        f"(ratio now {_budget.ratio_for(target_model):.2f} chars/tok)")
                                except Exception as cal_exc:  # calibration must never break a stream
                                    logger.debug(f"[LemonadeClient] calibration skipped: {cal_exc}")
                            choices = data.get("choices", [])
                            reasoning, content = self._extract_reasoning_and_content(choices)

                            if reasoning:
                                if not in_reasoning:
                                    in_reasoning = True
                                    yield "<thinking>"
                                yield reasoning

                            if content:
                                if in_reasoning:
                                    in_reasoning = False
                                    yield "</thinking>"
                                yield content
                        except json.JSONDecodeError:
                            continue

                if in_reasoning:
                    yield "</thinking>"
        except LLMBackendError:
            raise
        except Exception as e:
            # Never invent a reply here: a made-up line would be shown as the character's turn and saved as memory.
            logger.error(f"[LemonadeClient] Stream connection error: {e}")
            raise LLMBackendError(f"LLM backend unreachable: {e}") from e

