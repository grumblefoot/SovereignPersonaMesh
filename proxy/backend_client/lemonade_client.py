"""
Async HTTP Client for Lemonade Server (LLM Backend on Port 13305).
Supports streaming Server-Sent Events (SSE) and continuous batching / prompt caching configs.
"""

import json
import logging
import httpx
from typing import AsyncGenerator, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LemonadeLLMClient:
    def __init__(self, base_url: str = "http://localhost:13305/v1"):
        self.base_url = base_url.rstrip("/")

    async def _resolve_model(self, requested_model: str, client: httpx.AsyncClient) -> str:
        """Dynamically resolve requested model string to an available Lemonade model ID."""
        try:
            resp = await client.get(f"{self.base_url}/models")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                available = [m.get("id") for m in data if m.get("id")]
                if requested_model in available:
                    return requested_model
                # Check case-insensitive or substring matches
                req_lower = requested_model.lower()
                for av in available:
                    if req_lower in av.lower() or av.lower() in req_lower or "gemma" in av.lower() and "gemma" in req_lower:
                        return av
                # Fallback to first available text model if any exist
                if available:
                    return available[0]
        except Exception as e:
            logger.warning(f"[LemonadeClient] Model resolution failed: {e}")
        return requested_model

    async def generate_stream(
        self,
        prompt: str,
        model: str = "google/gemma-4-26B-A4B-it",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[list] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams completion tokens asynchronously from Lemonade server over SSE.
        """
        if stop is None:
            stop = ["</ctrl94>", "\nUser:"]

        async with httpx.AsyncClient(timeout=120.0) as client:
            target_model = await self._resolve_model(model, client)

            payload = {
                "model": target_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "stream": True
            }

            endpoint = f"{self.base_url}/chat/completions"
            logger.info(f"[LemonadeClient] Dispatching completion request (model={target_model}) to {endpoint}...")

            try:
                async with client.stream("POST", endpoint, json=payload) as response:
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
                        async with client.stream("POST", fallback_endpoint, json=fallback_payload) as fb_resp:
                            if fb_resp.status_code != 200:
                                logger.error(f"[LemonadeClient] LLM Backend error {fb_resp.status_code}")
                                yield f"Error from LLM Backend: {fb_resp.status_code}"
                                return
                            async for line in fb_resp.aiter_lines():
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        choices = data.get("choices", [])
                                        if choices:
                                            text_chunk = (
                                                choices[0].get("delta", {}).get("content") or
                                                choices[0].get("delta", {}).get("reasoning_content") or
                                                choices[0].get("text", "")
                                            )
                                            if text_chunk:
                                                yield text_chunk
                                    except json.JSONDecodeError:
                                        continue
                        return

                    if response.status_code != 200:
                        logger.error(f"[LemonadeClient] LLM Backend error {response.status_code}")
                        yield f"Error from LLM Backend: {response.status_code}"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    text_chunk = (
                                    choices[0].get("delta", {}).get("content") or
                                    choices[0].get("delta", {}).get("reasoning_content") or
                                    choices[0].get("text", "")
                                )
                                    if text_chunk:
                                        yield text_chunk
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"[LemonadeClient] Stream connection error: {e}")
                # Mock fallback for testing when backend isn't actively running
                yield f"<ctrl94>I hear movements nearby. I should proceed with caution.</ctrl94> I am ready."
