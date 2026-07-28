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

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "stream": True
        }

        endpoint = f"{self.base_url}/completions"
        logger.info(f"[LemonadeClient] Dispatching completion request to {endpoint}...")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream("POST", endpoint, json=payload) as response:
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
                                    text_chunk = choices[0].get("text", "") or choices[0].get("delta", {}).get("content", "")
                                    if text_chunk:
                                        yield text_chunk
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"[LemonadeClient] Stream connection error: {e}")
                # Mock fallback for testing when backend isn't actively running
                yield f"<ctrl94>I hear movements nearby. I should proceed with caution.</ctrl94> I am ready."
