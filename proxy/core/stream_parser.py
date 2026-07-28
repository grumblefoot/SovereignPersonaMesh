"""
Two-State Monologue Token Parsing State Machine with Fail-Safe Passthrough.
Parses <ctrl94> thoughts, writing monologue to private memory buffer, and streams public text to SillyTavern.
Includes fail-safe passthrough (>500 tokens / malformed tags / unexpected EOS auto-close).
"""

import logging
from typing import AsyncGenerator, Tuple

logger = logging.getLogger(__name__)

OPEN_TAG = "<ctrl94>"
CLOSE_TAG = "</ctrl94>"
MAX_MONOLOGUE_TOKENS = 500


class MonologueStreamParser:
    def __init__(self):
        self.state = 0  # 0: Monologue, 1: Public Dialogue
        self.inner_monologue_buffer: str = ""
        self.public_response_buffer: str = ""
        self.monologue_token_count: int = 0
        self.is_failsafe_triggered: bool = False

    async def process_token_stream(
        self, token_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Processes streaming token chunks from LLM backend.
        Yields public SSE output chunks to SillyTavern while stripping private inner monologue.
        """
        async for chunk in token_generator:
            if self.is_failsafe_triggered:
                # Raw Passthrough mode
                self.public_response_buffer += chunk
                yield chunk
                continue

            if self.state == 0:
                # Accumulating inner monologue
                if CLOSE_TAG in chunk or "</" in chunk:
                    # Transition to State 1 (Public)
                    parts = chunk.split(CLOSE_TAG)
                    self.inner_monologue_buffer += parts[0].replace(OPEN_TAG, "").strip()
                    self.state = 1
                    logger.info(f"[StreamParser] Monologue complete ({len(self.inner_monologue_buffer)} chars). Transitioning to State 1 (Public).")
                    if len(parts) > 1 and parts[1]:
                        self.public_response_buffer += parts[1]
                        yield parts[1]
                else:
                    self.inner_monologue_buffer += chunk
                    self.monologue_token_count += 1

                    # Check Fail-Safe Passthrough trigger (>500 monologue tokens without closing tag)
                    if self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                        logger.warning(f"[StreamParser] Fail-Safe Passthrough Triggered (>500 tokens in monologue). Auto-closing tag and switching to passthrough mode.")
                        self.is_failsafe_triggered = True
                        self.state = 1
                        # Yield remaining buffered text as public output to avoid client hangs
                        yield f"\n> {self.inner_monologue_buffer}\n"
            else:
                # State 1: Public Dialogue
                self.public_response_buffer += chunk
                yield chunk

    def get_final_buffers(self) -> Tuple[str, str]:
        """Returns (inner_monologue, public_response)."""
        return self.inner_monologue_buffer.strip(), self.public_response_buffer.strip()
