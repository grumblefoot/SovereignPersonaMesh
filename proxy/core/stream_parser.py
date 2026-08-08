"""
Two-State Monologue Token Parsing State Machine with Fail-Safe Passthrough.
Parses <ctrl94> thoughts, writing monologue to private memory buffer, and streams public text to SillyTavern.
Includes fail-safe passthrough (>500 tokens / malformed tags / unexpected EOS auto-close).
"""

import logging
from typing import AsyncGenerator, List, Tuple

logger = logging.getLogger(__name__)

OPEN_TAG = "<ctrl94>"
CLOSE_TAG = "</ctrl94>"
MAX_MONOLOGUE_TOKENS = 8192


class MonologueStreamParser:
    def __init__(self, max_public_tokens: Optional[int] = None):
        self.state = 1  # 1: Public Dialogue (default), 0: Monologue
        self.inner_monologue_buffer: str = ""
        self.public_response_buffer: str = ""
        self.monologue_token_count: int = 0
        self.public_token_count: int = 0
        self.max_public_tokens: Optional[int] = max_public_tokens
        self.is_public_truncated: bool = False
        self.is_failsafe_triggered: bool = False
        self._monologue_sections: List[str] = []
        self._in_monologue: bool = False

    def _check_public_limit(self, chunk: str) -> bool:
        """Helper to increment public token count and check if limit exceeded. Returns True if truncated."""
        tokens = len(chunk.split()) if chunk.strip() else 1
        self.public_token_count += tokens
        if self.max_public_tokens is not None and self.public_token_count >= self.max_public_tokens:
            self.is_public_truncated = True
            logger.info(f"[StreamParser] Reached max_public_tokens ({self.public_token_count} >= {self.max_public_tokens}). Truncating.")
            return True
        return False

    async def process_token_stream(
        self, token_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Processes streaming token chunks from LLM backend.
        Yields ONLY public canon character response to SillyTavern while capturing inner monologue for SPM monitoring.
        """
        try:
            async for chunk in token_generator:
                if self.state == 0:
                    # State 0: Private Monologue accumulation
                    self.inner_monologue_buffer += chunk
                    self.monologue_token_count += len(chunk.split()) if chunk.strip() else 1
                    self._in_monologue = True

                    if CLOSE_TAG in self.inner_monologue_buffer:
                        # Monologue closed -> transition to State 1 (Public)
                        parts = self.inner_monologue_buffer.split(CLOSE_TAG, 1)
                        mono_text = parts[0].replace(OPEN_TAG, "").strip()
                        if mono_text:
                            self._monologue_sections.append(mono_text)
                        self.state = 1
                        self.inner_monologue_buffer = ""
                        logger.info(
                            f"[StreamParser] Monologue complete ({len(mono_text)} chars). "
                            f"Transitioning to State 1 (Public)."
                        )
                        public_suffix = parts[1]
                        if public_suffix:
                            self.public_response_buffer += public_suffix
                            yield public_suffix
                            if self._check_public_limit(public_suffix):
                                break
                    elif self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                        logger.warning(
                            f"[StreamParser] Max monologue tokens reached (>500 tokens). Closing monologue."
                        )
                        mono_text = self.inner_monologue_buffer.replace(OPEN_TAG, "").strip()
                        if mono_text:
                            self._monologue_sections.append(mono_text)
                        self.state = 1
                        self.inner_monologue_buffer = ""

                else:
                    # State 1: Public Canon Dialogue (default)
                    if OPEN_TAG in chunk:
                        parts = chunk.split(OPEN_TAG, 1)
                        if parts[0]:
                            self.public_response_buffer += parts[0]
                            yield parts[0]
                            if self._check_public_limit(parts[0]):
                                break
                        self.state = 0
                        self.inner_monologue_buffer = parts[1] if len(parts) > 1 else ""
                        self.monologue_token_count = len(self.inner_monologue_buffer.split()) if self.inner_monologue_buffer.strip() else 1
                        self._in_monologue = True
                    else:
                        self.public_response_buffer += chunk
                        yield chunk
                        if self._check_public_limit(chunk):
                            break
        finally:
            # End of Stream (EOS) cleanup: save any pending monologue to log, NEVER dump as public text
            if self.state == 0 and self.inner_monologue_buffer:
                mono_text = self.inner_monologue_buffer.replace(OPEN_TAG, "").replace(CLOSE_TAG, "").strip()
                if mono_text:
                    self._monologue_sections.append(mono_text)
                self.inner_monologue_buffer = ""
                self.state = 1
                logger.info(f"[StreamParser] EOS reached in monologue mode. Monologue saved to log (0 chars dumped to public).")

    def get_final_buffers(self) -> Tuple[str, str]:
        """Returns (inner_monologue, public_response)."""
        if self.inner_monologue_buffer.strip() and self._in_monologue:
            self._monologue_sections.append(self.inner_monologue_buffer.strip())
        all_monologue = "\n".join(self._monologue_sections)
        return all_monologue, self.public_response_buffer.strip()
