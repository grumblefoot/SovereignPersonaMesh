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
MAX_MONOLOGUE_TOKENS = 500


class MonologueStreamParser:
    def __init__(self, max_public_tokens: Optional[int] = None):
        self.state = 0  # 0: Monologue, 1: Public Dialogue
        self.inner_monologue_buffer: str = ""
        self.public_response_buffer: str = ""
        self.monologue_token_count: int = 0
        self.public_token_count: int = 0
        self.max_public_tokens: Optional[int] = max_public_tokens
        self.is_public_truncated: bool = False
        self.is_failsafe_triggered: bool = False
        self._monologue_sections: List[str] = []  # Accumulate all monologue sections
        self._in_monologue: bool = False  # True once we've entered monologue mode
        self._open_tag_seen: bool = False  # True once we've seen <ctrl94> in current State 0

    def _enter_state_0(self):
        """Mark that we've entered monologue mode."""
        self.state = 0
        self.inner_monologue_buffer = ""
        self.monologue_token_count = 0
        self._in_monologue = True
        self._open_tag_seen = False

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
        Yields public SSE output chunks to SillyTavern while stripping private inner monologue.

        Fail-safe rules:
          1. >MAX_MONOLOGUE_TOKENS tokens in State 0 without closing tag -> passthrough.
          2. Unexpected EOS (stream ends) while in State 0 -> auto-close monologue,
             flush buffer as public, switch to State 1.
          3. Truncates public stream cleanly if max_public_tokens is reached.
        """
        try:
            async for chunk in token_generator:
                if self.is_failsafe_triggered:
                    self.public_response_buffer += chunk
                    yield chunk
                    if self._check_public_limit(chunk):
                        break
                    continue

                if self.state == 0:
                    # State 0: Monologue accumulation
                    self.inner_monologue_buffer += chunk
                    self.monologue_token_count += len(chunk.split()) if chunk.strip() else 1
                    self._in_monologue = True

                    if CLOSE_TAG in self.inner_monologue_buffer:
                        # Normal close: transition to State 1 (Public)
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
                            f"[StreamParser] Fail-Safe Passthrough Triggered (>500 tokens in monologue). "
                            f"Auto-closing tag and switching to passthrough mode."
                        )
                        self.is_failsafe_triggered = True
                        self.state = 1
                        mono_text = self.inner_monologue_buffer.replace(OPEN_TAG, "").strip()
                        self.public_response_buffer += mono_text
                        yield mono_text
                        if self._check_public_limit(mono_text):
                            break
                        self.inner_monologue_buffer = ""

                else:
                    # State 1: Public Dialogue
                    if OPEN_TAG in chunk:
                        logger.info(
                            f"[StreamParser] New monologue section detected in State 1. "
                            f"Re-entering State 0."
                        )
                        self.state = 0
                        self.inner_monologue_buffer = chunk
                        self.monologue_token_count = len(chunk.split()) if chunk.strip() else 1
                        self._in_monologue = True
                    else:
                        self.public_response_buffer += chunk
                        yield chunk
                        if self._check_public_limit(chunk):
                            break
        finally:
            # Unexpected EOS: stream ended without closing tag while in State 0
            if self.state == 0 and self.inner_monologue_buffer:
                mono_text = self.inner_monologue_buffer.replace(OPEN_TAG, "").replace(CLOSE_TAG, "").strip()
                if mono_text:
                    self._monologue_sections.append(mono_text)
                self.state = 1
                self.is_failsafe_triggered = True
                logger.warning(
                    f"[StreamParser] Unexpected EOS in monologue after "
                    f"{self.monologue_token_count} tokens. Auto-closing and yielding public text."
                )
                yield mono_text
                self.public_response_buffer += mono_text
                self.inner_monologue_buffer = ""

    def get_final_buffers(self) -> Tuple[str, str]:
        """Returns (inner_monologue, public_response)."""
        # Append final buffer if we entered monologue mode and have remaining content
        if self.inner_monologue_buffer.strip() and self._in_monologue:
            self._monologue_sections.append(self.inner_monologue_buffer.strip())
        all_monologue = " ".join(self._monologue_sections)
        return all_monologue, self.public_response_buffer.strip()
