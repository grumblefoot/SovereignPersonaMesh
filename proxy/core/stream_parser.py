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
    def __init__(self):
        self.state = 0  # 0: Monologue, 1: Public Dialogue
        self.inner_monologue_buffer: str = ""
        self.public_response_buffer: str = ""
        self.monologue_token_count: int = 0
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

    async def process_token_stream(
        self, token_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Processes streaming token chunks from LLM backend.
        Yields public SSE output chunks to SillyTavern while stripping private inner monologue.

        Fail-safe rules:
          1. >MAX_MONOLOGUE_TOKENS tokens in State 0 without closing tag -> passthrough.
          2. Unexpected EOS (stream ends) while in State 0 -> auto-close monologue,
             flush entire buffer as public, switch to State 1.
          3. Malformed or unclosed open tag -> treat as public output.
        """
        try:
            async for chunk in token_generator:
                if self.is_failsafe_triggered:
                    self.public_response_buffer += chunk
                    yield chunk
                    continue

                if self.state == 0:
                    # State 0: Monologue accumulation
                    if CLOSE_TAG in chunk:
                        # Normal close: transition to State 1 (Public)
                        parts = chunk.split(CLOSE_TAG)
                        self.inner_monologue_buffer += parts[0].replace(OPEN_TAG, "").strip()
                        self.state = 1
                        self._open_tag_seen = False  # reset for next State 0 entry
                        # Save completed section
                        if self.inner_monologue_buffer.strip():
                            self._monologue_sections.append(self.inner_monologue_buffer.strip())
                        logger.info(
                            f"[StreamParser] Monologue complete ({len(self.inner_monologue_buffer)} chars). "
                            f"Transitioning to State 1 (Public)."
                        )
                        if len(parts) > 1 and parts[1]:
                            self.public_response_buffer += parts[1]
                            yield parts[1]
                    elif "<ctrl" in chunk and OPEN_TAG not in chunk:
                        # Malformed open tag (e.g. "<ctrl9" without closing) -> passthrough
                        logger.warning(
                            f"[StreamParser] Malformed tag detected in monologue chunk. "
                            f"Switching to passthrough mode."
                        )
                        self.is_failsafe_triggered = True
                        self.state = 1
                        self.public_response_buffer += self.inner_monologue_buffer
                        yield self.inner_monologue_buffer
                        self.inner_monologue_buffer = ""
                        self.public_response_buffer += chunk
                        yield chunk
                    else:
                        if OPEN_TAG in chunk:
                            # First chunk with the open tag — accumulate it
                            self.inner_monologue_buffer += chunk
                            self.monologue_token_count += 1
                            self._open_tag_seen = True

                            if self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                                logger.warning(
                                    f"[StreamParser] Fail-Safe Passthrough Triggered (>500 tokens in monologue). "
                                    f"Auto-closing tag and switching to passthrough mode."
                                )
                                self.is_failsafe_triggered = True
                                self.state = 1
                                yield f"\n> {self.inner_monologue_buffer}\n"
                        elif self._open_tag_seen:
                            # We've seen <ctrl94> already — accumulate in monologue
                            self.inner_monologue_buffer += chunk
                            self.monologue_token_count += 1

                            if self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                                logger.warning(
                                    f"[StreamParser] Fail-Safe Passthrough Triggered (>500 tokens in monologue). "
                                    f"Auto-closing tag and switching to passthrough mode."
                                )
                                self.is_failsafe_triggered = True
                                self.state = 1
                                yield f"\n> {self.inner_monologue_buffer}\n"
                        else:
                            # No tag seen yet: yield as public (idle State 0)
                            self.public_response_buffer += chunk
                            yield chunk
                else:
                    # State 1: Public Dialogue
                    if OPEN_TAG in chunk:
                        logger.info(
                            f"[StreamParser] New monologue section detected in State 1. "
                            f"Re-entering State 0."
                        )
                        self._enter_state_0()
                        # Process this chunk as State 0
                        if CLOSE_TAG in chunk:
                            parts = chunk.split(CLOSE_TAG)
                            self.inner_monologue_buffer += parts[0].replace(OPEN_TAG, "").strip()
                            self.state = 1
                            self._open_tag_seen = False
                            if self.inner_monologue_buffer.strip():
                                self._monologue_sections.append(self.inner_monologue_buffer.strip())
                            if len(parts) > 1 and parts[1]:
                                self.public_response_buffer += parts[1]
                                yield parts[1]
                        elif "<ctrl" in chunk and OPEN_TAG not in chunk:
                            # Malformed
                            logger.warning(
                                f"[StreamParser] Malformed tag detected in monologue chunk. "
                                f"Switching to passthrough mode."
                            )
                            self.is_failsafe_triggered = True
                            self.state = 1
                            self.public_response_buffer += self.inner_monologue_buffer
                            yield self.inner_monologue_buffer
                            self.inner_monologue_buffer = ""
                            self.public_response_buffer += chunk
                            yield chunk
                        else:
                            self.inner_monologue_buffer += chunk
                            self.monologue_token_count += 1
                    else:
                        self.public_response_buffer += chunk
                        yield chunk
        finally:
            # Unexpected EOS: stream ended without closing tag
            # Only flush if we actually entered monologue mode (saw an open tag and accumulated content)
            if self.state == 0 and self.inner_monologue_buffer:
                logger.warning(
                    f"[StreamParser] Unexpected EOS in monologue (State 0) after "
                    f"{self.monologue_token_count} tokens. Auto-closing and flushing."
                )
                self.state = 1
                self.is_failsafe_triggered = True
                self._monologue_sections.append(self.inner_monologue_buffer.strip())
                yield f"\n> {self.inner_monologue_buffer}\n"
                self.public_response_buffer = self.inner_monologue_buffer
                self.inner_monologue_buffer = ""

    def get_final_buffers(self) -> Tuple[str, str]:
        """Returns (inner_monologue, public_response)."""
        # Append final buffer if we entered monologue mode and have remaining content
        if self.inner_monologue_buffer.strip() and self._in_monologue:
            self._monologue_sections.append(self.inner_monologue_buffer.strip())
        all_monologue = " ".join(self._monologue_sections)
        return all_monologue, self.public_response_buffer.strip()
