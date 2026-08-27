"""
Two-State Monologue Token Parsing State Machine with Fail-Safe Passthrough.
Parses monologue thoughts, writing monologue to private memory buffer, and streams public text to SillyTavern.
Includes fail-safe passthrough (>8192 tokens / malformed tags / unexpected EOS auto-close).
"""

import logging
import re
from typing import AsyncGenerator, List, Optional, Tuple

logger = logging.getLogger(__name__)

OPEN_TAGS = [
    "<ctrl94>", "<think>", "<thinking>", "<thought>", "<monologue>",
    "<channel:monologue>", "<channel:thought>", "<inner_monologue>", "<private>", "<system>",
    "<reasoning>", "<reason>", "<scratchpad>", "<details>",
    "<|thought|>", "<|start_thought|>", "<|channel:thought|>", "<|monologue|>", "<|ctrl94|>", "<|reasoning|>",
    "[Thought]", "[Monologue]", "[Thinking]", "[Inner Monologue]", "[Thought Process]", "[Reasoning]"
]
CLOSE_TAGS = [
    "</ctrl94>", "</think>", "</thinking>", "</thought>", "</monologue>",
    "</channel:monologue>", "</channel:thought>", "</inner_monologue>", "</private>", "</system>",
    "</reasoning>", "</reason>", "</scratchpad>", "</details>",
    "<|end_thought|>", "<|end_of_thought|>", "<|end_monologue|>", "</|thought|>", "</|monologue|>",
    "[/Thought]", "[/Monologue]", "[/Thinking]", "[/Inner Monologue]", "[/Thought Process]", "[/Reasoning]"
]
OPEN_TAG = "<ctrl94>"
CLOSE_TAG = "</ctrl94>"
MAX_MONOLOGUE_TOKENS = 8192

# Robust Regexes to catch malformed tags, missing brackets, markdown backticks, prompt directive echoes, and section headers
OPEN_TAG_REGEX = re.compile(
    r'(?:'
    r'`?\s*<ctrl94\b[^>]*>?|'
    r'<think\b[^>]*>?|'
    r'<thought\b[^>]*>?|'
    r'<monologue\b[^>]*>?|'
    r'<channel:monologue\b[^>]*>?|'
    r'<channel:thought\b[^>]*>?|'
    r'<inner_monologue\b[^>]*>?|'
    r'<private\b[^>]*>?|'
    r'<system\b[^>]*>?|'
    r'<reasoning\b[^>]*>?|'
    r'<reason\b[^>]*>?|'
    r'<scratchpad\b[^>]*>?|'
    r'<details\b[^>]*>?|'
    r'<\|thought\|>|<\|start_thought\|>|<\|channel:thought\|>|<\|monologue\|>|<\|ctrl94\|>|<\|reasoning\|>|'
    r'\[Thought\]|\[Monologue\]|\[Thinking\]|\[Inner Monologue\]|\[Thought Process\]|\[Reasoning\]|'
    r'"?You MUST begin your response immediately with\s*<ctrl94[^"\n]*"?|'
    r'\[Dialogue and Narration\]'
    r')',
    re.IGNORECASE
)

CLOSE_TAG_REGEX = re.compile(
    r'(?:'
    r'</ctrl94>?|'
    r'</think>?|'
    r'</thinking>?|'
    r'</thought>?|'
    r'</monologue>?|'
    r'</channel:monologue>?|'
    r'</channel:thought>?|'
    r'</inner_monologue>?|'
    r'</private>?|'
    r'</system>?|'
    r'</reasoning>?|'
    r'</reason>?|'
    r'</scratchpad>?|'
    r'</details>?|'
    r'<\|end_thought\|>|<\|end_of_thought\|>|<\|end_monologue\|>|</\|thought\|>|</\|monologue\|>|'
    r'\[/Thought\]|\[/Monologue\]|\[/Thinking\]|\[/Inner Monologue\]|\[/Thought Process\]|\[/Reasoning\]'
    r')',
    re.IGNORECASE
)

PARTIAL_TAG_REGEX = re.compile(r'(?:</?|</?\||\[)[a-zA-Z0-9_:|\-\s]{0,25}$')

MONOLOGUE_HEADER_REGEX = re.compile(
    r'^\s*(?:'
    r'>\s*|'
    r'[\*\-_•]\s*|'
    r'\d+[\.\)]\s*|'
    r'\([^\)]*(?:Thought|Monologue|Thinking|Self-Correction|Check|Drafting|Persona)[^\)]*\)|'
    r'\[[^\]]*(?:Thought|Monologue|Thinking|Self-Correction|Check|Drafting|Persona)[^\]]*\]|'
    r'\*+(?:Thought|Public|Monologue|Thinking|Check|Self-Correction|Drafting):\*+|'
    r'(?:Thought|Public|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning):|'
    r'System Directive:|'
    r'\[Dialogue and Narration\]'
    r')',
    re.IGNORECASE
)


class MonologueStreamParser:
    def __init__(self, max_public_tokens: Optional[int] = None, initial_state: int = 1):
        self.state = initial_state  # 1: Public Dialogue (default), 0: Monologue
        self.inner_monologue_buffer: str = ""
        self.public_response_buffer: str = ""
        self.monologue_token_count: int = 0
        self.public_token_count: int = 0
        self.max_public_tokens: Optional[int] = max_public_tokens
        self.is_public_truncated: bool = False
        self.is_failsafe_triggered: bool = False
        self.passthrough: bool = False
        self._monologue_sections: List[str] = []
        self._in_monologue: bool = (initial_state == 0)
        self._chunk_carryover: str = ""
        self._line_buffer: str = ""
        self._stream_buffer: str = ""

    def _check_public_limit(self, chunk: str) -> bool:
        """Helper to increment public token count and check if limit exceeded. Returns True if truncated."""
        tokens = len(chunk.split()) if chunk.strip() else 1
        self.public_token_count += tokens
        if self.max_public_tokens is not None and self.public_token_count >= self.max_public_tokens:
            self.is_public_truncated = True
            logger.info(f"[StreamParser] Reached max_public_tokens ({self.public_token_count} >= {self.max_public_tokens}). Truncating.")
            return True
        return False

    def _clean_monologue(self, text: str) -> str:
        """Removes open/close tags and leading blockquote symbols from monologue string."""
        cleaned = OPEN_TAG_REGEX.sub("", text)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")
        lines = [line[2:] if line.startswith("> ") else line for line in cleaned.split("\n")]
        return "\n".join(lines).strip()

    def _clean_public(self, text: str) -> str:
        """Filters out tags, blockquotes (>), and monologue headers from public response."""
        if not text:
            return ""
        cleaned = OPEN_TAG_REGEX.sub("", text)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")
        if not cleaned:
            return ""
        lines = []
        has_content = False
        for line in cleaned.split("\n"):
            if MONOLOGUE_HEADER_REGEX.search(line) or line.strip() == ".":
                continue
            lines.append(line)
            if line:
                has_content = True
        if not has_content:
            return ""
        return "\n".join(lines)

    def _strip_monologue_bleed(self, text: str) -> str:
        """Sanitizes public response buffer of leftover tags, trailing blockquote thoughts, or monologue bleed."""
        if not text:
            return ""
        cleaned = OPEN_TAG_REGEX.sub("", text)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")
        lines = []
        for line in cleaned.split("\n"):
            if MONOLOGUE_HEADER_REGEX.search(line) or line.strip() == ".":
                continue
            lines.append(line)
        result = "\n".join(lines).strip()
        result = re.sub(r'^\s*>\s*', '', result)
        return result

    def _is_partial_tag_prefix(self, text: str) -> bool:
        """Returns True if text is a valid prefix of any registered open or close tag."""
        if not text:
            return False
        for tag in OPEN_TAGS + CLOSE_TAGS:
            if tag.startswith(text):
                return True
        return False

    def _process_public_text(self, text: str) -> Tuple[str, str]:
        """Splits trailing partial open/close tag into carryover buffer."""
        if not text:
            return "", ""
        m = PARTIAL_TAG_REGEX.search(text)
        if m and self._is_partial_tag_prefix(m.group(0)):
            idx = m.start()
            return text[:idx], text[idx:]
        return text, ""

    async def process_token_stream(
        self, token_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Processes streaming token chunks from LLM backend.
        Yields ONLY public canon character response to SillyTavern while capturing inner monologue for SPM monitoring.
        """
        stop_stream = False
        try:
            async for raw_chunk in token_generator:
                if stop_stream:
                    break
                self._stream_buffer += self._chunk_carryover + raw_chunk
                self._chunk_carryover = ""

                if self.passthrough:
                    out = self._stream_buffer
                    self._stream_buffer = ""
                    truncated = self._check_public_limit(out)
                    self.public_response_buffer += out
                    yield out
                    if truncated:
                        stop_stream = True
                        break
                    continue

                while True:
                    if self.state == 1:
                        m_open = OPEN_TAG_REGEX.search(self._stream_buffer)
                        if m_open:
                            pre = self._stream_buffer[:m_open.start()]
                            if pre:
                                c_pre = self._clean_public(pre)
                                if c_pre:
                                    truncated = self._check_public_limit(c_pre)
                                    self.public_response_buffer += c_pre
                                    yield c_pre
                                    if truncated:
                                        stop_stream = True
                                        break
                            self._stream_buffer = self._stream_buffer[m_open.end():]
                            self.state = 0
                            self._in_monologue = True
                        else:
                            clean_txt, carry = self._process_public_text(self._stream_buffer)
                            if carry:
                                self._stream_buffer = carry
                            else:
                                self._stream_buffer = ""
                            if clean_txt:
                                c_clean = self._clean_public(clean_txt)
                                if c_clean:
                                    truncated = self._check_public_limit(c_clean)
                                    self.public_response_buffer += c_clean
                                    yield c_clean
                                    if truncated:
                                        stop_stream = True
                                        break
                            break

                    if self.state == 0:
                        m_close = CLOSE_TAG_REGEX.search(self._stream_buffer)
                        if m_close:
                            mono_part = self._stream_buffer[:m_close.start()]
                            full_mono = self.inner_monologue_buffer + mono_part
                            m_txt = self._clean_monologue(full_mono)
                            if m_txt:
                                self._monologue_sections.append(m_txt)
                            self.inner_monologue_buffer = ""
                            self._stream_buffer = self._stream_buffer[m_close.end():]
                            self.state = 1
                        else:
                            self.inner_monologue_buffer += self._stream_buffer
                            tokens = len(self._stream_buffer.split()) if self._stream_buffer.strip() else 1
                            self.monologue_token_count += tokens
                            self._stream_buffer = ""
                            self._in_monologue = True
                            if self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                                logger.warning(f"[StreamParser] Max monologue tokens reached (>8192 tokens). Passthrough.")
                                self.is_failsafe_triggered = True
                                self.passthrough = True
                                m_txt = self._clean_monologue(self.inner_monologue_buffer)
                                if m_txt:
                                    self._monologue_sections.append(m_txt)
                                    prefixed = f"\n> {m_txt}\n"
                                    self.public_response_buffer += prefixed
                                    yield prefixed
                                self.state = 1
                                self.inner_monologue_buffer = ""
                            break

            if self._stream_buffer and not stop_stream:
                if self.state == 1:
                    c_tail = self._clean_public(self._stream_buffer)
                    if c_tail:
                        if not self._check_public_limit(c_tail):
                            self.public_response_buffer += c_tail
                            yield c_tail
                else:
                    self.inner_monologue_buffer += self._stream_buffer
                self._stream_buffer = ""

        finally:
            if self.state == 0 and self.inner_monologue_buffer:
                mono_text = self._clean_monologue(self.inner_monologue_buffer)
                if mono_text:
                    self._monologue_sections.append(mono_text)
                    if not self.public_response_buffer:
                        self.is_failsafe_triggered = True
                        self.public_response_buffer += mono_text
                        yield mono_text
                self.inner_monologue_buffer = ""
                self.state = 1
                logger.info(f"[StreamParser] EOS reached in monologue mode. Monologue captured.")

    def get_final_buffers(self) -> Tuple[str, str]:
        """Returns (inner_monologue, public_response)."""
        if self.inner_monologue_buffer.strip() and self._in_monologue:
            clean = self._clean_monologue(self.inner_monologue_buffer)
            if clean:
                self._monologue_sections.append(clean)
            self.inner_monologue_buffer = ""
        all_monologue = "\n\n".join([s for s in self._monologue_sections if s])
        clean_public = self._strip_monologue_bleed(self.public_response_buffer)
        return all_monologue, clean_public
