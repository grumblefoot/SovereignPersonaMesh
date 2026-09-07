"""
Two-State Monologue Token Parsing State Machine with Fail-Safe Passthrough.
Parses monologue thoughts, writing monologue to private memory buffer, and streams public text to SillyTavern.
Includes fail-safe passthrough (>8192 tokens / malformed tags / unexpected EOS auto-close).
"""

import logging
import re
import json
from typing import AsyncGenerator, List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

OPEN_TAGS = [
    "<think\b>", "<think>", "<thinking>", "<thought>", "<monologue>",
    "<channel:monologue>", "<channel:thought>", "<inner_monologue>", "<private>", "<system>",
    "<reasoning>", "<reason>", "<scratchpad>", "<details>",
    "<|thought|>", "<|start_thought|>", "<|channel:thought|>", "<|monologue|>", "<|think|>", "<|reasoning|>",
    "[Thought]", "[Monologue]", "[Thinking]", "[Inner Monologue]", "[Thought Process]", "[Reasoning]"
]
CLOSE_TAGS = [
    "</thinking>", "</think>", "</thinking>", "</thought>", "</monologue>",
    "</channel:monologue>", "</channel:thought>", "</inner_monologue>", "</private>", "</system>",
    "</reasoning>", "</reason>", "</scratchpad>", "</details>",
    "<|end_thought|>", "<|end_of_thought|>", "<|end_monologue|>", "</|thought|>", "</|monologue|>",
    "[/Thought]", "[/Monologue]", "[/Thinking]", "[/Inner Monologue]", "[/Thought Process]", "[/Reasoning]"
]
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
MAX_MONOLOGUE_TOKENS = 8192

# Robust Regexes to catch malformed tags, missing brackets, markdown backticks, prompt directive echoes, and section headers
OPEN_TAG_REGEX = re.compile(
    r'(?:'
    r'`?\s*<thinking\b[^>]*>?|'
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
    r'<\|thought\|>|<\|start_thought\|>|<\|channel:thought\|>|<\|monologue\|>|<\|think\|>|<\|reasoning\|>|'
    r'\[Thought\]|\[Monologue\]|\[Thinking\]|\[Inner Monologue\]|\[Thought Process\]|\[Reasoning\]|'
    r'\*+(?:Thought|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning|Plan|Analysis|Strategy):\*+|'
    r'(?:\*|\b)(?:Thought|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning|Plan|Analysis|Strategy):\s*|'
    r'\b(?:[A-Z][a-z0-9_\'\s]{0,30})?reaction should be\b|'
    r'"?You MUST begin your response immediately with\s*<thinking[^"\n]*"?|'
    r'\[Dialogue and Narration\]'
    r')',
    re.IGNORECASE
)

CLOSE_TAG_REGEX = re.compile(
    r'(?:'
    r'</thinking>?|'
    r'</think\b>?|'
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
    r'\[/Thought\]|\[/Monologue\]|\[/Thinking\]|\[/Inner Monologue\]|\[/Thought Process\]|\[/Reasoning\]|'
    r'\[Public\]|\[Public Response\]|\[Public Dialogue\]|\[Dialogue\]|\[Response\]|\[Plan\]|'
    r'\*+(?:Public|Public Response|Public Dialogue|Dialogue|Response|Narration|Canon Response|Plan):\*+|'
    r'(?:\*|\b)(?:Public|Public Response|Public Dialogue|Dialogue|Response|Narration|Canon Response|Plan):\s*|'
    r'\[SCENE START\]|\[SCENE\]|\[NARRATION\]|\[RP\]|\[CHARACTER\]|\[PERSPECTIVE\]|'
    r'(?:^|\n)---+\s*(?:\n|$)'
    r')',
    re.IGNORECASE
)

PARTIAL_TAG_REGEX = re.compile(r'(?:</?|</?\||\[)[a-zA-Z0-9_:|\-\s]{0,25}$')

MONOLOGUE_HEADER_REGEX = re.compile(
    r'^\s*(?:'
    r'>\s*|'
    r'[\*\-_•]\s*|'
    r'\d+[\.\)]\s*|'
    r'\([^\)]*(?:Thought|Monologue|Thinking|Self-Correction|Check|Drafting|Persona|Plan|Analysis)[^\)]*\)|'
    r'\[[^\]]*(?:Thought|Monologue|Thinking|Self-Correction|Check|Drafting|Persona|Plan|Analysis)[^\]]*\]|'
    r'\*+(?:Thought|Public|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning|Plan|Analysis|Strategy):\*+|'
    r'(?:Thought|Public|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning|Plan|Analysis|Strategy):|'
    r'System Directive:|'
    r'\[Dialogue and Narration\]|'
    r'a mix of\b|a blend of\b|a combination of\b'
    r')',
    re.IGNORECASE
)

GM_ACTION_REGEX = re.compile(r'\[GM_ACTION:\s*(\{.*?\})\]', re.DOTALL | re.IGNORECASE)


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

    def extract_gm_actions(self) -> List[Dict]:
        """Extracts and parses all JSON [GM_ACTION: {...}] blocks from the inner monologue."""
        actions = []
        all_text = "\n\n".join([s for s in self._monologue_sections if s]) + "\n" + self.inner_monologue_buffer
        for match in GM_ACTION_REGEX.finditer(all_text):
            try:
                action_data = json.loads(match.group(1))
                actions.append(action_data)
            except json.JSONDecodeError as e:
                logger.error(f"[StreamParser] Failed to parse GM Action JSON: {e} | Payload: {match.group(1)}")
        return actions

    def _clean_monologue(self, text: str) -> str:
        """Removes open/close tags and leading blockquote symbols from monologue string."""
        cleaned = OPEN_TAG_REGEX.sub("", text)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")
        lines = [line[2:] if line.startswith("> ") else line for line in cleaned.split("\n")]
        return "\n".join(lines).strip()

    def _clean_public_line(self, line: str) -> Optional[str]:
        """
        Cleans a single line of public text.
        Returns the cleaned string, or None if the entire line should be dropped (e.g. it's meta-commentary).
        """
        if not line.strip():
            return line
            
        cleaned = OPEN_TAG_REGEX.sub("", line)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")
            
        if not cleaned.strip():
            return ""
            
        if MONOLOGUE_HEADER_REGEX.search(cleaned) or cleaned.strip() == ".":
            return None
            
        if re.search(
            r'\b(?:reaction should be|should lean into|should be a blend of|internal plan|planning notes|perceives (?:her|him|them)self as|has just insulted|is vain and)\b',
            cleaned,
            re.IGNORECASE
        ):
            return None
            
        if re.search(
            r'^\s*(?:a mix of|a blend of|a combination of|an expression of|an array of|reacting to|responding to|given that|in this turn)\b|'
            r'\b(?:perceives (?:her|him|them)self|insulted (?:her|his|their) appearance|echoing common|peasant misconceptions|supreme elegance|meta-commentary|character motivation|vibe profiling|has just insulted)\b',
            cleaned,
            re.IGNORECASE
        ):
            return None
            
        if re.match(r'^\s*(?:Plan|Strategy|Analysis|Draft|Notes):\s*$', cleaned, re.IGNORECASE):
            return None
            
        return cleaned

    def _process_public_output(self, text: str, force_flush: bool = False) -> List[str]:
        """
        Appends text to the line buffer and extracts cleaned complete lines.
        If force_flush is True, also extracts any remaining text without a newline.
        """
        self._line_buffer += text
        outputs = []
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            c_line = self._clean_public_line(line)
            if c_line is not None:
                outputs.append(c_line + "\n")
        
        if force_flush and self._line_buffer:
            c_tail = self._clean_public_line(self._line_buffer)
            if c_tail is not None:
                outputs.append(c_tail)
            self._line_buffer = ""
            
        return outputs

    def _strip_monologue_bleed(self, text: str) -> str:
        """Sanitizes public response buffer of leftover tags, trailing blockquote thoughts, or monologue bleed."""
        if not text:
            return ""
        # If text contains an explicit 'Plan:' or section divider, extract narrative content after it
        m_plan = re.search(r'\b(?:Plan|Strategy|Analysis|Draft|Notes|Planning|Internal Monologue):\s*', text, re.IGNORECASE)
        if m_plan:
            after_plan = text[m_plan.end():].strip()
            if after_plan:
                text = after_plan

        cleaned = OPEN_TAG_REGEX.sub("", text)
        cleaned = CLOSE_TAG_REGEX.sub("", cleaned)
        for tag in OPEN_TAGS + CLOSE_TAGS:
            cleaned = cleaned.replace(tag, "")

        # Split into paragraphs to detect un-tagged opening meta-analysis / character state summaries
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            p0 = paragraphs[0]
            # Check if first paragraph is meta-analysis / prompt reflection / character state breakdown
            if re.search(
                r'^\s*(?:a mix of|a blend of|a combination of|an expression of|an array of|reacting to|responding to|given that|in this turn|the user\'s response|the user is|i need to|the room should|room details)\b|'
                r'\b(?:perceives (?:her|him|them)self|insulted (?:her|his|their) appearance|echoing common|peasant misconceptions|supreme elegance|meta-commentary|character motivation|vibe profiling|has just insulted|is vain and)\b',
                p0,
                re.IGNORECASE
            ):
                cleaned = "\n\n".join(paragraphs[1:])

        lines = []
        for line in cleaned.split("\n"):
            if line.strip().startswith("**GM Warning:**"):
                lines.append(line)
                continue
            if MONOLOGUE_HEADER_REGEX.search(line) or line.strip() == ".":
                continue
            # Drop lines that are LLM internal prompt analysis or guidelines
            if re.search(
                r'\b(?:reaction should be|should lean into|should be a blend of|internal plan|planning notes|perceives (?:her|him|them)self as|has just insulted)\b',
                line,
                re.IGNORECASE
            ):
                continue
            # Aggressively drop bulleted meta-commentary lines
            if re.match(r'^\s*[-*]\s+(?:Arvenia\'s )?(?:Reaction|Characterization|Action|Goal|Internal|Response|Note):', line, re.IGNORECASE):
                continue
            # Also drop stray bullet points that just say "She should..." or "He is..." if it looks like planning
            if re.match(r'^\s*[-*]\s+(?:She|He|It) (?:should|needs to|will) ', line, re.IGNORECASE):
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
                            if pre or self._line_buffer:
                                outputs = self._process_public_output(pre, force_flush=True)
                                for out in outputs:
                                    truncated = self._check_public_limit(out)
                                    self.public_response_buffer += out
                                    yield out
                                    if truncated:
                                        stop_stream = True
                                        break
                                if stop_stream:
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
                                outputs = self._process_public_output(clean_txt, force_flush=False)
                                for out in outputs:
                                    truncated = self._check_public_limit(out)
                                    self.public_response_buffer += out
                                    yield out
                                    if truncated:
                                        stop_stream = True
                                        break
                                if stop_stream:
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
                    outputs = self._process_public_output(self._stream_buffer, force_flush=True)
                    for out in outputs:
                        truncated = self._check_public_limit(out)
                        self.public_response_buffer += out
                        yield out
                        if truncated:
                            break
                else:
                    self.inner_monologue_buffer += self._stream_buffer
                self._stream_buffer = ""

            if self.state == 1 and self._line_buffer and not stop_stream:
                outputs = self._process_public_output("", force_flush=True)
                for out in outputs:
                    truncated = self._check_public_limit(out)
                    self.public_response_buffer += out
                    yield out
                    if truncated:
                        break

        finally:
            if self.state == 0 and self.inner_monologue_buffer:
                mono_text = self._clean_monologue(self.inner_monologue_buffer)
                if mono_text:
                    self._monologue_sections.append(mono_text)
                    if not self.public_response_buffer:
                        # Check if mono_text contains an implicit public section split
                        m_close = CLOSE_TAG_REGEX.search(mono_text)
                        if m_close:
                            clean_pub = self._strip_monologue_bleed(mono_text[m_close.end():])
                            if clean_pub:
                                self.public_response_buffer += clean_pub
                                yield clean_pub
                        else:
                            # Pivot Parser Approach: scan for the *last* [GM_ACTION] as a pivot point
                            m_gm_matches = list(GM_ACTION_REGEX.finditer(mono_text))
                            if m_gm_matches:
                                last_gm = m_gm_matches[-1]
                                logger.info("[StreamParser] Unclosed monologue tag at EOF. Using last GM_ACTION as pivot.")
                                clean_pub = self._strip_monologue_bleed(mono_text[last_gm.end():])
                                if clean_pub:
                                    self.public_response_buffer += clean_pub
                                    yield clean_pub
                            else:
                                logger.warning("[StreamParser] Unclosed monologue tag at EOF. Defaulting buffer to public.")
                                self.is_failsafe_triggered = True
                                clean_pub = self._strip_monologue_bleed(mono_text)
                                if clean_pub:
                                    self.public_response_buffer += clean_pub
                                    yield clean_pub
                self.inner_monologue_buffer = ""
                self.state = 1
                logger.info(f"[StreamParser] EOS reached in monologue mode. Monologue captured.")

            # Extract GM Warning from monologue sections and yield it at the very end
            all_monologue = "\n\n".join([s for s in self._monologue_sections if s])
            m_warn = re.search(r'\[GM WARNING:\s*(.*?)\]', all_monologue, re.IGNORECASE | re.DOTALL)
            if m_warn:
                warning_text = f"\n\n**GM Warning:** {m_warn.group(1).strip()}"
                self.public_response_buffer += warning_text
                yield warning_text

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
