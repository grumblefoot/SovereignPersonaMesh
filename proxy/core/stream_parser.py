"""
Two-State Monologue Token Parsing State Machine with Fail-Safe Passthrough.
Parses monologue thoughts, writing monologue to private memory buffer, and streams public text to SillyTavern.
Includes fail-safe passthrough (>8192 tokens / malformed tags / unexpected EOS auto-close).
"""

import difflib
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
    # Bare "Plan:" / "Thought:" only at the start of a line: mid-sentence they are dialogue
    # ("The plan: we go at dawn") and used to truncate the reply (Hermes review H2).
    r'(?<=\n)[ \t]*\*?(?:Thought|Monologue|Thinking|Check|Self-Correction|Drafting|Thought Process|Internal Monologue|Reasoning|Plan|Analysis|Strategy):[ \t]*|'
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
    # "Plan" is NOT a close marker: a "Plan:" line inside the scratchpad is planning, and
    # treating it as the start of the public reply sent the plan to the user (QA Q5).
    r'\[Public\]|\[Public Response\]|\[Public Dialogue\]|\[Dialogue\]|\[Response\]|'
    r'\*+(?:Public|Public Response|Public Dialogue|Dialogue|Response|Narration|Canon Response):\*+|'
    r'(?<=\n)[ \t]*(?:Public|Public Response|Public Dialogue|Dialogue|Response|Narration|Canon Response):[ \t]*|'
    r'\[SCENE START\]|\[SCENE\]|\[NARRATION\]|\[RP\]|\[CHARACTER\]|\[PERSPECTIVE\]|'
    r'(?:^|\n)---+\s*(?:\n|$)'
    r')',
    re.IGNORECASE
)

def _line_search(regex, text: str, at_line_start: bool = True):
    """Search with knowledge of whether `text` begins at a line start. Line-start markers
    are written as (?<=\n); a streamed chunk usually begins mid-line, so the caller says.
    Returns (start, end) in `text`, or None."""
    probe = ("\n" if at_line_start else "\x00") + text
    m = regex.search(probe)
    if not m:
        return None
    return max(m.start() - 1, 0), max(m.end() - 1, 0)


def _line_sub(regex, text: str) -> str:
    """regex.sub("", text) treating the start of `text` as a line start."""
    res = regex.sub("", "\n" + text)
    return res[1:] if res.startswith("\n") else res


PARTIAL_TAG_REGEX = re.compile(r'(?:</?|</?\||\[)[a-zA-Z0-9_:|\-\s]{0,25}$')

MONOLOGUE_HEADER_REGEX = re.compile(
    r'^\s*(?:'
    r'>\s*|'
    # A bullet is "- ", "• ", "_ " or "* " (marker then space). "*Seraphina tilts her head*"
    # is an action in the "speech" *action* convention and must survive (QA Q5, 2026-10-13:
    # every line opening with an action was deleted, leaving an empty reply).
    r'(?:[\-_•]|\*(?=\s))\s*|'
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

# A whole public line that is a GM_ACTION tag (models sometimes emit these after closing the monologue).
GM_ACTION_LINE_REGEX = re.compile(r'^\s*\[GM_ACTION:.*\]\s*$', re.IGNORECASE)

# Leftover planning lines seen after the monologue closes: stray code fences, "Then the narrative.", "Done."
PUBLIC_META_LINE_REGEX = re.compile(
    r'^\s*(?:`+|(?:then|now)\s+(?:the\s+)?(?:narrative|narration|response|scene)\.?|done\.?)\s*$',
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
        self.is_failsafe_triggered: bool = False   # diagnostic: some salvage/failsafe path ran
        self.is_tainted: bool = False              # the reply must NOT become world state/memory
        self.passthrough: bool = False
        self._discard_rest: bool = False   # overflow failsafe: swallow the rest of the
                                           # stream into the (private) monologue sections
        self._monologue_sections: List[str] = []
        self._in_monologue: bool = (initial_state == 0)
        self._chunk_carryover: str = ""
        self._line_buffer: str = ""
        self._stream_buffer: str = ""
        self._public_gm_actions: List[str] = []
        self._at_line_start: bool = True   # whether the unconsumed stream begins a line

    def _note_consumed(self, text: str) -> None:
        if text:
            self._at_line_start = text.endswith("\n")

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
        """Extracts and parses all JSON [GM_ACTION: {...}] blocks from the inner monologue and public text.

        Duplicates (the same action written in both places) are dispatched once.
        """
        actions = []
        seen = set()
        all_text = "\n\n".join([s for s in self._monologue_sections if s]) + "\n" + self.inner_monologue_buffer
        all_text += "\n" + "\n".join(self._public_gm_actions)
        for match in GM_ACTION_REGEX.finditer(all_text):
            try:
                action_data = json.loads(match.group(1))
                key = json.dumps(action_data, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    actions.append(action_data)
            except json.JSONDecodeError as e:
                logger.error(f"[StreamParser] Failed to parse GM Action JSON: {e} | Payload: {match.group(1)}")
        return actions

    def _clean_monologue(self, text: str) -> str:
        """Removes open/close tags and leading blockquote symbols from monologue string."""
        cleaned = _line_sub(OPEN_TAG_REGEX, text)
        cleaned = _line_sub(CLOSE_TAG_REGEX, cleaned)
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

        # GM actions are world-engine commands, never narration: keep them for dispatch, hide them from the client.
        if GM_ACTION_LINE_REGEX.match(line):
            self._public_gm_actions.append(line.strip())
            return None

        if PUBLIC_META_LINE_REGEX.match(line):
            return None

        cleaned = _line_sub(OPEN_TAG_REGEX, line)
        cleaned = _line_sub(CLOSE_TAG_REGEX, cleaned)
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

    _PLANNING_START = re.compile(
        r"^\s*(?:Since I\b|Let me\b|Now I\b|Okay[,.]|Alright[,.]|"
        r"I(?:'ll| will| need to| should| am going to| cannot| can't| must)\b)", re.IGNORECASE)
    _META_VOCAB = re.compile(
        r"\b(?:describe|write|writing|reply|respond|response|the user|the prompt|"
        r"perspective|narrat\w*|scene|instruction)\b", re.IGNORECASE)
    SALVAGE_NOTICE = ("*[SPM: the model returned only planning notes, so this reply was "
                      "withheld and not saved. Regenerate to retry.]*")

    # A line that is the model talking to itself about the task, not narration.
    _PLAN_LINE = re.compile(
        r"^\s*(?:(?:Notes?|Plan|Goal|Reminder|Constraints?)\s*:|"
        r"(?:the user|the user's|user's|the player)\b|"
        r"I(?:'ll| will| must| need to| should| cannot| can't| am going to| have to)\b|"
        r"(?:Since I|Let me|Now I|Okay[,.]|Alright[,.]))", re.IGNORECASE)

    # First-person task talk inside a paragraph with no dialogue: "I will move Mei to the
    # kitchen", "I'll describe the room". In-character first person lives inside quotes.
    _PLAN_TASK = re.compile(
        r"\bI(?:'ll| will| must| need to| should| am going to)\s+"
        r"(?:move|describe|write|respond|reply|narrate|keep|maintain|portray|have [A-Z]\w+|focus|avoid|not)\b")
    # Future-tense stage direction without dialogue ("Lian will respond with a smile"): a plan.
    _STAGE_DIRECTION = re.compile(r"^\s*(?:I|[A-Z][\w’'\-]*(?:\s[A-Z][\w’'\-]*)?)\s+will\s+\w+")

    @staticmethod
    def _is_aside(para: str) -> bool:
        p = para.strip()
        return len(p) > 2 and p[0] in "([" and p[-1] in ")]." and p.rstrip(".")[-1] in ")]"

    def _is_planning_paragraph(self, para: str) -> bool:
        has_dialogue = any(q in para for q in ('"', '“'))
        return (bool(re.search(r"\bthe user\b", para, re.IGNORECASE))
                or any(self._PLAN_LINE.match(line.lstrip("([*- ")) for line in para.split("\n"))
                or (not has_dialogue and (self._is_aside(para)
                                          or bool(self._PLAN_TASK.search(para))
                                          or bool(self._STAGE_DIRECTION.match(para)))))

    def _is_prose_paragraph(self, para: str) -> bool:
        if self._is_aside(para):
            return False
        return len(para) >= 160 or any(q in para for q in ('"', '“', '*'))

    @staticmethod
    def _keep_final_draft(text: str) -> str:
        """Inside an unclosed scratchpad the model often writes a draft, then the final reply
        (present tense, then past tense). The pivot hands over both (QA 2026-10-13 Q3: the
        user saw Lian's reply twice). If the text ends with a near-copy of an earlier run of
        paragraphs, that earlier run and everything before it were drafting: keep the copy."""
        paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        n = len(paras)
        norm = [re.sub(r"\W+", " ", p.lower()).strip() for p in paras]
        # At least two paragraphs repeated, each a close rewrite (Hermes review H3: a single
        # refrain, or one loose match beside an exact copy, collapsed legitimate replies).
        for k in range(n // 2, 1, -1):
            tail = norm[n - k:]
            for i in range(0, n - 2 * k + 1):
                ratios = [difflib.SequenceMatcher(None, norm[i + j], tail[j]).ratio() for j in range(k)]
                if sum(ratios) / k >= 0.8 and all(r >= 0.7 for r in ratios):
                    logger.info(f"[StreamParser] Salvaged reply contained a draft and a final version; "
                                f"kept the final {k} paragraph(s), dropped {n - k}.")
                    return "\n\n".join(paras[n - k:])
        return text

    def _drop_leading_planning(self, text: str) -> str:
        """The model often writes GM actions first and keeps planning after them, so the
        last-GM_ACTION pivot hands over "Note: The user's action... I will describe..."
        followed by the real prose (QA N8, 2026-10-12). Drop the paragraphs before the
        first prose paragraph, but only when one of them is planning and prose follows:
        a short legitimate opening line is kept."""
        paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        for i, para in enumerate(paras):
            if not self._is_planning_paragraph(para) and self._is_prose_paragraph(para):
                head = paras[:i]
                planning = [p for p in head if self._is_planning_paragraph(p)]
                if planning:
                    # Only the planning paragraphs go; short narration among them stays
                    # (Hermes review H1: "Lian set the pot down…" was deleted with the aside).
                    logger.info(f"[StreamParser] Dropped {len(planning)} leading planning paragraph(s) "
                                f"from salvaged reply.")
                    return "\n\n".join([p for p in head if p not in planning] + paras[i:])
                return text
        return text

    def _guard_salvage(self, text: str) -> str:
        """Salvage paths only (the model never closed its private block): if what is
        left is the model talking about the WRITING TASK ("Since I cannot speak for
        Vardus, I will describe Lian's presence..."), withhold it and taint the turn
        instead of showing planning as the character's reply (QA F14, 2026-10-05).
        Needs both a planning-style opening and task vocabulary, so in-character
        first person ("I will make the tea.") is kept."""
        if not text:
            return text
        text = self._drop_leading_planning(self._keep_final_draft(text))
        first = text.strip().split("\n\n", 1)[0]
        # "the user" / a leading "Note:" never appears in narration (QA N8): withhold
        # too, without requiring the old planning-style opening.
        if ((self._PLANNING_START.match(first) and self._META_VOCAB.search(first))
                or re.search(r"\bthe user\b", first, re.IGNORECASE)
                or re.match(r"^\s*Notes?\s*:", first, re.IGNORECASE)):
            self.is_failsafe_triggered = True
            self.is_tainted = True
            logger.warning("[StreamParser] Salvaged text was planning, not a reply; withheld.")
            return self.SALVAGE_NOTICE
        return text

    def _strip_monologue_bleed(self, text: str) -> str:
        """Sanitizes public response buffer of leftover tags, trailing blockquote thoughts, or monologue bleed."""
        if not text:
            return ""

        # 1. NEW LOGIC: aggressively strip unclosed meta-headers that might have leaked into public text
        # Find all uppercase headers in brackets, e.g. [CHARACTER BEHAVIOR: ARVENIA], [ANTI-PUPPETING CHECK]
        # Ignore blocks that contain '{' or '"' which are usually JSON payloads like GM_ACTION
        m_blocks = list(re.finditer(r'^\[[A-Z0-9\s\-_:]+\]', text, flags=re.MULTILINE))
        if m_blocks:
            last_block = m_blocks[-1]
            # Assume everything up to and including the last block + its paragraph is bleed
            after_last = text[last_block.end():]
            # Find the end of the block's associated paragraph
            m_newline = re.search(r'\n\s*\n', after_last)
            if m_newline:
                text = after_last[m_newline.end():].strip()
            else:
                text = after_last.strip()

        # If text contains an explicit 'Plan:' or section divider, extract narrative content after it
        # Only a heading at the start of a line counts ("Plan:", "* Plan: offer tea",
        # "[Analysis]"), and its whole paragraph is planning. It used to match the word
        # anywhere and keep the rest of the line (QA Q5: "offer tea" leaked).
        # Only when the text OPENS with such a heading (Hermes review H5: an in-world note
        # quoted later in the reply, "Plan: meet me at the pier", was cut).
        m_plan = re.match(
            r'\s*(?:[\*\-•][ \t]+)?\[?(?:Plan|Strategy|Analysis|Draft|Notes|Planning|Internal Monologue)\b[^\]\n]*\]?:',
            text, re.IGNORECASE)
        if m_plan:
            m_end = re.search(r'\n\s*\n', text[m_plan.end():])
            after_plan = text[m_plan.end() + m_end.end():].strip() if m_end else ""
            if after_plan:
                text = after_plan

        # Strip explicit scene description headers but keep the content
        text = re.sub(r'\[SCENE DESCRIPTION\]\s*', '', text, flags=re.IGNORECASE)

        cleaned = _line_sub(OPEN_TAG_REGEX, text)
        cleaned = _line_sub(CLOSE_TAG_REGEX, cleaned)
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
            # Future-tense stage direction with no dialogue ("Mei will bow slightly,
            # acknowledge his request, and then move toward the kitchen.") is the
            # model's plan, not narration: RP prose is past tense. Leaked to the
            # user via the GM-action salvage path (QA F12, 2026-10-05).
            elif '"' not in p0 and '“' not in p0 and re.match(
                    r"^\s*(?:I|[A-Z][\w’'\-]*(?:\s[A-Z][\w’'\-]*)?)\s+will\s+\w+", p0):
                cleaned = "\n\n".join(paragraphs[1:])

        lines = []
        for line in cleaned.split("\n"):
            if line.strip().startswith("**GM Warning:**"):
                lines.append(line)
                continue
            if MONOLOGUE_HEADER_REGEX.search(line) or line.strip() == ".":
                continue
            # A bare label line such as "Lian's behavior:" is a planning header, never
            # narration (leaked as the first line of a reply, QA 2026-10-05).
            if re.match(r"^\s*[\w’' .-]{1,40}['’]s\s+(?:behaviou?r|response|reaction|actions?|reply|plan|state|"
                        r"next move|intentions?|perspective|pov|point of view|turn|dialogue|narration)\s*:\s*$",
                        line, re.IGNORECASE):
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

                if self._discard_rest:
                    # Still inside the runaway scratchpad: everything that follows is
                    # private planning. Keep it for the admin thoughts tab, never the user.
                    self.inner_monologue_buffer += self._stream_buffer
                    self._stream_buffer = ""
                    continue

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
                        m_open = _line_search(OPEN_TAG_REGEX, self._stream_buffer, self._at_line_start)
                        if m_open:
                            pre = self._stream_buffer[:m_open[0]]
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
                            self._note_consumed(self._stream_buffer[:m_open[1]])
                            self._stream_buffer = self._stream_buffer[m_open[1]:]
                            self.state = 0
                            self._in_monologue = True
                        else:
                            clean_txt, carry = self._process_public_text(self._stream_buffer)
                            self._note_consumed(clean_txt)
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
                        m_close = _line_search(CLOSE_TAG_REGEX, self._stream_buffer, self._at_line_start)
                        if m_close:
                            mono_part = self._stream_buffer[:m_close[0]]
                            full_mono = self.inner_monologue_buffer + mono_part
                            
                            m_txt = self._clean_monologue(full_mono)
                            if m_txt:
                                self._monologue_sections.append(m_txt)
                            self.inner_monologue_buffer = ""
                            self._note_consumed(self._stream_buffer[:m_close[1]])
                            self._stream_buffer = self._stream_buffer[m_close[1]:]
                            self.state = 1
                        else:
                            self.inner_monologue_buffer += self._stream_buffer
                            self._note_consumed(self._stream_buffer)
                            tokens = len(self._stream_buffer.split()) if self._stream_buffer.strip() else 1
                            self.monologue_token_count += tokens
                            self._stream_buffer = ""
                            self._in_monologue = True
                            if self.monologue_token_count > MAX_MONOLOGUE_TOKENS:
                                # Sprint 3 bleed hardening: the monologue is PRIVATE.
                                # The old failsafe dumped the whole scratchpad to the
                                # user as a blockquote — thousands of tokens of GM
                                # planning, lore checks and GM_ACTION JSON. Now it is
                                # kept for the admin thoughts tab, the user gets one
                                # notice, and the taint flag keeps the turn out of
                                # world state and memory (same as the EOF failsafe).
                                logger.warning("[StreamParser] Max monologue tokens reached "
                                               f"(>{MAX_MONOLOGUE_TOKENS}); monologue withheld from the stream.")
                                self.is_failsafe_triggered = True
                                self.is_tainted = True
                                self._discard_rest = True
                                m_txt = self._clean_monologue(self.inner_monologue_buffer)
                                if m_txt:
                                    self._monologue_sections.append(m_txt)
                                notice = ("\n*[SPM: the model never left its private scratchpad; "
                                          "this turn was withheld and not saved. Regenerate to retry.]*\n")
                                self.public_response_buffer += notice
                                yield notice
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
            if self._discard_rest and self.inner_monologue_buffer:
                # Overflow failsafe: the swallowed tail is monologue, never public.
                m_txt = self._clean_monologue(self.inner_monologue_buffer)
                if m_txt:
                    self._monologue_sections.append(m_txt)
                self.inner_monologue_buffer = ""
            if self.state == 0 and self.inner_monologue_buffer:
                mono_text = self._clean_monologue(self.inner_monologue_buffer)
                if mono_text:
                    self._monologue_sections.append(mono_text)
                    if not self.public_response_buffer:
                        # Check if mono_text contains an implicit public section split
                        m_close = _line_search(CLOSE_TAG_REGEX, mono_text)
                        if m_close:
                            clean_pub = self._guard_salvage(self._strip_monologue_bleed(mono_text[m_close[1]:]))
                            if clean_pub:
                                self.public_response_buffer += clean_pub
                                yield clean_pub
                        else:
                            # Pivot Parser Approach: scan for the explicit separator
                            m_sep = re.search(r'\n-+\n', mono_text)
                            if m_sep:
                                logger.info("[StreamParser] Unclosed monologue tag at EOF. Using separator '--' as pivot.")
                                clean_pub = self._guard_salvage(self._strip_monologue_bleed(mono_text[m_sep.end():]))
                                if clean_pub:
                                    self.public_response_buffer += clean_pub
                                    yield clean_pub
                            else:
                                # Fallback: scan for the *last* [GM_ACTION] as a pivot point
                                m_gm_matches = list(GM_ACTION_REGEX.finditer(mono_text))
                                if m_gm_matches:
                                    last_gm = m_gm_matches[-1]
                                    logger.info("[StreamParser] Unclosed monologue tag at EOF. Using last GM_ACTION as pivot.")
                                    clean_pub = self._guard_salvage(self._strip_monologue_bleed(mono_text[last_gm.end():]))
                                    if clean_pub:
                                        self.public_response_buffer += clean_pub
                                        yield clean_pub
                                else:
                                    logger.warning("[StreamParser] Unclosed monologue tag at EOF. Defaulting buffer to public.")
                                    self.is_failsafe_triggered = True   # diagnostic: a salvage path ran
                                    # F5 (QA 2026-10-05): a salvage path no longer means "don't save".
                                    # Gemma-4 leaves <think> unclosed on 5 of 8 replies, so tainting every
                                    # salvage made characters forget most of what they said. Only
                                    # is_tainted (planning guard / runaway scratchpad) blocks saving.
                                    clean_pub = self._guard_salvage(self._strip_monologue_bleed(mono_text))
                                    if clean_pub:
                                        self.public_response_buffer += clean_pub
                                        yield clean_pub
                    if not self.public_response_buffer:
                        # Salvage found nothing to show from a non-empty scratchpad. Never send
                        # a silent empty reply (QA Q5). If it holds planning, withhold it; if
                        # not (an in-world list the bullet rules deleted, Hermes review H4),
                        # the text IS the reply: pass it through, minus GM actions.
                        body = GM_ACTION_REGEX.sub("", mono_text).strip()
                        paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
                        if body and not any(self._is_planning_paragraph(p) for p in paras):
                            logger.info("[StreamParser] Salvage filters removed everything but no "
                                        "planning was found; passing the text through.")
                            self.public_response_buffer = body
                            yield body
                        else:
                            logger.warning("[StreamParser] Salvage produced no reply text; withheld.")
                            self.is_failsafe_triggered = True
                            self.is_tainted = True
                            self.public_response_buffer = self.SALVAGE_NOTICE
                            yield self.SALVAGE_NOTICE
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
        if self.public_response_buffer.strip() == self.SALVAGE_NOTICE:
            return all_monologue, self.SALVAGE_NOTICE     # Hermes review H8
        clean_public = self._strip_monologue_bleed(self.public_response_buffer)
        return all_monologue, clean_public
