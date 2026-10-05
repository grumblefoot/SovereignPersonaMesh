"""Deterministic action parser for sensory gating (Sprint 2, chunk 1).

Parses one user message into an ordered list of world-action hints. Pure
Python: no LLM, no IO, no clock — deterministic by construction.

Design contract (docs/plans/gating.md §3; docs/plans/SPRINT_PLAN.md §6
decision 11 as amended): markers are FAIL-OPEN HINTS, never requirements.
Anything malformed — unclosed quote/emote, swapped delimiters, an unknown
``[tag]`` — degrades the whole span to plain ``speak`` at low confidence.
Content is never dropped: for a fully malformed message the emitted speak
contents reconstruct the input character for character.

Conventions (owner decision 11):
- ``"double quotes"`` -> SPEAK (curly ``\u201c\u201d`` accepted too)
- ``*asterisks*``     -> EMOTE; a movement verb plus a preposition phrase
                         (or a bare stair direction) upgrades it to MOVE with
                         the destination as ``target``
- ``'single quotes'`` -> THOUGHT (contractions like ``don't`` are safe: the
                         opener must sit at a word boundary)
- ``**OOC**``, ``((OOC))``, ``[OOC: ...]`` -> stripped; raises the ``ooc``
                         flag (see ``parse_message``), never a world event
- explicit tags ``[whisper:NAME]``, ``[move:PLACE]``, ``[move:A->PLACE]``,
  ``[shout]`` anywhere in the message win over every heuristic; a pending
  tag governs the next spoken/prose span (not emotes, thoughts, or moves)

Confidence scale (fixed vocabulary, asserted by tests):
- 1.0   explicit tag
- 0.95  well-formed quoted speech / thought
- 0.9   well-formed emote
- 0.7   MOVE with a parsed destination
- 0.6   all-caps-with-bang prose shout
- 0.5   plain prose speech
- 0.3   fail-open degraded span
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal["speak", "whisper", "shout", "move", "emote", "thought"]

# Confidence vocabulary (keep in sync with the docstring and tests).
CONF_TAG = 1.0
CONF_QUOTED = 0.95
CONF_EMOTE = 0.9
CONF_MOVE = 0.7
CONF_CAPS_SHOUT = 0.6
CONF_PLAIN = 0.5
CONF_FAIL_OPEN = 0.3

_OPEN_DOUBLE = "\"\u201c"
_CLOSE_DOUBLE = "\"\u201d"
_OPEN_SINGLE = "'\u2018\u2019"

_TAG_RE = re.compile(
    r"\[\s*"
    r"(?:(?P<kind>whisper|move)\s*:\s*(?P<value>[^\[\]\n]*[^\s\[\]])"
    r"|(?P<bare>shout))"
    r"\s*\]",
    re.IGNORECASE,
)
_OOC_BRACKET_RE = re.compile(r"\[\s*OOC\s*:([^\[\]\n]*?)\s*\]", re.IGNORECASE)
_UNKNOWN_BRACKET_RE = re.compile(r"\[[^\[\]\n]{1,80}\]")
_EMOTE_RE = re.compile(r"\*(?!\s)([^*\n]+?)(?<!\s)\*")
_OOC_STARS_RE = re.compile(r"\*\*(?!\s)([^*\n]+?)(?<!\s)\*\*")
_OOC_PAREN_RE = re.compile(r"\(\([^()\n]*?\)\)")
_THOUGHT_RE = re.compile(r"(['\u2018])(.+?)(['\u2019])(?![\w])")

_MOTION_VERBS = (
    r"go(?:es|ing)?|went|walk(?:s|ed|ing)?|head(?:s|ed|ing)?|"
    r"mov(?:e|es|ed|ing)|run(?:s|ing)?|ran|rush(?:es|ed|ing)?|"
    r"step(?:s|ped|ping)?|come(?:s|ing)?|came|return(?:s|ed|ing)?|"
    r"retreat(?:s|ed|ing)?|depart(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"leav(?:e|es|ing)|enter(?:s|ed|ing)?|approach(?:es|ed|ing)?|"
    r"advance|storm(?:s|ed|ing)?|charge[sd]?|charge(?:d|ing)?|"
    r"sneak(?:s|ed|ing)?|slip(?:s|ped|ping)?|dash(?:es|ed|ing)?|fly(?:s|ing)?|"
    r"slink(?:s|ed|ing)?|tiptoe(?:s|d|ing)?|prowl(?:s|ed|ing)?|mosey(?:s|d|ing)?"
)
_PREP_RE = re.compile(
    r"\b(?:" + _MOTION_VERBS + r")\b[^.!?\n]*?\b"
    r"(?:out\s+of|back\s+to|into|onto|towards|toward|from|for|to|in|up|down)\s+"
    r"(?:the\s+|a\s+|an\s+)?([A-Za-z][A-Za-z0-9 _'\-]{1,40}?)"
    r"(?=[\s,.;:!?]|\Z)",
    re.IGNORECASE,
)
_DIR_RE = re.compile(r"\b(upstairs|downstairs)\b", re.IGNORECASE)
_DOWNSTAIRS_RE = re.compile(r"\b(?:down|up)\s+(?:the\s+)?stairs\b", re.IGNORECASE)

_VOCATIVE_RE = re.compile(r"^\s*([A-Z][A-Za-z]+)[,:]")
_TO_NAME_RE = re.compile(r"\bto\s+([A-Z][a-z]+)\b(?=\s*[.,;:!?]|\s*$)")
_IN_EAR_RE = re.compile(r"in\s+([A-Z][a-z]+)'s\s+ear")


class ParsedAction(BaseModel):
    """One world-action hint parsed from a message span."""

    action_type: ActionType
    content: str = ""
    target: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ParsedMessage(BaseModel):
    """Full parse: ordered actions plus the OOC flag (markers stripped)."""

    actions: list[ParsedAction] = Field(default_factory=list)
    ooc: bool = False


def parse_user_message(text: str) -> list[ParsedAction]:
    """Parse a chat message into ordered action hints. Never raises."""
    return parse_message(text).actions


def parse_message(text: str) -> ParsedMessage:
    """Parse a chat message into ordered action hints plus the OOC flag."""
    result = ParsedMessage()
    if not text or not text.strip():
        return result

    actions: list[ParsedAction] = []
    ooc = False
    # Pending explicit tag (kind, target); governs the next spoken span.
    pending: Optional[tuple[ActionType, Optional[str]]] = None

    def emit_fail_open(content: str) -> None:
        """Degraded span: merge into adjacent prose — it is all speech."""
        content = content.strip()
        if not content:
            return
        if actions:
            prev = actions[-1]
            if (prev.action_type == "speak"
                    and prev.confidence in (CONF_PLAIN, CONF_FAIL_OPEN)):
                actions[-1] = ParsedAction(
                    action_type="speak",
                    content=(prev.content + " " + content).strip(),
                    target=prev.target, confidence=CONF_FAIL_OPEN)
                return
        actions.append(ParsedAction(action_type="speak", content=content,
                                    confidence=CONF_FAIL_OPEN))

    def emit_spoken(pos_content: str, confidence: float,
                    allow_target: bool) -> None:
        nonlocal pending
        content = pos_content.strip()
        if not content:
            return
        if pending:
            kind, target = pending
            if kind == "shout":
                actions.append(ParsedAction(action_type="shout",
                                            content=content,
                                            confidence=CONF_TAG))
            else:
                actions.append(ParsedAction(action_type="whisper",
                                            content=content, target=target,
                                            confidence=CONF_TAG))
            pending = None
            return
        if allow_target and _is_all_caps_shout(content):
            actions.append(ParsedAction(
                action_type="shout", content=content,
                target=_extract_target(content), confidence=CONF_CAPS_SHOUT))
            return
        if confidence == CONF_PLAIN and actions:
            prev = actions[-1]
            if prev.action_type == "speak" and prev.confidence == CONF_PLAIN:
                # Split prose (around a stray bracket) is one speech act.
                actions[-1] = ParsedAction(
                    action_type="speak",
                    content=(prev.content + " " + content).strip(),
                    target=prev.target, confidence=CONF_PLAIN)
                return
        actions.append(ParsedAction(
            action_type="speak", content=content,
            target=_extract_target(content) if allow_target else None,
            confidence=confidence))

    n = len(text)
    i = 0
    while i < n:
        c = text[i]

        # --- bracket forms ---------------------------------------------------
        if c == "[":
            m = _OOC_BRACKET_RE.match(text, i)
            if m:
                ooc = True
                i = m.end()
                continue
            m = _TAG_RE.match(text, i)
            if m:
                if m.group("bare"):  # [shout]
                    pending = ("shout", None)
                else:
                    kind = m.group("kind").lower()
                    value = m.group("value").strip()
                    if kind == "whisper":
                        pending = ("whisper", value or None)
                    else:  # [move:PLACE] / [move:Actor->PLACE]
                        dest = value
                        if "->" in dest:
                            dest = dest.split("->", 1)[1]
                        actions.append(ParsedAction(
                            action_type="move", content=value,
                            target=_titleize(dest) or None,
                            confidence=CONF_TAG))
                        pending = None
                i = m.end()
                continue
            m = _UNKNOWN_BRACKET_RE.match(text, i)
            if m:
                # Known shape, unknown name: fail-open as literal speech.
                emit_fail_open(m.group(0))
                i = m.end()
                continue
            # Stray '[': ordinary prose, keep the character.
            nxt = _next_marker(text, i + 1)
            emit_spoken(text[i:nxt], CONF_PLAIN, allow_target=True)
            i = nxt
            continue

        if c == "(" and text.startswith("((", i):
            m = _OOC_PAREN_RE.match(text, i)
            if m:
                ooc = True
                i = m.end()
                continue
            nxt = _next_marker(text, i + 2)
            emit_fail_open(text[i:nxt])
            i = nxt
            continue

        # --- **OOC** vs *emote* ----------------------------------------------
        if c == "*" and text.startswith("**", i):
            m = _OOC_STARS_RE.match(text, i)
            if m:
                ooc = True
                i = m.end()
                continue
            nxt = _next_marker(text, i + 2)
            emit_fail_open(text[i:nxt])
            i = nxt
            continue

        if c == "*":
            m = _EMOTE_RE.match(text, i)
            if m:
                inner = m.group(1).strip()
                dest = _match_move(inner)
                if dest:
                    actions.append(ParsedAction(
                        action_type="move", content=inner, target=dest,
                        confidence=CONF_MOVE))
                elif inner:
                    actions.append(ParsedAction(
                        action_type="emote", content=inner,
                        confidence=CONF_EMOTE))
                i = m.end()
                continue
            nxt = _next_marker(text, i + 1)  # unclosed emote: fail-open
            emit_fail_open(text[i:nxt])
            i = nxt
            continue

        # --- "speech" ----------------------------------------------------------
        if c in _OPEN_DOUBLE:
            closer = _find_closer(text, i + 1, _CLOSE_DOUBLE)
            if closer != -1:
                emit_spoken(text[i + 1:closer], CONF_QUOTED, allow_target=True)
                i = closer + 1
                continue
            nxt = _next_marker(text, i + 1)  # unclosed quote: fail-open
            emit_fail_open(text[i:nxt])
            i = nxt
            continue

        # --- 'thought' -----------------------------------------------------------
        if c in _OPEN_SINGLE and _is_thought_opener(text, i):
            m = _THOUGHT_RE.match(text, i)
            if (m is not None
                    and ((c == "'" and m.group(3) == "'")
                         or (c == "\u2018" and m.group(3) == "\u2019"))
                    and not _is_possessive_plural(text, m.end(3))):
                inner = m.group(2).strip()
                if inner:
                    actions.append(ParsedAction(action_type="thought",
                                                content=inner,
                                                confidence=CONF_QUOTED))
                i = m.end()
                continue
            nxt = _next_marker(text, i + 1)  # unclosed thought quote: fail-open
            emit_fail_open(text[i:nxt])
            i = nxt
            continue

        # --- plain prose ---------------------------------------------------------
        nxt = _next_marker(text, i + 1)
        emit_spoken(text[i:nxt], CONF_PLAIN, allow_target=True)
        i = nxt

    # A trailing explicit tag with no following span still fires its intent.
    if pending:
        kind, target = pending
        if kind == "shout":
            actions.append(ParsedAction(action_type="shout", content="",
                                        confidence=CONF_TAG))
        else:
            actions.append(ParsedAction(action_type="whisper", content="",
                                        target=target, confidence=CONF_TAG))

    result.actions = actions
    result.ooc = ooc
    return result


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _find_closer(text: str, start: int, closers: str) -> int:
    for k in range(start, len(text)):
        if text[k] in closers:
            return k
    return -1


def _is_thought_opener(text: str, i: int) -> bool:
    """True if text[i] can open a thought quote (never a contraction mark)."""
    if i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_'\u2019`"):
        return False
    nxt = text[i + 1] if i + 1 < len(text) else ""
    return bool(nxt) and not nxt.isspace()


def _is_possessive_plural(text: str, close_end: int) -> bool:
    """`the dogs' bones` — a closer followed by a word char is not a quote."""
    prev = text[close_end - 1] if close_end > 0 else ""
    nxt = text[close_end] if close_end < len(text) else ""
    return prev == "'" and nxt.isalnum()


def _next_marker(text: str, start: int) -> int:
    """First position at/after `start` where a marker could begin, else EOF."""
    for k in range(start, len(text)):
        c = text[k]
        if c == "*" or c in _OPEN_DOUBLE:
            return k
        if c == "(" and text.startswith("((", k):
            return k
        if c == "[":
            if (_TAG_RE.match(text, k) or _OOC_BRACKET_RE.match(text, k)
                    or _UNKNOWN_BRACKET_RE.match(text, k)):
                return k
        if c in _OPEN_SINGLE and _is_thought_opener(text, k):
            return k
    return len(text)


_MOTION_CTX_RE = re.compile(r"\b(?:" + _MOTION_VERBS + r")\b", re.IGNORECASE)


def _match_move(inner: str) -> Optional[str]:
    """Destination for an emote span, or None if it is just an emote."""
    dm = _DIR_RE.search(inner)
    if not dm:
        stm = _DOWNSTAIRS_RE.search(inner)
        if stm and _MOTION_CTX_RE.search(inner):
            return ("Downstairs" if stm.group(0).lower().startswith("down")
                    else "Upstairs")
    pm = _PREP_RE.search(inner)
    if pm:
        if dm and dm.start() < pm.start():
            return dm.group(1).capitalize()
        return _titleize(pm.group(1))
    if dm and _MOTION_CTX_RE.search(inner):
        return dm.group(1).capitalize()
    return None


def _titleize(dest: str) -> str:
    dest = dest.strip().strip(".!?,;:").strip()
    if not dest:
        return ""
    if dest.isupper() or dest.islower():
        return " ".join(w.capitalize() for w in dest.split())
    return dest


def _is_all_caps_shout(span: str) -> bool:
    letters = [c for c in span if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and "!" in span


_VOCATIVE_STOP = {"He", "She", "It", "They", "We", "You", "I", "But", "And",
                  "Then", "So", "No", "Yes", "Not", "This", "That", "There"}


def _extract_target(quote: str) -> Optional[str]:
    m = _IN_EAR_RE.search(quote)
    if m:
        return m.group(1)
    m = _VOCATIVE_RE.match(quote)
    if m and m.group(1) not in _VOCATIVE_STOP:
        return m.group(1)
    m = _TO_NAME_RE.search(quote)
    if m:
        return m.group(1)
    return None
