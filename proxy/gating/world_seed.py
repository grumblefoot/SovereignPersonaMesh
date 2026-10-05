"""Deterministic world seeding from chat opening context (Sprint 2, chunk 2).

When GM_ACTION is off the world must still be seeded from the chat's opening
context without an LLM. Pure Python: no LLM, no IO of our own, no clock, no
randomness — deterministic by construction. The proxy wires this in later;
nothing imports it yet.

Design contract (docs/plans/gating.md §6, GM-off path):
1. Explicit tag ``[scene:TEMPLATE_KEY]`` anywhere in the system or first user
   text wins (confidence 1.0). An *unknown* template key inside the tag is
   treated as no tag at all — fail-open, fall through to keywords.
2. Keyword match: content words extracted from both texts (markup stripped
   with simple regexes) are scored via
   ``HybridWorldBuilder().match_template(keywords)``, which returns
   ``"generic_void"`` below its score floor. Confidence 0.65.
3. Fallback: ``"generic_void"``, confidence 0.3.

Placements: every named character (optional ``characters`` parameter) is
placed in the matched template's FIRST room (sorted room id) as a
deterministic default.
"""
from __future__ import annotations

import re
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from evennia_world.hybrid_builder import HybridWorldBuilder

# Confidence vocabulary for seeding (keep in sync with the docstring/tests).
CONF_TAG = 1.0
CONF_KEYWORDS = 0.65
CONF_FALLBACK = 0.3

FALLBACK_TEMPLATE = "generic_void"

SeedSource = Literal["tag", "keywords", "fallback"]

# Explicit seeding tag: [scene:TEMPLATE_KEY] (whitespace tolerated, case-free).
_SCENE_TAG_RE = re.compile(r"\[\s*scene\s*:\s*([A-Za-z0-9_\-]+)\s*\]",
                           re.IGNORECASE)

# Markup scrubbed before keyword extraction (OOC forms, emphasis, tags).
_OOC_BRACKET_RE = re.compile(r"\[\s*\*{0,2}\s*OOC\b[^\[\]\n]*?\*{0,2}\s*\]",
                             re.IGNORECASE)
_OOC_PAREN_RE = re.compile(r"\(\([^()\n]*?\)\)")
_OOC_STARS_RE = re.compile(r"\*{2}\s*OOC\b[^*\n]*?\*{2}", re.IGNORECASE)
_SCENE_TAG_STRIP_RE = re.compile(r"\[\s*scene\s*:[^\[\]\n]*?\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[[^\[\]\n]{0,80}\]")
_STAR_RE = re.compile(r"\*{1,2}")
_PAREN_RE = re.compile(r"\((?:[^()\n]*)\)")

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Function words that carry no scene signal.
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
he her here hers him his how i if in is it its me might my no not of on our
out she should so some than that the their them then there these they this
those to too us was we were what when where which who will with would you your
am about above after again all also any because before being below between both
cannot down during each few further more most other ours over own same s t
until very via while don dont didn isnt aren wont cant
""".split())

# Tokens shorter than this are dropped (unless they are template keywords —
# the builder decides; two-letter words are noise for scene matching anyway).
_MIN_TOKEN_LEN = 3


class SeedPlacement(BaseModel):
    """One deterministic character -> room placement in the seeded world."""

    character_id: str
    room_id: str


class SeedProposal(BaseModel):
    """What template to seed and who starts where (GM-off seeding path)."""

    template_key: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: SeedSource
    placements: list[SeedPlacement] = Field(default_factory=list)


def propose_world_seed(system_text: str,
                       first_user_text: str,
                       characters: Optional[Sequence[str]] = None,
                       builder: Optional[HybridWorldBuilder] = None) -> SeedProposal:
    """Propose a world seed from the opening context. Never raises.

    Priority: explicit ``[scene:KEY]`` tag > keyword match > generic_void.
    ``characters`` (if given) are all placed in the template's first room
    (sorted room id) preserving input order, deduplicated.
    """
    system_text = system_text or ""
    first_user_text = first_user_text or ""
    builder = builder if builder is not None else HybridWorldBuilder()

    template_key, confidence, source = _match_template(
        system_text, first_user_text, builder)

    return SeedProposal(
        template_key=template_key,
        confidence=confidence,
        source=source,
        placements=_placements(template_key, characters or (), builder),
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _match_template(system_text: str, user_text: str,
                    builder: HybridWorldBuilder):
    """Resolve (template_key, confidence, source) per the priority rules."""
    known = set(builder.templates.keys())

    # 1. Explicit tag — first occurrence in system text, then user text.
    for text in (system_text, user_text):
        for m in _SCENE_TAG_RE.finditer(text):
            key = m.group(1)
            if key in known:
                return key, CONF_TAG, "tag"
            # Unknown key: fail-open, keep scanning / fall through.

    # 2. Keyword match (match_template returns generic_void below its floor).
    keywords = extract_keywords(system_text, user_text)
    if keywords:
        matched = builder.match_template(keywords)
        if matched != FALLBACK_TEMPLATE:
            return matched, CONF_KEYWORDS, "keywords"

    # 3. Fallback.
    return FALLBACK_TEMPLATE, CONF_FALLBACK, "fallback"


def extract_keywords(*texts: str) -> list[str]:
    """Lowercased content words from markup-streaked chat text.

    Deterministic: OOC/emote/tag markup is stripped with simple regexes,
    tokens are stop-worded and deduplicated preserving first-seen order.
    """
    keywords: list[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = _strip_markup(text)
        for m in _TOKEN_RE.finditer(cleaned):
            word = m.group(0).lower().replace("'", "")
            if len(word) < _MIN_TOKEN_LEN or word in _STOPWORDS:
                continue
            if word not in seen:
                seen.add(word)
                keywords.append(word)
    return keywords


def _strip_markup(text: str) -> str:
    """Remove OOC spans, scene tags, brackets, emphasis stars, parentheses."""
    if not text:
        return ""
    text = _OOC_BRACKET_RE.sub(" ", text)
    text = _OOC_PAREN_RE.sub(" ", text)
    text = _OOC_STARS_RE.sub(" ", text)
    text = _SCENE_TAG_STRIP_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = _PAREN_RE.sub(" ", text)
    text = _STAR_RE.sub(" ", text)
    return text


def _placements(template_key: str, characters: Sequence[str],
                builder: HybridWorldBuilder) -> list[SeedPlacement]:
    """Place every named character in the template's first (sorted) room."""
    rooms = sorted(builder.templates.get(template_key, {}).keys())
    if not rooms:
        return []
    first_room = rooms[0]
    placements: list[SeedPlacement] = []
    seen: set[str] = set()
    for raw in characters:
        cid = (raw or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        placements.append(SeedPlacement(character_id=cid, room_id=first_room))
    return placements
