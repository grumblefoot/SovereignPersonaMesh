"""Character identifiers that are safe to embed in PostgreSQL table names.

Character names come from SillyTavern requests (system prompt, message `name`) and admin URLs, and are
formatted into `csa_memory_{id}` / `csa_lore_rules_{id}` table names. Every such name must go through
safe_char_id() first: it blocks SQL injection and keeps multi-word names ("Mira Vale") usable.
"""
import hashlib
import re

# PostgreSQL identifiers max out at 63 bytes; the longest prefix is "csa_lore_rules_" (15).
_MAX_LEN = 48


def safe_char_id(name: str, default: str = "default") -> str:
    """Lowercase and reduce to [a-z0-9_]; runs of anything else become a single underscore.

    A name with no Latin letters or digits at all (e.g. "美美", "Мира") used to collapse
    to `default`, so every such character shared ONE memory table. It now maps to a
    stable hash id instead (QA 2026-10-05)."""
    raw = (name or "").strip()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")[:_MAX_LEN].strip("_")
    if cleaned:
        return cleaned
    if any(ch.isalnum() for ch in raw):      # a real name in a non-Latin script
        return "c_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return default                            # empty or pure punctuation
