"""Character identifiers that are safe to embed in PostgreSQL table names.

Character names come from SillyTavern requests (system prompt, message `name`) and admin URLs, and are
formatted into `csa_memory_{id}` / `csa_lore_rules_{id}` table names. Every such name must go through
safe_char_id() first: it blocks SQL injection and keeps multi-word names ("Mira Vale") usable.
"""
import re

# PostgreSQL identifiers max out at 63 bytes; the longest prefix is "csa_lore_rules_" (15).
_MAX_LEN = 48


def safe_char_id(name: str, default: str = "default") -> str:
    """Lowercase and reduce to [a-z0-9_]; runs of anything else become a single underscore."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")[:_MAX_LEN].strip("_")
    return cleaned or default
