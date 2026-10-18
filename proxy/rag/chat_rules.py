"""Store module for ``spm_chat_rules`` — one rules table for all chats (brief A1).

GM rules move from one table per character (``csa_lore_rules_<char>``) to a single
table with three levels:

- ``world``               — every character in the chat
- ``character_internal``  — only that character
- ``character_external``  — that character, plus characters in the same room

A character's prompt gets: world rules, its own internal and external rules, the
external rules of the other characters in its room, and its room's active rules
(from the world engine snapshot; the caller supplies those, they live in the
engine, not this table). Only ``active`` rules, only this chat.

Every function takes an asyncpg ``conn`` as its first argument. No other module
touches ``spm_chat_rules`` directly.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

LEVELS = ("world", "character_internal", "character_external")
RULE_TYPES = ("invariant", "conditional_trigger", "game_over")
STATUSES = ("pending", "active")

# Cosine distance below which a new rule restates an existing one. Same constant
# the per-character extractor used (lore_extractor.NEAR_DUPLICATE_DISTANCE, QA O4):
# measured paraphrases <= 0.17, distinct rules >= 0.25.
NEAR_DUPLICATE_DISTANCE = 0.20

# Cap on the room's (engine snapshot) rules appended per turn — brief A1 spec 3.
ROOM_RULES_PER_TURN = 5

_UPDATABLE_FIELDS = ("rule_text", "rule_type", "level", "character_id", "status")


def _validate_level(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"invalid rule level {level!r}; expected one of {LEVELS}")


def _validate_rule_type(rule_type: str) -> None:
    if rule_type not in RULE_TYPES:
        raise ValueError(f"invalid rule_type {rule_type!r}; expected one of {RULE_TYPES}")


def _validate_status(status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"invalid rule status {status!r}; expected one of {STATUSES}")


def _embedding_str(embedding: Sequence[float]) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"


def _rule_dict(row: Any, *, full: bool = False) -> Dict[str, Any]:
    rule = {
        "id": str(row["id"]),
        "rule_text": row["rule_text"],
        "rule_type": row["rule_type"],
        "level": row["level"],
        "character_id": row["character_id"],
    }
    if full:
        rule["status"] = row["status"]
        rule["source"] = row["source"]
        ts = row["created_at"] if "created_at" in row.keys() else None
        rule["created_at"] = ts.isoformat() if ts else None
    if "cosine_distance" in row.keys():
        rule["cosine_distance"] = row["cosine_distance"]
    return rule


async def add_rule(
    conn,
    *,
    session_id: str,
    level: str,
    character_id: Optional[str],
    rule_text: str,
    rule_type: str,
    status: str = "pending",
    source: str = "extracted",
    embedding: Optional[Sequence[float]] = None,
    space_id: Optional[int] = None,
) -> Optional[str]:
    """Insert one rule; returns its id (str), or None if it duplicates an
    existing rule of this chat.

    Duplicate = same ``session_id`` and the same audience (a ``world`` rule, or a
    rule for the same character — a world rule already covers every character, so
    a character restating one is a duplicate, while restating a single character's
    rule at world level covers new characters and is allowed), and either the
    exact same text (case-insensitive, trimmed) or — when ``embedding`` and
    ``space_id`` are both given — cosine distance < NEAR_DUPLICATE_DISTANCE
    within the same embedding space.
    """
    _validate_level(level)
    _validate_rule_type(rule_type)
    _validate_status(status)
    if not rule_text or not rule_text.strip():
        raise ValueError("rule_text must not be empty")
    if level == "world":
        character_id = None
    elif not character_id:
        raise ValueError(f"level {level!r} requires a character_id")

    # Duplicate scope: world rules are visible to every character of the chat,
    # character rules only to their own audience. For a world-level new rule
    # ``character_id`` is NULL, so only existing world rows can match.
    dup_where = "session_id = $1 AND (level = 'world' OR character_id = $2)"
    existing = await conn.fetchval(
        f"SELECT 1 FROM spm_chat_rules WHERE {dup_where} "
        "AND LOWER(TRIM(rule_text)) = LOWER(TRIM($3)) LIMIT 1",
        session_id, character_id, rule_text)
    if existing:
        return None

    if embedding is not None and space_id is not None:
        # The space filter runs inside the OFFSET 0 fence, before the distance
        # operator: the vector column is dimension-free and <=> errors on a
        # dimension mismatch between spaces.
        near = await conn.fetchval(
            f"""
            SELECT 1 FROM (
                SELECT rule_embedding FROM spm_chat_rules
                 WHERE session_id = $2
                   AND (level = 'world' OR character_id = $3)
                   AND rule_embedding IS NOT NULL
                   AND embedding_space_id IS NOT DISTINCT FROM $4
                OFFSET 0
            ) candidates
            WHERE (rule_embedding <=> $1::vector) < $5
            LIMIT 1
            """,
            _embedding_str(embedding), session_id, character_id, space_id,
            NEAR_DUPLICATE_DISTANCE)
        if near:
            return None

    rule_id = await conn.fetchval(
        """
        INSERT INTO spm_chat_rules
            (session_id, level, character_id, rule_text, rule_type, status, source,
             rule_embedding, embedding_space_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)
        RETURNING id
        """,
        session_id, level, character_id, rule_text, rule_type, status, source,
        None if embedding is None else _embedding_str(embedding), space_id)
    return str(rule_id)


async def list_rules(
    conn,
    session_id: str,
    *,
    level: Optional[str] = None,
    character_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[dict]:
    """Rules of one chat, optionally filtered by level, character_id and status."""
    if level is not None:
        _validate_level(level)
    if status is not None:
        _validate_status(status)
    rows = await conn.fetch(
        """
        SELECT id, rule_text, rule_type, level, character_id, status, source, created_at
        FROM spm_chat_rules
        WHERE session_id = $1
          AND ($2::varchar IS NULL OR level = $2)
          AND ($3::varchar IS NULL OR character_id = $3)
          AND ($4::varchar IS NULL OR status = $4)
        ORDER BY created_at, id
        """,
        session_id, level, character_id, status)
    return [_rule_dict(r, full=True) for r in rows]


async def update_rule(conn, session_id: str, rule_id: str, confirm_edit: bool = False,
                      **fields) -> bool:
    """Update allowed fields of one rule, scoped by chat. Returns False when no
    rule with that id exists in this session (another chat's id included).

    Accepts only: rule_text, rule_type, level, character_id, status.
    ``confirm_edit``: an edit of a pending rule that does not set status itself
    confirms it (owner decision 2026-10-18, QA Q8: editing a rule is reviewing it).
    Only the dashboard's edit passes this; programmatic callers never auto-confirm.
    """
    unknown = set(fields) - set(_UPDATABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"update_rule accepts only {_UPDATABLE_FIELDS}, got {sorted(unknown)}")
    if not fields:
        return False
    try:
        rule_uuid = str(uuid.UUID(str(rule_id)))
    except (ValueError, AttributeError):
        return False

    row = await conn.fetchrow(
        "SELECT level, character_id, status FROM spm_chat_rules WHERE id = $1 AND session_id = $2",
        rule_uuid, session_id)
    if row is None:
        return False
    if (confirm_edit and "status" not in fields and row["status"] == "pending"
            and any(k in fields for k in ("rule_text", "rule_type", "level", "character_id"))):
        fields = dict(fields, status="active")

    if "level" in fields:
        _validate_level(fields["level"])
    if "rule_type" in fields:
        _validate_rule_type(fields["rule_type"])
    if "status" in fields:
        _validate_status(fields["status"])
    if "rule_text" in fields and (not fields["rule_text"]
                                  or not str(fields["rule_text"]).strip()):
        raise ValueError("rule_text must not be empty")

    # The level/character_id constraint is checked on the EFFECTIVE row: a level
    # change must not strand a character-level rule without a character, and
    # world level never keeps one.
    effective_level = fields.get("level", row["level"])
    effective_char = fields["character_id"] if "character_id" in fields else row["character_id"]
    if effective_level == "world":
        effective_char = None
    elif not effective_char:
        raise ValueError(
            f"level {effective_level!r} requires a character_id")
    fields = dict(fields, character_id=effective_char)

    sets = ["updated_at = CURRENT_TIMESTAMP"]
    values: list = [rule_uuid, session_id]
    for name in _UPDATABLE_FIELDS:
        if name in fields:
            values.append(fields[name])
            sets.append(f"{name} = ${len(values)}")
    result = await conn.execute(
        f"UPDATE spm_chat_rules SET {', '.join(sets)} "
        "WHERE id = $1 AND session_id = $2",
        *values)
    return result == "UPDATE 1"


async def delete_rule(conn, session_id: str, rule_id: str) -> bool:
    """Delete one rule of this chat. Returns False when no such rule exists here."""
    try:
        rule_uuid = str(uuid.UUID(str(rule_id)))
    except (ValueError, AttributeError):
        return False
    result = await conn.execute(
        "DELETE FROM spm_chat_rules WHERE id = $1 AND session_id = $2",
        rule_uuid, session_id)
    return result == "DELETE 1"


def _audience_sql(character_param: int, co_present_param: int) -> str:
    """This character sees: world rules; its own internal/external rules; and the
    external rules of the other characters it shares a room with."""
    return (
        f"(level = 'world' "
        f"OR character_id = ${character_param} "
        f"OR (level = 'character_external' AND character_id = ANY(${co_present_param}::text[])))"
    )


async def rules_for_turn(
    conn,
    *,
    session_id: str,
    character_id: str,
    co_present: Optional[Sequence[str]],
    query_embedding: Optional[Sequence[float]],
    space_id: Optional[int],
    max_cosine_distance: float,
) -> Dict[str, List[dict]]:
    """The active rules of this chat that apply to ``character_id`` this turn.

    Invariants always; ``conditional_trigger``/``game_over`` only when
    ``query_embedding`` is given and lands within ``max_cosine_distance`` in
    ``space_id`` (same space-filtered OFFSET 0 fence as the memory recall).
    Returns ``{"invariants": [...], "triggers": [...]}``.
    """
    co = list(co_present or ())
    audience = _audience_sql(2, 3)
    invariant_rows = await conn.fetch(
        f"""
        SELECT id, rule_text, rule_type, level, character_id, status, source
        FROM spm_chat_rules
        WHERE session_id = $1 AND status = 'active' AND rule_type = 'invariant'
          AND {audience}
        ORDER BY created_at, id
        """,
        session_id, character_id, co)

    trigger_rows = []
    if query_embedding is not None:
        trigger_rows = await conn.fetch(
            f"""
            SELECT id, rule_text, rule_type, level, character_id, status, source,
                   (rule_embedding <=> $1::vector) AS cosine_distance
            FROM (
                SELECT id, rule_text, rule_type, level, character_id, status, source,
                       rule_embedding
                FROM spm_chat_rules
                WHERE session_id = $4 AND status = 'active'
                  AND rule_type IN ('conditional_trigger', 'game_over')
                  AND rule_embedding IS NOT NULL
                  AND embedding_space_id IS NOT DISTINCT FROM $5
                  AND {audience}
                OFFSET 0
            ) candidates
            WHERE (rule_embedding <=> $1::vector) < $6
            ORDER BY cosine_distance ASC
            """,
            _embedding_str(query_embedding), character_id, co,
            session_id, space_id, max_cosine_distance)
    else:
        logger.info(
            "[chat_rules] No query embedding: trigger rules skipped for %s "
            "(invariants still apply).", character_id)

    return {"invariants": [_rule_dict(r) for r in invariant_rows],
            "triggers": [_rule_dict(r) for r in trigger_rows]}
