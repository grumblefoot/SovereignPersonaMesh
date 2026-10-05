"""Perception recording and per-character gated history (Sprint 2 chunks 3-4 foundation).

One row per (turn, recipient) in spm_perception: what each character actually perceived
after spatial gating. The gated history a character's prompt may contain is rebuilt from
these rows — never from the raw transcript (docs/plans/gating.md §gated history).
"""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def record_turn_perceptions(conn, *, session_id: str, tick: int,
                                  turn_id: Optional[str], actor_id: str,
                                  action_type: str,
                                  consequences: List[Dict[str, Any]]) -> int:
    """Insert one perception row per consequence recipient. Returns rows written.

    Blackout rows ARE recorded (with empty perceived_text): "X perceived nothing of this
    turn" is itself the fact the gated-history builder needs."""
    rows = 0
    for c in consequences:
        recipient = c.get("recipient_id")
        if not recipient:
            continue
        await conn.execute(
            """
            INSERT INTO spm_perception
                (session_id, tick, turn_id, actor_id, action_type, recipient_id,
                 gating_level, perceived_text, distance_ft, barriers)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            """,
            session_id, tick, turn_id, actor_id, action_type, recipient,
            str(c.get("gating_level", "direct")).lower(),
            c.get("sensory_feed") or "",
            float(c["distance_ft"]) if c.get("distance_ft") is not None else None,
            json.dumps(list(c.get("barriers") or [])),
        )
        rows += 1
    return rows


async def gated_history(conn, *, session_id: str, recipient_id: str,
                        limit: int = 30) -> List[Dict[str, Any]]:
    """The last `limit` perception rows for one character, oldest first.

    direct   -> the perceived text verbatim
    degraded -> the (already muffled) perceived text
    blackout -> an empty row the prompt builder renders as a gap marker, never content.
    """
    rows = await conn.fetch(
        """
        SELECT tick, turn_id, actor_id, action_type, gating_level, perceived_text,
               distance_ft
        FROM spm_perception
        WHERE session_id = $1 AND recipient_id = $2
        ORDER BY tick DESC, id DESC
        LIMIT $3
        """,
        session_id, recipient_id, limit,
    )
    return [dict(r) for r in reversed(rows)]
