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


GAP_MARKER = "[Time passes. You perceive nothing of what happens.]"


def render_history_rows(rows: List[Dict[str, Any]], target_char: str) -> List[Dict[str, str]]:
    """Turn a character's perception rows into chat messages for THEIR prompt.

    - rows the character produced (gating 'self') -> assistant turns, verbatim;
    - rows they perceived directly -> user turns (prefixed with the actor when it
      isn't the player, so group scenes stay attributable);
    - degraded rows -> the engine's muffled text, marked as such;
    - blackout rows -> one gap marker, consecutive blackouts collapsed.
    The raw transcript never appears here: this list IS the character's knowledge.
    """
    out: List[Dict[str, str]] = []
    for r in rows:
        gating = str(r.get("gating_level", "direct")).lower()
        actor = r.get("actor_id") or "someone"
        text = r.get("perceived_text") or ""
        if gating == "self":
            out.append({"role": "assistant", "content": text})
            continue
        if gating == "blackout" or not text:
            if out and out[-1]["content"] == GAP_MARKER:
                continue
            out.append({"role": "user", "content": GAP_MARKER})
            continue
        prefix = "" if actor == "user" else f"[{actor}] "
        if gating == "degraded":
            out.append({"role": "user", "content": f"{prefix}(indistinct) {text}"})
        else:
            out.append({"role": "user", "content": f"{prefix}{text}"})
    return out
