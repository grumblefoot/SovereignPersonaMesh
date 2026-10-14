"""Perception backfill for history SPM never saw (imported chats, chats begun before SPM).

A character's prompt history is rebuilt from spm_perception, never from the raw transcript.
A chat imported with 30 turns has no perception rows for them, so the first reply was
written from the card and the last message alone (live test 2026-10-13, imported chat).

SPM corrects this itself:
- plan_rows() turns the transcript's turns before the first turn SPM recorded into the rows
  live turns would have produced. Everyone in the scene perceived them directly: they
  happened before SPM tracked who was where, and the transcript is all we know.
- The turn that finds the gap uses the planned rows in its prompt at once (no blind reply).
- persist() writes them in the background at the lowest scheduler priority (an idle-time
  job a chat turn preempts); a preempted or failed write is retried on a later turn.

Backfilled turn ids are '<session>:<N>:bf...', so rewinds prune them like live turns
(prune_later_turns parses the leading number) and the action-digest reuse never
matches them. All rows use tick 0, which orders them before live turns (ticks >= 1).
"""
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.identifiers import safe_char_id

logger = logging.getLogger(__name__)

MARK = ":bf"
# SPM's own notices are not story: a bypassed character's placeholder, a withheld reply.
_SPM_NOTICE = re.compile(r"hears only muffled sounds from elsewhere|withheld and not saved|"
                         r"^\s*\*?\[SPM:", re.IGNORECASE)
# Group chats prefix each reply with the speaker's display name ("莲 Lian: ...").
_PREFIX = re.compile(r"^\s*([^\n:]{1,40}?):\s")


def _turn_no(session_id: str, turn_id: Optional[str]) -> Optional[int]:
    if not turn_id or not turn_id.startswith(session_id + ":"):
        return None
    m = re.match(r"\d+", turn_id[len(session_id) + 1:])
    return int(m.group(0)) if m else None


def coverage(session_id: str, turn_ids: Iterable[str]) -> Tuple[bool, Optional[int], bool]:
    """(already backfilled, first live turn number >= 1, greeting recorded as turn 0)."""
    done, first, greeting = False, None, False
    for t in turn_ids:
        if MARK in (t or ""):
            done = True
            continue
        n = _turn_no(session_id, t)
        if n == 0:
            greeting = True
        elif n is not None and (first is None or n < first):
            first = n
    return done, first, greeting


def group_speakers(messages: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """{display name: char id} for names that open at least two replies: a group chat.
    One-off openings ("Note: ...") are not speakers."""
    seen: Dict[str, int] = {}
    for role, content in messages:
        if role == "assistant" and content:
            m = _PREFIX.match(content)
            if m and 0 < len(m.group(1).split()) <= 3:
                seen[m.group(1).strip()] = seen.get(m.group(1).strip(), 0) + 1
    return {name: safe_char_id(name) for name, n in seen.items() if n >= 2}


def plan_rows(session_id: str, messages: Sequence[Tuple[str, str]], *, target_char: str,
              current_turn: int, first_live_turn: Optional[int], has_greeting: bool,
              redact: Callable[[str], str] = lambda t: t) -> List[Dict[str, Any]]:
    """Perception rows for the turns before `first_live_turn` (or before the current turn
    when SPM has recorded none). `messages` are (role, content) pairs, user and assistant
    only, ending with the current user message. Returns [] when nothing is missing."""
    stop = first_live_turn if first_live_turn is not None else current_turn
    if stop <= 1 and has_greeting:
        return []
    n_user = sum(1 for r, _ in messages if r == "user")
    k = max(0, current_turn - n_user)        # ST may have trimmed the oldest messages
    speakers = group_speakers(messages)
    chars: Set[str] = {target_char, *speakers.values()}
    rows: List[Dict[str, Any]] = []
    j = 0
    for role, content in messages:
        if role == "user":
            k += 1
            j = 0
        if k >= stop:
            break
        text = (content or "").strip()
        if not text or _SPM_NOTICE.search(text):
            continue
        if role == "user":
            tid = f"{session_id}:{k}{MARK}"
            said = redact(text)
            rows += [dict(turn_id=tid, actor_id="user", recipient_id=c, gating_level="direct",
                          perceived_text=said) for c in sorted(chars)]
            continue
        if k == 0 and has_greeting:
            continue                         # the greeting is already turn 0
        speaker = target_char
        m = _PREFIX.match(text)
        if m and m.group(1).strip() in speakers:
            speaker = speakers[m.group(1).strip()]
            text = text[m.end():].strip()
        j += 1
        tid = f"{session_id}:{k}{MARK}#reply:{speaker}:{j}"
        rows.append(dict(turn_id=tid, actor_id=speaker, recipient_id=speaker,
                         gating_level="self", perceived_text=text))
        rows += [dict(turn_id=tid, actor_id=speaker, recipient_id=c, gating_level="direct",
                      perceived_text=text) for c in sorted(chars | {"user"}) if c != speaker]
    for r in rows:
        r.update(session_id=session_id, tick=0, action_type="speak")
    return rows


def for_recipient(rows: List[Dict[str, Any]], recipient: str) -> List[Dict[str, Any]]:
    """The planned rows one character perceived, shaped like gated_history() rows."""
    return [{"id": None, "tick": 0, "turn_id": r["turn_id"], "actor_id": r["actor_id"],
             "action_type": "speak", "gating_level": r["gating_level"],
             "perceived_text": r["perceived_text"], "distance_ft": None}
            for r in rows if r["recipient_id"] == recipient]


async def persist(conn, session_id: str, rows: List[Dict[str, Any]]) -> int:
    """Write the planned rows once (a concurrent or earlier backfill wins)."""
    if not rows:
        return 0
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", "spm_backfill:" + session_id)
        done = await conn.fetchval(
            "SELECT 1 FROM spm_perception WHERE session_id = $1 AND position($2 in turn_id) > 0 LIMIT 1",
            session_id, MARK)
        if done:
            return 0
        await conn.executemany(
            """INSERT INTO spm_perception (session_id, tick, turn_id, actor_id, action_type,
                   recipient_id, gating_level, perceived_text)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            [(r["session_id"], r["tick"], r["turn_id"], r["actor_id"], r["action_type"],
              r["recipient_id"], r["gating_level"], r["perceived_text"]) for r in rows])
    return len(rows)
