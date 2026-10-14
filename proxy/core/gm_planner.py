"""World-state changes read from a finished exchange, in a separate small call.

Used when the reply prompt carries no GM duties (reasoning_mode none, or native without
inline GM tags). The model sees the rooms, who is where, the player's message and the reply,
and answers with JSON only. The result goes through the usual validator.
"""
import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["MOVE", "CREATE_ROOM"]},
                    "entity": {"type": "string"},
                    "room_id": {"type": "string"},
                    "name": {"type": "string"},
                    "desc": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                # entity is required for every item (CREATE_ROOM uses ""): without it Gemma
                # dropped "who moves" from 18 moves (replay round 3).
                "required": ["type", "entity", "room_id", "evidence"],
            },
        }
    },
    "required": ["actions"],
}


def messages_for(rooms: List[Dict[str, str]], where: Dict[str, str], player: str,
                 player_text: str, speaker: str, reply: str, allow_create: bool) -> List[Dict[str, str]]:
    room_list = "\n".join(f"- {r['room_id']} ({r.get('name') or r['room_id']})" for r in rooms) or "- none yet"
    positions = "\n".join(f"- {e}: {r}" for e, r in sorted(where.items())) or "- unknown"
    kinds = "MOVE or CREATE_ROOM" if allow_create else "MOVE"
    task = (
        "Read this roleplay exchange and list the location changes it shows.\n\n"
        f"Known rooms (use these exact ids):\n{room_list}\n\n"
        f"Where everyone is now:\n{positions}\n\n"
        f"The player (entity id 'user', called {player}) wrote:\n{player_text}\n\n"
        f"{speaker} replied:\n{reply}\n\n"
        f"Answer with JSON: {{\"actions\": [...]}}, each action of type {kinds}.\n"
        "- MOVE {\"type\": \"MOVE\", \"entity\": id, \"room_id\": id, \"evidence\": quote} only "
        "when the text says someone arrives in or leaves for another place in THIS exchange. Talking "
        "about a place, wanting to go, or looking around is not a move.\n"
        "- evidence: copy the exact words (5-15 words) from the player's message or the reply that "
        "show the move or the new place. No quote, no action.\n"
        "- entity is who moves: 'user' for the player, the ids above for others. Every MOVE "
        "needs an entity, e.g. {\"type\": \"MOVE\", \"entity\": \"user\", \"room_id\": "
        "\"kitchen\", \"evidence\": \"I walk into the kitchen and sit down\"}. For CREATE_ROOM "
        "use entity \"\".\n"
        "- If the place matches a known room (same place, any wording), use that room's id; never "
        "create a second room for a place that already exists.\n"
        + ("- A place not in the list: first {\"type\": \"CREATE_ROOM\", \"room_id\": "
           "lowercase_with_underscores, \"name\": short name, \"desc\": one sentence, \"evidence\": "
           "quote}, then the MOVE.\n"
           if allow_create else "")
        + "- If nobody changed location: {\"actions\": []}")
    return [{"role": "system", "content": "You track locations in a roleplay. Answer with JSON only."},
            {"role": "user", "content": task}]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


_FIELDS = {"MOVE": {"type", "entity", "room_id"}, "CREATE_ROOM": {"type", "room_id", "name", "desc"}}


def grounded(actions: List[Dict[str, Any]], *texts: str) -> List[Dict[str, Any]]:
    """Keep only actions whose evidence quote really appears in the exchange (checked in
    code, not trusted from the model). An invented move (Lian 'to the kitchen' with no such
    line, replay 2026-10-13) has no quote to point at. Evidence is dropped before dispatch."""
    hay = " ".join(_norm(t) for t in texts)

    def ok(a):
        ev = _norm(str(a.get("evidence") or ""))
        return len(ev.split()) >= 3 and ev in hay

    moves_into = {a.get("room_id") for a in actions if a.get("type") == "MOVE" and ok(a)}
    kept = []
    for a in actions:
        # A new room stands if its own quote checks out, or a grounded move goes into it.
        if ok(a) or (a.get("type") == "CREATE_ROOM" and a.get("room_id") in moves_into):
            # Only the fields the validator accepts for the type (extra fields are rejected):
            # the schema makes CREATE_ROOM carry entity "" and MOVE may carry name/desc "".
            logger.info(f"[GMPlanner] grounded {a.get('type')} {a.get('entity', '')}->{a.get('room_id')}: "
                        f"\"{str(a.get('evidence'))[:120]}\"")
            keep = _FIELDS.get(str(a.get("type")), set())
            kept.append({k: v for k, v in a.items() if k in keep and (k == "type" or v)})
        else:
            logger.info(f"[GMPlanner] dropped ungrounded action (evidence not in the text): {a}")
    return kept


_STOP = {"the", "a", "an", "of", "s", "room", "area", "place", "main"}


def _words(t: str) -> List[str]:
    out = []
    for w in re.split(r"[^a-z0-9]+", (t or "").lower()):
        if not w or w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)   # principals -> principal
    return out


def _same_place(a_id: str, a_name: str, b_id: str, b_name: str) -> bool:
    """Close names that end in the same noun: "principals_office" ~ "Principal Nezu's
    Office", but "UA High School Main Gates" is not "UA High School"."""
    a = _words(a_name or a_id) or _words(a_id)
    b = _words(b_name or b_id) or _words(b_id)
    if not a or not b or a[-1] != b[-1]:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) >= 0.6


def reuse_rooms(actions: List[Dict[str, Any]], rooms: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Map a proposed new room onto an existing room for the same place (deterministic):
    the model kept creating near-duplicates (ua_academy / ua_high_school; principals_office /
    principal_nezus_office) although the list was in the prompt."""
    known = {r["room_id"]: r.get("name") or r["room_id"] for r in rooms}
    alias = {}
    out = []
    for a in actions:
        if a.get("type") == "CREATE_ROOM" and a.get("room_id") not in known:
            match = next((rid for rid, nm in known.items()
                          if _same_place(a.get("room_id", ""), a.get("name", ""), rid, nm)), None)
            if match:
                alias[a["room_id"]] = match
                logger.info(f"[GMPlanner] '{a['room_id']}' is the existing room '{match}'; reusing it.")
                continue
        out.append(a)
    for a in out:
        if a.get("type") == "MOVE" and a.get("room_id") in alias:
            a["room_id"] = alias[a["room_id"]]
    # One move per entity: the last one is where they end up (Lian kitchen -> Vardus's room in
    # one reply came back as two moves). Rooms only an earlier, dropped move used are dropped too.
    last = {a.get("entity"): i for i, a in enumerate(out) if a.get("type") == "MOVE"}
    out = [a for i, a in enumerate(out) if a.get("type") != "MOVE" or last.get(a.get("entity")) == i]
    used = {a["room_id"] for a in out if a.get("type") == "MOVE"}
    return [a for a in out if a.get("type") != "CREATE_ROOM" or a.get("room_id") in used or a.get("room_id") in known]


def parse(text: str) -> List[Dict[str, Any]]:
    """The actions from the model's answer; [] when nothing usable comes back."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    acts = data.get("actions") if isinstance(data, dict) else None
    return [a for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []


async def plan(client, model: str, msgs: List[Dict[str, str]], local: bool,
               **stream_kwargs) -> List[Dict[str, Any]]:
    """Ask for the JSON. Structured output first; plain text if the server refuses it.
    Thinking is switched off (local backends only: cloud APIs reject unknown fields)."""
    base = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}} if local else {}
    attempts = [dict(base, response_format={"type": "json_schema", "json_schema": {
                    "name": "location_changes", "schema": SCHEMA}}), base]
    for extra in attempts:
        text, thinking = "", False
        try:
            async for chunk in client.generate_stream(messages=msgs, model=model, temperature=0.1,
                                                      max_tokens=500, extra_body=extra or None,
                                                      **stream_kwargs):
                if chunk in ("<thinking>", "</thinking>"):
                    thinking = chunk == "<thinking>"
                elif not thinking:
                    text += chunk
        except Exception as e:
            logger.info(f"[GMPlanner] call failed ({'schema' if 'response_format' in extra else 'plain'}): {e}")
            continue
        acts = parse(text)
        if acts or re.search(r'"actions"\s*:\s*\[\s*\]', text):
            return acts
    return []
