"""World-state changes read from a finished exchange, in a separate small call.

Used when the reply prompt carries no GM duties (reasoning_mode none, or native without
inline GM tags). The model sees the rooms, who is where, the player's message and the reply,
and answers with JSON only. The result goes through the usual validator.
"""
import asyncio
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
        "about a place, wanting to go, being told to go, or looking around is not a move. Quote "
        "the narration, not dialogue.\n"
        "- evidence: copy the exact words (5-15 words) from the player's message or the reply that "
        "show the person going (walks into, steps out, arrives at, makes her way to). Where someone "
        "already stands is not a move. For the player's own move, quote the player's message. "
        "No quote, no action.\n"
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


# A move's quote must say someone goes somewhere. Quotes that only say where someone is
# ("the figure standing at the mouth of the alley", "pause her movements in the kitchen") made
# 26 of 63 no-move turns move someone in the offline GM check (2026-10-13). A missed move costs
# less than a false one, which splits the scene and muffles everyone.
_P = r"(?:my|his|her|their|your|our|the)"
_MOTION = re.compile(
    r"\b(?:enter\w*|walk\w*|step(?:s|ped|ping)?|strid\w*|strode|wander\w*|stroll\w*|"
    r"climb\w*|arriv\w*|exit(?:s|ed|ing)?|rush\w*|hurr\w*|sprint\w*|march\w*|travel\w*|"
    r"ascend\w*|descend\w*|emerg\w*|stumbl\w*|sneak\w*|snuck|dash\w*|disappear\w*|"
    r"vanish\w*|teleport\w*|escort\w*|"
    rf"(?:make|makes|made|making) {_P} way|find(?:s)? (?:yourself|myself|himself|herself|themselves)|"
    r"(?:go|goes|going|gone|went|head|heads|headed|heading|run|runs|ran|running|move|moves|moved|"
    r"moving|come|comes|came|coming|return|returns|returned|returning|back) "
    r"(?:to|into|in|inside|outside|out|back|home|through|toward|towards|down|up|over|away|across|for)\b|"
    r"(?:leave|leaves|leaving|left) (?:the|for|through|his|her|their|my|your)\b|"
    r"(?:reach|reaches|reached|reaching) (?:the|a|an)\b|"
    r"(?:follow|follows|followed|following) (?:him|her|them|you|me|us|behind)\b|"
    r"(?:lead|leads|led|leading) (?:him|her|them|you|me|us|the way)\b|"
    r"cross(?:es|ed|ing)? (?:the|into)\b|now (?:in|inside|at)\b)", re.IGNORECASE)

# Dialogue may run over several lines (review C3); an unpaired quote strips more narration,
# which only loses a move (the safe direction).
# Leaving says where someone WAS, not where they go: "I bow and open the door to leave his
# office" moved the player INTO the office (tester session 2026-10-13, T5). A quote that
# leaves and never arrives anywhere moves nobody.
_LEAVING = re.compile(r"\b(?:leav(?:e|es|ing)|left|exit(?:s|ed|ing)?|depart\w*|out of|away from)\b")
_ARRIVING = re.compile(
    r"\b(?:enter\w*|into|inside|arriv\w*|reach\w*|towards?|"
    r"(?:go|goes|going|went|head\w*|walk\w*|run|runs|ran|running|return\w*|step\w*|come|comes|came|"
    r"coming|move\w*|lead\w*|led|follow\w*) (?:to|in|into|over|back|up|down|inside|for)|"
    r"(?:leave|leaves|leaving|left|depart\w*) for|"
    r"(?:make|makes|made|making) \w+ way)\b")


def _departure_only(text: str) -> bool:
    return bool(_LEAVING.search(text)) and not _ARRIVING.search(text)


_QUOTED = re.compile(r'"[^"]{1,600}"|\u201c[^\u201d]{1,600}\u201d')


def grounded(actions: List[Dict[str, Any]], player_text: str, *texts: str) -> List[Dict[str, Any]]:
    """Keep only actions the exchange supports (checked in code, not trusted from the model).
    `player_text` is the player's message; `texts` are the replies. Evidence is dropped
    before dispatch.

    - The quote must appear in the narration. Quoted dialogue does not count: an order in
      speech ("Go to the guest room", Lian to Mei, replay 2026-10-13) moved Mei although she
      never left. An invented move (Lian "to the kitchen" with no such line) has no quote.
    - A MOVE's quote must describe movement (_MOTION), not where someone already stands.
    - The player's own MOVE also stands when the player's narration describes movement: the
      model often quoted the reply's scenery ("the gates loom ahead") although the player wrote
      "I go to the training grounds" (offline check: 9 of 18 moves lost).
    - A CREATE_ROOM stands if its own quote checks out or a kept MOVE goes into it."""
    hay = " ".join(_norm(_QUOTED.sub(" | ", t or "")) for t in (player_text, *texts))
    said = _norm(_QUOTED.sub(" | ", player_text or ""))
    player_moves = bool(_MOTION.search(said)) and not _departure_only(said)
    said_words = set(_words(said))

    def ok(a):
        ev = _norm(str(a.get("evidence") or ""))
        if a.get("type") == "MOVE" and a.get("entity") != "user":
            # Someone else moves only if the exchange names them (review C1: a quote about
            # Lian moved Vardus). "mei_lin" matches "Mei-Lin"; the first name is enough.
            name = _norm(str(a.get("entity") or "")).split()
            if not name or not re.search(rf"\b{re.escape(name[0])}\b", hay):
                return False
        if (a.get("type") == "MOVE" and a.get("entity") == "user" and player_moves
                and said_words & set(_words(str(a.get("room_id") or "")))):
            # The player's narration moves them AND names the place ("I go to the training
            # grounds" -> pe_grounds). "The key is back in the box" names no place (review C1).
            return True
        if a.get("type") == "MOVE" and (not _MOTION.search(ev) or _departure_only(ev)):
            return False
        return len(ev.split()) >= 3 and ev in hay

    moves_into = {a.get("room_id") for a in actions if a.get("type") == "MOVE" and ok(a)}
    kept = []
    for a in actions:
        if ok(a) or (a.get("type") == "CREATE_ROOM" and a.get("room_id") in moves_into):
            logger.info(f"[GMPlanner] grounded {a.get('type')} {a.get('entity', '')}->{a.get('room_id')}: "
                        f"\"{str(a.get('evidence'))[:120]}\"")
            # Only the fields the validator accepts for the type (it rejects extra fields):
            # the schema makes CREATE_ROOM carry entity "" and MOVE may carry name/desc "".
            keep = _FIELDS.get(str(a.get("type")), set())
            kept.append({k: v for k, v in a.items() if k in keep and (k == "type" or v)})
        else:
            logger.info(f"[GMPlanner] dropped ungrounded action: {a}")
    return kept


_STOP = {"the", "a", "an", "of", "s", "room", "area", "place", "main"}


def _words(t: str) -> List[str]:
    out = []
    for w in re.split(r"[^a-z0-9]+", (t or "").lower()):
        if not w or w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)   # principals -> principal
    return out


def _place_words(t: str) -> frozenset:
    # "U.A." -> "ua": join runs of single letters before comparing.
    t = re.sub(r"\b([a-z])[^a-z0-9]+(?=[a-z]\b)", r"\1", (t or "").lower())
    return frozenset(_words(t))


def _same_place(a_id: str, a_name: str, b_id: str, b_name: str) -> bool:
    """The same words after dropping filler and plurals: "principals_office" ~ "Principal's
    Office", "ua_high_school" ~ "U.A. High School". Any extra word means a different place
    ("Vice Principal's Office" is not "Principal's Office", review C4): a duplicate room is
    recoverable, a wrong merge is not."""
    a = {_place_words(a_id), _place_words(a_name)} - {frozenset()}
    b = {_place_words(b_id), _place_words(b_name)} - {frozenset()}
    return bool(a & b)


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


def drop_no_ops(actions: List[Dict[str, Any]], where: Dict[str, str]) -> List[Dict[str, Any]]:
    """Drop a MOVE into the room the entity is already in (tester session T6: the player was
    'moved' into Class 1-A twice), and a new room nobody then moves into."""
    out = [a for a in actions if not (a.get("type") == "MOVE" and a.get("entity") in where
                                      and where[a["entity"]] == a.get("room_id"))]
    used = {a.get("room_id") for a in out if a.get("type") == "MOVE"}
    return [a for a in out if a.get("type") != "CREATE_ROOM" or a.get("room_id") in used]


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


_NO_SCHEMA: set = set()        # (base_url, model) pairs that rejected structured output


PLAN_TIMEOUT_S = 45.0   # the GM call holds the only LLM slot (review C2); a lost move is cheap


async def plan(client, model: str, msgs: List[Dict[str, str]], local: bool,
               **stream_kwargs) -> List[Dict[str, Any]]:
    try:
        return await asyncio.wait_for(_plan(client, model, msgs, local, **stream_kwargs),
                                      timeout=PLAN_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(f"[GMPlanner] no answer within {PLAN_TIMEOUT_S:.0f}s; no location changes this turn")
        return []


async def _plan(client, model: str, msgs: List[Dict[str, str]], local: bool,
                **stream_kwargs) -> List[Dict[str, Any]]:
    """Ask for the JSON. Structured output first; plain text if the server refuses it.
    Thinking is switched off (local backends only: cloud APIs reject unknown fields)."""
    base = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}} if local else {}
    key = (getattr(client, "base_url", None), model)
    attempts = [] if key in _NO_SCHEMA else [dict(base, response_format={
        "type": "json_schema", "json_schema": {"name": "location_changes", "schema": SCHEMA}})]
    attempts.append(base)
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
            if "response_format" in extra and re.search(r"\b(?:400|422)\b", str(e)):
                # gufo answers 400 to json_schema on every call: remember it, skip the wasted call.
                _NO_SCHEMA.add(key)
            continue
        acts = parse(text)
        if acts or re.search(r'"actions"\s*:\s*\[\s*\]', text):
            return acts
    return []
