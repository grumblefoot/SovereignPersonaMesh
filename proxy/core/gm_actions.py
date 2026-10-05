"""
GM_ACTION validation (OPEN-005, gm_actions_and_lore_scope.md §A.3-A.4, deviation SD-01).

Pure Python, engine-agnostic: the LLM's proposed world actions pass through here
before anything reaches the world engine. The Zero-LLM rule holds because this layer
is deterministic — the LLM only ever *proposes*; schema, caps and semantic checks
decide. Rejections carry a reason code (schema | unknown_entity | unknown_room |
cap | mode) for telemetry.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SLUG_RE = re.compile(r"^[a-z0-9_]{1,48}$")
_TAG_RE = re.compile(r"<[^>]*>")
_NESTED_GM_RE = re.compile(r"\[GM_[^\]]*\]?", re.IGNORECASE)

REASON_SCHEMA = "schema"
REASON_UNKNOWN_ENTITY = "unknown_entity"
REASON_UNKNOWN_ROOM = "unknown_room"
REASON_CAP = "cap"
REASON_MODE = "mode"


def slugify(value: str) -> str:
    """Normalise a proposed room id: spaces/dashes to underscores, lower-case,
    strip everything else. Returns '' when nothing valid remains."""
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[\s\-]+", "_", value.strip())
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value[:48]


def sanitize_text(value: str, limit: int) -> str:
    """Strip control characters, <...> tags and nested [GM_...] from names/descriptions."""
    value = _TAG_RE.sub("", str(value or ""))
    value = _NESTED_GM_RE.sub("", value)
    value = "".join(c for c in value if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return value.strip()[:limit]


class MoveAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["MOVE"]
    entity: str = Field(min_length=1, max_length=80)
    room_id: str

    @field_validator("room_id")
    @classmethod
    def _slug(cls, v):
        slug = slugify(v)
        if not SLUG_RE.match(slug or ""):
            raise ValueError(f"invalid room slug: {v!r}")
        return slug


class CreateRoomAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["CREATE_ROOM"]
    room_id: str
    name: str = Field(default="", max_length=200)
    desc: str = Field(default="", max_length=1000)

    @field_validator("room_id")
    @classmethod
    def _slug(cls, v):
        slug = slugify(v)
        if not SLUG_RE.match(slug or ""):
            raise ValueError(f"invalid room slug: {v!r}")
        return slug

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return sanitize_text(v, 60)

    @field_validator("desc")
    @classmethod
    def _desc(cls, v):
        return sanitize_text(v, 400)


GMAction = Union[MoveAction, CreateRoomAction]


@dataclass
class Rejection:
    action: dict
    reason: str                     # schema | unknown_entity | unknown_room | cap | mode
    detail: str = ""


@dataclass
class ValidatedBatch:
    """CREATE_ROOM actions first, then MOVEs — the order dispatch must use."""
    creates: List[CreateRoomAction] = field(default_factory=list)
    moves: List[MoveAction] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)

    @property
    def accepted(self) -> List[GMAction]:
        return [*self.creates, *self.moves]


def validate_batch(
    raw_actions: List[dict],
    *,
    mode: str = "full",
    known_rooms: Set[str] = frozenset(),
    known_entities: Set[str] = frozenset(),
    target_char: str = "",
    user_aliases: Set[str] = frozenset(),
    max_per_turn: int = 4,
    rooms_in_session: int = 0,
    max_rooms_per_session: int = 40,
) -> ValidatedBatch:
    """Validate one turn's proposed actions against the session world.

    - mode 'off' rejects everything (reason=mode); 'move_only' rejects CREATE_ROOM.
    - Schema: discriminated pydantic models, extra fields forbidden, slugs normalised.
    - Semantics: MOVE entity must be the target character, 'user', or a known entity;
      MOVE room must exist or be created earlier in the same batch; CREATE_ROOM of an
      existing id is dropped as a harmless no-op (reason=schema, detail='exists').
    - Caps: max_per_turn counts ACCEPTED actions; room cap counts session rooms.
    """
    out = ValidatedBatch()
    if mode == "off":
        out.rejections = [Rejection(a, REASON_MODE, "gm_actions_mode=off") for a in raw_actions]
        return out

    allowed_entities = {e.lower() for e in known_entities} | {"user"}
    if target_char:
        allowed_entities.add(target_char.lower())
    rooms = {slugify(r) for r in known_rooms}
    created_now: Set[str] = set()
    room_count = rooms_in_session
    accepted_count = 0

    parsed: List[Tuple[dict, Optional[GMAction], Optional[Rejection]]] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            parsed.append((raw, None, Rejection(raw, REASON_SCHEMA, "not an object")))
            continue
        a_type = raw.get("type")
        model = {"MOVE": MoveAction, "CREATE_ROOM": CreateRoomAction}.get(a_type)
        if model is None:
            parsed.append((raw, None, Rejection(raw, REASON_SCHEMA, f"unknown type {a_type!r}")))
            continue
        if a_type == "CREATE_ROOM" and mode == "move_only":
            parsed.append((raw, None, Rejection(raw, REASON_MODE, "gm_actions_mode=move_only")))
            continue
        try:
            parsed.append((raw, model(**raw), None))
        except ValidationError as e:
            parsed.append((raw, None, Rejection(raw, REASON_SCHEMA, str(e.errors()[0].get("msg", e)))))

    # CREATE first, then MOVE (a MOVE may target a room created this turn).
    ordered = ([p for p in parsed if isinstance(p[1], CreateRoomAction)]
               + [p for p in parsed if isinstance(p[1], MoveAction)]
               + [p for p in parsed if p[1] is None])
    for raw, action, rejection in ordered:
        if rejection is not None:
            out.rejections.append(rejection)
            continue
        if accepted_count >= max_per_turn:
            out.rejections.append(Rejection(raw, REASON_CAP, f"max {max_per_turn} per turn"))
            continue
        if isinstance(action, CreateRoomAction):
            if action.room_id in rooms or action.room_id in created_now:
                out.rejections.append(Rejection(raw, REASON_SCHEMA, "exists"))
                continue
            if room_count >= max_rooms_per_session:
                out.rejections.append(Rejection(raw, REASON_CAP,
                                                f"max {max_rooms_per_session} rooms per session"))
                continue
            created_now.add(action.room_id)
            room_count += 1
            out.creates.append(action)
            accepted_count += 1
        else:
            if action.entity.lower() in {a.lower() for a in user_aliases}:
                # The LLM names the player's persona ("Tom"); the engine id is
                # "user" (plan A.3: the persona name maps to user).
                action = action.model_copy(update={"entity": "user"})
            if action.entity.lower() not in allowed_entities:
                out.rejections.append(Rejection(raw, REASON_UNKNOWN_ENTITY, action.entity))
                continue
            if action.room_id not in rooms and action.room_id not in created_now:
                out.rejections.append(Rejection(raw, REASON_UNKNOWN_ROOM, action.room_id))
                continue
            out.moves.append(action)
            accepted_count += 1
    return out


def idempotency_key(session_id: str, turn_anchor: str, action: GMAction) -> str:
    """sha256(session | turn_anchor | canonical action): a regenerate of the same turn
    re-derives the same key, so the engine drops the duplicate (replaces uuid4)."""
    canonical = json.dumps(action.model_dump(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{session_id}|{turn_anchor}|{canonical}".encode()).hexdigest()
    return f"gm-{digest[:40]}"
