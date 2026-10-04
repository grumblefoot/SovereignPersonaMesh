"""
Typed models and the WorldEngine Protocol for the SPM engine interface contract (v1.0).

Source designs: docs/plans/world_engine.md §6 (Protocol and data model) and
docs/plans/gating.md §1 (snapshot/apply/clock shape: edges ``{a, b, barrier, state,
distance_ft}``, occupants with optional posture flags), reconciled by
docs/plans/SPRINT_PLAN.md §1.1 — one contract WITHOUT ``perceive()``; perception is a
pure function in the proxy over the graph snapshot.

Semantics (world_engine.md §6, SPRINT_PLAN.md §6 decision 10):
- **Ticks** are per-session user-message counts. ``advance_tick(session_id, turn_id)``
  increments only for a new ``turn_id``; a repeated ``turn_id`` (regeneration/swipe)
  returns the same tick and does not advance. ``submit_action`` never advances the tick.
- **Idempotency.** ``apply_mutation`` with a repeated ``idempotency_key`` returns the
  original result and applies nothing.
- **Isolation.** Every call is scoped by ``session_id``; there is no global
  "current world".
- **Errors.** ``WorldNotFound``, ``InvalidMutation``, ``MutationConflict``,
  ``EngineUnavailable``; adapters that cannot express a contract method over their
  transport raise ``EngineCapabilityError``.

Model style mirrors evennia_world/models.py (str Enums + Pydantic BaseModel).
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.0"

# Id format shared with core.identifiers.safe_char_id output.
ID_PATTERN = r"^[a-z0-9_]{1,48}$"

MutationOrigin = Literal["gm", "system", "user"]


# ── Enums ───────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    SPEAK = "speak"
    WHISPER = "whisper"
    SHOUT = "shout"
    MOVE = "move"
    MANIPULATE = "manipulate"


class GatingLevel(str, Enum):
    DIRECT = "direct"
    DEGRADED = "degraded"
    BLACKOUT = "blackout"


class BarrierType(str, Enum):
    NONE = "none"
    OPEN_DOOR = "open_door"
    CLOSED_DOOR = "closed_door"
    LOCKED_DOOR = "locked_door"
    DRYWALL = "drywall"
    SOLID_WALL = "solid_wall"
    METAL_PARTITION = "metal_partition"


class BarrierState(str, Enum):
    """Current state of an edge's barrier (doors and partitions are stateful)."""
    OPEN = "open"
    CLOSED = "closed"
    LOCKED = "locked"


class MutationType(str, Enum):
    PLACE = "PLACE"
    MOVE = "MOVE"
    CREATE_ROOM = "CREATE_ROOM"
    SET_BARRIER = "SET_BARRIER"


class EntityKind(str, Enum):
    CHARACTER = "character"
    OBJECT = "object"


# ── Typed errors ────────────────────────────────────────────────────────

class EngineError(Exception):
    """Base class for contract errors."""


class WorldNotFound(EngineError):
    """The session has no world (and the adapter cannot lazily create one)."""


class InvalidMutation(EngineError):
    """The mutation is malformed or references ids that do not exist (HTTP 422)."""


class MutationConflict(EngineError):
    """A repeated idempotency_key arrived with a *different* mutation body (HTTP 409)."""


class EngineUnavailable(EngineError):
    """The engine cannot be reached; the proxy degrades the turn gracefully (OPEN-010)."""


class EngineCapabilityError(EngineError):
    """The adapter's transport cannot express this contract method yet.

    Each raise names the missing capability; the gaps are Sprint 1 Track A work items.
    """

    def __init__(self, capability: str, detail: str = ""):
        self.capability = capability
        self.detail = detail
        super().__init__(f"engine capability not available: {capability}"
                         + (f" ({detail})" if detail else ""))


# ── World data model ────────────────────────────────────────────────────

class Room(BaseModel):
    """A room node in the world graph (world_engine.md §6 `Room`)."""
    id: str = Field(pattern=ID_PATTERN)
    name: str
    desc: str = ""
    tags: List[str] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    size_ft: float = 20.0


class Edge(BaseModel):
    """An undirected adjacency between two rooms (gating.md snapshot shape).

    ``barrier`` is what separates the rooms; ``state`` is its current position.
    ``distance_ft`` is the representative distance between occupants across the edge.
    """
    a: str = Field(pattern=ID_PATTERN)
    b: str = Field(pattern=ID_PATTERN)
    barrier: BarrierType = BarrierType.NONE
    state: BarrierState = BarrierState.OPEN
    distance_ft: float = 15.0

    def key(self) -> Tuple[str, str]:
        """Canonical undirected key."""
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


class Occupant(BaseModel):
    """An entity placed in a room, with optional posture flags (e.g. listening, asleep)."""
    entity_id: str = Field(pattern=ID_PATTERN)
    room_id: str = Field(pattern=ID_PATTERN)
    kind: EntityKind = EntityKind.CHARACTER
    posture: List[str] = Field(default_factory=list)


class WorldGraph(BaseModel):
    """Rooms, edges and occupants for one session: everything perception needs,
    in one call (gating.md §1: one snapshot call per request, not N action calls)."""
    session_id: str
    tick: int = 0
    rooms: List[Room] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    occupants: List[Occupant] = Field(default_factory=list)

    def room_ids(self) -> List[str]:
        return [r.id for r in self.rooms]

    def occupants_of(self, room_id: str) -> List[Occupant]:
        return [o for o in self.occupants if o.room_id == room_id]

    def find_occupant(self, entity_id: str) -> Optional[Occupant]:
        for o in self.occupants:
            if o.entity_id == entity_id:
                return o
        return None


# The snapshot *is* the graph; both plan documents use both names for the same thing.
WorldSnapshot = WorldGraph


class WorldSeed(BaseModel):
    """Initial world description for ensure_world.

    Either explicit ``rooms``/``edges``/``placements``, or a ``template_key`` /
    ``keywords`` hint for template-driven construction — or nothing, in which case the
    engine creates a minimal single-room world.
    """
    template_key: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    placements: Dict[str, str] = Field(default_factory=dict)  # entity_id -> room_id


class EntityState(BaseModel):
    """Spatial state of one entity inside a session world."""
    entity_id: str
    kind: EntityKind = EntityKind.CHARACTER
    room_id: Optional[str] = None
    posture: List[str] = Field(default_factory=list)
    distances_ft: Dict[str, float] = Field(default_factory=dict)  # other entity -> ft
    attrs: Dict[str, Any] = Field(default_factory=dict)


# ── Actions ─────────────────────────────────────────────────────────────

class Action(BaseModel):
    """A character action submitted for world-state effects (world_engine.md §6)."""
    actor_id: str = Field(pattern=ID_PATTERN)
    type: ActionType
    target_id: Optional[str] = None
    text: str = ""
    volume: Optional[float] = None


class Consequence(BaseModel):
    """Per-recipient outcome of an action — a superset of PRD §3.2.3."""
    recipient_id: str
    gating: GatingLevel
    sensory_feed: str = ""
    distance_ft: float = 0.0
    barriers: List[BarrierType] = Field(default_factory=list)
    path: List[str] = Field(default_factory=list)  # room ids actor → recipient


class ActionResult(BaseModel):
    success: bool = True
    tick: int
    consequences: List[Consequence] = Field(default_factory=list)


# ── Mutations ───────────────────────────────────────────────────────────

class Mutation(BaseModel):
    """A closed union over MutationType; fields used per type:

    - PLACE:       entity_id, room_id (put an entity in a room; no-op if already there)
    - MOVE:        entity_id, room_id (remove from every room, add to room_id)
    - CREATE_ROOM: room (full Room), optional edges to link it
    - SET_BARRIER: a, b, barrier, state, optional distance_ft (creates the edge if absent)
    """
    type: MutationType
    entity_id: Optional[str] = None
    room_id: Optional[str] = None
    room: Optional[Room] = None
    edges: List[Edge] = Field(default_factory=list)
    a: Optional[str] = None
    b: Optional[str] = None
    barrier: Optional[BarrierType] = None
    state: Optional[BarrierState] = None
    distance_ft: Optional[float] = None


class MutationResult(BaseModel):
    applied: bool = True          # False when the idempotency_key was already seen
    idempotency_key: str
    origin: MutationOrigin
    type: MutationType
    tick: int = 0
    detail: str = ""


# ── Tick / health ───────────────────────────────────────────────────────

class TickResult(BaseModel):
    tick: int
    turn_id: str
    advanced: bool  # False when the turn_id had been seen before (regeneration/swipe)


class EngineHealth(BaseModel):
    ok: bool
    engine: str                   # adapter name, e.g. "in_process" or "http"
    contract_version: str = CONTRACT_VERSION
    sessions: int = 0
    detail: str = ""


# ── The Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class WorldEngine(Protocol):
    """The ONE engine contract (SPRINT_PLAN.md §1.1). No perceive(): perception is
    computed in the proxy from get_graph()'s snapshot."""

    contract_version: str  # "1.0"

    async def ensure_world(self, session_id: str, seed: Optional[WorldSeed] = None) -> WorldSnapshot:
        """Create the session's world if missing (from seed, or minimal) and return its snapshot."""
        ...

    async def advance_tick(self, session_id: str, turn_id: str) -> TickResult:
        """Idempotent per turn_id: a new turn_id advances the per-session tick by one;
        a repeated turn_id returns the same tick with advanced=False."""
        ...

    async def submit_action(self, session_id: str, action: Action, turn_id: str) -> ActionResult:
        """Apply a character action's world-state effects. Never advances the tick."""
        ...

    async def get_graph(self, session_id: str) -> WorldGraph:
        """Full snapshot (rooms, edges, occupants, tick) for one session."""
        ...

    async def get_entity_state(self, session_id: str, entity_id: str) -> EntityState:
        ...

    async def apply_mutation(self, session_id: str, mutation: Mutation,
                             idempotency_key: str, origin: MutationOrigin) -> MutationResult:
        """Idempotent per idempotency_key: a repeat returns the original result unapplied."""
        ...

    async def reset(self, session_id: Optional[str]) -> None:
        """Drop one session's world, or (None, admin only) every session."""
        ...

    async def health(self) -> EngineHealth:
        ...
