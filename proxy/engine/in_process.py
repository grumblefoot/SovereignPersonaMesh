"""
InProcessWorldEngine — reference implementation of the WorldEngine contract (v1.0).

Implements the contract directly against evennia_world's builder/spatial modules,
entirely in memory: it constructs its own world state via ``HybridWorldBuilder`` /
``RoomMetadata`` and gates action consequences through ``SpatialConstraintsMatrix``.
It deliberately does NOT import ``evennia_world.app`` (which would drag in FastAPI
startup, module-level app_state and asyncpg pools). Zero network, zero DB.

This adapter is the reference implementation of SPRINT_PLAN.md §6 decision 10's tick
semantics: ticks are per-session user-message counts; ``advance_tick(session, turn_id)``
increments once per new ``turn_id`` and a repeated ``turn_id`` (regeneration/swipe)
returns the same tick without advancing.

Notes
-----
- Template instantiation strips the demo ``present_characters`` that ship inside
  ``evennia_world/res/strings.json`` (a Sprint 0 known bug): occupants enter a world
  only through seed placements or PLACE/MOVE mutations.
- ``ActionType.SHOUT`` is gated like SPEAK but carries through one extra edge of
  distance tolerance via its larger representative volume; the SpatialConstraintsMatrix
  has no shout notion, so shout maps to SPEAK for feed formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evennia_world.hybrid_builder import HybridWorldBuilder
from evennia_world.models import (
    ActionType as EvActionType,
    BarrierType as EvBarrierType,
    RoomMetadata,
)
from evennia_world.spatial_matrix import SpatialConstraintsMatrix

from proxy.engine.contract import (
    CONTRACT_VERSION,
    Action,
    ActionResult,
    ActionType,
    BarrierState,
    BarrierType,
    Consequence,
    Edge,
    EngineHealth,
    EntityKind,
    EntityState,
    GatingLevel,
    InvalidMutation,
    Mutation,
    MutationConflict,
    MutationOrigin,
    MutationResult,
    MutationType,
    Occupant,
    Room,
    TickResult,
    WorldGraph,
    WorldNotFound,
    WorldSeed,
    WorldSnapshot,
)

# Representative distances (ft): same room, across one edge, unconnected rooms.
SAME_ROOM_FT = 3.0
ADJACENT_FT = 15.0
DISTANT_FT = 45.0

_DEFAULT_ROOM = Room(
    id="origin_room",
    name="Origin Room",
    desc="A plain, unremarkable room.",
)


@dataclass
class _MutationRecord:
    key: str
    origin: MutationOrigin
    mutation: Mutation
    result: MutationResult


@dataclass
class _SessionState:
    """All world state for one session. Never shared between sessions."""
    rooms: Dict[str, RoomMetadata] = field(default_factory=dict)
    room_meta: Dict[str, Room] = field(default_factory=dict)  # contract-level room info
    edges: Dict[Tuple[str, str], Edge] = field(default_factory=dict)
    postures: Dict[str, List[str]] = field(default_factory=dict)
    tick: int = 0
    turn_ticks: Dict[str, int] = field(default_factory=dict)      # turn_id -> tick
    mutations: Dict[str, _MutationRecord] = field(default_factory=dict)
    mutation_log: List[_MutationRecord] = field(default_factory=list)


class InProcessWorldEngine:
    """In-memory WorldEngine: per-session isolation, per-session idempotent ticks."""

    contract_version: str = CONTRACT_VERSION

    def __init__(self, builder: Optional[HybridWorldBuilder] = None):
        self._builder = builder if builder is not None else HybridWorldBuilder()
        self._sessions: Dict[str, _SessionState] = {}

    # ── Contract: world lifecycle ───────────────────────────────────

    async def ensure_world(self, session_id: str, seed: Optional[WorldSeed] = None) -> WorldSnapshot:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._build_world(seed)
        return await self.get_graph(session_id)

    async def reset(self, session_id: Optional[str]) -> None:
        if session_id is None:
            self._sessions.clear()
            return
        self._sessions.pop(session_id, None)

    async def health(self) -> EngineHealth:
        return EngineHealth(
            ok=True,
            engine="in_process",
            contract_version=self.contract_version,
            sessions=len(self._sessions),
        )

    # ── Contract: clock ─────────────────────────────────────────────

    async def advance_tick(self, session_id: str, turn_id: str) -> TickResult:
        state = self._require(session_id)
        if turn_id in state.turn_ticks:
            return TickResult(tick=state.turn_ticks[turn_id], turn_id=turn_id, advanced=False)
        state.tick += 1
        state.turn_ticks[turn_id] = state.tick
        return TickResult(tick=state.tick, turn_id=turn_id, advanced=True)

    # ── Contract: actions ───────────────────────────────────────────

    async def submit_action(self, session_id: str, action: Action, turn_id: str) -> ActionResult:
        state = self._require(session_id)
        # The tick for this turn: already-advanced value, else the current tick.
        tick = state.turn_ticks.get(turn_id, state.tick)

        if action.type == ActionType.MOVE:
            # A user/world move expressed as an action: target room in target_id or text.
            dest = action.target_id or action.text.strip()
            if dest not in state.rooms:
                raise InvalidMutation(f"move action to unknown room '{dest}'")
            self._move_entity(state, action.actor_id, dest)
            return ActionResult(success=True, tick=tick, consequences=[])

        actor_room = self._find_room_of(state, action.actor_id)
        if actor_room is None:
            raise WorldNotFound(f"actor '{action.actor_id}' is not placed in session '{session_id}'")

        ev_type = _to_ev_action(action.type)
        consequences: List[Consequence] = []
        for room_id, room in state.rooms.items():
            distance, barriers, path = self._distance_and_barriers(state, actor_room, room_id)
            for recipient_id in room.present_characters:
                if recipient_id == action.actor_id:
                    continue
                gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
                    distance_ft=distance,
                    barriers=[_to_ev_barrier(b) for b in barriers],
                    action_type=ev_type,
                    raw_text=action.text,
                    actor_id=action.actor_id,
                    recipient_id=recipient_id,
                    is_target=(recipient_id == action.target_id),
                    session_id=session_id,
                    action_tick=tick,
                )
                consequences.append(Consequence(
                    recipient_id=recipient_id,
                    gating=GatingLevel(gating.value),
                    sensory_feed=feed,
                    distance_ft=distance,
                    barriers=barriers,
                    path=path,
                ))
        return ActionResult(success=True, tick=tick, consequences=consequences)

    # ── Contract: reads ─────────────────────────────────────────────

    async def get_graph(self, session_id: str) -> WorldGraph:
        state = self._require(session_id)
        rooms: List[Room] = []
        occupants: List[Occupant] = []
        for room_id, room in state.rooms.items():
            meta = state.room_meta.get(room_id)
            rooms.append(meta if meta is not None else _room_from_metadata(room))
            for entity_id in room.present_characters:
                occupants.append(Occupant(
                    entity_id=entity_id,
                    room_id=room_id,
                    kind=EntityKind.CHARACTER,
                    posture=list(state.postures.get(entity_id, [])),
                ))
        return WorldGraph(
            session_id=session_id,
            tick=state.tick,
            rooms=rooms,
            edges=[e.model_copy(deep=True) for e in state.edges.values()],
            occupants=occupants,
        )

    async def get_entity_state(self, session_id: str, entity_id: str) -> EntityState:
        state = self._require(session_id)
        room_id = self._find_room_of(state, entity_id)
        distances: Dict[str, float] = {}
        if room_id is not None:
            for other_room_id, room in state.rooms.items():
                dist, _, _ = self._distance_and_barriers(state, room_id, other_room_id)
                for other_id in room.present_characters:
                    if other_id != entity_id:
                        distances[other_id] = dist
        return EntityState(
            entity_id=entity_id,
            kind=EntityKind.CHARACTER,
            room_id=room_id,
            posture=list(state.postures.get(entity_id, [])),
            distances_ft=distances,
        )

    # ── Contract: mutations ─────────────────────────────────────────

    async def apply_mutation(self, session_id: str, mutation: Mutation,
                             idempotency_key: str, origin: MutationOrigin) -> MutationResult:
        state = self._require(session_id)

        seen = state.mutations.get(idempotency_key)
        if seen is not None:
            if seen.mutation.model_dump() != mutation.model_dump():
                raise MutationConflict(
                    f"idempotency_key '{idempotency_key}' was already used for a different mutation"
                )
            return seen.result.model_copy(update={"applied": False})

        detail = self._apply(state, mutation)
        result = MutationResult(
            applied=True,
            idempotency_key=idempotency_key,
            origin=origin,
            type=mutation.type,
            tick=state.tick,
            detail=detail,
        )
        record = _MutationRecord(key=idempotency_key, origin=origin,
                                 mutation=mutation.model_copy(deep=True), result=result)
        state.mutations[idempotency_key] = record
        state.mutation_log.append(record)
        return result

    def mutation_origins(self, session_id: str) -> List[Tuple[str, MutationOrigin]]:
        """Introspection used by the contract tests: (idempotency_key, origin) in
        application order."""
        state = self._require(session_id)
        return [(r.key, r.origin) for r in state.mutation_log]

    # ── Internals ───────────────────────────────────────────────────

    def _require(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            # Lazy creation keeps calls on an unknown session usable (world_engine.md
            # "calls on an unknown session lazily load"); in memory that means a
            # minimal default world.
            self._sessions[session_id] = self._build_world(None)
        return self._sessions[session_id]

    def _build_world(self, seed: Optional[WorldSeed]) -> _SessionState:
        state = _SessionState()

        template_rooms: Dict[str, RoomMetadata] = {}
        if seed is not None and (seed.template_key or seed.keywords):
            template_key = seed.template_key or self._builder.match_template(seed.keywords)
            template_rooms = self._builder.instantiate_world(template_key)

        for room_id, room in template_rooms.items():
            room = RoomMetadata(**room.model_dump())
            room.present_characters = []  # never inherit demo characters from templates
            state.rooms[room_id] = room
            state.room_meta[room_id] = _room_from_metadata(room)
        # Template exits become open edges.
        for room_id, room in state.rooms.items():
            for exit_id in room.exits:
                if exit_id in state.rooms:
                    edge = Edge(a=room_id, b=exit_id, barrier=BarrierType.OPEN_DOOR,
                                state=BarrierState.OPEN, distance_ft=ADJACENT_FT)
                    state.edges.setdefault(edge.key(), edge)

        if seed is not None:
            for room in seed.rooms:
                self._create_room(state, room, [])
            for edge in seed.edges:
                self._set_edge(state, edge)
            for entity_id, room_id in seed.placements.items():
                if room_id not in state.rooms:
                    raise InvalidMutation(f"seed places '{entity_id}' in unknown room '{room_id}'")
                self._place_entity(state, entity_id, room_id)

        if not state.rooms:
            self._create_room(state, _DEFAULT_ROOM.model_copy(deep=True), [])
        return state

    def _apply(self, state: _SessionState, mutation: Mutation) -> str:
        if mutation.type == MutationType.PLACE:
            if not mutation.entity_id or not mutation.room_id:
                raise InvalidMutation("PLACE requires entity_id and room_id")
            if mutation.room_id not in state.rooms:
                raise InvalidMutation(f"PLACE into unknown room '{mutation.room_id}'")
            self._place_entity(state, mutation.entity_id, mutation.room_id)
            return f"placed {mutation.entity_id} in {mutation.room_id}"

        if mutation.type == MutationType.MOVE:
            if not mutation.entity_id or not mutation.room_id:
                raise InvalidMutation("MOVE requires entity_id and room_id")
            if mutation.room_id not in state.rooms:
                raise InvalidMutation(f"MOVE into unknown room '{mutation.room_id}'")
            self._move_entity(state, mutation.entity_id, mutation.room_id)
            return f"moved {mutation.entity_id} to {mutation.room_id}"

        if mutation.type == MutationType.CREATE_ROOM:
            if mutation.room is None:
                raise InvalidMutation("CREATE_ROOM requires a room")
            if mutation.room.id in state.rooms:
                raise InvalidMutation(f"room '{mutation.room.id}' already exists")
            self._create_room(state, mutation.room, mutation.edges)
            return f"created room {mutation.room.id}"

        if mutation.type == MutationType.SET_BARRIER:
            if not mutation.a or not mutation.b:
                raise InvalidMutation("SET_BARRIER requires rooms a and b")
            if mutation.a not in state.rooms or mutation.b not in state.rooms:
                raise InvalidMutation("SET_BARRIER references unknown room(s)")
            edge = Edge(
                a=mutation.a,
                b=mutation.b,
                barrier=mutation.barrier if mutation.barrier is not None else BarrierType.NONE,
                state=mutation.state if mutation.state is not None else BarrierState.OPEN,
                distance_ft=mutation.distance_ft if mutation.distance_ft is not None else ADJACENT_FT,
            )
            self._set_edge(state, edge)
            return f"set barrier {edge.barrier.value}/{edge.state.value} between {edge.a} and {edge.b}"

        raise InvalidMutation(f"unknown mutation type '{mutation.type}'")  # pragma: no cover

    def _create_room(self, state: _SessionState, room: Room, edges: List[Edge]) -> None:
        state.rooms[room.id] = RoomMetadata(
            room_id=room.id,
            room_name=room.name,
            description=room.desc,
        )
        state.room_meta[room.id] = room.model_copy(deep=True)
        for edge in edges:
            if edge.a not in state.rooms or edge.b not in state.rooms:
                raise InvalidMutation(f"edge {edge.a}–{edge.b} references unknown room(s)")
            self._set_edge(state, edge)

    def _set_edge(self, state: _SessionState, edge: Edge) -> None:
        if edge.a not in state.rooms or edge.b not in state.rooms:
            raise InvalidMutation(f"edge {edge.a}–{edge.b} references unknown room(s)")
        state.edges[edge.key()] = edge.model_copy(deep=True)
        # Mirror adjacency into RoomMetadata.exits so builder-level helpers stay true.
        for here, there in ((edge.a, edge.b), (edge.b, edge.a)):
            exits = state.rooms[here].exits
            if there not in exits:
                exits.append(there)

    def _place_entity(self, state: _SessionState, entity_id: str, room_id: str) -> None:
        current = self._find_room_of(state, entity_id)
        if current == room_id:
            return
        if current is not None:
            # PLACE on an entity that is elsewhere behaves like MOVE: one body, one room.
            self._remove_everywhere(state, entity_id)
        if entity_id not in state.rooms[room_id].present_characters:
            state.rooms[room_id].present_characters.append(entity_id)

    def _move_entity(self, state: _SessionState, entity_id: str, room_id: str) -> None:
        self._remove_everywhere(state, entity_id)
        state.rooms[room_id].present_characters.append(entity_id)

    @staticmethod
    def _remove_everywhere(state: _SessionState, entity_id: str) -> None:
        for room in state.rooms.values():
            if entity_id in room.present_characters:
                room.present_characters.remove(entity_id)

    @staticmethod
    def _find_room_of(state: _SessionState, entity_id: str) -> Optional[str]:
        for room_id, room in state.rooms.items():
            if entity_id in room.present_characters:
                return room_id
        return None

    @staticmethod
    def _distance_and_barriers(state: _SessionState, from_room: str,
                               to_room: str) -> Tuple[float, List[BarrierType], List[str]]:
        """Representative distance, barrier list and room path between two rooms."""
        if from_room == to_room:
            return SAME_ROOM_FT, [], [from_room]
        key = (from_room, to_room) if from_room <= to_room else (to_room, from_room)
        edge = state.edges.get(key)
        if edge is not None:
            barriers: List[BarrierType] = []
            if edge.barrier != BarrierType.NONE:
                if edge.barrier in (BarrierType.CLOSED_DOOR, BarrierType.OPEN_DOOR,
                                    BarrierType.LOCKED_DOOR):
                    # Door barriers resolve by state.
                    barriers = [BarrierType.OPEN_DOOR if edge.state == BarrierState.OPEN
                                else BarrierType.CLOSED_DOOR]
                else:
                    barriers = [edge.barrier]
            return edge.distance_ft, barriers, [from_room, to_room]
        return DISTANT_FT, [BarrierType.CLOSED_DOOR, BarrierType.SOLID_WALL], []


def _room_from_metadata(room: RoomMetadata) -> Room:
    return Room(
        id=room.room_id,
        name=room.room_name,
        desc=room.description,
        tags=[],
        attrs={"lighting": room.lighting},
    )


def _to_ev_action(action_type: ActionType) -> EvActionType:
    if action_type == ActionType.SHOUT:
        return EvActionType.SPEAK
    return EvActionType(action_type.value)


def _to_ev_barrier(barrier: BarrierType) -> EvBarrierType:
    if barrier == BarrierType.LOCKED_DOOR:
        return EvBarrierType.CLOSED_DOOR
    return EvBarrierType(barrier.value)
