"""
HttpWorldEngine — the WorldEngine contract mapped onto the :4005 HTTP API through
``proxy.backend_client.evennia_client.EvenniaWorldClient``.

Sprint 1 Track A closed the original ten gaps; the full mapping is now:

- ``ensure_world``       → GET /world/snapshot (create-on-read) + seed composition:
                           POST /world/rooms, POST /world/barrier, POST /world/move.
                           Seeding is idempotent: if every seed room already exists,
                           the existing world is returned untouched.
- ``advance_tick``       → POST /world/tick (per-session, idempotent per turn_id).
- ``submit_action``      → POST /world/action (shout native since A1; turn_id forwarded).
- ``get_graph``          → GET /world/snapshot.
- ``get_entity_state``   → GET /world/state.
- ``apply_mutation``     → MOVE/PLACE → /world/move; CREATE_ROOM → /world/rooms
                           (+ /world/barrier per attached edge); SET_BARRIER → /world/barrier.
                           Idempotency and origin ride in the request BODY; the server
                           replays duplicates (applied=False) and 409s conflicting reuse.
- ``reset``              → DELETE /world/admin/reset[?session_id=].
- ``health``             → GET /health.

Remaining deliberate limits:
- ``Occupant.posture`` is not on the wire (always []).
- ``Consequence.path`` is not on the wire (always []).
- ``MutationResult.tick`` costs one extra snapshot GET per mutation.
"""

from typing import Optional

import httpx

from proxy.backend_client.evennia_client import EvenniaWorldClient
from proxy.engine.contract import (
    CONTRACT_VERSION,
    Action,
    ActionResult,
    ActionType,
    BarrierState,
    BarrierType,
    Consequence,
    EngineCapabilityError,
    EngineHealth,
    EngineUnavailable,
    EntityKind,
    EntityState,
    GatingLevel,
    InvalidMutation,
    Mutation,
    MutationConflict,
    MutationOrigin,
    MutationResult,
    MutationType,
    Room,
    Edge,
    Occupant,
    TickResult,
    WorldGraph,
    WorldNotFound,
    WorldSeed,
    WorldSnapshot,
)


class HttpWorldEngine:
    """WorldEngine adapter over the :4005 service via EvenniaWorldClient."""

    contract_version: str = CONTRACT_VERSION

    def __init__(self, client: EvenniaWorldClient, template_key: str = "dynamic"):
        self._client = client
        self._template_key = template_key

    # ── Internals ───────────────────────────────────────────────────

    async def _snapshot(self, session_id: str) -> WorldGraph:
        try:
            resp = await self._client.client.get(
                f"{self._client.base_url}/world/snapshot",
                params={"session_id": session_id, "template_key": self._template_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc
        return WorldGraph(
            session_id=data["session_id"],
            tick=int(data.get("tick", 0)),
            rooms=[Room(id=r["room_id"], name=r.get("name", r["room_id"]),
                        desc=r.get("description", "")) for r in data.get("rooms", [])],
            edges=[Edge(a=e["a"], b=e["b"], barrier=BarrierType(e.get("barrier", "none")),
                        state=BarrierState(e.get("state", "closed")),
                        distance_ft=float(e.get("distance_ft", 15.0)))
                   for e in data.get("edges", [])],
            occupants=[Occupant(entity_id=o["entity_id"], room_id=o["room_id"])
                       for o in data.get("occupants", [])],
        )

    @staticmethod
    async def _call(coro, *, not_found=WorldNotFound):
        """Run a client call, translating HTTP failures into contract errors."""
        try:
            return await coro
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (400, 404):
                raise not_found(exc.response.text) from exc
            if code == 409:
                raise MutationConflict(exc.response.text) from exc
            raise EngineUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc

    # ── World lifecycle ─────────────────────────────────────────────

    async def ensure_world(self, session_id: str, seed: Optional[WorldSeed] = None) -> WorldSnapshot:
        graph = await self._snapshot(session_id)  # create-on-read
        if seed is None:
            if not graph.rooms:
                # The dynamic template starts empty; the contract promises a minimal world.
                await self._call(self._client.create_room(
                    room_id="origin", name="Origin", desc="A featureless starting point.",
                    session_id=session_id,
                    idempotency_key=f"seed:{session_id}:room:origin", origin="system",
                ), not_found=InvalidMutation)
                graph = await self._snapshot(session_id)
            return graph
        if seed.template_key or seed.keywords:
            raise EngineCapabilityError(
                "ensure_world(template seed)",
                "template/keyword seeding joins Sprint 2 world seeding; pass explicit rooms",
            )
        existing = set(graph.room_ids())
        if seed.rooms and all(r.id in existing for r in seed.rooms):
            return graph  # idempotent re-ensure: never reset live state
        for room in seed.rooms:
            if room.id in existing:
                continue
            await self._call(self._client.create_room(
                room_id=room.id, name=room.name, desc=room.desc, session_id=session_id,
                idempotency_key=f"seed:{session_id}:room:{room.id}", origin="system",
            ), not_found=InvalidMutation)
        for i, edge in enumerate(seed.edges):
            await self._call(self._client.set_barrier(
                a=edge.a, b=edge.b, barrier=edge.barrier.value, state=edge.state.value,
                distance_ft=edge.distance_ft, session_id=session_id,
                template_key=self._template_key,
                idempotency_key=f"seed:{session_id}:edge:{i}", origin="system",
            ), not_found=InvalidMutation)
        for entity_id, room_id in (seed.placements or {}).items():
            await self._call(self._client.move_character(
                character_id=entity_id, room_id=room_id, session_id=session_id,
                idempotency_key=f"seed:{session_id}:place:{entity_id}", origin="system",
            ), not_found=InvalidMutation)
        return await self._snapshot(session_id)

    async def reset(self, session_id: Optional[str]) -> None:
        params = {"session_id": session_id} if session_id is not None else {}
        try:
            resp = await self._client.client.delete(
                f"{self._client.base_url}/world/admin/reset", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc

    async def health(self) -> EngineHealth:
        root = self._client.base_url
        for suffix in ("/api/v1", "/api"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        try:
            resp = await self._client.client.get(f"{root}/health")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return EngineHealth(ok=False, engine="http", detail=str(exc))
        return EngineHealth(
            ok=data.get("status") in ("healthy", "ok", "online"),
            engine="http",
            contract_version=self.contract_version,
            sessions=int(data.get("active_sessions", 0) or 0),
            detail=str(data.get("status", "")),
        )

    # ── Clock ───────────────────────────────────────────────────────

    async def advance_tick(self, session_id: str, turn_id: str) -> TickResult:
        try:
            resp = await self._client.client.post(
                f"{self._client.base_url}/world/tick",
                json={"session_id": session_id, "turn_id": turn_id})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc
        return TickResult(tick=int(data["tick"]), turn_id=data["turn_id"],
                          advanced=bool(data["advanced"]))

    # ── Actions ─────────────────────────────────────────────────────

    async def submit_action(self, session_id: str, action: Action, turn_id: str) -> ActionResult:
        if action.type == ActionType.MOVE:
            raise EngineCapabilityError(
                "submit_action(move)",
                "POST /world/action treats move as flavour only; use apply_mutation(MOVE)",
            )
        data = await self._call(self._client.submit_action(
            character_id=action.actor_id,
            action_type=action.type.value,  # shout is native since Track A1
            raw_text=action.text,
            target_id=action.target_id,
            session_id=session_id,
            turn_id=turn_id,
        ))
        consequences = [
            Consequence(
                recipient_id=c["recipient_id"],
                gating=GatingLevel(c["gating_level"]),
                sensory_feed=c.get("sensory_feed", ""),
                distance_ft=float(c.get("distance_ft", 0.0)),
                barriers=[BarrierType(b) for b in c.get("barriers", [])],
                path=[],  # not on the wire
            )
            for c in data.get("consequences", [])
        ]
        return ActionResult(
            success=bool(data.get("success", True)),
            tick=int(data.get("action_tick", 0)),
            consequences=consequences,
        )

    # ── Reads ───────────────────────────────────────────────────────

    async def get_graph(self, session_id: str) -> WorldGraph:
        return await self._snapshot(session_id)

    async def get_entity_state(self, session_id: str, entity_id: str) -> EntityState:
        data = await self._call(
            self._client.get_character_state(entity_id, session_id=session_id))
        room = data.get("current_room") or {}
        distances = {k: float(v) for k, v in (data.get("distances") or {}).items()
                     if k != entity_id}
        return EntityState(
            entity_id=entity_id,
            kind=EntityKind.CHARACTER,
            room_id=room.get("room_id"),
            distances_ft=distances,
        )

    # ── Mutations ───────────────────────────────────────────────────

    async def apply_mutation(self, session_id: str, mutation: Mutation,
                             idempotency_key: str, origin: MutationOrigin) -> MutationResult:
        if mutation.type in (MutationType.MOVE, MutationType.PLACE):
            if not mutation.entity_id or not mutation.room_id:
                raise InvalidMutation(f"{mutation.type.value} requires entity_id and room_id")
            data = await self._call(self._client.move_character(
                character_id=mutation.entity_id, room_id=mutation.room_id,
                session_id=session_id, idempotency_key=idempotency_key,
                origin=str(origin)), not_found=InvalidMutation)
            detail = f"{mutation.type.value.lower()}d {mutation.entity_id} to {mutation.room_id}"
        elif mutation.type == MutationType.CREATE_ROOM:
            if mutation.room is None:
                raise InvalidMutation("CREATE_ROOM requires a room")
            data = await self._call(self._client.create_room(
                room_id=mutation.room.id, name=mutation.room.name, desc=mutation.room.desc,
                session_id=session_id, idempotency_key=idempotency_key,
                origin=str(origin)), not_found=InvalidMutation)
            for i, edge in enumerate(mutation.edges or []):
                await self._call(self._client.set_barrier(
                    a=edge.a, b=edge.b, barrier=edge.barrier.value, state=edge.state.value,
                    distance_ft=edge.distance_ft, session_id=session_id,
                    template_key=self._template_key,
                    idempotency_key=f"{idempotency_key}:edge:{i}",
                    origin=str(origin)), not_found=InvalidMutation)
            detail = f"created room {mutation.room.id}"
        elif mutation.type == MutationType.SET_BARRIER:
            data = await self._call(self._client.set_barrier(
                a=mutation.a, b=mutation.b, barrier=mutation.barrier.value,
                state=mutation.state.value, distance_ft=mutation.distance_ft,
                session_id=session_id, template_key=self._template_key,
                idempotency_key=idempotency_key, origin=str(origin)),
                not_found=InvalidMutation)
            detail = f"set barrier {mutation.a}~{mutation.b} {mutation.barrier.value}/{mutation.state.value}"
        else:  # pragma: no cover - MutationType is a closed enum
            raise EngineCapabilityError(f"apply_mutation({mutation.type})")

        graph = await self._snapshot(session_id)
        return MutationResult(
            applied=not bool(data.get("duplicate", False)),
            idempotency_key=idempotency_key,
            origin=origin,
            type=mutation.type,
            tick=graph.tick,
            detail=detail,
        )
