"""
HttpWorldEngine — the WorldEngine contract mapped onto the EXISTING :4005 HTTP API
through ``proxy.backend_client.evennia_client.EvenniaWorldClient``.

What maps cleanly today
-----------------------
- ``submit_action``      → POST /api/v1/world/action (shout downgraded to speak; see gap 8)
- ``get_entity_state``   → GET  /api/v1/world/state
- ``apply_mutation``:
    * MOVE               → POST /api/v1/world/move  (client.move_character)
    * PLACE              → POST /api/v1/world/move  (placement == move in today's API;
                           POST /api/v1/world/characters exists server-side but the
                           client does not wrap it)
    * CREATE_ROOM        → POST /api/v1/world/rooms (client.create_room)
- ``ensure_world`` (no seed) → lazy: today's engine creates session worlds on first
  touch, so this is a no-op server-side — but it cannot RETURN a snapshot (gap 1),
  so it raises ``EngineCapabilityError`` until GET /world/snapshot ships.
- ``reset(None)``        → DELETE /api/v1/world/admin/reset (global wipe)
- ``health``             → GET /health (server root)

Contract gaps in today's HTTP API — Sprint 1 Track A work items
---------------------------------------------------------------
1.  ``get_graph`` / ``ensure_world`` return value: no ``GET /api/v1/world/snapshot?session_id=``
    endpoint exists; there is no way to fetch rooms+edges+occupants in one call.
2.  ``ensure_world(seed)``: no session-scoped seeding endpoint.
    ``POST /world/configure`` exists but uses the template key as the session id
    (known Sprint 0 bug), so it cannot express per-session seeding.
3.  ``advance_tick``: no tick endpoint at all. The server tick is global, seeded at
    1420, and bumps on every POST /world/action including regenerations — the
    opposite of decision 10's per-session, per-turn_id idempotent clock.
4.  ``SET_BARRIER``: no endpoint for creating or changing edges/exits/barrier state.
5.  Mutation idempotency: ``EvenniaWorldClient`` sends ``X-Idempotency-Key`` but the
    server ignores it, so repeats re-apply over HTTP. This adapter keeps a client-side
    replay cache as a stopgap, which only holds within one adapter instance.
6.  Mutation ``origin`` ("gm" | "system" | "user"): nothing on the wire or in the DB
    records it; this adapter echoes it back client-side only.
7.  ``MOVE``/``PLACE`` session scoping: the server's local ``CharacterMovePayload`` in
    evennia_world/app.py has no ``session_id`` field, so every move lands in
    ``default_session`` (known Sprint 0 bug) even though the client sends it.
8.  ``ActionType.SHOUT``: the HTTP enum has no "shout"; this adapter downgrades it to
    "speak" before dispatch.
9.  ``reset(session_id)``: DELETE /api/v1/world/admin/reset wipes every session;
    there is no per-session reset.
10. ``turn_id`` on ``submit_action``: POST /world/action takes no turn identifier and
    advances the global tick per call, so replayed turns double-count server-side.

Every gap raises ``EngineCapabilityError`` naming the capability, so callers (and the
Sprint 1 contract suite) can tell "not implemented over HTTP yet" apart from a real
engine failure.
"""

from typing import Dict, Optional

import httpx

from proxy.backend_client.evennia_client import EvenniaWorldClient
from proxy.engine.contract import (
    CONTRACT_VERSION,
    Action,
    ActionResult,
    ActionType,
    BarrierType,
    Consequence,
    EngineCapabilityError,
    EngineHealth,
    EngineUnavailable,
    EntityKind,
    EntityState,
    GatingLevel,
    Mutation,
    MutationConflict,
    MutationOrigin,
    MutationResult,
    MutationType,
    TickResult,
    WorldGraph,
    WorldNotFound,
    WorldSeed,
    WorldSnapshot,
)


class HttpWorldEngine:
    """WorldEngine adapter over the existing :4005 service via EvenniaWorldClient."""

    contract_version: str = CONTRACT_VERSION

    def __init__(self, client: EvenniaWorldClient):
        self._client = client
        # Client-side replay cache (stopgap for gap 5: the server ignores
        # X-Idempotency-Key today). Keyed per session.
        self._mutation_cache: Dict[str, Dict[str, MutationResult]] = {}

    # ── World lifecycle ─────────────────────────────────────────────

    async def ensure_world(self, session_id: str, seed: Optional[WorldSeed] = None) -> WorldSnapshot:
        if seed is not None:
            raise EngineCapabilityError(
                "ensure_world(seed)",
                "no session-scoped seeding endpoint; POST /world/configure uses the "
                "template key as the session id (Sprint 1 Track A)",
            )
        # The server creates session worlds lazily on first touch, but there is no
        # snapshot endpoint to return the contract's WorldSnapshot from (gap 1).
        raise EngineCapabilityError(
            "ensure_world",
            "needs GET /api/v1/world/snapshot?session_id= (Sprint 1 Track A)",
        )

    async def reset(self, session_id: Optional[str]) -> None:
        if session_id is not None:
            raise EngineCapabilityError(
                "reset(session_id)",
                "DELETE /api/v1/world/admin/reset is global-only; no per-session reset "
                "endpoint (Sprint 1 Track A)",
            )
        try:
            resp = await self._client.client.delete(f"{self._client.base_url}/world/admin/reset")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc
        self._mutation_cache.clear()

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
        raise EngineCapabilityError(
            "advance_tick",
            "no tick endpoint; the server tick is global, seeded at 1420, and bumps on "
            "every /world/action including regenerations (Sprint 1 Track A)",
        )

    # ── Actions ─────────────────────────────────────────────────────

    async def submit_action(self, session_id: str, action: Action, turn_id: str) -> ActionResult:
        # Gap 8: the HTTP enum has no "shout". Gap 10: the server takes no turn_id.
        action_type = "speak" if action.type == ActionType.SHOUT else action.type.value
        if action.type == ActionType.MOVE:
            raise EngineCapabilityError(
                "submit_action(move)",
                "POST /world/action treats move as flavour only; use apply_mutation(MOVE)",
            )
        try:
            data = await self._client.submit_action(
                character_id=action.actor_id,
                action_type=action_type,
                raw_text=action.text,
                target_id=action.target_id,
                session_id=session_id,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise WorldNotFound(f"actor '{action.actor_id}' in session '{session_id}'") from exc
            raise EngineUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc

        consequences = [
            Consequence(
                recipient_id=c["recipient_id"],
                gating=GatingLevel(c["gating_level"]),
                sensory_feed=c.get("sensory_feed", ""),
                distance_ft=float(c.get("distance_ft", 0.0)),
                barriers=[BarrierType(b) for b in c.get("barriers", [])],
                path=[],  # not on the wire today
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
        raise EngineCapabilityError(
            "get_graph",
            "needs GET /api/v1/world/snapshot?session_id= returning rooms, edges "
            "{a, b, barrier, state, distance_ft} and occupants (Sprint 1 Track A)",
        )

    async def get_entity_state(self, session_id: str, entity_id: str) -> EntityState:
        try:
            data = await self._client.get_character_state(entity_id, session_id=session_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise WorldNotFound(f"entity '{entity_id}' in session '{session_id}'") from exc
            raise EngineUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc

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
        cache = self._mutation_cache.setdefault(session_id, {})
        seen = cache.get(idempotency_key)
        if seen is not None:
            # Client-side stopgap for gap 5; the server has no replay protection.
            return seen.model_copy(update={"applied": False})

        if mutation.type in (MutationType.MOVE, MutationType.PLACE):
            if not mutation.entity_id or not mutation.room_id:
                raise EngineCapabilityError(
                    f"apply_mutation({mutation.type.value})",
                    "entity_id and room_id are required",
                )
            await self._call(
                self._client.move_character(
                    character_id=mutation.entity_id,
                    room_id=mutation.room_id,
                    session_id=session_id,       # dropped server-side today (gap 7)
                    idempotency_key=idempotency_key,  # ignored server-side today (gap 5)
                )
            )
            detail = f"{mutation.type.value.lower()}d {mutation.entity_id} to {mutation.room_id}"
        elif mutation.type == MutationType.CREATE_ROOM:
            if mutation.room is None:
                raise EngineCapabilityError("apply_mutation(CREATE_ROOM)", "a room is required")
            if mutation.edges:
                raise EngineCapabilityError(
                    "apply_mutation(CREATE_ROOM with edges)",
                    "POST /world/rooms cannot link rooms; SET_BARRIER has no endpoint "
                    "(Sprint 1 Track A)",
                )
            await self._call(
                self._client.create_room(
                    room_id=mutation.room.id,
                    name=mutation.room.name,
                    desc=mutation.room.desc,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                )
            )
            detail = f"created room {mutation.room.id}"
        elif mutation.type == MutationType.SET_BARRIER:
            raise EngineCapabilityError(
                "apply_mutation(SET_BARRIER)",
                "no exits/barrier endpoint on the HTTP API (Sprint 1 Track A)",
            )
        else:  # pragma: no cover - MutationType is a closed enum
            raise EngineCapabilityError(f"apply_mutation({mutation.type})")

        result = MutationResult(
            applied=True,
            idempotency_key=idempotency_key,
            origin=origin,  # echoed client-side only; not recorded server-side (gap 6)
            type=mutation.type,
            tick=0,         # the HTTP API exposes no per-session tick (gap 3)
            detail=detail,
        )
        cache[idempotency_key] = result
        return result

    @staticmethod
    async def _call(coro):
        try:
            return await coro
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise WorldNotFound(str(exc)) from exc
            if exc.response.status_code == 409:
                raise MutationConflict(str(exc)) from exc
            raise EngineUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailable(str(exc)) from exc
