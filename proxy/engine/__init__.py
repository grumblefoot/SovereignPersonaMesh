"""Sovereign Persona Mesh — world-engine interface contract (Sprint 0).

One contract, built from the world-engine plan's `Protocol` (docs/plans/world_engine.md §6)
reconciled with the gating plan's engine interface (docs/plans/gating.md §1) per
SPRINT_PLAN.md §1.1: the engine stores and mutates world state; **perception lives in the
proxy** (`core/gating`, Sprint 2) and works on the graph snapshot, so there is no
`perceive()` here.

Adapters:
- ``InProcessWorldEngine`` (proxy.engine.in_process): the reference implementation,
  in memory, zero network, zero DB.
- ``HttpWorldEngine`` (proxy.engine.http_adapter): today's :4005 HTTP API via
  ``EvenniaWorldClient``; contract methods the HTTP API cannot express raise
  ``EngineCapabilityError`` (Sprint 1 Track A work items).
"""

from proxy.engine.contract import (
    CONTRACT_VERSION,
    Action,
    ActionResult,
    ActionType,
    BarrierState,
    BarrierType,
    Consequence,
    Edge,
    EngineCapabilityError,
    EngineError,
    EngineHealth,
    EngineUnavailable,
    EntityState,
    GatingLevel,
    InvalidMutation,
    Mutation,
    MutationConflict,
    MutationResult,
    MutationType,
    Occupant,
    Room,
    TickResult,
    WorldEngine,
    WorldGraph,
    WorldNotFound,
    WorldSeed,
    WorldSnapshot,
)

__all__ = [
    "CONTRACT_VERSION",
    "Action",
    "ActionResult",
    "ActionType",
    "BarrierState",
    "BarrierType",
    "Consequence",
    "Edge",
    "EngineCapabilityError",
    "EngineError",
    "EngineHealth",
    "EngineUnavailable",
    "EntityState",
    "GatingLevel",
    "InvalidMutation",
    "Mutation",
    "MutationConflict",
    "MutationResult",
    "MutationType",
    "Occupant",
    "Room",
    "TickResult",
    "WorldEngine",
    "WorldGraph",
    "WorldNotFound",
    "WorldSeed",
    "WorldSnapshot",
]
