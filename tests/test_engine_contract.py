"""Contract test suite for the WorldEngine interface (Sprint 0, SPRINT_PLAN.md §1.1).

Written against the Protocol in proxy/engine/contract.py and parameterized over engine
implementations. Today it runs fully against InProcessWorldEngine (the reference
implementation of decision 10's tick semantics). When Sprint 1 Track A lands the
missing :4005 endpoints, add HttpWorldEngine to ENGINE_FACTORIES below — e.g.

    ENGINE_FACTORIES["http"] = lambda: HttpWorldEngine(
        EvenniaWorldClient(base_url=test_server_base_url)
    )

— and the same suite becomes the HTTP adapter's acceptance gate.

These are behavioral tests: nothing inside the engine is mocked.
"""

import itertools

import pytest

from proxy.engine.contract import (
    Action,
    ActionType,
    BarrierState,
    BarrierType,
    Edge,
    EngineHealth,
    GatingLevel,
    InvalidMutation,
    Mutation,
    MutationConflict,
    MutationType,
    Room,
    WorldEngine,
    WorldSeed,
)
from proxy.engine.in_process import InProcessWorldEngine

# ── Engine parameterization ─────────────────────────────────────────────

ENGINE_FACTORIES = {
    "in_process": InProcessWorldEngine,
    # "http": lambda: HttpWorldEngine(EvenniaWorldClient(...)),  # Sprint 1 Track A
}

_session_counter = itertools.count()


@pytest.fixture(params=sorted(ENGINE_FACTORIES))
def engine(request):
    return ENGINE_FACTORIES[request.param]()


@pytest.fixture
def sid():
    """A unique session id per test, so SpatialConstraintsMatrix's class-level
    per-session gating history never bleeds across tests."""
    return f"contract_test_{next(_session_counter)}"


def two_room_seed() -> WorldSeed:
    """kitchen —(closed door, 15 ft)— hall; alice and bob in the kitchen, carol in the hall."""
    return WorldSeed(
        rooms=[
            Room(id="kitchen", name="Kitchen", desc="A small kitchen."),
            Room(id="hall", name="Hall", desc="A long hall."),
        ],
        edges=[
            Edge(a="kitchen", b="hall", barrier=BarrierType.CLOSED_DOOR,
                 state=BarrierState.CLOSED, distance_ft=15.0),
        ],
        placements={"alice": "kitchen", "bob": "kitchen", "carol": "hall"},
    )


# ── Protocol conformance ────────────────────────────────────────────────

def test_engine_satisfies_protocol(engine):
    assert isinstance(engine, WorldEngine)
    assert engine.contract_version == "1.0"


# ── ensure_world ────────────────────────────────────────────────────────

async def test_ensure_world_without_seed_creates_minimal_world(engine, sid):
    snapshot = await engine.ensure_world(sid)
    assert snapshot.session_id == sid
    assert snapshot.tick == 0
    assert len(snapshot.rooms) >= 1
    assert snapshot.occupants == []  # no demo characters leak in from templates


async def test_ensure_world_with_seed(engine, sid):
    snapshot = await engine.ensure_world(sid, seed=two_room_seed())
    assert sorted(snapshot.room_ids()) == ["hall", "kitchen"]
    assert len(snapshot.edges) == 1
    edge = snapshot.edges[0]
    assert {edge.a, edge.b} == {"kitchen", "hall"}
    assert edge.barrier == BarrierType.CLOSED_DOOR
    assert edge.state == BarrierState.CLOSED
    assert edge.distance_ft == 15.0
    assert {o.entity_id: o.room_id for o in snapshot.occupants} == {
        "alice": "kitchen", "bob": "kitchen", "carol": "hall",
    }


async def test_ensure_world_is_idempotent_and_keeps_state(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    await engine.apply_mutation(
        sid, Mutation(type=MutationType.MOVE, entity_id="alice", room_id="hall"),
        idempotency_key="mv_1", origin="user",
    )
    # A second ensure_world must not re-seed or roll back the move.
    snapshot = await engine.ensure_world(sid, seed=two_room_seed())
    assert snapshot.find_occupant("alice").room_id == "hall"
    assert len(snapshot.rooms) == 2
    assert len([o for o in snapshot.occupants if o.entity_id == "alice"]) == 1


# ── Session isolation ───────────────────────────────────────────────────

async def test_two_sessions_same_room_ids_independent_state(engine, sid):
    s1, s2 = f"{sid}_a", f"{sid}_b"
    await engine.ensure_world(s1, seed=two_room_seed())
    await engine.ensure_world(s2, seed=two_room_seed())

    await engine.apply_mutation(
        s1, Mutation(type=MutationType.MOVE, entity_id="alice", room_id="hall"),
        idempotency_key="mv_alice", origin="user",
    )
    await engine.apply_mutation(
        s2, Mutation(type=MutationType.CREATE_ROOM,
                     room=Room(id="attic", name="Attic", desc="Dusty.")),
        idempotency_key="mk_attic", origin="gm",
    )

    g1 = await engine.get_graph(s1)
    g2 = await engine.get_graph(s2)
    # s1's move did not touch s2, and s2's new room did not appear in s1.
    assert g1.find_occupant("alice").room_id == "hall"
    assert g2.find_occupant("alice").room_id == "kitchen"
    assert "attic" in g2.room_ids()
    assert "attic" not in g1.room_ids()


async def test_two_sessions_tick_independently(engine, sid):
    s1, s2 = f"{sid}_a", f"{sid}_b"
    await engine.ensure_world(s1)
    await engine.ensure_world(s2)

    await engine.advance_tick(s1, "turn_1")
    await engine.advance_tick(s1, "turn_2")
    r2 = await engine.advance_tick(s2, "turn_1")

    assert (await engine.get_graph(s1)).tick == 2
    assert (await engine.get_graph(s2)).tick == 1
    assert r2.tick == 1  # s2 was not dragged along by s1


# ── Tick semantics (decision 10) ────────────────────────────────────────

async def test_advance_tick_increments_per_new_turn_id(engine, sid):
    await engine.ensure_world(sid)
    r1 = await engine.advance_tick(sid, "turn_1")
    r2 = await engine.advance_tick(sid, "turn_2")
    assert (r1.tick, r1.advanced) == (1, True)
    assert (r2.tick, r2.advanced) == (2, True)


async def test_advance_tick_idempotent_per_turn_id(engine, sid):
    """A regeneration/swipe reuses its turn_id and must not advance the clock."""
    await engine.ensure_world(sid)
    await engine.advance_tick(sid, "turn_1")
    first = await engine.advance_tick(sid, "turn_2")
    for _ in range(3):  # three regenerations of the same user message
        again = await engine.advance_tick(sid, "turn_2")
        assert again.advanced is False
        assert again.tick == first.tick == 2
    # The clock is where it was, and the next real turn advances normally.
    assert (await engine.get_graph(sid)).tick == 2
    nxt = await engine.advance_tick(sid, "turn_3")
    assert (nxt.tick, nxt.advanced) == (3, True)


async def test_submit_action_does_not_advance_tick(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    await engine.advance_tick(sid, "turn_1")
    result = await engine.submit_action(
        sid, Action(actor_id="alice", type=ActionType.SPEAK, text="Hello"), turn_id="turn_1",
    )
    assert result.tick == 1
    assert (await engine.get_graph(sid)).tick == 1


# ── apply_mutation: idempotency, origins, effects ───────────────────────

async def test_apply_mutation_idempotent_per_key(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    mut = Mutation(type=MutationType.MOVE, entity_id="alice", room_id="hall")

    first = await engine.apply_mutation(sid, mut, idempotency_key="k1", origin="gm")
    repeat = await engine.apply_mutation(sid, mut, idempotency_key="k1", origin="gm")

    assert first.applied is True
    assert repeat.applied is False          # replay returns the original result
    assert repeat.idempotency_key == "k1"
    assert repeat.type == MutationType.MOVE
    assert repeat.detail == first.detail

    graph = await engine.get_graph(sid)
    alices = [o for o in graph.occupants if o.entity_id == "alice"]
    assert len(alices) == 1 and alices[0].room_id == "hall"


async def test_apply_mutation_same_key_different_body_conflicts(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    await engine.apply_mutation(
        sid, Mutation(type=MutationType.MOVE, entity_id="alice", room_id="hall"),
        idempotency_key="k1", origin="gm",
    )
    with pytest.raises(MutationConflict):
        await engine.apply_mutation(
            sid, Mutation(type=MutationType.MOVE, entity_id="bob", room_id="hall"),
            idempotency_key="k1", origin="gm",
        )


async def test_mutation_origins_recorded(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    r_gm = await engine.apply_mutation(
        sid, Mutation(type=MutationType.MOVE, entity_id="alice", room_id="hall"),
        idempotency_key="k_gm", origin="gm",
    )
    r_sys = await engine.apply_mutation(
        sid, Mutation(type=MutationType.CREATE_ROOM,
                      room=Room(id="cellar_x", name="Cellar", desc="Dark.")),
        idempotency_key="k_sys", origin="system",
    )
    r_usr = await engine.apply_mutation(
        sid, Mutation(type=MutationType.PLACE, entity_id="dave", room_id="kitchen"),
        idempotency_key="k_usr", origin="user",
    )
    assert (r_gm.origin, r_sys.origin, r_usr.origin) == ("gm", "system", "user")

    # Engines that expose their mutation log must have recorded origins in order.
    if hasattr(engine, "mutation_origins"):
        assert engine.mutation_origins(sid) == [
            ("k_gm", "gm"), ("k_sys", "system"), ("k_usr", "user"),
        ]


async def test_create_room_place_move_visible_in_graph(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())

    await engine.apply_mutation(
        sid,
        Mutation(type=MutationType.CREATE_ROOM,
                 room=Room(id="pantry", name="Pantry", desc="Shelves of jars."),
                 edges=[Edge(a="pantry", b="kitchen", barrier=BarrierType.OPEN_DOOR,
                             state=BarrierState.OPEN, distance_ft=10.0)]),
        idempotency_key="mk_pantry", origin="gm",
    )
    await engine.apply_mutation(
        sid, Mutation(type=MutationType.PLACE, entity_id="dave", room_id="pantry"),
        idempotency_key="pl_dave", origin="system",
    )
    await engine.apply_mutation(
        sid, Mutation(type=MutationType.MOVE, entity_id="bob", room_id="pantry"),
        idempotency_key="mv_bob", origin="user",
    )

    graph = await engine.get_graph(sid)
    assert "pantry" in graph.room_ids()
    pantry_room = next(r for r in graph.rooms if r.id == "pantry")
    assert pantry_room.name == "Pantry"
    assert {(e.key()) for e in graph.edges} >= {("hall", "kitchen"), ("kitchen", "pantry")}
    occupant_rooms = {o.entity_id: o.room_id for o in graph.occupants}
    assert occupant_rooms["dave"] == "pantry"
    assert occupant_rooms["bob"] == "pantry"      # moved out of the kitchen...
    assert occupant_rooms["alice"] == "kitchen"   # ...without disturbing alice
    assert len(graph.occupants) == len(occupant_rooms)  # nobody duplicated


async def test_set_barrier_creates_and_updates_edge(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    await engine.apply_mutation(
        sid,
        Mutation(type=MutationType.SET_BARRIER, a="kitchen", b="hall",
                 barrier=BarrierType.OPEN_DOOR, state=BarrierState.OPEN, distance_ft=12.0),
        idempotency_key="open_door", origin="gm",
    )
    graph = await engine.get_graph(sid)
    edge = next(e for e in graph.edges if e.key() == ("hall", "kitchen"))
    assert edge.barrier == BarrierType.OPEN_DOOR
    assert edge.state == BarrierState.OPEN
    assert edge.distance_ft == 12.0


async def test_mutation_to_unknown_room_is_invalid(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    with pytest.raises(InvalidMutation):
        await engine.apply_mutation(
            sid, Mutation(type=MutationType.MOVE, entity_id="alice", room_id="nowhere"),
            idempotency_key="bad_mv", origin="user",
        )


# ── Actions and consequences ────────────────────────────────────────────

async def test_speak_consequences_follow_spatial_gating(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    # Age the clock past the hysteresis window so raw gating shows through.
    await engine.advance_tick(sid, "turn_1")
    await engine.advance_tick(sid, "turn_2")
    await engine.advance_tick(sid, "turn_3")

    result = await engine.submit_action(
        sid, Action(actor_id="alice", type=ActionType.SPEAK, text="The key is under the mat"),
        turn_id="turn_3",
    )
    by_recipient = {c.recipient_id: c for c in result.consequences}
    assert set(by_recipient) == {"bob", "carol"}

    bob = by_recipient["bob"]          # same room: hears it verbatim
    assert bob.gating == GatingLevel.DIRECT
    assert "The key is under the mat" in bob.sensory_feed

    carol = by_recipient["carol"]      # adjacent through a closed door: degraded, no leak
    assert carol.gating == GatingLevel.DEGRADED
    assert "The key is under the mat" not in carol.sensory_feed


async def test_entity_state_reports_room_and_distances(engine, sid):
    await engine.ensure_world(sid, seed=two_room_seed())
    state = await engine.get_entity_state(sid, "alice")
    assert state.entity_id == "alice"
    assert state.room_id == "kitchen"
    assert state.distances_ft["bob"] <= 5.0        # same room
    assert state.distances_ft["carol"] == 15.0     # across the one edge


# ── reset ───────────────────────────────────────────────────────────────

async def test_reset_single_session_leaves_others_alone(engine, sid):
    s1, s2 = f"{sid}_a", f"{sid}_b"
    await engine.ensure_world(s1, seed=two_room_seed())
    await engine.ensure_world(s2, seed=two_room_seed())
    await engine.advance_tick(s2, "turn_1")

    await engine.reset(s1)

    # s1 comes back fresh; s2 kept its occupants and clock.
    fresh = await engine.ensure_world(s1)
    assert fresh.occupants == []
    assert fresh.tick == 0
    g2 = await engine.get_graph(s2)
    assert g2.find_occupant("alice") is not None
    assert g2.tick == 1


async def test_reset_all_sessions(engine, sid):
    s1, s2 = f"{sid}_a", f"{sid}_b"
    await engine.ensure_world(s1, seed=two_room_seed())
    await engine.ensure_world(s2, seed=two_room_seed())

    await engine.reset(None)

    for s in (s1, s2):
        fresh = await engine.ensure_world(s)
        assert fresh.occupants == []
        assert fresh.tick == 0


# ── health ──────────────────────────────────────────────────────────────

async def test_health(engine, sid):
    await engine.ensure_world(sid)
    health = await engine.health()
    assert isinstance(health, EngineHealth)
    assert health.ok is True
    assert health.contract_version == "1.0"
    assert health.engine
    assert health.sessions >= 1
