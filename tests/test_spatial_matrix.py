"""
Unit tests for Spatial & Acoustic Constraints Matrix.
"""

import pytest
from evennia_world.spatial_matrix import SpatialConstraintsMatrix
from evennia_world.models import GatingLevel, BarrierType, ActionType


@pytest.fixture(autouse=True)
def clear_history():
    """Clear the class-level gating history before each test."""
    SpatialConstraintsMatrix._gating_history.clear()
    yield
    SpatialConstraintsMatrix._gating_history.clear()


# ── Basic gating behaviour ──────────────────────────────────────────────────

def test_direct_whisper_same_room():
    gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[],
        action_type=ActionType.WHISPER,
        raw_text="I hear a venomcrawler.",
        actor_id="rowan",
        recipient_id="domino",
        is_target=True
    )
    assert gating == GatingLevel.DIRECT
    assert 'Rowan whispers to you: "I hear a venomcrawler."' in feed


def test_degraded_whisper_distance():
    gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=12.0,
        barriers=[],
        action_type=ActionType.WHISPER,
        raw_text="I hear a venomcrawler.",
        actor_id="rowan",
        recipient_id="luna",
        is_target=False
    )
    assert gating == GatingLevel.DEGRADED
    assert "murmur quietly" in feed


def test_blackout_wall_barrier():
    gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=20.0,
        barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK,
        raw_text="Hello?",
        actor_id="rowan",
        recipient_id="seamus",
        is_target=False,
        action_tick=10
    )
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


# ── Bug 1: Session isolation ────────────────────────────────────────────────

def test_session_isolation_different_sessions():
    """
    Two different session_ids must NOT share gating history.
    Session A sets a DIRECT gating; Session B should see a fresh
    history (no leftover state from Session A).
    """
    # Session A: establish DIRECT gating at tick 0
    gating_a, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[],
        action_type=ActionType.SPEAK,
        raw_text="hello",
        actor_id="rowan",
        recipient_id="domino",
        session_id="session_A",
        action_tick=0
    )
    assert gating_a == GatingLevel.DIRECT

    # Session B: same physical params but different session.
    gating_b, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=2.0,
        barriers=[],
        action_type=ActionType.SPEAK,
        raw_text="hello",
        actor_id="rowan",
        recipient_id="domino",
        session_id="session_B",
        action_tick=1
    )
    assert gating_b == GatingLevel.DIRECT


def test_session_isolation_no_cross_pollution():
    """
    After session A goes DIRECT->BLACKOUT, session B must NOT be
    affected by session A's cooldown window.
    """
    # Session A: DIRECT at tick 0
    SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0, barriers=[],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="alpha", action_tick=0
    )

    # Session A: BLACKOUT at tick 1 (within cooldown)
    gating_a_cooldown, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="alpha", action_tick=1
    )

    # Session B: same params, fresh session
    gating_b_fresh, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="beta", action_tick=1
    )

    # Both sessions should have the same gating for identical params
    assert gating_a_cooldown == gating_b_fresh


# ── Bug 2: Hysteresis logic ─────────────────────────────────────────────────

def test_hysteresis_direct_to_blackout_cooldown():
    """
    When transitioning from DIRECT to BLACKOUT within the cooldown
    window (2 ticks), the result should be DEGRADED, not BLACKOUT.
    """
    # tick 0: DIRECT
    SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0, barriers=[],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst", action_tick=0
    )

    # tick 1: BLACKOUT (within 2-tick window) -> should be DEGRADED
    gating, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst", action_tick=1
    )
    assert gating == GatingLevel.DEGRADED, (
        f"Hysteresis should suppress BLACKOUT within cooldown; got {gating}"
    )


def test_hysteresis_direct_to_blackout_after_cooldown():
    """
    After the cooldown window expires, BLACKOUT should apply normally.
    """
    # tick 0: DIRECT
    SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0, barriers=[],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst2", action_tick=0
    )

    # tick 5: BLACKOUT (outside 2-tick window) -> should be BLACKOUT
    gating, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0,
        barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst2", action_tick=5
    )
    assert gating == GatingLevel.BLACKOUT


def test_hysteresis_reverse_blackout_to_direct_cooldown():
    """
    The reverse transition (BLACKOUT -> DIRECT) must also respect hysteresis.
    Without it, a character can instantly go from COMPLETE SILENCE to hearing
    everything clearly.
    """
    # tick 0: BLACKOUT (solid wall)
    SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=50.0, barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst3", action_tick=0
    )

    # tick 1: DIRECT (no wall, close range) - within cooldown -> should be DEGRADED
    gating, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0, barriers=[],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst3", action_tick=1
    )
    assert gating == GatingLevel.DEGRADED, (
        f"Reverse hysteresis should suppress DIRECT within cooldown; got {gating}"
    )


def test_hysteresis_reverse_blackout_to_direct_after_cooldown():
    """
    After the cooldown expires, DIRECT should apply normally.
    """
    # tick 0: BLACKOUT
    SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=50.0, barriers=[BarrierType.SOLID_WALL],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst4", action_tick=0
    )

    # tick 10: DIRECT (outside window) -> should be DIRECT
    gating, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=3.0, barriers=[],
        action_type=ActionType.SPEAK, raw_text="hi",
        actor_id="rowan", recipient_id="domino",
        session_id="hyst4", action_tick=10
    )
    assert gating == GatingLevel.DIRECT
