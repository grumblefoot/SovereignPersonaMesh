"""
Unit tests for Spatial & Acoustic Constraints Matrix.

Sprint 1 Track A semantics (SRD 3.3.2, SPRINT_PLAN.md §6 decision 9):
- ≤5 ft DIRECT, >5–15 ft DEGRADED, >15 ft BLACKOUT.
- CLOSED_DOOR / METAL_PARTITION / SOLID_WALL blackout regardless of distance;
  OPEN_DOOR does not gate; DRYWALL caps at DEGRADED.
- Hysteresis was REMOVED (decision 9): gating is a pure function of the
  current geometry, so these tests no longer clear any shared history.
- SHOUT (decision 11) lands one gating level better than speak at the same
  geometry, but never through a barrier blackout.
"""

import pytest
from evennia_world.spatial_matrix import SpatialConstraintsMatrix
from evennia_world.models import GatingLevel, BarrierType, ActionType


def speak(distance_ft=3.0, barriers=None, **kw):
    return SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=distance_ft,
        barriers=barriers or [],
        action_type=ActionType.SPEAK,
        raw_text="hello",
        actor_id="rowan",
        recipient_id="domino",
        **kw,
    )


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


# ── Threshold edges (SRD 3.3.2: DIRECT ≤5, DEGRADED >5–15, BLACKOUT >15) ───

def test_direct_at_exactly_five_feet():
    gating, _ = speak(distance_ft=5.0)
    assert gating == GatingLevel.DIRECT


def test_degraded_just_beyond_five_feet():
    gating, _ = speak(distance_ft=5.1)
    assert gating == GatingLevel.DEGRADED


def test_degraded_at_exactly_fifteen_feet():
    gating, _ = speak(distance_ft=15.0)
    assert gating == GatingLevel.DEGRADED


def test_blackout_just_beyond_fifteen_feet():
    gating, feed = speak(distance_ft=15.1)
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


def test_twenty_feet_is_now_blackout():
    """The old implementation had DEGRADED out to 20 ft; SRD 3.3.2 ends it at 15."""
    gating, _ = speak(distance_ft=20.0)
    assert gating == GatingLevel.BLACKOUT


# ── Barrier rules (SRD 3.3.2 / PRD 4.2) ─────────────────────────────────────

def test_closed_door_blackout_even_same_room_distance():
    gating, feed = speak(distance_ft=3.0, barriers=[BarrierType.CLOSED_DOOR])
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


def test_metal_partition_blackout_even_same_room_distance():
    gating, feed = speak(distance_ft=3.0, barriers=[BarrierType.METAL_PARTITION])
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


def test_open_door_does_not_gate():
    gating, _ = speak(distance_ft=3.0, barriers=[BarrierType.OPEN_DOOR])
    assert gating == GatingLevel.DIRECT


def test_open_door_near_band_still_degrades_by_distance():
    gating, _ = speak(distance_ft=10.0, barriers=[BarrierType.OPEN_DOOR])
    assert gating == GatingLevel.DEGRADED


def test_drywall_caps_at_degraded_never_direct():
    """DRYWALL is DEGRADED-at-best: muffled up close, never a clean signal."""
    gating, feed = speak(distance_ft=2.0, barriers=[BarrierType.DRYWALL])
    assert gating == GatingLevel.DEGRADED
    assert "hello" not in feed


def test_drywall_beyond_speak_band_is_blackout():
    gating, _ = speak(distance_ft=20.0, barriers=[BarrierType.DRYWALL])
    assert gating == GatingLevel.BLACKOUT


# ── No hysteresis (decision 9) ──────────────────────────────────────────────

def test_no_hysteresis_direct_to_blackout_is_instant():
    """
    A character who just left a room must NOT keep hearing it: a DIRECT at
    tick 0 followed by blackout geometry at tick 1 is BLACKOUT immediately —
    no 2-tick DEGRADED carryover.
    """
    gating_direct, _ = speak(session_id="hyst", action_tick=0)
    assert gating_direct == GatingLevel.DIRECT

    gating, _ = speak(
        distance_ft=3.0,
        barriers=[BarrierType.SOLID_WALL],
        session_id="hyst",
        action_tick=1,
    )
    assert gating == GatingLevel.BLACKOUT


def test_no_hysteresis_blackout_to_direct_is_instant():
    """Reverse direction: entering a room restores DIRECT on the next tick."""
    speak(distance_ft=50.0, barriers=[BarrierType.SOLID_WALL],
          session_id="hyst_rev", action_tick=0)

    gating, _ = speak(session_id="hyst_rev", action_tick=1)
    assert gating == GatingLevel.DIRECT


def test_gating_is_pure_function_no_shared_state():
    """Identical parameters produce identical gating regardless of order."""
    results = set()
    for i, session in enumerate(["s1", "s2", "s3"]):
        for dist in (3.0, 10.0, 20.0):
            gating, _ = speak(distance_ft=dist, session_id=session, action_tick=i)
            results.add((session, dist, gating))
    assert {g for (_, _, g) in results} == {
        GatingLevel.DIRECT, GatingLevel.DEGRADED, GatingLevel.BLACKOUT,
    }
    for session, dist, gating in results:
        expected = (GatingLevel.DIRECT if dist <= 5 else
                    GatingLevel.DEGRADED if dist <= 15 else GatingLevel.BLACKOUT)
        assert gating == expected


# ── SHOUT (decision 11; gap 8) ──────────────────────────────────────────────

def shout(distance_ft=10.0, barriers=None, **kw):
    return SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=distance_ft,
        barriers=barriers or [],
        action_type=ActionType.SHOUT,
        raw_text="GUARDS!",
        actor_id="rowan",
        recipient_id="domino",
        shout=True,
        **kw,
    )


def test_shout_degraded_band_becomes_direct():
    """Where speak would be DEGRADED (5–15 ft), a shout is DIRECT."""
    speak_gating, _ = speak(distance_ft=10.0)
    shout_gating, feed = shout(distance_ft=10.0)
    assert speak_gating == GatingLevel.DEGRADED
    assert shout_gating == GatingLevel.DIRECT
    assert "GUARDS!" in feed


def test_shout_distance_blackout_band_becomes_degraded():
    """>15 ft but ≤30 ft: speak is BLACKOUT, a shout is DEGRADED (no verbatim)."""
    shout_gating, feed = shout(distance_ft=20.0)
    assert shout_gating == GatingLevel.DEGRADED
    assert "GUARDS!" not in feed


def test_shout_at_exactly_thirty_feet_is_degraded():
    gating, _ = shout(distance_ft=30.0)
    assert gating == GatingLevel.DEGRADED


def test_shout_beyond_thirty_feet_is_blackout():
    gating, feed = shout(distance_ft=31.0)
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


def test_shout_cannot_lift_closed_door_blackout():
    """Barrier blackouts are never lifted by volume (PRD 4.2)."""
    gating, feed = shout(distance_ft=3.0, barriers=[BarrierType.CLOSED_DOOR])
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""


def test_shout_cannot_lift_metal_partition_blackout():
    gating, _ = shout(distance_ft=3.0, barriers=[BarrierType.METAL_PARTITION])
    assert gating == GatingLevel.BLACKOUT


def test_shout_cannot_lift_solid_wall_blackout():
    gating, _ = shout(distance_ft=3.0, barriers=[BarrierType.SOLID_WALL])
    assert gating == GatingLevel.BLACKOUT


def test_shout_drywall_capped_at_degraded():
    gating, feed = shout(distance_ft=3.0, barriers=[BarrierType.DRYWALL])
    assert gating == GatingLevel.DEGRADED
    assert "GUARDS!" not in feed


def test_shout_flag_matches_action_type_path():
    """app.py passes shout=True alongside action_type=SHOUT; both routes agree."""
    gating, _ = SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=10.0,
        barriers=[],
        action_type=ActionType.SHOUT,
        raw_text="GUARDS!",
        actor_id="rowan",
        recipient_id="domino",
        shout=True,
    )
    assert gating == GatingLevel.DIRECT
