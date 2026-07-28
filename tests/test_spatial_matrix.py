"""
Unit tests for Spatial & Acoustic Constraints Matrix.
"""

import pytest
from evennia_world.spatial_matrix import SpatialConstraintsMatrix
from evennia_world.models import GatingLevel, BarrierType, ActionType


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
        is_target=False
    )
    assert gating == GatingLevel.BLACKOUT
    assert feed == ""
