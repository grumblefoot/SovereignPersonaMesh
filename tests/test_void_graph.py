"""
Unit tests for FR-005 (Issue #5): Generic Fallback Template ("The Void Graph") for Unmapped Environments.
"""

import pytest
from evennia_world.hybrid_builder import HybridWorldBuilder
from evennia_world.models import RoomMetadata, CharacterWorldState, GatingLevel, BarrierType, ActionType
from evennia_world.spatial_matrix import SpatialConstraintsMatrix
from evennia_world.app import _compute_distance_and_barriers, current_world, session_worlds
from proxy.rag.prompt_builder import CognitivePromptBuilder


class TestTemplateConfidenceScoring:
    def test_rigid_template_match_above_threshold(self):
        builder = HybridWorldBuilder()
        # "dungeon cellar prison" scores well over 0.65
        result = builder.match_template(["dungeon", "cellar", "prison"])
        assert result == "dungeon_cellar"

    def test_unmapped_setting_falls_back_to_generic_void(self):
        builder = HybridWorldBuilder()
        # Abstract/unmapped setting with no matching keywords
        result = builder.match_template(["freefall", "zero-gravity", "mindscape", "quantum"])
        assert result == "generic_void"

    def test_fuzzy_match_unmapped_text_fallback(self):
        builder = HybridWorldBuilder()
        result = builder.match_template_fuzzy("Falling endlessly through a surreal void of swirling colors.")
        assert result == "generic_void"


class TestVoidGraphInstantiation:
    def test_generic_void_instantiates_three_nodes(self):
        builder = HybridWorldBuilder()
        world = builder.instantiate_world("generic_void")
        assert len(world) == 3
        assert "central_nexus" in world
        assert "node_alpha" in world
        assert "node_beta" in world

    def test_void_graph_exit_topology(self):
        builder = HybridWorldBuilder()
        world = builder.instantiate_world("generic_void")
        central = world["central_nexus"]
        assert "node_alpha" in central.exits
        assert "node_beta" in central.exits
        assert world["node_alpha"].exits == ["central_nexus"]
        assert world["node_beta"].exits == ["central_nexus"]


class TestFlavorTextMetadata:
    def test_room_metadata_flavor_text_field(self):
        room = RoomMetadata(
            room_id="central_nexus",
            room_name="Central Nexus",
            description="Abstract nexus",
            flavor_text="Zero-gravity freefall chamber with flickering neon particle trails.",
        )
        assert room.flavor_text == "Zero-gravity freefall chamber with flickering neon particle trails."

    def test_prompt_builder_injects_flavor_text(self):
        builder = CognitivePromptBuilder()
        prompt = builder.build_csa_prompt(
            system_prompt="You are Rowan.",
            sensory_feed="You hear nothing.",
            retrieved_memories=[],
            chat_history=[],
            spatial_context="Location: Central Nexus",
            flavor_text="Surreal neon mindscape.",
        )
        assert "[CURRENT SPATIAL & SENSORY ENVIRONMENT]" in prompt
        assert "Environmental Atmosphere: Surreal neon mindscape." in prompt


class TestVoidGraphSpatialGating:
    def test_adjacent_nodes_enforce_solid_wall_blackout(self):
        builder = HybridWorldBuilder()
        world = builder.instantiate_world("generic_void")
        # Temporarily populate current_world for test
        current_world.clear()
        current_world.update(world)

        dist, barriers = _compute_distance_and_barriers("central_nexus", "node_alpha", ActionType.SPEAK)
        assert BarrierType.SOLID_WALL in barriers

        gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
            distance_ft=dist,
            barriers=barriers,
            action_type=ActionType.SPEAK,
            raw_text="Can anyone hear me?",
            actor_id="rowan",
            recipient_id="luna",
            action_tick=10,
        )
        assert gating == GatingLevel.BLACKOUT
        assert feed == ""
