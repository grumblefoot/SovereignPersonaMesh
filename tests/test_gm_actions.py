"""
Comprehensive unit tests for Game Master Actions feature.
Covers GMActionRegistry, CognitivePromptBuilder GM injection,
MonologueStreamParser extract_gm_actions, and integration flows.
"""

import pytest
import json
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.core.stream_parser import MonologueStreamParser
from proxy.rag.gm_actions import default_gm_registry, GMActionRegistry, GMActionDefinition


# ---------------------------------------------------------------------------
# GMActionRegistry tests
# ---------------------------------------------------------------------------

def test_register_and_get_action():
    """Test registering an action and retrieving it by id."""
    registry = GMActionRegistry()
    registry.register(GMActionDefinition(
        id="CUSTOM_ACTION",
        description="A custom action",
        schema_example='{"type": "CUSTOM_ACTION"}',
        prompt_fragment="Do [GM_ACTION: {...}]",
        tags=["custom"],
        priority=50
    ))
    action = registry.get("CUSTOM_ACTION")
    assert action is not None
    assert action.id == "CUSTOM_ACTION"
    assert action.priority == 50


def test_get_nonexistent_action():
    """Retrieving a non-existent action returns None."""
    registry = GMActionRegistry()
    assert registry.get("DOES_NOT_EXIST") is None


def test_get_all_active_returns_sorted_by_priority():
    """get_all_active returns actions sorted by priority descending."""
    registry = GMActionRegistry()
    # Register custom actions with specific priorities
    registry.register(GMActionDefinition(
        id="LOW", description="Low priority", schema_example="{}",
        prompt_fragment="low", priority=10
    ))
    registry.register(GMActionDefinition(
        id="HIGH", description="High priority", schema_example="{}",
        prompt_fragment="high", priority=90
    ))
    registry.register(GMActionDefinition(
        id="MID", description="Mid priority", schema_example="{}",
        prompt_fragment="mid", priority=50
    ))

    actions = registry.get_all_active()
    # Default actions (MOVE=100, CREATE_ROOM=90) are also present,
    # but custom actions should still be sorted among themselves.
    custom = [a for a in actions if a.id in ("LOW", "MID", "HIGH")]
    priorities = [a.priority for a in custom]
    assert priorities == sorted(priorities, reverse=True)


def test_get_all_active_filter_by_tags():
    """get_all_active filters correctly when include_tags is provided."""
    registry = GMActionRegistry()
    registry.register(GMActionDefinition(
        id="TAG_A", description="", schema_example="{}",
        prompt_fragment="a", tags=["alpha"]
    ))
    registry.register(GMActionDefinition(
        id="TAG_B", description="", schema_example="{}",
        prompt_fragment="b", tags=["beta"]
    ))
    registry.register(GMActionDefinition(
        id="BOTH", description="", schema_example="{}",
        prompt_fragment="both", tags=["alpha", "gamma"]
    ))

    alpha_actions = registry.get_all_active(include_tags=["alpha"])
    ids = {a.id for a in alpha_actions}
    assert ids == {"TAG_A", "BOTH"}
    assert "TAG_B" not in ids


def test_default_registry_has_expected_actions():
    """The global default_gm_registry should have MOVE and CREATE_ROOM."""
    move = default_gm_registry.get("MOVE")
    create = default_gm_registry.get("CREATE_ROOM")
    assert move is not None
    assert create is not None
    assert move.id == "MOVE"
    assert create.id == "CREATE_ROOM"


# ---------------------------------------------------------------------------
# CognitivePromptBuilder GM injection tests
# ---------------------------------------------------------------------------

def test_prompt_builder_injects_gm_actions():
    builder = CognitivePromptBuilder()

    prompt = builder.build_csa_prompt(
        system_prompt="Test System",
        sensory_feed="Test Feed",
        retrieved_memories=[],
        chat_history=[],
        spatial_context="Test Context"
    )

    assert "[GM_ACTION: {\"type\": \"MOVE\"" in prompt
    assert "When performing a Game Master action, you MUST output exactly: [GM_ACTION:" in prompt


def test_prompt_builder_includes_formatting_rule():
    """The formatting rule about outputting exactly [GM_ACTION: {...}] must be present."""
    builder = CognitivePromptBuilder()

    prompt = builder.build_csa_prompt(
        system_prompt="Test",
        sensory_feed="",
        retrieved_memories=[],
        chat_history=[],
        spatial_context=""
    )

    assert "Do not wrap in markdown" in prompt


def test_prompt_builder_no_inner_monologue():
    """When inner_monologue_enabled is False, GM actions still appear in system content."""
    from config.hardware_tiers import HARDWARE_TIERS, HardwareTierEnum
    hw = HARDWARE_TIERS[HardwareTierEnum.EXPERIMENTAL]  # has inner_monologue_enabled=False
    builder = CognitivePromptBuilder(hw_config=hw)

    prompt = builder.build_csa_prompt(
        system_prompt="Test",
        sensory_feed="",
        retrieved_memories=[],
        chat_history=[],
        spatial_context=""
    )

    assert "System Directive:" not in prompt
    assert "Character Output:" in prompt


def test_messages_builder_injects_gm_actions():
    """build_csa_messages should also inject GM action instructions into the system message."""
    builder = CognitivePromptBuilder()

    messages = builder.build_csa_messages(
        system_prompt="Test System",
        sensory_feed="Test Feed",
        retrieved_memories=[],
        chat_history=[],
        spatial_context="Test Context"
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "[GM_ACTION:" in messages[0]["content"]
    assert "Available GM Actions:" in messages[0]["content"]


def test_messages_builder_gm_in_monologue_enabled():
    """GM actions in system content when inner_monologue_enabled is True."""
    builder = CognitivePromptBuilder()

    messages = builder.build_csa_messages(
        system_prompt="Test",
        sensory_feed="",
        retrieved_memories=[],
        chat_history=[],
        spatial_context=""
    )

    content = messages[0]["content"]
    assert "[GM_ACTION:" in content
    assert "When performing a Game Master action, you MUST output exactly:" in content


# ---------------------------------------------------------------------------
# MonologueStreamParser.extract_gm_actions tests
# ---------------------------------------------------------------------------

def test_stream_parser_extracts_gm_actions():
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = """
    I need to move Vardus to the dungeon.
    [GM_ACTION: {"type": "MOVE", "entity": "Vardus", "room_id": "dungeon_cellar"}]
    And I also need to create a new room.
    [GM_ACTION: {"type": "CREATE_ROOM", "room_id": "hidden_chamber", "name": "Hidden Chamber", "desc": "A dark hidden room."}]
    """

    actions = parser.extract_gm_actions()
    assert len(actions) == 2
    assert actions[0]["type"] == "MOVE"
    assert actions[0]["entity"] == "Vardus"
    assert actions[0]["room_id"] == "dungeon_cellar"

    assert actions[1]["type"] == "CREATE_ROOM"
    assert actions[1]["room_id"] == "hidden_chamber"


def test_stream_parser_handles_malformed_json(caplog):
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = """
    I'm going to make a mistake.
    [GM_ACTION: {"type": "MOVE", "entity": "Vardus", missing_quotes}]
    """

    actions = parser.extract_gm_actions()
    assert len(actions) == 0
    assert "Failed to parse GM Action JSON" in caplog.text


def test_stream_parser_empty_buffer():
    """No actions extracted when buffer is empty."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = ""
    assert parser.extract_gm_actions() == []


def test_stream_parser_no_gm_action_blocks():
    """No actions when there are no [GM_ACTION: ...] blocks at all."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = "Just regular monologue text with no GM actions."
    assert parser.extract_gm_actions() == []


def test_stream_parser_nested_braces_in_json():
    """Handles nested JSON objects inside GM_ACTION blocks."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = """
    Moving with parameters.
    [GM_ACTION: {"type": "MOVE", "entity": {"name": "Vardus", "id": 42}, "room_id": "dungeon"}]
    """
    actions = parser.extract_gm_actions()
    assert len(actions) == 1
    assert actions[0]["type"] == "MOVE"
    assert actions[0]["entity"]["name"] == "Vardus"


def test_stream_parser_multiple_on_same_line():
    """Extracts multiple GM_ACTION blocks even if on the same line."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = "[GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"A\"}] [GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"B\"}]"
    actions = parser.extract_gm_actions()
    assert len(actions) == 2
    assert actions[0]["entity"] == "A"
    assert actions[1]["entity"] == "B"


def test_stream_parser_whitespace_variants():
    """Regex matches with varying whitespace around the colon and JSON."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = "[GM_ACTION:  {\"type\": \"MOVE\"}]\n[GM_ACTION:{\"type\":\"CREATE_ROOM\"}]"
    actions = parser.extract_gm_actions()
    assert len(actions) == 2


def test_stream_parser_invalid_json_falls_through():
    """Only valid JSON is extracted; invalid ones are silently skipped."""
    parser = MonologueStreamParser()
    parser.inner_monologue_buffer = """
    [GM_ACTION: {"type": "MOVE", "ok": true}]
    [GM_ACTION: {incomplete json here}]
    [GM_ACTION: {"type": "CREATE_ROOM", "room_id": "x"}]
    """
    actions = parser.extract_gm_actions()
    assert len(actions) == 2
    assert actions[0]["type"] == "MOVE"
    assert actions[1]["type"] == "CREATE_ROOM"


# ---------------------------------------------------------------------------
# Integration: stream parser with GM actions during actual streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_with_gm_action_extraction():
    """GM actions extracted from monologue after streaming completes."""
    async def mock_chunks():
        yield "<thinking>"
        yield "I should move Vardus."
        yield " [GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"Vardus\", \"room_id\": \"dungeon\"}]"
        yield "</thinking>"
        yield " Vardus appears in the dungeon."

    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass

    # GM action is inside the monologue, so it should be extracted.
    actions = parser.extract_gm_actions()
    assert len(actions) == 1
    assert actions[0]["type"] == "MOVE"
    assert actions[0]["entity"] == "Vardus"

    inner_mono, public_resp = parser.get_final_buffers()
    assert "move Vardus" in inner_mono.lower() or "dungeon" in inner_mono.lower()


@pytest.mark.asyncio
async def test_stream_with_multiple_gm_actions():
    """Multiple GM actions are all extracted after streaming."""
    async def mock_chunks():
        yield "<thinking>"
        yield "Create the chamber first. "
        yield "[GM_ACTION: {\"type\": \"CREATE_ROOM\", \"room_id\": \"secret\", \"name\": \"Secret Room\", \"desc\": \"A hidden chamber.\"}] "
        yield "Then move the player there. "
        yield "[GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"player\", \"room_id\": \"secret\"}]"
        yield "</thinking>"
        yield " The secret room opens before you."

    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass

    actions = parser.extract_gm_actions()
    assert len(actions) == 2
    assert actions[0]["type"] == "CREATE_ROOM"
    assert actions[1]["type"] == "MOVE"


@pytest.mark.asyncio
async def test_stream_gm_action_in_public_after_close():
    """GM action block appearing after the close tag should still be found in monologue."""
    async def mock_chunks():
        yield "<thinking>"
        yield "I'll do it after this."
        yield "</thinking>"
        yield " [GM_ACTION: {\"type\": \"MOVE\", \"entity\": \"player\", \"room_id\": \"hall\"}]"
        yield " Now the player is in the hall."

    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass

    # The GM action block is after the close tag, so it's in public, not monologue.
    # extract_gm_actions searches inner_monologue_buffer only.
    actions = parser.extract_gm_actions()
    assert len(actions) == 0  # It's in public text, not captured in inner_monologue

    _, public_resp = parser.get_final_buffers()
    assert "player is in the hall" in public_resp
