import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from proxy.core.st_parser import parse_sillytavern_context
from proxy.rag.lore_extractor import LoreExtractionWorker, INITIAL_RULES_PROMPT
import json

# Mock messages matching SillyTavern payload structure
MOCK_PAYLOAD = [
    {"role": "system", "content": "Write Arvenia's next reply in a fictional chat between Arvenia and Vardus."},
    {"role": "system", "content": "[Vardus is a tall and fit human male in his late 20's. He has a muscular swimmers build...]"},
    {"role": "system", "content": '[character("Arvenia")\nFrom("Her Last Piece")\nPersonality("Possessive" + "Lonely")]'},
    {"role": "system", "content": "[Goal: Player's Eternal, Obsessive Love and Companionship, and the consumption of the Players Soul]\n[Method: Psychological manipulation]"},
    {"role": "system", "content": "[Example Chat]"},
    {"role": "system", "content": "1. Initial Encounter:\nArvenia steps closer..."},
    {"role": "system", "content": "[Example Chat]"},
    {"role": "system", "content": "2. Seduction:\nArvenia glides beside Vardus..."},
    {"role": "system", "content": "[Start a new Chat]"},
    {"role": "assistant", "content": "Vardus wakes up in a cold, damp dungeon cell..."},
    {"role": "user", "content": '*Vardus looks up at the strange but beautiful woman* "Who are you?"'}
]

def test_parse_sillytavern_context():
    parsed = parse_sillytavern_context(MOCK_PAYLOAD)
    
    # Assert noise is dropped
    assert "Write Arvenia's next reply" not in parsed
    assert "[Example Chat]" not in parsed
    assert "1. Initial Encounter" not in parsed
    assert "2. Seduction" not in parsed
    assert "[Start a new Chat]" not in parsed
    
    # Assert essential context remains
    assert "Vardus is a tall and fit" in parsed
    assert '[character("Arvenia")' in parsed
    assert "Goal: Player's Eternal" in parsed
    assert "Vardus wakes up in a cold" in parsed
    
    # Assert user messages are dropped
    assert '*Vardus looks up' not in parsed

@pytest.mark.asyncio
async def test_live_extraction_goal_and_physical():
    """
    Live LLM test: Calls the actual LLM (via LemonadeClient) using the parsed
    context, and asserts that the extracted JSON rules contain the character's 
    goal (consuming the soul) or physical traits.
    This test connects to the local Lemonade backend on localhost:13305.
    """
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    db_pool_mock.acquire.return_value.__aenter__.return_value = conn_mock
    
    extractor = LoreExtractionWorker(db_pool_mock)
    
    # We don't want to actually write to the DB or embed in this test
    extractor.embedding_engine.generate_embedding = AsyncMock(return_value=[0.1]*3584)
    
    # Parse the mock payload to get clean context
    clean_context = parse_sillytavern_context(MOCK_PAYLOAD)
    
    try:
        await extractor.extract_initial_rules("test_session", "arvenia", clean_context, model="google/gemma-4-26B-A4B-it")
    except Exception as e:
        pytest.fail(f"Live LLM call failed. Is the local model running? Error: {e}")
        
    # Check what was sent to conn_mock.execute during the INSERT
    insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO csa_lore_rules_arvenia" in c[0][0]]
    
    assert len(insert_calls) > 0, "No rules were extracted by the live LLM"
    
    # Check if the extracted rules contain keywords from the Goal or Personality
    all_rules_text = " ".join([c[0][1] for c in insert_calls]).lower()
    
    # We expect the LLM to extract the goal or personality because it was in the Character Card & Scenario block
    assert "soul" in all_rules_text or "love" in all_rules_text or "possessive" in all_rules_text, \
        f"LLM failed to extract key goal/personality traits. Extracted texts: {all_rules_text}"
