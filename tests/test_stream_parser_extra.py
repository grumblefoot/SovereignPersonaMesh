import pytest
import asyncio
from proxy.core.stream_parser import MonologueStreamParser

@pytest.mark.asyncio
async def test_max_public_tokens_truncation():
    parser = MonologueStreamParser(max_public_tokens=5, initial_state=1)
    
    async def token_gen():
        yield "This is a test "
        yield "sentence that exceeds five tokens."
        
    outputs = []
    async for token in parser.process_token_stream(token_gen()):
        outputs.append(token)
        
    assert parser.is_public_truncated
    # It stops yielding once the limit is hit.
    assert len(parser.public_token_count) >= 5 if isinstance(parser.public_token_count, list) else parser.public_token_count >= 5

@pytest.mark.asyncio
async def test_extract_gm_actions_json_error():
    parser = MonologueStreamParser(initial_state=0)
    parser.inner_monologue_buffer = "[GM_ACTION: {bad json}]"
    
    # Should catch the JSONDecodeError and return empty list
    actions = parser.extract_gm_actions()
    assert len(actions) == 0

@pytest.mark.asyncio
async def test_clean_public_line_exclusions():
    parser = MonologueStreamParser()
    
    # These should return None (dropped entirely)
    assert parser._clean_public_line("planning notes for the character") is None
    assert parser._clean_public_line("a mix of anger and joy") is None
    assert parser._clean_public_line("Strategy:") == ""
    assert parser._clean_public_line(".") is None
    
    # Valid lines
    assert parser._clean_public_line("Hello world") == "Hello world"

@pytest.mark.asyncio
async def test_strip_monologue_bleed_exclusions():
    parser = MonologueStreamParser()
    
    # Test specific exclusion branches
    assert parser._strip_monologue_bleed("Plan: Just say hello") == "Just say hello"
    
    # Test dropping meta-analysis in first paragraph
    text = "i need to respond well\n\nActual response here."
    assert parser._strip_monologue_bleed(text) == "Actual response here."
    
    text2 = "response.\n**GM Warning:** bad\n- She should do this\n- Arvenia's Reaction: sad"
    res2 = parser._strip_monologue_bleed(text2)
    assert "**GM Warning:** bad" in res2
    assert "She should do this" not in res2
    assert "Arvenia's Reaction: sad" not in res2

@pytest.mark.asyncio
async def test_partial_tag_handling():
    parser = MonologueStreamParser()
    
    assert parser._is_partial_tag_prefix("<th") is True
    assert parser._is_partial_tag_prefix("</chan") is True
    assert parser._is_partial_tag_prefix("Not a tag") is False
    
    clean, carry = parser._process_public_text("Hello <th")
    assert clean == "Hello "
    assert carry == "<th"

@pytest.mark.asyncio
async def test_gm_warning_extraction():
    parser = MonologueStreamParser(initial_state=0)
    
    async def token_gen():
        yield "Some thoughts.\n[GM WARNING: you broke the rule]"
        
    outputs = []
    async for token in parser.process_token_stream(token_gen()):
        outputs.append(token)
        
    assert len(outputs) >= 1
    assert any("**GM Warning:** you broke the rule" in out for out in outputs)

@pytest.mark.asyncio
async def test_get_final_buffers_cleans():
    parser = MonologueStreamParser(initial_state=0)
    parser.inner_monologue_buffer = "test buffer "
    
    mono, pub = parser.get_final_buffers()
    assert "test buffer" in mono
