"""
Comprehensive unit tests for Monologue Token Stream Parser.
Covers normal operation, fail-safe passthrough, unexpected EOS, and malformed tags.
"""

import pytest
from proxy.core.stream_parser import MonologueStreamParser, OPEN_TAG, CLOSE_TAG, MAX_MONOLOGUE_TOKENS


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_monologue_then_public():
    """Standard flow: monologue tags, transition to public, yield only public."""
    async def mock_chunks():
        yield "<thinking>I should stay hidden."
        yield " The shadows protect me.</thinking>"
        yield " I step forward from the dark."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "I should stay hidden" in inner_mono
    assert "shadows protect me" in inner_mono
    assert "I step forward" in public_resp
    assert "".join(public_chunks) == " I step forward from the dark."


@pytest.mark.asyncio
async def test_multiple_monologue_sections():
    """Two separate monologue-public cycles in one stream."""
    async def mock_chunks():
        yield "<thinking>First thought.</thinking>"
        yield " First public line."
        yield "<thinking>Second thought.</thinking>"
        yield " Second public line."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "First thought" in inner_mono
    assert "Second thought" in inner_mono
    public_text = "".join(public_chunks)
    assert "First public line" in public_text
    assert "Second public line" in public_text


@pytest.mark.asyncio
async def test_no_monologue_all_public():
    """Stream contains no tags at all — everything is public."""
    async def mock_chunks():
        yield "Hello there."
        yield " How are you?"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert inner_mono == ""
    assert public_resp == "Hello there. How are you?"
    assert "".join(public_chunks) == "Hello there. How are you?"


@pytest.mark.asyncio
async def test_empty_stream():
    """Empty token generator yields nothing."""
    async def mock_chunks():
        return
        yield  # unreachable, just makes it a generator

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert public_chunks == []
    inner_mono, public_resp = parser.get_final_buffers()
    assert inner_mono == ""
    assert public_resp == ""


@pytest.mark.asyncio
async def test_close_tag_immediately():
    """Monologue is empty; close tag arrives on first chunk."""
    async def mock_chunks():
        yield "<thinking></thinking>Immediate public."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert "".join(public_chunks) == "Immediate public."


@pytest.mark.asyncio
async def test_public_text_after_close_in_same_chunk():
    """Public text immediately follows closing tag in same chunk."""
    async def mock_chunks():
        yield "<thinking>thought text.</thinking>Public right away."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "thought text" in inner_mono
    assert public_resp == "Public right away."


# ---------------------------------------------------------------------------
# Fail-safe: >MAX_MONOLOGUE_TOKENS passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_monologue_tokens_triggers_passthrough():
    """
    When monologue token count exceeds MAX_MONOLOGUE_TOKENS without closing tag,
    the parser switches to passthrough mode and yields the buffered monologue
    prefixed with '> ...'.
    """
    async def mock_chunks():
        yield "<thinking>"
        for i in range(MAX_MONOLOGUE_TOKENS):
            yield f"word{i} "
        yield "overflow "

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert parser.is_failsafe_triggered is True
    # The overflow token and everything after should be passthrough
    combined = "".join(public_chunks)
    assert "overflow" in combined


@pytest.mark.asyncio
async def test_max_monologue_yields_prefixed_monologue():
    """The buffered monologue is emitted prefixed with '> ...' before passthrough."""
    async def mock_chunks():
        yield "<thinking>"
        for i in range(MAX_MONOLOGUE_TOKENS + 1):
            yield "word "

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    combined = "".join(public_chunks)
    # First yield from fail-safe is the prefixed monologue
    assert "\n> " in combined


# ---------------------------------------------------------------------------
# Unexpected EOS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unexpected_eos_in_monologue():
    """Stream ends while still in State 0 — auto-close and flush monologue as public."""
    async def mock_chunks():
        yield "<thinking>I never finished my thought"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert parser.is_failsafe_triggered is True
    combined = "".join(public_chunks)
    assert "I never finished my thought" in combined


@pytest.mark.asyncio
async def test_unexpected_eos_empty_buffer():
    """Stream ends but buffer is empty — no flush needed."""
    async def mock_chunks():
        return
        yield  # makes this an async generator

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert parser.is_failsafe_triggered is False
    assert public_chunks == []


@pytest.mark.asyncio
async def test_unexpected_eos_does_not_double_yield():
    """If stream ends after State 1 transition, no extra flush happens."""
    async def mock_chunks():
        yield "<thinking>done.</thinking>Public."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    # Only the public text should appear once
    assert "".join(public_chunks) == "Public."


# ---------------------------------------------------------------------------
# Malformed / unexpected tags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_ctrl_tag_triggers_passthrough():
    """A malformed <ctrl9 tag (no closing tag) triggers passthrough."""
    async def mock_chunks():
        yield "<thinking>partial "
        yield "<ctrl9 incoherent text"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert parser.is_failsafe_triggered is True
    combined = "".join(public_chunks)
    assert "<ctrl9" in combined


@pytest.mark.asyncio
async def test_normal_close_tag_is_not_malformed():
    """A proper </thinking> close tag does NOT trigger passthrough."""
    async def mock_chunks():
        yield "<thinking>thought</thinking>public text"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    assert parser.is_failsafe_triggered is False
    assert "".join(public_chunks) == "public text"


@pytest.mark.asyncio
async def test_passthrough_mode_ignores_future_tags():
    """Once in passthrough mode, subsequent tags are passed through as-is."""
    async def mock_chunks():
        yield "<thinking>"
        for i in range(MAX_MONOLOGUE_TOKENS + 1):
            yield "word "
        yield "<thinking>fake monologue</thinking>should appear"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    combined = "".join(public_chunks)
    assert "<thinking>fake monologue</thinking>" in combined


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_final_buffers_returns_tuple():
    """get_final_buffers returns a (str, str) tuple."""
    async def mock_chunks():
        yield "<thinking>mono</thinking>pub"

    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass

    result = parser.get_final_buffers()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.asyncio
async def test_markdown_thought_and_public_headers():
    """Test parsing markdown formatted headers *Thought:* and *Public:* without leaking thoughts."""
    async def mock_chunks():
        yield "*Thought:* \"A butler... how quaint.\"\n\n*Public:* Arvenia remains by the cell door."

    parser = MonologueStreamParser(initial_state=0)
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "butler" in inner_mono
    assert "Thought" not in public_resp
    assert "Arvenia remains by the cell door." in public_resp


@pytest.mark.asyncio
async def test_prose_analysis_and_plan_header_bleed_prevention():
    """Test that LLM internal prose analysis and 'Plan:' headers are completely stripped from public canon response."""
    async def mock_chunks():
        yield "Arvenia's reaction should be a blend of amusement and a subtle, predatory grace. Instead, she should lean into the irony.\n\nPlan:\n\nArvenia froze for a heartbeat, her hand hovering just inches from your skin."

    parser = MonologueStreamParser(initial_state=0)
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "reaction should be" not in public_resp
    assert "Plan:" not in public_resp
    assert "Arvenia froze for a heartbeat" in public_resp


@pytest.mark.asyncio
async def test_untagged_opening_meta_commentary_strip():
    """Test that un-tagged opening meta-commentary paragraphs (e.g. 'a mix of amusement... Arvenia is vain...') are stripped."""
    async def mock_chunks():
        text = "a mix of amusement, offense, and a display of power. Amanda has just insulted her appearance by echoing common peasant misconceptions. Arvenia is vain and perceives herself as a creature of supreme elegance.\n\nArvenia’s laughter ripples through the damp air, a sound like silver bells being struck in a graveyard."
        # simulate streaming in small 10-char chunks
        for i in range(0, len(text), 10):
            yield text[i:i+10]

    parser = MonologueStreamParser(initial_state=1)
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "peasant misconceptions" not in public_resp
    assert "supreme elegance" not in public_resp
    assert "Arvenia’s laughter ripples through the damp air" in public_resp
    
    streamed_public = "".join(public_chunks)
    assert "peasant misconceptions" not in streamed_public
    assert "supreme elegance" not in streamed_public

@pytest.mark.asyncio
async def test_eof_failsafe_with_explicit_separator():
    """Test that when </think> is missing, EOF failsafe pivots on explicit '--' separator."""
    async def mock_chunks():
        yield "<think>internal thoughts\n[INTERNAL MONOLOGUE - ARVENIA]\n--\npublic dialogue"
    
    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass
    
    inner_mono, public_resp = parser.get_final_buffers()
    assert "internal thoughts" in inner_mono
    assert "INTERNAL MONOLOGUE" not in public_resp
    assert "public dialogue" in public_resp

@pytest.mark.asyncio
async def test_strip_scene_description_header():
    """Test that explicit [SCENE DESCRIPTION] headers are stripped from public response."""
    async def mock_chunks():
        yield "<think>thoughts</think>\n[SCENE DESCRIPTION]\nThe room is dark."
    
    parser = MonologueStreamParser()
    async for chunk in parser.process_token_stream(mock_chunks()):
        pass
    
    inner_mono, public_resp = parser.get_final_buffers()
    assert "[SCENE DESCRIPTION]" not in public_resp
    assert "The room is dark." in public_resp



