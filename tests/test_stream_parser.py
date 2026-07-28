"""
Unit tests for Monologue Token Stream Parser.
"""

import pytest
from proxy.core.stream_parser import MonologueStreamParser


@pytest.mark.asyncio
async def test_stream_parser_normal():
    async def mock_chunks():
        yield "<ctrl94>I should stay hidden."
        yield " The shadows protect me.</ctrl94>"
        yield " I step forward from the dark."

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    inner_mono, public_resp = parser.get_final_buffers()
    assert "I should stay hidden" in inner_mono
    assert public_resp == "I step forward from the dark."
    assert "".join(public_chunks) == " I step forward from the dark."
