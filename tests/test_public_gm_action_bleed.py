"""
Regression: GM_ACTION tags and planning lines emitted AFTER the monologue closes leaked to SillyTavern.

Captured live on 2026-10-03 with Qwen3.8-27B-GGUF: the model closed its reasoning, then wrote a stray
backtick, "Then the narrative." / "Done.", and four [GM_ACTION: ...] lines before the narration.
"""
import pytest

from proxy.core.stream_parser import MonologueStreamParser

CREATE = '[GM_ACTION: {"type": "CREATE_ROOM", "room_id": "outside_lighthouse", "name": "Outside the Lighthouse", "desc": "A stone path."}]'
MOVE = '[GM_ACTION: {"type": "MOVE", "entity": "User", "room_id": "outside_lighthouse"}]'
NARRATION = "The sound of the knock echoes through the hollow stone walls."


async def _run(chunks):
    async def gen():
        for c in chunks:
            yield c

    parser = MonologueStreamParser()
    streamed = "".join([out async for out in parser.process_token_stream(gen())])
    return parser, streamed


@pytest.mark.asyncio
async def test_gm_actions_after_close_tag_do_not_reach_client():
    chunks = [
        "<thinking>The user knocks. I should create the room.\n", CREATE, "\n", MOVE, "\n</thinking>",
        "`\nThen the narrative.\nDone.\n\n\n",
        CREATE[:20], CREATE[20:] + "\n", MOVE + "\n",
        NARRATION + "\n",
    ]
    parser, streamed = await _run(chunks)

    assert "GM_ACTION" not in streamed
    assert "Then the narrative" not in streamed
    assert "Done." not in streamed
    assert "`" not in streamed
    assert NARRATION in streamed

    _, public = parser.get_final_buffers()
    assert "GM_ACTION" not in public
    assert public.strip() == NARRATION


@pytest.mark.asyncio
async def test_public_gm_actions_are_dispatched_once():
    # Actions appear both in the monologue and again in public text: each must be dispatched exactly once.
    chunks = ["<thinking>", CREATE, "\n", MOVE, "</thinking>\n", CREATE + "\n", MOVE + "\n", NARRATION]
    parser, _ = await _run(chunks)

    actions = parser.extract_gm_actions()
    assert [a["type"] for a in actions] == ["CREATE_ROOM", "MOVE"]


@pytest.mark.asyncio
async def test_public_only_gm_action_is_still_dispatched():
    parser, streamed = await _run([MOVE + "\n", NARRATION])

    assert "GM_ACTION" not in streamed
    assert [a["type"] for a in parser.extract_gm_actions()] == ["MOVE"]


@pytest.mark.asyncio
async def test_narration_mentioning_done_is_kept():
    text = "Done with the dishes, she turns to face you.\n"
    _, streamed = await _run([text])
    assert streamed == text
