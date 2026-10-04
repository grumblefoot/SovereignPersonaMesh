"""BUG-008: the user's SillyTavern persona must not reach lore extraction as if it were NPC lore."""
import json
from pathlib import Path

from proxy.api.routes import ChatCompletionMessage, _extract_target_char, _strip_user_persona

FIXTURE = Path(__file__).parent / "fixtures" / "test_chat_payload.json"


def _msgs(*pairs):
    return [ChatCompletionMessage(role=r, content=c) for r, c in pairs]


def test_real_sillytavern_payload_drops_persona_keeps_character():
    payload = json.loads(FIXTURE.read_text())
    msgs = [ChatCompletionMessage(**m) for m in payload["messages"]]
    char = _extract_target_char(msgs)
    assert char == "arvenia"

    kept = _strip_user_persona(msgs, char)
    text = "\n".join(m.content for m in kept)
    assert "Vardus is a tall and fit human male" not in text      # persona block removed
    assert '[character("Arvenia")' in text                         # character card kept
    assert len(kept) == len(msgs) - 1
    assert any(m.role == "user" for m in kept)                     # chat turns untouched


def test_possessive_persona_heading_is_dropped():
    msgs = _msgs(("system", "Write Mira's next reply in a fictional chat between Mira and Tom."),
                 ("system", "[Tom's persona: a travelling merchant]"),
                 ("system", "[Character: Mira] Tavern keeper."))
    assert [m.content for m in _strip_user_persona(msgs, "mira")] == [msgs[0].content, msgs[2].content]


def test_unknown_names_leave_messages_unchanged():
    msgs = _msgs(("system", "[Character: Mira]"), ("system", "[Tom is a merchant]"), ("user", "hi"))
    assert _strip_user_persona(msgs, "mira") == msgs


def test_user_chat_lines_starting_with_user_name_are_kept():
    msgs = _msgs(("system", "A chat between Mira and Tom."), ("user", "Tom walks in."))
    assert _strip_user_persona(msgs, "mira") == msgs
