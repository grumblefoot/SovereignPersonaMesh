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


def test_persona_written_as_a_sheet_is_stripped():
    # QA 2026-10-17: the persona began "Name: Gu Changi"; the card that only mentions
    # the player further down is kept.
    msgs = _msgs(
        ("system", "Write Alba's next reply in a fictional chat between Alba and Changi."),
        ("system", "Name: Gu Changi\n\nRace: Demonic Cultivator\nPowers: flight"),
        ("system", "Alba is the witch of Belwyck.\nShe once met Changi at the fair."),
        ("user", "*Changi steps into the circle.*"),
    )
    kept = [m.content for m in _strip_user_persona(msgs, "alba")]
    assert not any(c.startswith("Name: Gu Changi") for c in kept)
    assert any(c.startswith("Alba is the witch") for c in kept)
    assert len(kept) == 3
