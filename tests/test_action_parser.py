"""Behavioral tests for proxy.gating.action_parser (Sprint 2, chunk 1).

Markers are FAIL-OPEN HINTS (SPRINT_PLAN.md §6 decision 11): anything
malformed degrades the span to plain speak, never drops content.
"""
import pytest

from proxy.gating.action_parser import (
    CONF_CAPS_SHOUT,
    CONF_EMOTE,
    CONF_FAIL_OPEN,
    CONF_MOVE,
    CONF_PLAIN,
    CONF_QUOTED,
    CONF_TAG,
    ParsedAction,
    ParsedMessage,
    parse_message,
    parse_user_message,
)


def types(actions):
    return [a.action_type for a in actions]


def one(text):
    actions = parse_user_message(text)
    assert len(actions) == 1, actions
    return actions[0]


# --------------------------------------------------------------------------
# Pydantic model
# --------------------------------------------------------------------------

def test_model_shape_and_validation():
    a = ParsedAction(action_type="speak", content="hi", confidence=0.5)
    assert a.target is None
    with pytest.raises(Exception):
        ParsedAction(action_type="telepathy", content="x", confidence=0.5)  # pyright: ignore
    with pytest.raises(Exception):
        ParsedAction(action_type="speak", content="x", confidence=1.5)


# --------------------------------------------------------------------------
# Empty / trivial input
# --------------------------------------------------------------------------

def test_empty_and_whitespace_produce_nothing():
    assert parse_user_message("") == []
    assert parse_user_message("   \n ") == []
    assert parse_message("").ooc is False


# --------------------------------------------------------------------------
# Quoted speech
# --------------------------------------------------------------------------

def test_double_quotes_are_speak():
    a = one('"Hello there."')
    assert (a.action_type, a.content, a.confidence) == ("speak", "Hello there.", CONF_QUOTED)


def test_curly_quotes_are_speak():
    a = one("\u201cHello there.\u201d")
    assert (a.action_type, a.content) == ("speak", "Hello there.")


def test_apostrophes_are_not_thoughts():
    # The headline fail-open requirement: ordinary chat prose stays speak.
    for line in [
        "I don't think that's wise, is it?",
        "It's over. Vardus won't survive the night.",
        "The dogs' bones were buried under the well.",
    ]:
        acts = parse_user_message(line)
        assert types(acts) == ["speak"], line
        assert acts[0].confidence == CONF_PLAIN
        assert acts[0].content == line


def test_apostrophe_inside_quote_does_not_split():
    a = one("\"don't go\"")
    assert (a.action_type, a.content) == ("speak", "don't go")


# --------------------------------------------------------------------------
# Thoughts (decision 11: single quotes mean thoughts)
# --------------------------------------------------------------------------

def test_single_quotes_are_thoughts():
    a = one("'I must not show fear.'")
    assert (a.action_type, a.content, a.confidence) == ("thought", "I must not show fear.", CONF_QUOTED)


def test_thoughts_never_carry_a_target():
    a = one("'Domino, he suspects everything.'")
    assert a.action_type == "thought"
    assert a.target is None


def test_thought_mid_sentence():
    acts = parse_user_message("she said, 'he is mine' quietly")
    assert types(acts) == ["speak", "thought", "speak"]
    assert acts[1].content == "he is mine"


def test_curly_single_quotes_are_thoughts():
    a = one("\u2018He knows.\u2019")
    assert a.action_type == "thought"
    assert a.content == "He knows."


# --------------------------------------------------------------------------
# Emotes and moves
# --------------------------------------------------------------------------

def test_asterisks_are_emotes():
    a = one("*waves at the crowd*")
    assert (a.action_type, a.content, a.confidence) == ("emote", "waves at the crowd", CONF_EMOTE)


def test_emote_with_movement_verb_and_prep_is_move():
    a = one("*walks to the bar*")
    assert a.action_type == "move"
    assert a.target == "Bar"
    assert a.confidence == CONF_MOVE


@pytest.mark.parametrize("text,dest", [
    ("*goes to the kitchen*", "Kitchen"),
    ("*heads into the cellar*", "Cellar"),
    ("*steps out of the cell*", "Cell"),
    ("*runs toward the door*", "Door"),
    ("*returns to the tavern*", "Tavern"),
    ("*leaves for the hall*", "Hall"),
    ("*went back to the room*", "Room"),
])
def test_movement_lexicon_yields_move(text, dest):
    a = one(text)
    assert a.action_type == "move", text
    assert a.target == dest


def test_bare_stair_direction_is_move():
    assert one("*goes upstairs*").target == "Upstairs"
    assert one("*slinks downstairs*").target == "Downstairs"


def test_non_motion_emotes_stay_emotes():
    for text in ["*nods slowly*", "*smiles at Domino*", "*touches the locket*",
                 "*looks up at the strange but beautiful woman*"]:
        assert one(text).action_type == "emote", text


def test_emote_without_destination_noun_stays_emote():
    # 'walks' with no preposition phrase is not a move.
    assert one("*walks slowly*").action_type == "emote"


# --------------------------------------------------------------------------
# OOC: stripped to a flag, never a world event
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "**OOC: brb**",
    "((can you redo that roll))",
    "[OOC: scene change next turn]",
])
def test_ooc_forms_are_flags_not_events(text):
    res = parse_message(text)
    assert res.ooc is True
    assert res.actions == []


def test_ooc_stripped_around_real_content():
    res = parse_message("**OOC: one sec** \"Where is the key?\"")
    assert res.ooc is True
    assert types(res.actions) == ["speak"]
    assert res.actions[0].content == "Where is the key?"


def test_unclosed_double_star_fails_open_as_speak():
    acts = parse_user_message("text **unclosed ooc")
    assert types(acts) == ["speak"]
    assert acts[0].confidence == CONF_FAIL_OPEN
    assert "unclosed ooc" in acts[0].content


# --------------------------------------------------------------------------
# Explicit tags win over heuristics
# --------------------------------------------------------------------------

def test_whisper_tag_applies_to_following_prose():
    a = one("[whisper:Domino] Meet me at midnight.")
    assert (a.action_type, a.target, a.content, a.confidence) == (
        "whisper", "Domino", "Meet me at midnight.", CONF_TAG)


def test_whisper_tag_overrides_quote_heuristic():
    a = one('[whisper:Domino] "Meet me at midnight."')
    assert a.action_type == "whisper"
    assert a.confidence == CONF_TAG


def test_shout_tag():
    a = one("[shout] GET OUT!")
    assert (a.action_type, a.content, a.confidence) == ("shout", "GET OUT!", CONF_TAG)


def test_move_tag_and_actor_arrow_form():
    a = one("[move:kitchen]")
    assert (a.action_type, a.target, a.confidence) == ("move", "Kitchen", CONF_TAG)
    b = one("[move:Arvenia->hall]")
    assert (b.action_type, b.target) == ("move", "Hall")


def test_tag_is_case_insensitive_and_trims():
    a = one("[ Whisper : Domino ] come here")
    assert (a.action_type, a.target, a.content) == ("whisper", "Domino", "come here")


def test_tag_does_not_capture_emote_or_thought():
    # Emotes and thoughts keep their own type; the tag governs speech only.
    acts = parse_user_message('[whisper:Domino] *nods* "Come closer."')
    assert types(acts) == ["emote", "whisper"]
    assert acts[1].content == "Come closer."
    assert acts[1].target == "Domino"


def test_second_quote_consumes_pending_tag():
    acts = parse_user_message('[shout] "Run!" "Faster!"')
    assert types(acts) == ["shout", "speak"]


def test_trailing_tag_still_fires():
    a = one("[whisper:Seamus]")
    assert (a.action_type, a.target, a.content) == ("whisper", "Seamus", "")


def test_move_tag_then_speech():
    acts = parse_user_message("[move:hall] \"I'm leaving.\"")
    assert types(acts) == ["move", "speak"]
    assert acts[0].target == "Hall"


# --------------------------------------------------------------------------
# Fail-open: malformed markers degrade to speak, never drop content
# --------------------------------------------------------------------------

def test_unclosed_quote_fails_open():
    acts = parse_user_message('He said "hello and waved')
    assert types(acts) == ["speak"]
    assert acts[-1].confidence == CONF_FAIL_OPEN
    assert "hello and waved" in acts[-1].content


def test_unclosed_emote_fails_open():
    acts = parse_user_message("*waves at the crowd")
    assert types(acts) == ["speak"]
    assert acts[0].confidence == CONF_FAIL_OPEN
    assert acts[0].content == "*waves at the crowd"


def test_unclosed_thought_quote_fails_open():
    acts = parse_user_message("'what if this never closes")
    assert types(acts) == ["speak"]
    assert acts[0].confidence == CONF_FAIL_OPEN


def test_swapped_delimiters_fail_open():
    # Opened with a thought quote, closed with a speech quote: no thought.
    acts = parse_user_message("'hello\"")
    assert "thought" not in types(acts)
    assert all(a.confidence == CONF_FAIL_OPEN for a in acts)
    joined = "".join(a.content for a in acts)
    assert "hello" in joined


def test_unknown_bracket_tag_fails_open_as_speak():
    acts = parse_user_message("[dance:fire] hello")
    assert types(acts) == ["speak", "speak"]
    assert acts[0].content == "[dance:fire]"
    assert acts[0].confidence == CONF_FAIL_OPEN


def test_stray_bracket_survives_as_speak():
    acts = parse_user_message("a ] bracket [ in prose")
    assert "".join(a.content for a in acts) == "a ] bracket [ in prose"
    assert types(acts) == ["speak"]


def test_malformed_nothing_is_dropped():
    # Worst-case soup: every character of the input survives in some action.
    for text in [
        'He said "hello and waved *toward the',
        "'quoted but never closed, then **ooc",
        "[move: ] [whisper:] mismatch ][ tags",
        "*a* b \"c d 'e f **g",
    ]:
        acts = parse_user_message(text)
        assert acts, text
        rebuilt = "".join(a.content for a in acts)
        assert all(ch in rebuilt for ch in text if not ch.isspace()), text


# --------------------------------------------------------------------------
# Mixed styles and ordering
# --------------------------------------------------------------------------

def test_mixed_styles_preserve_order():
    acts = parse_user_message(
        '*Vardus looks up at the strange but beautiful woman* "Who are you?"')
    assert types(acts) == ["emote", "speak"]
    assert acts[1].content == "Who are you?"


def test_full_mixed_message():
    acts = parse_user_message(
        'mixed: "speech" *waves* \'thinks\' and prose')
    # The "mixed:" lead-in is its own prose speech act.
    assert types(acts) == ["speak", "speak", "emote", "thought", "speak"]
    assert acts[1].content == "speech"
    assert acts[2].content == "waves"
    assert acts[3].content == "thinks"


def test_move_then_speak_from_natural_text():
    acts = parse_user_message('*heads down the stairs* "Help!"')
    assert types(acts) == ["move", "speak"]
    assert acts[0].target == "Downstairs"


def test_mixed_styles_across_turns_are_independent():
    # Fixture requirement: styles bleeding between turns must not persist.
    t1 = parse_user_message('"closed quote" and prose')
    t2 = parse_user_message("unclosed prose with a quote \" dangling")
    t3 = parse_user_message("*emote* then 'thought'")
    assert types(t1) == ["speak", "speak"]
    assert types(t2) == ["speak"] and t2[0].confidence == CONF_FAIL_OPEN
    assert types(t3) == ["emote", "speak", "thought"]  # ' then ' is prose

# --------------------------------------------------------------------------
# Shout heuristics and targets
# --------------------------------------------------------------------------

def test_all_caps_with_bang_is_shout():
    a = one("HEY LOOK OUT!")
    assert a.action_type == "shout"
    assert a.confidence == CONF_CAPS_SHOUT


def test_all_caps_without_bang_is_speak():
    assert one("WHO ARE YOU").action_type == "speak"


def test_vocative_target_extraction():
    a = one('"Vardus, I need you."')
    assert (a.action_type, a.target) == ("speak", "Vardus")


def test_to_name_target_extraction():
    a = one('"I gave it to Arvenia."')
    assert a.target == "Arvenia"


def test_to_lowercase_word_is_not_a_target():
    assert one('"I went to market."').target is None


def test_in_ear_target_extraction():
    acts = parse_user_message('"He lied," she said in Domino\'s ear')
    assert types(acts) == ["speak", "speak"]
    assert acts[0].content == "He lied,"
    assert acts[1].target == "Domino"


def test_speak_without_target():
    assert one('"The weather is fine."').target is None


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_deterministic_repeat_parse():
    text = '[whisper:Domino] *leans in* "the key is under the stone" \'he must not see\''
    first = [(a.action_type, a.content, a.target, a.confidence)
             for a in parse_user_message(text)]
    for _ in range(5):
        again = [(a.action_type, a.content, a.target, a.confidence)
                 for a in parse_user_message(text)]
        assert again == first


def test_parsed_message_defaults():
    res = ParsedMessage()
    assert res.actions == [] and res.ooc is False


# ── Decision 11: malformed markers are hints that degrade to plain speak ────
# (missing closer, unknown tag, swapped/mixed delimiters — never a crash, never
# a privileged action, the words stay audible as ordinary speech)

@pytest.mark.parametrize("raw", [
    '[whisper:Mira "no closing bracket',
    '[shout "no closing bracket either',
    "'an unclosed thought quote",
    '[whisp:Mira] "typoed tag name"',
    '[move:] "empty move target"',
    '\'a\' then "b" then [shout unclosed at the end',
])
def test_malformed_markers_degrade_to_speak(raw):
    actions = parse_user_message(raw)            # must never raise
    assert actions, "fail-open means SOMETHING is emitted"
    for a in actions:
        # A malformed marker may never grant a privileged channel by accident:
        # whispers/shouts/moves require their tag to be well-formed and closed.
        if a.action_type in ("whisper", "shout", "move"):
            raise AssertionError(f"malformed input produced {a.action_type}: {raw!r}")


def test_malformed_marker_words_stay_audible():
    acts = parse_user_message('[whisper:Mira "secret plan without closer')
    spoken = " ".join(a.content for a in acts if a.action_type == "speak")
    assert "secret plan without closer" in spoken


def test_well_formed_thought_still_private_next_to_malformed_tag():
    acts = parse_user_message("'keep this private' [whisp:X] \"say this\"")
    assert [a.action_type for a in acts if a.content == "keep this private"] == ["thought"]
    assert any(a.action_type == "speak" and a.content == "say this" for a in acts)
