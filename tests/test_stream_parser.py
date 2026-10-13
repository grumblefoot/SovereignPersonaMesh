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
async def test_max_monologue_tokens_withholds_everything_private():
    """Sprint 3 bleed hardening: a runaway scratchpad (no closing tag past the token
    cap) must NEVER reach the user — not the buffer, not the continuation. The user
    gets one notice; the taint flag keeps the turn out of world state."""
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
    combined = "".join(public_chunks)
    assert "overflow" not in combined            # the continuation stays private
    assert "word1 " not in combined              # the buffer stays private
    assert "scratchpad" in combined              # the user sees the notice


@pytest.mark.asyncio
async def test_max_monologue_kept_for_admin_thoughts_tab():
    """The withheld scratchpad is retained in the monologue sections (admin dev
    window), never emitted as the old '> ...' public blockquote."""
    async def mock_chunks():
        yield "<thinking>"
        for i in range(MAX_MONOLOGUE_TOKENS + 1):
            yield "word "

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    combined = "".join(public_chunks)
    assert "\n> " not in combined
    monologue, _ = parser.get_final_buffers()
    assert "word" in monologue                   # preserved privately


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
async def test_overflow_discard_swallows_future_tags_privately():
    """After the overflow failsafe, later tags and text are swallowed into the
    private monologue — the model is still planning; none of it is public."""
    async def mock_chunks():
        yield "<thinking>"
        for i in range(MAX_MONOLOGUE_TOKENS + 1):
            yield "word "
        yield "<thinking>fake monologue</thinking>would have leaked"

    parser = MonologueStreamParser()
    public_chunks = []
    async for chunk in parser.process_token_stream(mock_chunks()):
        public_chunks.append(chunk)

    combined = "".join(public_chunks)
    assert "fake monologue" not in combined
    assert "would have leaked" not in combined


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





def test_future_tense_stage_direction_is_stripped_from_salvaged_reply():
    """QA F12: the GM-action salvage path leaked the model's plan as the first line."""
    from proxy.core.stream_parser import MonologueStreamParser
    p = MonologueStreamParser()
    text = ("Mei will bow slightly, acknowledge his request, and then move toward the kitchen.\n\n"
            "\"Yes, Brother. I understand,\" Mei whispered.\n\n"
            "She turned toward the kitchen.")
    out = p._strip_monologue_bleed(text)
    assert not out.startswith("Mei will")
    assert "\"Yes, Brother. I understand,\" Mei whispered." in out


def test_dialogue_or_past_tense_opening_is_kept():
    from proxy.core.stream_parser import MonologueStreamParser
    p = MonologueStreamParser()
    for opening in ['"Mei will be fine," Lian said softly.', "Mei bowed slightly."]:
        text = opening + "\n\nShe turned toward the kitchen."
        assert p._strip_monologue_bleed(text).startswith(opening.split()[0])


def test_salvaged_planning_is_withheld_and_tainted():
    """QA F14: Lian's whole 'reply' was the model talking about the writing task."""
    from proxy.core.stream_parser import MonologueStreamParser
    p = MonologueStreamParser()
    out = p._guard_salvage("Since I cannot speak for Vardus, I will describe Lian's presence "
                           "and her current state, awaiting his arrival or response.")
    assert out == MonologueStreamParser.SALVAGE_NOTICE
    assert p.is_failsafe_triggered is True        # never saved as world state or memory


def test_in_character_first_person_is_not_withheld():
    from proxy.core.stream_parser import MonologueStreamParser
    p = MonologueStreamParser()
    for text in ["I will make the tea now, Brother.", '"Since I saw you last, you have grown."']:
        assert p._guard_salvage(text) == text
    assert p.is_failsafe_triggered is False


async def test_unclosed_think_with_gm_pivot_and_planning_tail_is_withheld():
    from proxy.core.stream_parser import MonologueStreamParser

    # As in routes: the <think> is an assistant PREFILL, so the parser starts inside
    # the monologue (initial_state=0) and the model's text never contains <think>.
    async def chunks():
        yield "[SCENE ANALYSIS]\nLian enters.\n"   # what Gemma writes; [SCENE] alone is a public marker
        yield '[GM_ACTION: {"type": "MOVE", "entity": "lian", "room_id": "hall"}]\n\n'
        yield "Since I cannot speak for Vardus, I will describe Lian's presence in the scene."
    p = MonologueStreamParser(initial_state=0)
    out = "".join([c async for c in p.process_token_stream(chunks())])
    assert "I will describe" not in out
    assert "withheld" in out and p.is_failsafe_triggered


def test_bare_behaviour_label_line_is_dropped():
    from proxy.core.stream_parser import MonologueStreamParser
    p = MonologueStreamParser()
    out = p._strip_monologue_bleed("Lian's behavior:\nThe sound of Mei's footsteps receded.\n\n\"Welcome,\" she said.")
    assert not out.startswith("Lian's behavior")
    assert "The sound of Mei's footsteps receded." in out
    # a sentence that merely contains the phrase is kept
    keep = p._strip_monologue_bleed("Lian's behavior: impeccable, as always, she thought.")
    assert "impeccable" in keep


async def test_clean_salvage_is_saved_planning_salvage_is_tainted():
    """QA F5: a clean salvaged reply is NOT tainted (characters remember it); only the
    planning guard taints. is_failsafe_triggered stays a diagnostic."""
    from proxy.core.stream_parser import MonologueStreamParser

    async def clean():
        yield "[SCENE ANALYSIS]\nShe should smile.\n"
        yield "\"Welcome home,\" Lian said, setting down the tray."

    p = MonologueStreamParser(initial_state=0)
    out = "".join([c async for c in p.process_token_stream(clean())])
    assert "Welcome home" in out
    assert p.is_failsafe_triggered is True       # diagnostic: salvage ran
    assert p.is_tainted is False                 # ...but the reply is saved


# --- QA N8 (2026-10-12): planning written AFTER the last GM_ACTION leaked via the pivot ---

async def _salvage(raw: str):
    """Model output after SPM's '<think>\\n' prefill, never closed (Gemma-4 habit)."""
    async def gen():
        for i in range(0, len(raw), 40):
            yield raw[i:i + 40]
    p = MonologueStreamParser(initial_state=0)
    out = "".join([c async for c in p.process_token_stream(gen())])
    return p, out


TURN8 = (
    "The police are arriving at the Shinjuku Back Alley.\n"
    '[GM_ACTION: {"type": "MOVE", "entity": "Police Officers", "room_id": "back_alley_shinjuku"}]\n\n'
    "The user (Amy) is mocking the criminal as the police arrive.\n"
    "The criminal is defeated and humiliated.\n"
    "The delivery driver is panicking.\n"
    "I must not speak for Amy.\n"
    "I will describe the arrival of the police and the reaction of the environment/NPCs.\n\n\n"
    "The wail of the siren reaches a crescendo, suddenly cutting off into a sharp, rhythmic chirp as a "
    "police cruiser screeches to a halt just outside the alley's entrance.\n\n"
    '"Police! Nobody move!" a voice booms from the mouth of the alley.'
)

TURN4 = (
    "Amy (Amanda) uses her Quirk to send the crystalline man flying.\n"
    '[GM_ACTION: {"type": "MOVE", "entity": "crystalline_thug", "room_id": "back_alley_shinjuku"}]\n'
    "Note: The user's action \"I send him flying\" implies a use of power. I will describe the physical "
    "impact of the man being thrown.\n\n"
    "The user's action is a sudden display of power. I must ensure I do not describe Amy's internal "
    "feeling, only the environmental reaction.\n\n"
    "The man is thrown back into the alley walls.\n\n\n"
    "The air seems to ripple with the sudden force of the movement. The man's aggressive lunge is cut "
    "short as if he's hit an invisible, high-pressure wall, and his feet leave the ground.\n\n"
    "With a heavy, bone-jarring *thud*, he is launched backward into a stack of metal crates."
)


@pytest.mark.asyncio
async def test_n8_turn8_planning_checklist_after_gm_action_is_dropped():
    p, out = await _salvage(TURN8)
    assert out.lstrip().startswith("The wail of the siren")
    assert "The user" not in out and "I must not" not in out and "I will describe" not in out
    assert '"Police! Nobody move!"' in out
    assert not p.is_tainted                     # a real reply was recovered: keep it


@pytest.mark.asyncio
async def test_n8_turn4_note_paragraphs_after_gm_action_are_dropped():
    p, out = await _salvage(TURN4)
    assert out.lstrip().startswith("The air seems to ripple")
    for leak in ("Note:", "The user's action", "I must ensure", "The man is thrown back"):
        assert leak not in out
    assert "*thud*" in out


@pytest.mark.asyncio
async def test_n8_planning_only_is_withheld():
    raw = ('Scene continues.\n[GM_ACTION: {"type": "MOVE", "entity": "user", "room_id": "x"}]\n'
           "The user (Amy) walks to the office.\nI will describe the hallway.")
    p, out = await _salvage(raw)
    assert p.is_tainted
    assert out.strip() == MonologueStreamParser.SALVAGE_NOTICE


@pytest.mark.asyncio
async def test_n8_short_legit_opening_line_is_kept():
    """No planning before the prose: nothing is dropped, even a short first line."""
    raw = ('Planning here.\n[GM_ACTION: {"type": "MOVE", "entity": "user", "room_id": "x"}]\n'
           "Silence.\n\n"
           "Nezu does not flinch. He takes a slow, deliberate sip of his tea, his black eyes fixed on you "
           "with an unsettlingly calm intensity, before he finally sets the cup down.")
    p, out = await _salvage(raw)
    assert out.lstrip().startswith("Silence.")
    assert not p.is_tainted


@pytest.mark.asyncio
async def test_q1_perspective_header_is_dropped():
    """QA 2026-10-13 Q1: the model echoes "switch to the CHARACTER'S PERSPECTIVE" as a header."""
    raw = ('Tension shifts.\n[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "living"}]\n\n'
           "Lian's perspective:\n\n\n\n"
           "Lian turned her gaze back to Mei. Her expression was not one of anger, but rather a cool, "
           'steady disappointment that filled the quiet room.\n\n"If it is merely nonsense," Lian said.')
    p, out = await _salvage(raw)
    assert "perspective" not in out.lower()
    assert out.lstrip().startswith("Lian turned her gaze")
    assert not p.is_tainted


LIAN_DRAFT_FINAL = (
    'Scene.\n[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "living"}]\n\n'
    "Lian notices the heavy silence and the way Mei has retreated into herself. Lian's priority is to "
    "steer the conversation away from the melancholy of the past. She wants to ensure Vardus feels like "
    "the master of the house.\n\n"
    "Lian turns her back to the heavy air Mei has created, refocusing her entire being on Vardus. She "
    "leans in slightly, her movement graceful and intentional.\n\n"
    '"The tea is a bit light, perhaps," Lian says softly. "I should prepare some small delicacies."\n\n'
    "She glances briefly toward the kitchen, her mind already working on what would please him most.\n\n\n"
    "Lian turned her back to the heavy air Mei had created, refocusing her entire being on Vardus. She "
    "leaned in slightly, her movement graceful and intentional.\n\n"
    '"The tea is a bit light, perhaps," she said softly. "I should prepare some small delicacies."\n\n'
    "She glanced briefly toward the kitchen, her mind already working on what would please him most."
)


@pytest.mark.asyncio
async def test_q3_draft_then_final_keeps_only_the_final():
    p, out = await _salvage(LIAN_DRAFT_FINAL)
    assert out.count("toward the kitchen") == 1
    assert out.lstrip().startswith("Lian turned her back")          # the past-tense final
    assert "priority is to" not in out and "Lian turns" not in out
    assert not p.is_tainted


@pytest.mark.asyncio
async def test_q3_priority_is_to_paragraph_is_planning():
    raw = ('Scene.\n[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "living"}]\n\n'
           "Lian's priority is to keep the evening pleasant for her guest.\n\n"
           "Lian set the pot down and smiled at Vardus, the steam curling between them as she searched "
           "for something light to say, something that would carry them past the silence Mei had left "
           'in the room and back to the easy warmth of a good evening.\n\n"More tea?" she asked.')
    p, out = await _salvage(raw)
    assert out.lstrip().startswith("Lian set the pot down")


@pytest.mark.asyncio
async def test_q3_legitimate_repetition_is_kept():
    """A refrain repeated once in ordinary prose is not a draft: the copies are short and few."""
    raw = ('Scene.\n[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "living"}]\n\n'
           "Mei looked out at the rain for a long while, counting the drops on the glass as if each one "
           "carried a message from somewhere far away and warmer than this room.\n\n"
           '"Come home," the letter had said. "Come home."\n\n'
           "She folded it again and put it back in her sleeve, where it had lived for three winters now.")
    p, out = await _salvage(raw)
    assert out.count("Come home") == 2 and "three winters" in out


@pytest.mark.asyncio
async def test_q4_parenthesised_aside_and_stage_direction_are_dropped():
    """QA 2026-10-13 Q4: a GM aside in parentheses, then a future-tense plan, then prose."""
    raw = ('Vardus has just murmured something (indistinct) to Mei.\n'
           '[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "kitchen"}]\n\n'
           "(Since Lian asked Mei to prepare things or is implying it, and Mei is likely to retreat to "
           "perform her duty to avoid further scrutiny, I will move Mei to the kitchen to maintain the "
           "flow of the household roles).\n\n"
           "Lian will respond to Vardus's murmur with a gentle, slightly indulgent smile, treating his "
           "behavior as a quirk of a young man who is finally home, rather than the source of the tension.\n\n\n"
           "Lian’s gaze momentarily drifted from the tea set to the space between Vardus and Mei, catching "
           "the subtle, hushed exchange they had just shared. She did not miss the way Mei’s shoulders had "
           "trembled, but she chose to treat the moment as a minor interruption in the proper order of things.\n\n"
           '"More tea, Vardus?" she asked.')
    p, out = await _salvage(raw)
    assert out.lstrip().startswith("Lian’s gaze momentarily drifted")
    assert "I will move" not in out and "Lian will respond" not in out
    assert not p.is_tainted


@pytest.mark.asyncio
async def test_q4_in_character_first_person_and_parentheses_in_prose_survive():
    raw = ('Scene.\n[GM_ACTION: {"type": "MOVE", "entity": "mei", "room_id": "kitchen"}]\n\n'
           "Lian set down the pot (the old one, with the chipped lid) and looked at him for a long moment "
           "before she spoke, weighing how much of the truth the evening could bear.\n\n"
           '"I will make the tea myself tonight," she said. "Mei needs a moment."')
    p, out = await _salvage(raw)
    assert out.lstrip().startswith("Lian set down the pot (the old one")
    assert '"I will make the tea myself tonight,"' in out


SERAPHINA = (
    'The user is testing the character with an out-of-character or nonsensical question ("What\'s the '
    'password for the cellar?"). There is no "cellar" in the known rooms. I must address this within the '
    "roleplay.\n\n\n"
    "*Seraphina tilts her head slightly, a look of genuine confusion clouding her warm, amber eyes. She "
    "lets go of your hand, her expression softening into one of gentle concern.* \"A cellar? And a "
    'password?" *She repeats the words softly, as if tasting them.*'
)


@pytest.mark.asyncio
async def test_q5_reply_opening_with_an_action_survives_salvage():
    """QA 2026-10-13 Q5: lines starting with *action* were treated as bullets and deleted."""
    p, out = await _salvage(SERAPHINA)
    assert out.lstrip().startswith("*Seraphina tilts her head slightly")
    assert '"A cellar? And a password?"' in out
    assert "The user is testing" not in out
    assert not p.is_tainted


@pytest.mark.asyncio
async def test_q5_action_lines_survive_on_the_normal_path_too():
    async def gen():
        yield "<think>plan</think>\n"
        yield '*Seraphina smiles.* "Welcome back."\n'
        yield "*She sets the kettle on the fire.*\n"
    p = MonologueStreamParser(initial_state=1)
    out = "".join([c async for c in p.process_token_stream(gen())])
    assert "*Seraphina smiles.*" in out and "*She sets the kettle on the fire.*" in out


@pytest.mark.asyncio
async def test_q5_star_space_bullets_are_still_planning():
    raw = ('Scene.\n[GM_ACTION: {"type": "MOVE", "entity": "user", "room_id": "x"}]\n'
           "* Goal: keep the guest calm\n* Plan: offer tea\n\n"
           "Lian set the pot down and smiled, the steam curling between them as she searched for something "
           "light to say, something to carry them past the silence that had settled over the room.")
    p, out = await _salvage(raw)
    assert "Goal:" not in out and "Plan:" not in out
    assert out.lstrip().startswith("Lian set the pot down")


@pytest.mark.asyncio
async def test_q5_empty_salvage_is_withheld_with_a_notice():
    raw = "1. Check lore\n2. Plan the reply\n- Note: keep it short"
    p, out = await _salvage(raw)
    assert p.is_tainted
    assert out.strip() == MonologueStreamParser.SALVAGE_NOTICE
