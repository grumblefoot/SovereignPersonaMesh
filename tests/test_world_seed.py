"""Behavioral tests for proxy.gating.world_seed (Sprint 2, chunk 2).

GM-off seeding is deterministic and Zero-LLM (docs/plans/gating.md §6):
explicit [scene:KEY] tag > keyword match > generic_void fallback, with all
named characters placed in the template's first (sorted) room.
"""
import pytest

from proxy.gating.world_seed import (
    CONF_FALLBACK,
    CONF_KEYWORDS,
    CONF_TAG,
    FALLBACK_TEMPLATE,
    SeedPlacement,
    SeedProposal,
    extract_keywords,
    propose_world_seed,
)


# ── rule 1: explicit tag ────────────────────────────────────────────────

def test_tag_wins_in_system_text():
    p = propose_world_seed("[scene:forest_camp] You wake in a dungeon.",
                           "I draw my sword and seek the tavern ale.")
    assert p.template_key == "forest_camp"
    assert p.source == "tag"
    assert p.confidence == CONF_TAG == 1.0


def test_tag_wins_in_user_text():
    p = propose_world_seed("You are in a dark dungeon cellar.",
                           "[scene:castle_exterior] hails the knight")
    assert p.template_key == "castle_exterior"
    assert p.source == "tag"


def test_tag_wins_over_strong_keywords():
    # Text screams tavern_common; the tag says otherwise and must win.
    p = propose_world_seed("",
                           "[scene:forest_camp] The tavern serves cold ale "
                           "in the noisy inn bar.")
    assert p.template_key == "forest_camp"
    assert p.source == "tag"
    assert p.confidence == 1.0


def test_tag_case_insensitive_and_whitespace_tolerant():
    p = propose_world_seed("[ Scene : tavern_common ]", "")
    assert p.template_key == "tavern_common"
    assert p.source == "tag"


def test_unknown_tag_falls_through_to_keywords():
    # Unknown key is no tag at all; tavern keywords then carry the match.
    p = propose_world_seed("[scene:not_a_real_template]",
                           "The tavern ale flows at the inn bar.")
    assert p.template_key == "tavern_common"
    assert p.source == "keywords"
    assert p.confidence == CONF_KEYWORDS


def test_unknown_tag_everywhere_falls_to_fallback():
    p = propose_world_seed("[scene:nope]", "[scene:also_nope] qwerty zxcvbn")
    assert p.template_key == FALLBACK_TEMPLATE
    assert p.source == "fallback"
    assert p.confidence == CONF_FALLBACK


def test_known_tag_after_unknown_tag_in_same_text():
    p = propose_world_seed("", "[scene:bogus] [scene:forest_camp] hello")
    assert p.template_key == "forest_camp"
    assert p.source == "tag"


# ── rule 2: keyword match ───────────────────────────────────────────────

def test_tavern_text_matches_tavern_template():
    p = propose_world_seed("You enter a lively tavern.",
                           "The barkeep pours ale at the crowded inn.")
    assert p.template_key == "tavern_common"
    assert p.source == "keywords"
    assert p.confidence == CONF_KEYWORDS == 0.65


def test_forest_keywords_match_forest_camp():
    p = propose_world_seed("", "We make camp deep in the forest woods.")
    assert p.template_key == "forest_camp"
    assert p.source == "keywords"


def test_castle_keywords_match_castle_exterior():
    p = propose_world_seed("", "The castle fortress courtyard gleams; "
                               "knights guard the armory.")
    assert p.template_key == "castle_exterior"
    assert p.source == "keywords"


# ── rule 3: fallback ────────────────────────────────────────────────────

def test_gibberish_falls_back_to_generic_void():
    p = propose_world_seed("asdf qwer zxcv", "jjkl mnpq blth")
    assert p.template_key == "generic_void"
    assert p.source == "fallback"
    assert p.confidence == CONF_FALLBACK == 0.3


def test_empty_inputs_safe():
    p = propose_world_seed("", "")
    assert p.template_key == "generic_void"
    assert p.source == "fallback"
    assert p.placements == []


def test_none_inputs_safe():
    p = propose_world_seed(None, None)  # type: ignore[arg-type]
    assert p.template_key == "generic_void"
    assert p.source == "fallback"


# ── placements ───────────────────────────────────────────────────────────

def test_placements_all_in_first_room():
    p = propose_world_seed("[scene:castle_exterior]", "",
                           characters=["aria", "bram", "seraphine"])
    rooms = {"armory", "courtyard", "gate_house", "great_hall", "stables",
             "throne_room"}
    first = sorted(rooms)[0]
    assert [pl.room_id for pl in p.placements] == [first, first, first]
    assert [pl.character_id for pl in p.placements] == ["aria", "bram",
                                                        "seraphine"]


def test_placements_deterministic_and_deduped():
    chars = ["zed", "ann", "zed", "  ", "bo"]
    p1 = propose_world_seed("[scene:forest_camp]", "", characters=chars)
    p2 = propose_world_seed("[scene:forest_camp]", "", characters=chars)
    assert p1.placements == p2.placements
    ids = [pl.character_id for pl in p1.placements]
    assert ids == ["zed", "ann", "bo"]  # input order kept, dupes/blanks out
    first_room = sorted({"forest_clearing", "forest_path_north",
                         "forest_path_south", "thickets"})[0]
    assert all(pl.room_id == first_room for pl in p1.placements)


def test_no_characters_means_no_placements():
    p = propose_world_seed("[scene:tavern_common]", "")
    assert p.placements == []


def test_placements_on_fallback_template():
    p = propose_world_seed("", "qwerty asdf", characters=["solo"])
    assert p.template_key == "generic_void"
    first = sorted({"central_nexus", "node_alpha", "node_beta"})[0]
    assert p.placements == [SeedPlacement(character_id="solo",
                                          room_id=first)]


# ── determinism ──────────────────────────────────────────────────────────

def test_same_input_twice_same_output():
    sys_t = "[OOC: brb] You wake in the tavern cellar below the ale hall."
    usr_t = "*stretches* \"What is this place?\" the inn bar glimmers"
    a = propose_world_seed(sys_t, usr_t, characters=["ka", "el"])
    b = propose_world_seed(sys_t, usr_t, characters=["ka", "el"])
    assert a == b
    assert a.model_dump() == b.model_dump()


# ── keyword extraction through markup ───────────────────────────────────

def test_markup_heavy_input_still_extracts_keywords():
    text = ('[OOC: reroll] ((afk a sec)) **OOC brb** [whisper:Garrick] '
            '"The tavern ale is cold" *walks into the inn* '
            "I don't think so — [scene:whatever]")
    kws = extract_keywords(text)
    assert "tavern" in kws and "ale" in kws and "inn" in kws
    # OOC spans and tag innards are scrubbed, not mined for keywords.
    assert "reroll" not in kws and "afk" not in kws and "whatever" not in kws
    # Stopwords and short noise dropped; prose survives lowercased.
    assert "the" not in kws and "and" not in kws
    assert "think" in kws


def test_keyword_extraction_dedupes_preserving_order():
    kws = extract_keywords("tavern ale tavern", "ale castle courtyard")
    assert kws == ["tavern", "ale", "castle", "courtyard"]


def test_markup_only_input_falls_back():
    p = propose_world_seed("[OOC: hi]", "((waves)) **smiles**")
    assert p.template_key == "generic_void"
    assert p.source == "fallback"


# ── schema ───────────────────────────────────────────────────────────────

def test_proposal_is_pydantic_with_expected_shape():
    p = propose_world_seed("[scene:tavern_common]", "", characters=["x"])
    assert isinstance(p, SeedProposal)
    dumped = p.model_dump()
    assert set(dumped) == {"template_key", "confidence", "source",
                           "placements"}
    assert isinstance(dumped["placements"], list)
    assert set(dumped["placements"][0]) == {"character_id", "room_id"}


def test_confidence_out_of_range_rejected():
    with pytest.raises(Exception):
        SeedProposal(template_key="tavern_common", confidence=1.5,
                     source="tag")


# ── QA F25: incidental keywords in long lore must not pick a template ─────

def test_single_stray_keyword_in_long_lore_falls_back():
    """The live case: one 'nature' among hundreds of lorebook words seeded forest_camp."""
    lore = " ".join(f"quirk{i} hero{i} society{i}" for i in range(200)) + " the nature of quirks"
    p = propose_world_seed(lore, "Name: Vardus. Quirk: hypnotic appearance.")
    assert p.template_key == FALLBACK_TEMPLATE
    assert p.source == "fallback"


def test_keywords_diluted_by_a_huge_prompt_fall_back():
    import itertools, string
    # 900 DISTINCT alphabetic words (digits are stripped by the tokenizer, so
    # "word17" would collapse into one keyword and not dilute anything).
    words = ("".join(t) for t in itertools.product(string.ascii_lowercase, repeat=4))
    lore = " ".join(itertools.islice(words, 900)) + " forest camp"
    p = propose_world_seed(lore, "hello")
    assert p.template_key == FALLBACK_TEMPLATE        # 2 hits but density far below 0.02
