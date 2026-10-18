import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from proxy.rag.lore_extractor import LoreExtractionWorker, _place_rule


def _fake_pool(conn_mock):
    """db_pool mock whose `async with pool.acquire() as conn` yields conn_mock."""
    db_pool_mock = MagicMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    return db_pool_mock


def _insert_calls(conn_mock):
    """chat_rules.add_rule INSERTs through conn.fetchval, not conn.execute."""
    return [c for c in conn_mock.fetchval.call_args_list
            if "INSERT INTO spm_chat_rules" in c[0][0]]


@pytest.mark.asyncio
async def test_extract_initial_rules():
    """Extracted rules land in spm_chat_rules (brief B1: no more per-character tables)."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Character is scared of fire.", "rule_type": "invariant"}]\n'
        yield "```"

    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session123", "Tester", "context")

                # The old per-character table must no longer be touched at all.
                for mock in (conn_mock.execute, conn_mock.fetchval):
                    assert not [c for c in mock.call_args_list
                                if "csa_lore_rules" in c[0][0]]
                inserts = _insert_calls(conn_mock)
                assert len(inserts) == 1
                # add_rule INSERT binds (sql, session, level, character_id, rule_text,
                # rule_type, status, source, embedding, space_id). No level from the
                # model -> default 'world', character NULL.
                args = inserts[0][0]
                assert args[1] == "session123"
                assert args[2] == "world"
                assert args[3] is None
                assert args[4] == "Character is scared of fire."
                assert args[6] == "pending"
                assert args[7] == "extracted"


@pytest.mark.asyncio
async def test_extract_initial_rules_skips_existing():
    """add_rule returns None for a duplicate -> no INSERT (idempotency)."""
    conn_mock = AsyncMock()
    # add_rule's exact-duplicate fetchval hits; the INSERT never runs.
    conn_mock.fetchval = AsyncMock(return_value="1")
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Character is scared of fire.", "rule_type": "invariant"}]\n'
        yield "```"

    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session123", "Tester", "context")

                assert len(_insert_calls(conn_mock)) == 0


@pytest.mark.asyncio
async def test_extract_initial_rules_multiple_rules():
    """Test that multiple rules are extracted and persisted."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Rule 1", "rule_type": "invariant"}, {"rule_text": "Rule 2", "rule_type": "trigger"}]\n'
        yield "```"

    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session123", "Tester", "context")

                inserts = _insert_calls(conn_mock)
                assert len(inserts) == 2
                # Trigger should be converted to conditional_trigger
                assert inserts[1][0][5] == "conditional_trigger"


@pytest.mark.asyncio
async def test_periodic_review_rules():
    """Test that periodic_review_rules builds messages string and extracts rules."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "New rule from periodic review", "rule_type": "invariant"}]\n'
        yield "```"

    recent_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What is this place?"}
    ]

    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                with patch('proxy.rag.lore_extractor.strings') as mock_strings:
                    mock_strings.get.return_value = "mock prompt"
                    await extractor.periodic_review_rules("session123", "Tester", recent_messages)

                    mock_strings.get.assert_called()
                    first_call = mock_strings.get.call_args_list[0]
                    assert "recent_messages" in first_call[1]
                    assert "Hello" in first_call[1]["recent_messages"]
                    assert "Hi there" in first_call[1]["recent_messages"]
                    assert "What is this place?" in first_call[1]["recent_messages"]
                    assert "characters" in first_call[1]
                    assert "rooms" in first_call[1]


@pytest.mark.asyncio
async def test_periodic_review_rules_empty_messages():
    """Test that periodic_review_rules handles empty message list."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Empty review rule", "rule_type": "invariant"}]\n'
        yield "```"

    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                with patch('proxy.rag.lore_extractor.strings') as mock_strings:
                    mock_strings.get.return_value = "mock prompt"
                    await extractor.periodic_review_rules("session123", "Tester", [])

                    mock_strings.get.assert_called()
                    first_call = mock_strings.get.call_args_list[0]
                    assert first_call[1]["recent_messages"] == ""


class TestCleanLLMJSONResponse:
    """Tests for _clean_llm_json_response method."""

    def setup_method(self):
        db_pool_mock = MagicMock()
        self.extractor = LoreExtractionWorker(db_pool_mock)

    def test_clean_json_with_fenced_block(self):
        raw = "```json\n[{\"rule_text\": \"test\"}]\n```"
        result = self.extractor._clean_llm_json_response(raw)
        assert len(result) == 1
        assert result[0]["rule_text"] == "test"

    def test_clean_json_with_plain_code_block(self):
        raw = "```\n[{\"rule_text\": \"test\"}]\n```"
        result = self.extractor._clean_llm_json_response(raw)
        assert len(result) == 1
        assert result[0]["rule_text"] == "test"

    def test_clean_json_without_wrapping(self):
        raw = '[{"rule_text": "test"}]'
        result = self.extractor._clean_llm_json_response(raw)
        assert len(result) == 1
        assert result[0]["rule_text"] == "test"

    def test_clean_json_single_dict(self):
        raw = '{"rule_text": "single rule"}'
        result = self.extractor._clean_llm_json_response(raw)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["rule_text"] == "single rule"

    def test_clean_json_invalid_returns_empty_list(self):
        raw = "not valid json {{{"
        result = self.extractor._clean_llm_json_response(raw)
        assert result == []

    def test_clean_json_empty_string(self):
        result = self.extractor._clean_llm_json_response("")
        assert result == []


class TestLoreTypeNormalization:
    """Tests for lore rule type normalization in _execute_extraction."""

    @pytest.mark.asyncio
    async def test_trigger_converted_to_conditional_trigger(self):
        """Test that 'trigger' type is normalized to 'conditional_trigger'."""
        conn_mock = AsyncMock()
        conn_mock.fetchval = AsyncMock(return_value=None)
        extractor = LoreExtractionWorker(_fake_pool(conn_mock))

        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_text": "trigger rule", "rule_type": "trigger"}]\n'
            yield "```"

        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
                with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                    await extractor.extract_initial_rules("session1", "char", "ctx")

                    inserts = _insert_calls(conn_mock)
                    assert len(inserts) == 1
                    # INSERT binds (..., rule_text, rule_type, ...)
                    assert inserts[0][0][5] == "conditional_trigger"

    @pytest.mark.asyncio
    async def test_invalid_rule_type_defaults_to_invariant(self):
        """Test that invalid rule_type defaults to 'invariant'."""
        conn_mock = AsyncMock()
        conn_mock.fetchval = AsyncMock(return_value=None)
        extractor = LoreExtractionWorker(_fake_pool(conn_mock))

        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_text": "bad type", "rule_type": "invalid_type"}]\n'
            yield "```"

        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
                with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                    await extractor.extract_initial_rules("session1", "char", "ctx")

                    inserts = _insert_calls(conn_mock)
                    assert len(inserts) == 1
                    assert inserts[0][0][5] == "invariant"

    @pytest.mark.asyncio
    async def test_skips_rules_without_rule_text(self):
        """Test that rules missing 'rule_text' are skipped."""
        conn_mock = AsyncMock()
        conn_mock.fetchval = AsyncMock(return_value=None)
        extractor = LoreExtractionWorker(_fake_pool(conn_mock))

        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_type": "invariant"}, {"rule_text": "valid rule", "rule_type": "invariant"}]\n'
            yield "```"

        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None):
                with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                    await extractor.extract_initial_rules("session1", "char", "ctx")

                    assert len(_insert_calls(conn_mock)) == 1


@pytest.mark.asyncio
async def test_reasoning_text_is_skipped_and_reasoning_disabled():
    """Live 2026-10-03 (Qwen3.8-27B): reasoning prose landed before the JSON, so no rules were ever extracted."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield "<thinking>"
        yield "The user wants rules. Mira is a tavern keeper, so [maybe] an invariant."
        yield "</thinking>"
        yield '[{"rule_text": "Mira Vale is the tavern keeper.", "rule_type": "invariant"}]'

    with patch.object(extractor.embedding_engine, "generate_embedding", new_callable=AsyncMock, return_value=[0.1]), \
         patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None), \
         patch.object(extractor.llm_client, "generate_stream", return_value=mock_gen()) as gen:
        await extractor.extract_initial_rules("s", "mira_vale", "context")

    assert gen.call_args.kwargs["extra_body"] == {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    inserts = _insert_calls(conn_mock)
    assert len(inserts) == 1
    assert inserts[0][0][4] == "Mira Vale is the tavern keeper."


@pytest.mark.asyncio
async def test_extraction_output_fits_the_window_and_local_gets_both_switches(monkeypatch):
    """Live test 2026-10-13: Gemma loaded with an 8k window; max_tokens=8192 on top of the
    prompt made every extraction come back empty."""
    from unittest.mock import MagicMock
    from config.manager import get_settings_manager
    get_settings_manager().update({"context_window_override": 8192})
    seen = {}

    async def gen(**kw):
        seen.update(kw)
        yield "[]"
    client = MagicMock()
    client.base_url = "http://10.0.0.5:1234/v1"
    client.generate_stream = gen
    w = LoreExtractionWorker.__new__(LoreExtractionWorker)
    w.llm_client, w.logger = client, logging.getLogger("t")
    await w._execute_extraction("x " * 2000, "s", "c", "m")
    assert seen["max_tokens"] < 8192 - 1000
    assert seen["extra_body"] == {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


# --- _place_rule (brief B1 spec 3): pure placement decisions --------------------

ROOMS = {"principals_office": "Principal's Office",
         "ua_high_school": "U.A. High School"}


class TestPlaceRule:
    def _place(self, rule, **kw):
        opts = dict(character_id="turn_char", characters=("turn_char", "mira_vale"),
                    rooms=ROOMS, default_level="world")
        opts.update(kw)
        return _place_rule(rule, **opts)

    def test_world_level(self):
        assert self._place({"level": "world"}) == ("world", None, None)

    def test_character_internal_known(self):
        assert self._place({"level": "character_internal", "character": "Mira Vale"}) == \
            ("character_internal", "mira_vale", None)

    def test_character_external_known(self):
        assert self._place({"level": "character_external", "character": "MIRA VALE"}) == \
            ("character_external", "mira_vale", None)

    def test_character_name_other_script_maps_to_id(self):
        """A non-Latin name reduces to its stable hash id, which is how it is known."""
        from core.identifiers import safe_char_id
        name = "美美"
        cid = safe_char_id(name)
        assert self._place({"level": "character_internal", "character": name},
                           characters=("turn_char", cid)) == \
            ("character_internal", cid, None)

    def test_unknown_character_keeps_its_own_id(self):
        # An NPC not in the chat yet: inactive until a character with that id joins;
        # never handed to the turn character (QA 2026-10-17).
        assert self._place({"level": "character_internal", "character": "Nobody"}) == \
            ("character_internal", "nobody", None)

    def test_narrator_turn_files_unknown_names_under_the_narrator(self):
        # A narrator voices the NPCs (QA 2026-10-17 T3): their rules are the
        # narrator's knowledge, or they would never reach a prompt.
        assert self._place({"level": "character_external", "character": "Katsuki Bakugo"},
                           narrator=True) == ("character_external", "turn_char", None)
        # A character who is in the chat keeps their own rules even on a narrator turn.
        assert self._place({"level": "character_internal", "character": "Mira Vale"},
                           narrator=True) == ("character_internal", "mira_vale", None)
        # The player is still never a source of card rules.
        assert self._place({"level": "character_external", "character": "Brenda"},
                           player_ids=("brenda",), from_cards=True, narrator=True) is None

    def test_player_rules_from_the_cards_are_dropped(self):
        for name in ("Changi", "Gu Changi", "user"):
            assert self._place({"level": "character_external", "character": name},
                               player_ids=("changi", "user"), from_cards=True) is None

    def test_world_rule_about_the_player_is_not_world_knowledge(self):
        # QA 2026-10-17: "Changi can shatter magic circles" was filed as world, so
        # everyone would know what only Alba saw.
        rule = {"level": "world", "rule_text": "Changi can effortlessly shatter magical circles."}
        assert self._place(rule, player_ids=("changi",), from_cards=False) == \
            ("character_internal", "turn_char", None)
        assert self._place(rule, player_ids=("changi",), from_cards=True) is None
        assert self._place({"level": "world", "rule_text": "The user's wings are black."},
                           player_ids=("changi",), from_cards=True) is None
        # Only mentioning the player keeps a world rule a world rule.
        assert self._place({"level": "world",
                            "rule_text": "The witchfinders hunt anyone who helps Changi."},
                           player_ids=("changi",)) == ("world", None, None)
        assert self._place({"level": "world", "rule_text": "Changi's arrival shook the hut."},
                           player_ids=("changi",), from_cards=False)[0] == "character_internal"

    def test_player_rules_from_the_story_become_the_turn_characters_knowledge(self):
        assert self._place({"level": "character_external", "character": "Gu Changi"},
                           player_ids=("changi",), from_cards=False) == \
            ("character_internal", "turn_char", None)

    def test_missing_level_uses_default_world(self):
        assert self._place({"rule_text": "x"}) == ("world", None, None)

    def test_unknown_level_uses_default_world(self):
        assert self._place({"level": "galaxy"}) == ("world", None, None)

    def test_missing_level_uses_default_character_internal(self):
        assert self._place({}, default_level="character_internal") == \
            ("character_internal", "turn_char", None)

    def test_unknown_level_uses_default_character_internal(self):
        assert self._place({"level": "nonsense"}, default_level="character_internal") == \
            ("character_internal", "turn_char", None)

    def test_location_exact_room_id(self):
        assert self._place({"level": "location", "location": "principals_office"}) == \
            ("location", None, "principals_office")

    def test_location_by_name(self):
        """'Principal's Office' matches the room whose name it is."""
        assert self._place({"level": "location", "location": "Principal's Office"}) == \
            ("location", None, "principals_office")

    def test_location_by_name_wordforms(self):
        assert self._place({"level": "location", "location": "Principals Office"}) == \
            ("location", None, "principals_office")

    def test_unknown_location_becomes_world(self):
        assert self._place({"level": "location", "location": "the void"}) == \
            ("world", None, None)

    def test_location_without_rooms_becomes_world(self):
        assert self._place({"level": "location", "location": "Principal's Office"},
                           rooms=None) == ("world", None, None)

    def test_location_empty_value_becomes_world(self):
        assert self._place({"level": "location"}) == ("world", None, None)


# --- end-to-end extraction: levels land where the brief says --------------------

@pytest.mark.asyncio
async def test_extraction_stores_each_level_in_its_place():
    """Fake LLM answer with one rule per level: world/character rules go to
    spm_chat_rules with the right level and character (real test DB), the
    location rule reaches the engine (fake). Brief B1 tests."""
    import asyncpg
    from tests._testdb import TEST_DB_CONFIG

    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        await conn.execute("DELETE FROM spm_chat_rules;")

        class _Pool:
            def __init__(self, conn):
                self.conn = conn

            def acquire(self):
                outer = self

                class _Ctx:
                    async def __aenter__(self):
                        return outer.conn

                    async def __aexit__(self, *exc):
                        return False
                return _Ctx()

        extractor = LoreExtractionWorker(_Pool(conn))

        rules_json = ('['
                      '{"rule_text": "Magic hums at dusk.", "rule_type": "invariant", "level": "world"},'
                      '{"rule_text": "She hides the heirloom.", "rule_type": "invariant",'
                      ' "level": "character_internal", "character": "Mira Vale"},'
                      '{"rule_text": "He limps when it rains.", "rule_type": "invariant",'
                      ' "level": "character_external", "character": "Bramwell"},'
                      '{"rule_text": "The floorboard creaks.", "rule_type": "trigger",'
                      ' "level": "location", "location": "Principal\'s Office"}'
                      ']')

        async def mock_gen(**kw):
            yield rules_json

        class FakeEngine:
            def __init__(self):
                self.calls = []

            async def add_room_rule(self, room_id, rule_text, session_id, **kw):
                self.calls.append((room_id, rule_text, session_id, kw))

        engine = FakeEngine()

        with patch.object(extractor.embedding_engine, "generate_embedding",
                          new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]), \
             patch("proxy.rag.lore_extractor._space_id_for", new_callable=AsyncMock, return_value=None), \
             patch.object(extractor.llm_client, "generate_stream", new=mock_gen):
            await extractor.extract_initial_rules(
                "b1-e2e-session", "turn_char", "ctx",
                characters=("turn_char", "mira_vale"),
                rooms={"principals_office": "Principal's Office"},
                engine=engine)

        rows = await conn.fetch(
            "SELECT level, character_id, rule_text, rule_type, status, source "
            "FROM spm_chat_rules WHERE session_id = $1 ORDER BY rule_text",
            "b1-e2e-session")
        by_text = {r["rule_text"]: dict(r) for r in rows}

        assert set(by_text) == {"Magic hums at dusk.",
                                "She hides the heirloom.",
                                "He limps when it rains."}
        w = by_text["Magic hums at dusk."]
        assert (w["level"], w["character_id"], w["status"], w["source"]) == \
            ("world", None, "pending", "extracted")
        ci = by_text["She hides the heirloom."]
        assert (ci["level"], ci["character_id"]) == ("character_internal", "mira_vale")
        ce = by_text["He limps when it rains."]
        # 'Bramwell' is not in the chat -> the character whose turn it was.
        # Bramwell is not in the chat yet: his rule keeps his id (QA 2026-10-17), never
        # the turn character's.
        assert (ce["level"], ce["character_id"]) == ("character_external", "bramwell")

        assert len(engine.calls) == 1
        room_id, rule_text, session_id, kw = engine.calls[0]
        assert (room_id, rule_text, session_id) == \
            ("principals_office", "The floorboard creaks.", "b1-e2e-session")
        assert kw["rule_type"] == "conditional_trigger"
        assert kw["status"] == "pending" and kw["source"] == "extracted"
    finally:
        await conn.execute("DELETE FROM spm_chat_rules;")
        await conn.close()


@pytest.mark.asyncio
async def test_location_rule_without_engine_is_stored_as_world():
    """Spec 3: no engine -> the location rule is kept as world (pending), not lost."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield ('[{"rule_text": "The floorboard creaks.", "rule_type": "invariant",'
               ' "level": "location", "location": "principals_office"}]')

    with patch.object(extractor.embedding_engine, "generate_embedding",
                      new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]), \
         patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None), \
         patch.object(extractor.llm_client, "generate_stream", return_value=mock_gen()):
        await extractor.extract_initial_rules(
            "s", "turn_char", "ctx",
            characters=("turn_char",), rooms={"principals_office": "Principal's Office"})

    inserts = _insert_calls(conn_mock)
    assert len(inserts) == 1
    assert inserts[0][0][2] == "world"


@pytest.mark.asyncio
async def test_engine_error_skips_rule_and_never_raises():
    """Spec 4: an engine error is logged and the rule skipped; extraction continues."""
    conn_mock = AsyncMock()
    conn_mock.fetchval = AsyncMock(return_value=None)
    extractor = LoreExtractionWorker(_fake_pool(conn_mock))

    async def mock_gen():
        yield ('[{"rule_text": "Room rule that fails.", "rule_type": "invariant",'
               ' "level": "location", "location": "principals_office"},'
               ' {"rule_text": "Still stored.", "rule_type": "invariant", "level": "world"}]')

    class BoomEngine:
        async def add_room_rule(self, *a, **kw):
            raise RuntimeError("engine offline")

    with patch.object(extractor.embedding_engine, "generate_embedding",
                      new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]), \
         patch('proxy.rag.lore_extractor._space_id_for', new_callable=AsyncMock, return_value=None), \
         patch.object(extractor.llm_client, "generate_stream", return_value=mock_gen()):
        await extractor.extract_initial_rules(
            "s", "turn_char", "ctx", characters=("turn_char",),
            rooms={"principals_office": "Principal's Office"}, engine=BoomEngine())

    inserts = _insert_calls(conn_mock)
    assert len(inserts) == 1
    assert inserts[0][0][4] == "Still stored."
