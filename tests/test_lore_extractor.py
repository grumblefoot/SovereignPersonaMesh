import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from proxy.rag.lore_extractor import LoreExtractionWorker


@pytest.mark.asyncio
async def test_extract_initial_rules():
    """Test that extract_initial_rules calls LLM, parses rules, and persists them to DB."""
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value=None)

    extractor = LoreExtractionWorker(db_pool_mock)
    
    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Character is scared of fire.", "rule_type": "invariant"}]\n'
        yield "```"
    
    with patch.object(extractor, '_clean_llm_json_response', return_value=[
        {"rule_text": "Character is scared of fire.", "rule_type": "invariant"}
    ]):
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session123", "Tester", "context")
                
                conn_mock.execute.assert_any_call("SELECT create_csa_lore_rules_table($1);", "tester")
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO csa_lore_rules_tester" in c[0][0]]
                assert len(insert_calls) == 1
                assert "Character is scared of fire." in insert_calls[0][0][1]


@pytest.mark.asyncio
async def test_extract_initial_rules_skips_existing():
    """Test that duplicate rule_text is not inserted (idempotency)."""
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value="existing-id")

    extractor = LoreExtractionWorker(db_pool_mock)
    
    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Character is scared of fire.", "rule_type": "invariant"}]\n'
        yield "```"
    
    with patch.object(extractor, '_clean_llm_json_response', return_value=[
        {"rule_text": "Character is scared of fire.", "rule_type": "invariant"}
    ]):
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session123", "Tester", "context")
                
                select_calls = [c for c in conn_mock.execute.call_args_list if "create_csa_lore_rules_table" in c[0][0]]
                assert len(select_calls) == 1
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO" in c[0][0]]
                assert len(insert_calls) == 0


@pytest.mark.asyncio
async def test_extract_initial_rules_multiple_rules():
    """Test that multiple rules are extracted and persisted."""
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value=None)

    extractor = LoreExtractionWorker(db_pool_mock)
    
    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Rule 1", "rule_type": "invariant"}, {"rule_text": "Rule 2", "rule_type": "trigger"}]\n'
        yield "```"
    
    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
            await extractor.extract_initial_rules("session123", "Tester", "context")
            
            insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO" in c[0][0]]
            assert len(insert_calls) == 2
            # Trigger should be converted to conditional_trigger
            # INSERT args: (sql, rule_text, rule_type, emb_str)
            assert "conditional_trigger" in insert_calls[1][0][2]


@pytest.mark.asyncio
async def test_periodic_review_rules():
    """Test that periodic_review_rules builds messages string and extracts rules."""
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value=None)

    extractor = LoreExtractionWorker(db_pool_mock)
    
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
        with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
            with patch('proxy.rag.lore_extractor.strings') as mock_strings:
                mock_strings.get.return_value = "mock prompt"
                await extractor.periodic_review_rules("session123", "Tester", recent_messages)
                
                mock_strings.get.assert_called()
                call_kwargs = mock_strings.get.call_args
                assert "recent_messages" in call_kwargs.kwargs
                assert "Hello" in call_kwargs.kwargs["recent_messages"]
                assert "Hi there" in call_kwargs.kwargs["recent_messages"]
                assert "What is this place?" in call_kwargs.kwargs["recent_messages"]


@pytest.mark.asyncio
async def test_periodic_review_rules_empty_messages():
    """Test that periodic_review_rules handles empty message list."""
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value=None)

    extractor = LoreExtractionWorker(db_pool_mock)
    
    async def mock_gen():
        yield "```json\n"
        yield '[{"rule_text": "Empty review rule", "rule_type": "invariant"}]\n'
        yield "```"
    
    with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]
        with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
            with patch('proxy.rag.lore_extractor.strings') as mock_strings:
                mock_strings.get.return_value = "mock prompt"
                await extractor.periodic_review_rules("session123", "Tester", [])
                
                mock_strings.get.assert_called()
                call_kwargs = mock_strings.get.call_args
                assert call_kwargs.kwargs["recent_messages"] == ""


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
        db_pool_mock = MagicMock()
        conn_mock = AsyncMock()
        acquire_cm = AsyncMock()
        acquire_cm.__aenter__.return_value = conn_mock
        acquire_cm.__aexit__.return_value = False
        db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
        conn_mock.fetchval = AsyncMock(return_value=None)

        extractor = LoreExtractionWorker(db_pool_mock)
        
        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_text": "trigger rule", "rule_type": "trigger"}]\n'
            yield "```"
        
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session1", "char", "ctx")
                
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO" in c[0][0]]
                assert len(insert_calls) == 1
                # INSERT args: (sql, rule_text, rule_type, emb_str)
                assert "conditional_trigger" in insert_calls[0][0][2]
    
    @pytest.mark.asyncio
    async def test_invalid_rule_type_defaults_to_invariant(self):
        """Test that invalid rule_type defaults to 'invariant'."""
        db_pool_mock = MagicMock()
        conn_mock = AsyncMock()
        acquire_cm = AsyncMock()
        acquire_cm.__aenter__.return_value = conn_mock
        acquire_cm.__aexit__.return_value = False
        db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
        conn_mock.fetchval = AsyncMock(return_value=None)

        extractor = LoreExtractionWorker(db_pool_mock)
        
        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_text": "bad type", "rule_type": "invalid_type"}]\n'
            yield "```"
        
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session1", "char", "ctx")
                
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO" in c[0][0]]
                assert len(insert_calls) == 1
                assert "invariant" in insert_calls[0][0][2]
    
    @pytest.mark.asyncio
    async def test_skips_rules_without_rule_text(self):
        """Test that rules missing 'rule_text' are skipped."""
        db_pool_mock = MagicMock()
        conn_mock = AsyncMock()
        acquire_cm = AsyncMock()
        acquire_cm.__aenter__.return_value = conn_mock
        acquire_cm.__aexit__.return_value = False
        db_pool_mock.acquire = MagicMock(return_value=acquire_cm)
        conn_mock.fetchval = AsyncMock(return_value=None)

        extractor = LoreExtractionWorker(db_pool_mock)
        
        async def mock_gen():
            yield "```json\n"
            yield '[{"rule_type": "invariant"}, {"rule_text": "valid rule", "rule_type": "invariant"}]\n'
            yield "```"
        
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=mock_gen()):
                await extractor.extract_initial_rules("session1", "char", "ctx")
                
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO" in c[0][0]]
                assert len(insert_calls) == 1
