import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from proxy.rag.lore_extractor import LoreExtractionWorker

@pytest.mark.asyncio
async def test_extract_initial_rules():
    db_pool_mock = MagicMock()
    conn_mock = AsyncMock()
    db_pool_mock.acquire.return_value.__aenter__.return_value = conn_mock

    extractor = LoreExtractionWorker(db_pool_mock)
    
    with patch.object(extractor, '_clean_llm_json_response', return_value=[
        {"rule_text": "Character is scared of fire.", "rule_type": "invariant"}
    ]) as clean_mock:
        with patch.object(extractor.embedding_engine, 'generate_embedding', new_callable=AsyncMock) as embed_mock:
            embed_mock.return_value = [0.1, 0.2, 0.3]
            with patch.object(extractor.llm_client, 'generate_stream', return_value=AsyncMock()) as llm_mock:
                # Mock async generator
                async def mock_gen():
                    yield "```json\\n"
                    yield "[{\\"rule_text\\": \\"Character is scared of fire.\\", \\"rule_type\\": \\"invariant\\"}]\\n"
                    yield "```"
                llm_mock.return_value = mock_gen()
                
                await extractor.extract_initial_rules("session123", "Tester", "context")
                
                # Verify DB calls
                conn_mock.execute.assert_any_call("SELECT create_csa_lore_rules_table($1);", "tester")
                
                # Check if insert was called
                insert_calls = [c for c in conn_mock.execute.call_args_list if "INSERT INTO csa_lore_rules_tester" in c[0][0]]
                assert len(insert_calls) == 1
                assert "Character is scared of fire." in insert_calls[0][0][1]
