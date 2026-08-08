"""
Unit tests for FR-007 (Issue #7): Universal Token Decoupling & Strict Monologue Prompt Enforcement.
"""

import pytest
import os
import tempfile
from config.manager import SettingsManager
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.core.stream_parser import MonologueStreamParser


class TestBackendMaxTokensConfig:
    def test_default_backend_max_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            mgr = SettingsManager(config_path=cfg_path)
            settings = mgr.get_settings()
            assert "backend_max_tokens" in settings
            assert settings["backend_max_tokens"] == 2048

    def test_update_backend_max_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            mgr = SettingsManager(config_path=cfg_path)
            mgr.write_settings({"backend_max_tokens": 4096})
            updated = mgr.get_settings()
            assert updated["backend_max_tokens"] == 4096


class TestPromptSuffixInjection:
    def test_prompt_builder_injects_frontend_max_tokens_directive(self):
        builder = CognitivePromptBuilder()
        prompt = builder.build_csa_prompt(
            system_prompt="You are Luna.",
            sensory_feed="Direct feed",
            retrieved_memories=[],
            chat_history=[],
            spatial_context="Location: Cellar",
            frontend_max_tokens=150,
        )
        assert "System Directive: You must begin your response immediately with <ctrl94>." in prompt
        assert "Your public dialogue must not exceed 150 words." in prompt


class TestStreamParserTokenDecoupling:
    @pytest.mark.asyncio
    async def test_monologue_tokens_do_not_count_against_public_limit(self):
        parser = MonologueStreamParser(max_public_tokens=5)

        async def mock_stream():
            # Long monologue (100 tokens inside tags) followed by short public response
            yield "<ctrl94>"
            for i in range(20):
                yield f"thinking step {i} "
            yield "</ctrl94>"
            yield "Hello world, this is public text."

        results = []
        async for chunk in parser.process_token_stream(mock_stream()):
            results.append(chunk)

        # Monologue was parsed out into private buffer
        mono, pub = parser.get_final_buffers()
        assert len(mono) > 0
        assert "thinking step" in mono
        assert "Hello world" in pub
        # Public token count should track public words
        assert parser.public_token_count > 0
        assert parser.monologue_token_count > 0

    @pytest.mark.asyncio
    async def test_public_tokens_truncated_at_limit(self):
        parser = MonologueStreamParser(max_public_tokens=3)

        async def mock_stream():
            yield "<ctrl94>Planning response...</ctrl94>"
            yield "One "
            yield "two "
            yield "three "
            yield "four "
            yield "five "

        results = []
        async for chunk in parser.process_token_stream(mock_stream()):
            results.append(chunk)

        assert parser.is_public_truncated is True
        assert len(results) <= 3
