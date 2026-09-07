"""
Unit tests for FR-008 / Issue #8: Batch Chat Analyzer & Hierarchical Vibe Profiler.
"""

import pytest
import os
import json
import tempfile
from scripts.batch_analyzer import BatchChatAnalyzer
from proxy.rag.vibe_profiler import VibeProfiler, StyleCard
from proxy.rag.prompt_builder import CognitivePromptBuilder


class TestBatchChatAnalyzer:
    def test_parse_message_payload_clean(self):
        analyzer = BatchChatAnalyzer()
        raw = '<thinking>I should speak softly to Rowan.</thinking> "Hello Rowan," Luna whispered.'
        result = analyzer.parse_message_payload(raw)

        assert result["is_clean"] is True
        assert result["inner_monologue"] == "I should speak softly to Rowan."
        assert result["public_dialogue"] == '"Hello Rowan," Luna whispered.'
        assert result["anomalies"] == []

    def test_parse_message_payload_pre_thought_bleed(self):
        analyzer = BatchChatAnalyzer()
        raw = 'Garbage text before tag <thinking>Planning turn...</thinking> "Hello."'
        result = analyzer.parse_message_payload(raw)

        assert result["is_clean"] is False
        assert "PRE_PLANNING_BLEED" in result["anomalies"]

    def test_parse_message_payload_missing_closing_tag(self):
        analyzer = BatchChatAnalyzer()
        raw = '<thinking>Thinking about secret plans without closing tag'
        result = analyzer.parse_message_payload(raw)

        assert result["is_clean"] is False
        assert "MISSING_CLOSING_TAG" in result["anomalies"]

    def test_parse_message_payload_omniscient_bleed(self):
        analyzer = BatchChatAnalyzer(hidden_variables=["secret_key_123"])
        raw = '<thinking>I know the key.</thinking> "The secret_key_123 is hidden in the cellar."'
        result = analyzer.parse_message_payload(raw)

        assert result["is_clean"] is False
        assert "OMNISCIENT_BLEED" in result["anomalies"]

    def test_export_cot_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")

            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"name": "User", "mes": "<thinking>Plan A.</thinking> \"Public response.\""}) + "\n")
                f.write(json.dumps({"name": "User", "mes": "Bad text <thinking>Unclosed"}) + "\n")

            analyzer = BatchChatAnalyzer()
            count = analyzer.export_cot_dataset(input_path, output_path)
            assert count == 1

            with open(output_path, "r", encoding="utf-8") as out_f:
                lines = out_f.readlines()
                assert len(lines) == 1
                rec = json.loads(lines[0])
                assert rec["thought"] == "Plan A."
                assert rec["output"] == '"Public response."'


class TestVibeProfilerHeuristics:
    def test_zero_llm_heuristics_calculation(self):
        profiler = VibeProfiler()
        messages = [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": '"I am doing well," she said, *looking over the ancient map*.'},
            {"role": "assistant", "content": '"We must hurry," I whispered, *grabbing my leather satchel*.'},
        ]
        card = profiler.analyze_messages(messages)

        assert isinstance(card, StyleCard)
        assert 0.0 <= card.dialogue_action_ratio <= 1.0
        assert card.verbosity_baseline > 0
        assert card.dominant_perspective in ("first_person", "second_person", "third_person")
        assert "Stylistic Heuristics:" in card.style_instruction


class TestHierarchicalPromptAssembly:
    def test_hierarchical_prompt_merge_precedence(self):
        builder = CognitivePromptBuilder()
        profiler = VibeProfiler()
        card = profiler.analyze_messages([
            {"role": "assistant", "content": '"Hello," I said.'}
        ])

        prompt = builder.build_csa_prompt(
            system_prompt="[System Note: Write descriptively.]",
            sensory_feed="Direct feed",
            retrieved_memories=[],
            chat_history=[],
            spatial_context="Location: Cellar",
            frontend_max_tokens=250,
            style_card=card,
        )

        assert "[System Note: Write descriptively.]" in prompt
        assert "[NARRATIVE STYLE HEURISTICS]" in prompt
        assert card.style_instruction in prompt
        assert "System Directive: You must begin your response immediately with <think>." in prompt
