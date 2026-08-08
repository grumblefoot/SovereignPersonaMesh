"""
Zero-LLM Agnostic Vibe Profiler for Sovereign Persona Mesh (proxy/rag/vibe_profiler.py).

Relies purely on Python string heuristics (0 ms LLM overhead) to compute:
  - dialogue_action_ratio: Ratio of tokens in quotes ("...") vs asterisks (*...*).
  - verbosity_baseline: Average word length per assistant message.
  - perspective_and_tense: Tagging for pronouns (I/You/He/She) and verb tenses.
Outputs a structured StyleCard used by CognitivePromptBuilder for hierarchical merging.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class StyleCard:
    dialogue_action_ratio: float  # Ratio of dialogue to total (0.0 to 1.0)
    verbosity_baseline: int        # Average words per response
    dominant_perspective: str     # "first_person", "second_person", "third_person"
    tense: str                    # "past", "present"
    style_instruction: str        # Compiled string instruction for prompt builder

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VibeProfiler:
    """Zero-LLM heuristic vibe analyzer for character narratives."""

    def __init__(self, default_window: int = 10):
        self.default_window = default_window

    def analyze_messages(self, messages: List[Dict[str, str]], window: Optional[int] = None) -> StyleCard:
        """
        Analyze a list of message dicts (role, content) and calculate zero-LLM style metrics.
        """
        limit = window or self.default_window
        assistant_msgs = [
            m.get("content", "") for m in messages
            if isinstance(m, dict) and m.get("role") in ("assistant", "char", "model")
        ][-limit:]

        if not assistant_msgs:
            # Fallback if no assistant messages found
            assistant_msgs = [
                m.get("content", "") for m in messages
                if isinstance(m, dict) and m.get("content")
            ][-limit:]

        if not assistant_msgs:
            return StyleCard(
                dialogue_action_ratio=0.5,
                verbosity_baseline=150,
                dominant_perspective="third_person",
                tense="present",
                style_instruction="Maintain a balanced dialogue-to-action ratio.",
            )

        total_words = 0
        total_dialogue_words = 0
        total_action_words = 0

        first_person_count = 0
        second_person_count = 0
        third_person_count = 0

        past_tense_count = 0
        present_tense_count = 0

        for msg in assistant_msgs:
            words = msg.split()
            word_count = len(words)
            total_words += word_count

            # Dialogue in quotes "..."
            quotes = re.findall(r'"([^"]*)"', msg)
            dialogue_words = sum(len(q.split()) for q in quotes)
            total_dialogue_words += dialogue_words

            # Action in asterisks *...*
            asterisks = re.findall(r'\*([^*]*)\*', msg)
            action_words = sum(len(a.split()) for a in asterisks)
            total_action_words += action_words

            # Pronoun perspective tagging
            lower = msg.lower()
            first_person_count += len(re.findall(r'\b(i|me|my|mine|myself)\b', lower))
            second_person_count += len(re.findall(r'\b(you|your|yours|yourself)\b', lower))
            third_person_count += len(re.findall(r'\b(he|him|his|she|her|hers|they|them|their)\b', lower))

            # Tense tagging (simple verb ending heuristics)
            past_tense_count += len(re.findall(r'\b\w+ed\b', lower)) + len(re.findall(r'\b(was|were|had|did|said|walked|looked)\b', lower))
            present_tense_count += len(re.findall(r'\b\w+ing\b', lower)) + len(re.findall(r'\b(is|are|has|does|says|walks|looks)\b', lower))

        avg_verbosity = max(1, total_words // len(assistant_msgs))
        denom = max(1, total_dialogue_words + total_action_words)
        ratio = round(total_dialogue_words / denom, 2)
        if ratio > 1.0:
            ratio = 1.0

        # Perspective
        if first_person_count >= second_person_count and first_person_count >= third_person_count:
            perspective = "first_person"
        elif second_person_count >= third_person_count:
            perspective = "second_person"
        else:
            perspective = "third_person"

        tense = "past" if past_tense_count >= present_tense_count else "present"

        # Style directive string
        dial_pct = int(ratio * 100)
        act_pct = 100 - dial_pct
        instruction = (
            f"Stylistic Heuristics: Maintain a ~{dial_pct}/{act_pct} dialogue-to-action ratio. "
            f"Target length: ~{avg_verbosity} words. Perspective: {perspective.replace('_', ' ')}."
        )

        return StyleCard(
            dialogue_action_ratio=ratio,
            verbosity_baseline=avg_verbosity,
            dominant_perspective=perspective,
            tense=tense,
            style_instruction=instruction,
        )
