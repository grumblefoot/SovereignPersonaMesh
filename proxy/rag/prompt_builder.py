"""
Cognitive Prompt Assembly & 32K Token Budget Partitioning Matrix.
Formats custom prompts for Character Subagents (CSAs) strictly adhering to token budgets.
"""

from typing import List, Dict, Any
from config.hardware_tiers import HardwareConfig, HARDWARE_TIERS, HardwareTierEnum


class CognitivePromptBuilder:
    def __init__(self, hw_config: HardwareConfig = HARDWARE_TIERS[HardwareTierEnum.SOVEREIGN]):
        self.config = hw_config

    def build_csa_prompt(
        self,
        system_prompt: str,
        sensory_feed: str,
        retrieved_memories: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        spatial_context: str
    ) -> str:
        """
        Assembles structured prompt for Character Subagent turn execution.
        """
        # Format Long-Term RAG Memories
        memory_str = ""
        if retrieved_memories:
            mem_lines = [f"- {m['sensory_input']}" for m in retrieved_memories]
            memory_str = "\n".join(mem_lines)
        else:
            memory_str = "No specific relevant past memories recalled."

        formatted_prompt = f"""<system>
{system_prompt}

[RECALLED EPISODIC MEMORIES]
{memory_str}

[CURRENT SPATIAL & SENSORY ENVIRONMENT]
{spatial_context}
Sensory Feed: {sensory_feed}
</system>

[RECENT CONVERSATIONAL HISTORY]
"""
        for msg in chat_history[-15:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"\n{role.capitalize()}: {content}"

        if self.config.inner_monologue_enabled:
            formatted_prompt += """\n\n<ctrl94>Write your private inner monologue planning thoughts first inside <ctrl94> ... </ctrl94> tags, followed by your public dialogue response.</ctrl94>\n<ctrl94>"""
        else:
            formatted_prompt += "\n\nCharacter Output:"

        return formatted_prompt
