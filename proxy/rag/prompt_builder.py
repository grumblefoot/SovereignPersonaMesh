"""
Cognitive Prompt Assembly & 32K Token Budget Partitioning Matrix.
Formats custom prompts for Character Subagents (CSAs) strictly adhering to token budgets.
"""

from typing import List, Dict, Any
import re
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
        spatial_context: str,
        flavor_text: str = "",
        frontend_max_tokens: int = 300,
        style_card: Optional[Any] = None,
    ) -> str:
        """
        Assembles structured prompt for Character Subagent turn execution.
        Follows conflict resolution hierarchy:
          1. SPM Mechanics (<ctrl94> monologue system directive)
          2. Explicit Frontend (system_prompt & frontend_max_tokens)
          3. Implicit Heuristics (style_card directives)
        """
        # Format Long-Term RAG Memories
        memory_str = ""
        if retrieved_memories:
            mem_lines = [f"- {m['sensory_input']}" for m in retrieved_memories]
            memory_str = "\n".join(mem_lines)
        else:
            memory_str = "No specific relevant past memories recalled."

        env_block = f"{spatial_context}\nSensory Feed: {sensory_feed}"
        if flavor_text:
            env_block += f"\nEnvironmental Atmosphere: {flavor_text}"

        style_block = ""
        if style_card and hasattr(style_card, "style_instruction"):
            style_block = f"\n\n[NARRATIVE STYLE HEURISTICS]\n{style_card.style_instruction}"

        formatted_prompt = f"""<system>
{system_prompt}{style_block}

[RECALLED EPISODIC MEMORIES]
{memory_str}

[CURRENT SPATIAL & SENSORY ENVIRONMENT]
{env_block}
</system>

[RECENT CONVERSATIONAL HISTORY]
"""
        for msg in chat_history[-15:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"\n{role.capitalize()}: {content}"

        if self.config.inner_monologue_enabled:
            formatted_prompt += f"""\n\nSystem Directive: You must begin your response immediately with <ctrl94>. Place all internal thoughts and planning strictly inside these tags. You must close with </ctrl94> before writing your public dialogue. Your public dialogue must not exceed {frontend_max_tokens} words.\n<ctrl94>"""
        else:
            formatted_prompt += "\n\nCharacter Output:"

        return formatted_prompt

    def build_csa_messages(
        self,
        system_prompt: str,
        sensory_feed: str,
        retrieved_memories: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        spatial_context: str,
        flavor_text: str = "",
        frontend_max_tokens: int = 300,
        style_card: Optional[Any] = None,
    ) -> List[Dict[str, str]]:
        """
        Assembles OpenAI-native structured messages array for Character Subagent execution.
        Prevents system prompt dumping and keeps character dialogue 100% clean.
        """
        memory_str = ""
        if retrieved_memories:
            mem_lines = [f"- {m['sensory_input']}" for m in retrieved_memories]
            memory_str = "\n".join(mem_lines)
        else:
            memory_str = "No specific relevant past memories recalled."

        env_block = f"{spatial_context}\nSensory Feed: {sensory_feed}"
        if flavor_text:
            env_block += f"\nEnvironmental Atmosphere: {flavor_text}"

        style_block = ""
        if style_card and hasattr(style_card, "style_instruction"):
            style_block = f"\n\n[NARRATIVE STYLE HEURISTICS]\n{style_card.style_instruction}"

        if self.config.inner_monologue_enabled:
            system_content = f"""{system_prompt}{style_block}

[RECALLED EPISODIC MEMORIES]
{memory_str}

[CURRENT SPATIAL & SENSORY ENVIRONMENT]
{env_block}

System Directive: You MUST begin your response immediately with <ctrl94>. Place all internal thoughts and planning strictly inside these tags. You MUST close with </ctrl94> before writing your public dialogue. Your public dialogue must not exceed {frontend_max_tokens} words."""
        else:
            system_content = f"""{system_prompt}{style_block}

[RECALLED EPISODIC MEMORIES]
{memory_str}

[CURRENT SPATIAL & SENSORY ENVIRONMENT]
{env_block}

System Directive: Respond strictly in-character. Do not output system meta-instructions or prompt text. Your public output must not exceed {frontend_max_tokens} words."""

        messages = [{"role": "system", "content": system_content}]
        for msg in chat_history[-15:]:
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r == "assistant" and self.config.inner_monologue_enabled:
                c = re.sub(r'Internal Monologue/Planning:\s*\n*', '', c, flags=re.IGNORECASE)
                if "<ctrl94>" not in c:
                    c = f"<ctrl94> *Processing context...* </ctrl94>\n{c}"
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

        return messages
