"""
Cognitive Prompt Assembly & 32K Token Budget Partitioning Matrix.
Formats custom prompts for Character Subagents (CSAs) strictly adhering to token budgets.
"""

from typing import List, Dict, Any, Optional
import re
from config.hardware_tiers import HardwareConfig, HARDWARE_TIERS, HardwareTierEnum
from proxy.rag.gm_actions import default_gm_registry
from core.resource_manager import strings

OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"


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
          1. SPM Mechanics (`` thinking `` monologue system directive)
          2. Explicit Frontend (system_prompt & frontend_max_tokens)
          3. Implicit Heuristics (style_card directives)
        """
        # Format Long-Term RAG Memories
        memory_str = ""
        if retrieved_memories:
            mem_lines = [f"- {m['sensory_input']}" for m in retrieved_memories]
            memory_str = "\n".join(mem_lines)
        else:
            memory_str = strings.get("rag.no_memories_fallback")

        env_block = f"{spatial_context}\nSensory Feed: {sensory_feed}"
        if flavor_text:
            env_block += strings.get("rag.environmental_atmosphere", flavor_text=flavor_text)

        style_block = ""
        if style_card and hasattr(style_card, "style_instruction"):
            style_block = strings.get("rag.style_heuristics", style_instruction=style_card.style_instruction)

        formatted_prompt = strings.get(
            "rag.csa_prompt_system_block",
            system_prompt=system_prompt,
            style_block=style_block,
            memory_str=memory_str,
            env_block=env_block
        )

        for msg in chat_history[-15:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"\n{role.capitalize()}: {content}"

        gm_actions = default_gm_registry.get_all_active()
        gm_instructions = "\n".join([a.prompt_fragment for a in gm_actions])

        if self.config.inner_monologue_enabled:
            formatted_prompt += strings.get(
                "rag.monologue_directive",
                frontend_max_tokens=frontend_max_tokens,
                gm_instructions=gm_instructions
            )
        else:
            formatted_prompt += strings.get("rag.no_monologue_directive")

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
            memory_str = strings.get("rag.no_memories_fallback")

        env_block = f"{spatial_context}\nSensory Feed: {sensory_feed}"
        if flavor_text:
            env_block += strings.get("rag.environmental_atmosphere", flavor_text=flavor_text)

        style_block = ""
        if style_card and hasattr(style_card, "style_instruction"):
            style_block = strings.get("rag.style_heuristics", style_instruction=style_card.style_instruction)

        gm_actions = default_gm_registry.get_all_active()
        gm_instructions = "\n".join([a.prompt_fragment for a in gm_actions])

        system_content = strings.get(
            "rag.csa_messages_system_block",
            system_prompt=system_prompt,
            style_block=style_block,
            memory_str=memory_str,
            env_block=env_block
        )

        if self.config.inner_monologue_enabled:
            system_content += strings.get(
                "rag.monologue_directive_strict",
                frontend_max_tokens=frontend_max_tokens,
                gm_instructions=gm_instructions
            )
        else:
            system_content += strings.get(
                "rag.no_monologue_directive_strict",
                frontend_max_tokens=frontend_max_tokens
            )

        messages = [{"role": "system", "content": system_content}]
        for msg in chat_history[-15:]:
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r == "assistant" and self.config.inner_monologue_enabled:
                c = re.sub(r'Internal Monologue/Planning:\s*\n*', '', c, flags=re.IGNORECASE)
                if OPEN_THINK_TAG not in c:
                    c = strings.get("rag.dummy_think_block", open_tag=OPEN_THINK_TAG, close_tag=CLOSE_THINK_TAG, content=c)
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

        return messages
