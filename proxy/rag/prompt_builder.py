"""
Cognitive Prompt Assembly & 32K Token Budget Partitioning Matrix.
Formats custom prompts for Character Subagents (CSAs) strictly adhering to token budgets.
"""

from typing import List, Dict, Any, Optional
import logging
import os
import re
from config.hardware_tiers import HardwareConfig, HARDWARE_TIERS, HardwareTierEnum
from proxy.rag.gm_actions import default_gm_registry
from core.resource_manager import strings

OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"


class CognitivePromptBuilder:
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        # Resolved at call time: the old default argument froze SOVEREIGN at import,
        # ignoring SPM_HARDWARE_TIER entirely.
        if hw_config is None:
            tier_name = os.getenv("SPM_HARDWARE_TIER", "SOVEREIGN").upper()
            try:
                tier = HardwareTierEnum(tier_name)
            except ValueError:
                logging.getLogger(__name__).warning(
                    f"[PromptBuilder] Unknown SPM_HARDWARE_TIER '{tier_name}', using SOVEREIGN.")
                tier = HardwareTierEnum.SOVEREIGN
            hw_config = HARDWARE_TIERS[tier]
        self.config = hw_config

    @staticmethod
    def _gm_actions_block(gm_mode: str) -> str:
        """The GM_ACTION directive for this mode (OPEN-005 / SD-01). 'off' returns ''
        (~90 prompt tokens saved per turn); 'move_only' never advertises CREATE_ROOM."""
        mode = str(gm_mode or "full").lower()
        if mode == "off":
            return ""
        if mode == "move_only":
            fragments = [a.prompt_fragment for a in default_gm_registry.get_all_active()
                         if a.id == "MOVE"]
            return strings.get("rag.gm_actions_directive_move_only",
                               gm_instructions="\n".join(fragments))
        fragments = [a.prompt_fragment for a in default_gm_registry.get_all_active()]
        return strings.get("rag.gm_actions_directive",
                           gm_instructions="\n".join(fragments))

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
        gm_mode: str = "full",
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
            mem_lines = []
            for m in retrieved_memories:
                line = f"- Sensory: {m['sensory_input']}"
                if m.get('inner_monologue'):
                    line += f"\n  Past Thoughts: {m['inner_monologue']}"
                mem_lines.append(line)
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

        if self.config.inner_monologue_enabled:
            formatted_prompt += strings.get(
                "rag.monologue_directive",
                frontend_max_tokens=frontend_max_tokens,
            )
            formatted_prompt += self._gm_actions_block(gm_mode)
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
        gm_mode: str = "full",
        max_history: Optional[int] = 15,
    ) -> List[Dict[str, str]]:
        """
        Assembles OpenAI-native structured messages array for Character Subagent execution.
        Prevents system prompt dumping and keeps character dialogue 100% clean.
        gm_mode (off | move_only | full) decides whether and which GM_ACTION
        directive joins the prompt (OPEN-005 / SD-01): 'off' spends zero tokens on it.
        """
        memory_str = ""
        if retrieved_memories:
            mem_lines = []
            for m in retrieved_memories:
                line = f"- Sensory: {m['sensory_input']}"
                if m.get('inner_monologue'):
                    line += f"\n  Past Thoughts: {m['inner_monologue']}"
                mem_lines.append(line)
            memory_str = "\n".join(mem_lines)
        else:
            memory_str = strings.get("rag.no_memories_fallback")

        env_block = f"{spatial_context}\nSensory Feed: {sensory_feed}"
        if flavor_text:
            env_block += strings.get("rag.environmental_atmosphere", flavor_text=flavor_text)

        style_block = ""
        if style_card and hasattr(style_card, "style_instruction"):
            style_block = strings.get("rag.style_heuristics", style_instruction=style_card.style_instruction)

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
            )
            system_content += self._gm_actions_block(gm_mode)
        else:
            system_content += strings.get(
                "rag.no_monologue_directive_strict",
                frontend_max_tokens=frontend_max_tokens
            )

        messages = [{"role": "system", "content": system_content}]
        # None = the caller already budgeted the history (token budget P1); the
        # legacy fixed cap only applies when no allocator ran.
        for msg in (chat_history if max_history is None else chat_history[-max_history:]):
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r == "assistant" and self.config.inner_monologue_enabled:
                c = re.sub(r'Internal Monologue/Planning:\s*\n*', '', c, flags=re.IGNORECASE)
                if OPEN_THINK_TAG not in c:
                    c = strings.get("rag.dummy_think_block", open_tag=OPEN_THINK_TAG, close_tag=CLOSE_THINK_TAG, content=c)
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

        return messages
