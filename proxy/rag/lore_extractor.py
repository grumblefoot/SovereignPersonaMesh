import asyncio
import json
import logging
from typing import List, Dict, Any

import asyncpg

from proxy.backend_client.lemonade_client import LemonadeLLMClient, DEFAULT_CHAT_MODEL
from scripts.onnx_embedder import CPUEmbeddingEngine
from proxy.embeddings import space_id_for as _space_id_for

# Cosine distance below which a new rule restates an existing one (QA O4, 2026-10-06).
NEAR_DUPLICATE_DISTANCE = 0.20

logger = logging.getLogger(__name__)

from core.resource_manager import strings
from core.identifiers import safe_char_id


class LoreExtractionWorker:
    """
    Asynchronous worker responsible for extracting, embedding, and persisting 
    GM lore rules based on initial context or periodic chat history review.
    """

    def __init__(self, db_pool: asyncpg.Pool, llm_client=None):
        self.db_pool = db_pool
        # The SHARED scheduled client (OPEN-004): lore runs at background priority and
        # can never overlap an interactive turn on a single-slot backend.
        from proxy.core.llm_scheduler import get_scheduled_client
        self.llm_client = llm_client or get_scheduled_client()
        self.embedding_engine = CPUEmbeddingEngine()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _clean_llm_json_response(self, raw_response: str) -> Any:
        """Strips markdown code blocks and parses JSON safely."""
        cleaned = raw_response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        # sometimes LLM outputs a single dict instead of a list
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return [parsed]
            return parsed
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM JSON output: {e}\nRaw output: {raw_response}")
            return []

    async def _execute_extraction(self, prompt: str, session_id: str, character_id: str, model: str):
        self.logger.info(f"Executing extraction for session={session_id}, character={character_id} using model={model}")
        
        # Stream=False generates a full response via process_token_stream conceptually,
        # but LemonadeLLMClient only has generate_stream.
        # We can collect tokens from the generator.
        # JSON extraction needs no reasoning: with it on, Qwen3.8-27B took ~36 s per call instead of ~7 s.
        # Live test 2026-10-13: LM Studio loaded Gemma with an 8k window, and a fixed
        # max_tokens=8192 on top of the prompt made every extraction fail at once (empty
        # output). Size the output to the window like the chat turn does; switch thinking
        # off with both switches (LM Studio honours only reasoning_effort).
        from config.manager import get_settings_manager
        from proxy.rag import budget
        from proxy.rag.prompt_builder import CognitivePromptBuilder
        from proxy.core import reasoning
        settings = get_settings_manager().get_settings()
        max_tokens = budget.clamp_max_tokens(
            [{"role": "user", "content": prompt}], settings,
            CognitivePromptBuilder().config, model=model, configured_ceiling=8192)
        # Local backends get both switches; a cloud API only the field it always got.
        no_thinking = (reasoning.extra_body("none", settings, str(getattr(self.llm_client, "base_url", "") or ""))
                       or {"chat_template_kwargs": {"enable_thinking": False}})
        generator = self.llm_client.generate_stream(
            prompt=prompt, model=model, temperature=0.2, max_tokens=max_tokens,
            extra_body=no_thinking,
            job_kind="lore", session_id=session_id,
            coalesce_key=f"lore:{session_id}:{character_id}",
        )

        raw_response = ""
        in_reasoning = False
        try:
            async for chunk in generator:
                # The client wraps reasoning_content in <thinking> markers. Skip the reasoning itself, not just
                # the markers; otherwise the prose lands in front of the JSON and parsing fails at char 0.
                if chunk == "<thinking>":
                    in_reasoning = True
                elif chunk == "</thinking>":
                    in_reasoning = False
                elif not in_reasoning:
                    raw_response += chunk
        except Exception as e:
            self.logger.error(f"Error during LLM generation: {e}")
            return

        # Strip <thinking>...</thinking> block from response (LLM often wraps analysis in these)
        import re
        raw_response = re.sub(r'<thinking>.*?</thinking>', '', raw_response, flags=re.DOTALL)
        
        rules = self._clean_llm_json_response(raw_response)
        
        if not rules:
            self.logger.info("No valid rules extracted.")
            return

        self.logger.info(f"Extracted {len(rules)} rules. Vectorizing and persisting...")

        for rule in rules:
            rule_text = rule.get("rule_text")
            rule_type = rule.get("rule_type", "invariant")
            
            if not rule_text:
                continue
                
            if rule_type not in ("invariant", "conditional_trigger", "trigger", "game_over"):
                rule_type = "invariant"
                
            if rule_type == "trigger":
                rule_type = "conditional_trigger"
                
            try:
                emb = await self.embedding_engine.generate_embedding(rule_text)
                emb_str = None if emb is None else "[" + ",".join(map(str, emb)) + "]"
                
                table_name = f"csa_lore_rules_{safe_char_id(character_id)}"
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(strings.get("sql.create_csa_lore_rules_table"), safe_char_id(character_id))
                    
                    # Dedupe within this chat (or against character canon): the same
                    # rule may legitimately exist in two different chats (OPEN-007).
                    existing = await conn.fetchval(
                        strings.get("sql.check_lore_rule_exists").format(table_name=table_name),
                        rule_text, session_id)
                    space_id = await _space_id_for(self.embedding_engine, conn) if emb_str is not None else None
                    # Reworded restatements of a rule we already hold ("Characters can/may
                    # experience sudden bursts of strength") are duplicates too. Measured on
                    # the MHA lorebook: paraphrases <= 0.17, distinct rules >= 0.25 (QA O4).
                    if not existing and emb_str is not None and space_id is not None:
                        existing = await conn.fetchval(
                            strings.get("sql.check_lore_rule_near_duplicate").format(table_name=table_name),
                            emb_str, session_id, space_id, NEAR_DUPLICATE_DISTANCE)
                        if existing:
                            self.logger.info(f"Skipping near-duplicate lore rule: {rule_text[:80]}")
                    if not existing:
                        await conn.execute(
                            f"""
                            INSERT INTO {table_name} (rule_text, rule_type, rule_embedding,
                                                      embedding_space_id, status, session_id, scope)
                            VALUES ($1, $2, $3::vector, $4, 'pending', $5, 'chat');
                            """,
                            rule_text, rule_type, emb_str, space_id, session_id
                        )
            except Exception as e:
                self.logger.error(f"Error persisting rule '{rule_text}': {e}")
                
        self.logger.info(f"Successfully processed extraction for {character_id}")

    async def extract_initial_rules(self, session_id: str, character_id: str, context_text: str, model: str = DEFAULT_CHAT_MODEL):
        prompt = strings.get("rag.lore_extractor.initial_rules_prompt", context_text=context_text)
        await self._execute_extraction(prompt, session_id, character_id, model)

    async def periodic_review_rules(self, session_id: str, character_id: str, recent_messages: List[Dict], model: str = DEFAULT_CHAT_MODEL):
        messages_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_messages])
        prompt = strings.get("rag.lore_extractor.periodic_rules_prompt", recent_messages=messages_str)
        await self._execute_extraction(prompt, session_id, character_id, model)
