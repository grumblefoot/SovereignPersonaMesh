import asyncio
import json
import logging
from typing import List, Dict, Any

import asyncpg

from proxy.backend_client.lemonade_client import LemonadeLLMClient
from scripts.onnx_embedder import CPUEmbeddingEngine

logger = logging.getLogger(__name__)

from core.resource_manager import strings


class LoreExtractionWorker:
    """
    Asynchronous worker responsible for extracting, embedding, and persisting 
    GM lore rules based on initial context or periodic chat history review.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.llm_client = LemonadeLLMClient()
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
        generator = self.llm_client.generate_stream(prompt=prompt, model=model, temperature=0.2, max_tokens=8192)
        
        raw_response = ""
        try:
            async for chunk in generator:
                if chunk not in ("<thinking>", "</thinking>"):
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
                emb_str = "[" + ",".join(map(str, emb)) + "]"
                
                table_name = f"csa_lore_rules_{character_id.lower()}"
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(strings.get("sql.create_csa_lore_rules_table"), character_id.lower())
                    
                    # Check if rule exists
                    existing = await conn.fetchval(strings.get("sql.check_lore_rule_exists").format(table_name=table_name), rule_text)
                    if not existing:
                        await conn.execute(
                            f"""
                            INSERT INTO {table_name} (rule_text, rule_type, rule_embedding, status)
                            VALUES ($1, $2, $3::vector, 'pending');
                            """,
                            rule_text, rule_type, emb_str
                        )
            except Exception as e:
                self.logger.error(f"Error persisting rule '{rule_text}': {e}")
                
        self.logger.info(f"Successfully processed extraction for {character_id}")

    async def extract_initial_rules(self, session_id: str, character_id: str, context_text: str, model: str = "google/gemma-4-26B-A4B-it"):
        prompt = strings.get("rag.lore_extractor.initial_rules_prompt", context_text=context_text)
        await self._execute_extraction(prompt, session_id, character_id, model)

    async def periodic_review_rules(self, session_id: str, character_id: str, recent_messages: List[Dict], model: str = "google/gemma-4-26B-A4B-it"):
        messages_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_messages])
        prompt = strings.get("rag.lore_extractor.periodic_rules_prompt", recent_messages=messages_str)
        await self._execute_extraction(prompt, session_id, character_id, model)
