import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from proxy.backend_client.lemonade_client import LemonadeLLMClient, DEFAULT_CHAT_MODEL
from scripts.onnx_embedder import CPUEmbeddingEngine
from proxy.embeddings import space_id_for as _space_id_for
from proxy.rag import chat_rules

# Cosine distance below which a new rule restates an existing one (QA O4, 2026-10-06).
# Kept for the lore-scope SQL tests; the extractor itself now dedupes via
# chat_rules.add_rule (brief B1).
NEAR_DUPLICATE_DISTANCE = 0.20

logger = logging.getLogger(__name__)

from core.resource_manager import strings
from core.identifiers import safe_char_id

# Levels a rule may carry. `location` never lands in spm_chat_rules (it goes on
# the room in the world engine), but the model emits it, so it is valid input.
RULE_LEVELS = ("world", "location", "character_internal", "character_external")
CHARACTER_LEVELS = ("character_internal", "character_external")


def _place_rule(rule: Dict[str, Any], *, character_id: str,
                characters, rooms, default_level: str,
                player_ids=(), from_cards: bool = False) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """Decide where one extracted rule belongs: (level, character_id, room_id), or
    None to drop it.

    A character never reads another's card (owner rule, QA 2026-10-17):
    - a character-level rule about the player (`player_ids`: persona name, player
      name, "user") is dropped when it comes from the cards (`from_cards`, the
      first-turn pass); from the story (periodic review) it becomes what the turn
      character knows about the player, i.e. that character's internal rule;
    - a name that is not a known character (an NPC who has not appeared yet) keeps
      its own id and stays inactive until such a character joins the chat; it is
      never handed to the turn character.

    Pure and unit-testable; the engine is deliberately not an argument — a
    `location` rule whose room cannot be matched (unknown name, no rooms, no
    engine wired in) is stored as `world` (pending, so the owner sees it)
    instead of being lost (brief B1 spec 3).
    """
    level = (rule.get("level") or "").strip().lower()
    if level not in RULE_LEVELS:
        level = default_level

    players = {safe_char_id(str(p)) for p in (player_ids or ()) if str(p).strip()} | {"user"}
    if level not in CHARACTER_LEVELS and _about_player(str(rule.get("rule_text") or ""), players):
        # A world or location rule whose subject is the player ("Changi can shatter
        # magic circles", QA 2026-10-17) would tell everyone what only the people
        # present saw: same handling as a character-level rule about the player.
        return None if from_cards else ("character_internal", character_id, None)

    if level in CHARACTER_LEVELS:
        name = str(rule.get("character") or "").strip()
        if not name:
            # No name given: the character whose turn it was.
            return level, character_id, None
        target = safe_char_id(name)
        last = safe_char_id(name.split()[-1])          # "Gu Changi" -> changi
        if target in players or last in players:
            if from_cards:
                return None
            return "character_internal", character_id, None
        return level, target, None

    if level == "location":
        from proxy.core import gm_planner
        wanted = str(rule.get("location") or "").strip()
        matched = None
        if wanted and rooms:
            if wanted in rooms:
                matched = wanted
            else:
                wanted_id = safe_char_id(wanted)
                matched = next(
                    (rid for rid, name in rooms.items()
                     if gm_planner._same_place(wanted_id, wanted, rid, name or rid)), None)
        if matched is not None and rooms is not None:
            return "location", None, matched
        return "world", None, None

    return "world", None, None


def _about_player(rule_text: str, players) -> bool:
    """True when the rule's subject is the player: one of their names among its first
    three words ("Changi can ...", "Gu Changi's wings ..."). A rule that only mentions
    the player later ("The witchfinders hunt anyone who helps Changi") is not."""
    for word in rule_text.split()[:3]:
        w = re.sub(r"(?:'s|’s)$", "", word.strip(".,;:!?\"'()[]"), flags=re.IGNORECASE)
        if w and w[0].isalpha() and safe_char_id(w) in players:
            return True
    return False


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

    async def _execute_extraction(self, prompt: str, session_id: str, character_id: str, model: str,
                                  characters=(), rooms=None, engine=None,
                                  default_level: str = "world",
                                  player_ids=(), from_cards: bool = False):
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
        from proxy.backend_client import backend_profile
        settings = get_settings_manager().get_settings()
        # Brief W1b: size the output from the window the server reports, like the chat turn.
        win = await backend_profile.window_for(self.llm_client, model)
        max_tokens = budget.clamp_max_tokens(
            [{"role": "user", "content": prompt}], settings,
            CognitivePromptBuilder().config, model=model, configured_ceiling=8192,
            detected=win.tokens, detected_source=win.source)
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

            placed = _place_rule(
                rule, character_id=character_id, characters=characters,
                rooms=rooms, default_level=default_level,
                player_ids=player_ids, from_cards=from_cards)
            if placed is None:
                self.logger.info(f"Dropped a rule about the player from the cards: {rule_text[:80]}")
                continue
            level, rule_char, room_id = placed

            if level == "location" and engine is None:
                # Spec 3: without an engine there is no room to attach to; the
                # rule is still worth keeping, visible to the owner as world.
                level, room_id = "world", None

            if level == "location":
                # Location rules live on the room in the world engine (brief D1),
                # not in spm_chat_rules; no embedding. An engine error skips the
                # rule — extraction must never raise out of a background task.
                if engine is None:
                    self.logger.warning(
                        f"No engine for location rule, skipping: {rule_text[:80]}")
                    continue
                try:
                    await engine.add_room_rule(room_id, rule_text, session_id,
                                               rule_type=rule_type, status="pending",
                                               source="extracted")
                except Exception as e:
                    self.logger.error(
                        f"Error storing room rule '{rule_text}' on {room_id}: {e}")
                continue

            try:
                emb = await self.embedding_engine.generate_embedding(rule_text)
                space_id = None
                async with self.db_pool.acquire() as conn:
                    if emb is not None:
                        space_id = await _space_id_for(self.embedding_engine, conn)
                    # add_rule does the exact/near-duplicate checks itself
                    # (brief A1a): it returns None when the rule restates one
                    # this chat already holds.
                    stored = await chat_rules.add_rule(
                        conn, session_id=session_id, level=level,
                        character_id=rule_char, rule_text=rule_text,
                        rule_type=rule_type, status="pending", source="extracted",
                        embedding=emb, space_id=space_id)
                    if stored is None:
                        self.logger.info(f"Skipping duplicate lore rule: {rule_text[:80]}")
            except Exception as e:
                self.logger.error(f"Error persisting rule '{rule_text}': {e}")
                
        self.logger.info(f"Successfully processed extraction for {character_id}")

    async def extract_initial_rules(self, session_id: str, character_id: str, context_text: str,
                                    model: str = DEFAULT_CHAT_MODEL,
                                    characters=(), rooms=None, engine=None, player_ids=()):
        prompt = strings.get("rag.lore_extractor.initial_rules_prompt",
                             context_text=context_text,
                             characters=_name_list(characters),
                             rooms=_room_list(rooms),
                             player=_player_label(player_ids))
        await self._execute_extraction(prompt, session_id, character_id, model,
                                       characters=characters, rooms=rooms, engine=engine,
                                       default_level="world",
                                       player_ids=player_ids, from_cards=True)

    async def periodic_review_rules(self, session_id: str, character_id: str, recent_messages: List[Dict],
                                    model: str = DEFAULT_CHAT_MODEL,
                                    characters=(), rooms=None, engine=None, player_ids=()):
        messages_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_messages])
        prompt = strings.get("rag.lore_extractor.periodic_rules_prompt",
                             recent_messages=messages_str,
                             characters=_name_list(characters),
                             rooms=_room_list(rooms),
                             player=_player_label(player_ids))
        await self._execute_extraction(prompt, session_id, character_id, model,
                                       characters=characters, rooms=rooms, engine=engine,
                                       default_level="character_internal",
                                       player_ids=player_ids, from_cards=False)


def _player_label(player_ids) -> str:
    names = [str(p).strip() for p in (player_ids or ()) if str(p).strip() and str(p).lower() != "user"]
    return ", ".join(dict.fromkeys(names)) or "the user"


def _name_list(characters) -> str:
    names = [str(c) for c in (characters or ()) if c]
    return ", ".join(names) if names else "none yet"


def _room_list(rooms) -> str:
    if not rooms:
        return "none yet"
    return ", ".join(f"{name or rid} ({rid})" for rid, name in rooms.items())
