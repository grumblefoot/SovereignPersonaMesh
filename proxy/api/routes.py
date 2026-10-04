"""
OpenAI-Compatible API Routes for SPM Proxy (Port 5050).
Emulates /v1/chat/completions endpoint for SillyTavern, handling spatial routing,
sensory gating bypass, RAG retrieval, and real-time monologue stripping over SSE.
"""

import hashlib
import json
import os
import time
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from config.hardware_tiers import get_hardware_config, HardwareTierEnum
from proxy.core.st_parser import parse_sillytavern_context
from config.manager import get_settings_manager
from proxy.core.fifo_queue import InferenceFIFOQueue
from proxy.core.stream_parser import MonologueStreamParser, CLOSE_TAGS
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.rag.retriever import EpisodicRAGRetriever
from proxy.rag.import_worker import BulkImportWorker, get_import_worker, _compute_dynamic_batch_size, BULK_IMPORT_THRESHOLD
from proxy.rag.tier_manager import MemoryTierManager
from proxy.rag.lore_extractor import LoreExtractionWorker
from proxy.backend_client.lemonade_client import LemonadeLLMClient, LLMBackendError, DEFAULT_CHAT_MODEL, SPM_VIRTUAL_MODEL_ID
from proxy.backend_client.evennia_client import EvenniaWorldClient
from scripts.onnx_embedder import CPUEmbeddingEngine
from core.resource_manager import strings

from proxy.core.telemetry import get_telemetry_collector
from core.identifiers import safe_char_id

logger = logging.getLogger(__name__)

router = APIRouter()
_background_tasks = set()

# Module-level db_pool reference — set by tests via set_db_pool()
_db_pool = None
_db_pool_explicitly_set = False


def set_db_pool(pool):
    """Inject a db_pool into the routes module (used by tests)."""
    global _db_pool, _db_pool_explicitly_set
    _db_pool = pool
    _db_pool_explicitly_set = True

# Service components
fifo_queue = InferenceFIFOQueue()
prompt_builder = CognitivePromptBuilder()
lemonade_client = LemonadeLLMClient()
evennia_client = EvenniaWorldClient()
embedder = CPUEmbeddingEngine()



class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_CHAT_MODEL
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    # No default: SillyTavern always sends its own cap, and the old 128000 default made the
    # `or 300` fallback below unreachable while telling the backend to generate without limit.
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True
    stop: Optional[List[str]] = None
    # FR-001 chat identity: these were silently dropped before they were declared here,
    # because model_dump() only contains declared fields.
    session_id: Optional[str] = None
    user: Optional[str] = None


@router.get("/v1/models")
async def list_models():
    """SPM's virtual id plus Lemonade's chat models, so SillyTavern can offer the real choices."""
    ids = [SPM_VIRTUAL_MODEL_ID]
    try:
        backend_ids = await lemonade_client._chat_model_ids()
    except Exception as e:
        logger.warning(f"[SPMProxy] Could not list backend models: {e}")
        backend_ids = None
    ids += backend_ids if backend_ids else [DEFAULT_CHAT_MODEL]
    return {
        "object": "list",
        "data": [{"id": i, "object": "model", "owned_by": "spm" if i == SPM_VIRTUAL_MODEL_ID else "lemonade"} for i in ids],
    }


import re


def _extract_target_char(messages: List[ChatCompletionMessage]) -> str:
    """Extract the target character identifier from system prompts or message metadata."""
    if not messages:
        return "default"
    for msg in reversed(messages):
        if msg.role == "system" and msg.content:
            content = msg.content
            # Pattern 1: [CharName's Personality=...]
            match = re.search(r"\[([A-Za-z0-9_\-\s]+)'s\s+Personality=", content, re.IGNORECASE)
            if match:
                return safe_char_id(match.group(1))
            # Pattern 2: [Character: CharName] or Character: CharName
            match = re.search(r"(?:\[Character:\s*|Character:\s*)([A-Za-z0-9_\-\s]+)(?:\]|\n|$)", content, re.IGNORECASE)
            if match:
                return safe_char_id(match.group(1))
            # Pattern 3: [<CharName>:] or [<CharName>'s ...]
            match = re.search(r"\[([A-Za-z0-9_\-\s]+)(?:'s|:)", content)
            if match:
                char_name = safe_char_id(match.group(1))
                if char_name not in ("scenario", "system", "user", "assistant", "context"):
                    return char_name
        elif msg.name:
            return safe_char_id(msg.name)
    return "default"



async def _gather_public_response(prompt: str, model: str, temperature: float,
                                   max_tokens: int, stop: Optional[list]) -> str:
    """Non-streaming helper: gather all public tokens into a single response string."""
    parser = MonologueStreamParser()
    raw_stream = lemonade_client.generate_stream(
        prompt=prompt, model=model, temperature=temperature,
        max_tokens=max_tokens, stop=stop
    )
    public_chunks = []
    async for chunk in parser.process_token_stream(raw_stream):
        public_chunks.append(chunk)
    inner_mono, public_resp = parser.get_final_buffers()
    logger.info(f"[SPMProxy] Non-streaming turn finished. Public chars: {len(public_resp)}")
    return public_resp


# Shared, un-redactable context blocks (decision 12): ST's Summary and Author's Note go to every
# character's prompt, so by default they are stripped to keep hidden intent out of other characters.
_SHARED_NOTE_REGEX = re.compile(r"^\s*\[?\s*(summary|author'?s\s+note)\s*[:\]]", re.IGNORECASE)


def _assemble_system_prompt(messages: List[ChatCompletionMessage], settings: dict) -> str:
    """Join ALL system messages in order (the old code kept only the first, dropping world info,
    persona and scenario). Summary/Author's Note blocks are stripped unless the
    st_passthrough_shared_notes setting opts in (per decision 12)."""
    passthrough = bool(settings.get("st_passthrough_shared_notes", False))
    parts = []
    for m in messages:
        if m.role != "system" or not m.content:
            continue
        if not passthrough and _SHARED_NOTE_REGEX.match(m.content):
            logger.info("[SPMProxy] Stripped shared Summary/Author's Note block from character prompt.")
            continue
        parts.append(m.content)
    if not parts:
        return "You are a character inside the Sovereign Persona Mesh."
    return "\n\n".join(parts)


def _extract_session_id(request: Request, body: dict) -> str:
    """
    Extract session_id from request with precedence:
    X-SPM-Chat-ID (ST extension, stable per chat) > X-Session-ID > body session_id >
    X-Chat-ID/X-Conversation-ID > legacy st_{user}_{char} pair.
    Implements FR-001 session-bound context isolation.

    The legacy fallback embeds the target character, which gives every group-chat member its
    own world; a content fingerprint replacing it rides on the Sprint 2 perception table.
    """
    spm_chat_id = request.headers.get("X-SPM-Chat-ID")
    # A literal "{{spmChatId}}" means the ST extension failed to load and the macro never resolved.
    if spm_chat_id and not spm_chat_id.startswith("{{"):
        return f"st_chat_{safe_char_id(spm_chat_id, default='unidentified')}"
    header_session = request.headers.get("X-Session-ID") or request.headers.get("x-session-id")
    if header_session:
        return header_session
    body_session = body.get("session_id")
    if body_session:
        return str(body_session)
    chat_id = request.headers.get("X-Chat-ID") or request.headers.get("X-Conversation-ID")
    if chat_id:
        return chat_id

    user_name = body.get("user") or "user"
    msg_objs = [ChatCompletionMessage(**m) if isinstance(m, dict) else m for m in body.get("messages", [])]
    target_char = _extract_target_char(msg_objs)
    return f"st_{user_name}_{target_char}"


async def _check_bulk_import(
    request: ChatCompletionRequest,
    session_id: str,
    db_pool,
) -> bool:
    """
    Detect whether a bulk import is needed.

    Registers the import job synchronously (so the DB row exists immediately),
    then dispatches the actual processing via asyncio.create_task() so that
    response latency stays under the 5 ms SLA.

    Returns True if an import was spawned, False otherwise.
    Bulk import is triggered when:
      - session_id is new (not in spm_chat_imports)
      - message count exceeds BULK_IMPORT_THRESHOLD (10)
    """
    if db_pool is None or getattr(db_pool, "_closed", False):
        return False

    try:
        # Only check on first request to a new session
        worker = BulkImportWorker(db_pool)
        existing = await worker.check_import_status(session_id)
        if existing:
            return False  # Already being imported or completed
    except Exception as e:
        logger.warning(f"[ImportWorker] Check import status skipped: {e}")
        return False

    if len(request.messages) > BULK_IMPORT_THRESHOLD:
        target_char = _extract_target_char(request.messages)
        logger.info(
            f"[ImportWorker] Bulk import detected: "
            f"session={session_id}, messages={len(request.messages)} > {BULK_IMPORT_THRESHOLD}"
        )
        # Register synchronously so DB row is immediately visible
        await worker.register_import_job(
            session_id, target_char, len(request.messages)
        )
        # Spawn background task for actual processing (skip_registration since
        # we already registered above)
        task = asyncio.create_task(
            worker.process_bulk_import_background(
                session_id=session_id,
                character_id=target_char,
                messages=[m.model_dump() for m in request.messages],
                skip_registration=True,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return True

    return False


def _extract_location_from_messages(messages: List[Any], system_prompt: str = "") -> str:
    """Dynamically extract location/room name from system prompt or message history."""
    import re
    full_text = system_prompt + "\n" + "\n".join([getattr(m, 'content', '') or '' for m in messages if hasattr(m, 'content')])
    
    m = re.search(r'(?:Location|Setting|Room|Area):\s*([^\n\.,;\]]+)', full_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    m2 = re.search(r'\[(?:LOC|LOCATION|Setting):\s*([^\]]+)\]', full_text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()

    kw_match = re.search(r'\b(prison|dungeon|cell|cellar|chamber|vault|archive|room|hall|tower|courtyard|castle|tavern|inn|fortress)\b', full_text, re.IGNORECASE)
    if kw_match:
        matched_kw = kw_match.group(1).capitalize()
        if matched_kw.lower() in ["cell", "cellar", "dungeon", "prison"]:
            return "Underground Prison"
        return matched_kw

    return "Starting Location"


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completions endpoint intercepted by SPM Proxy.
    Supports session-bound context isolation via _extract_session_id().
    Triggers async bulk import when > 10 messages detected for a new session.
    """
    t0 = time.time()
    if os.getenv("SPM_LOG_HEADERS") == "1":
        logger.info(f"[SillyIntoSPMLog] Raw request headers: {dict(req.headers)}")
    logger.info(f"[SillyIntoSPMLog] Request payload: {request.model_dump()}")

    # Extract session ID for FR-001 session isolation
    session_id = _extract_session_id(req, request.model_dump())

    # Extract target character identifier
    target_char = _extract_target_char(request.messages)
    last_msg = request.messages[-1] if request.messages else ChatCompletionMessage(role="user", content="")
    user_text = last_msg.content

    # --- FR-002: Bulk Import Detection ---
    is_bulk = await _check_bulk_import(request, session_id, _db_pool)

    # --- Step 1: spatial routing via Evennia ---
    world_res = await evennia_client.submit_action(
        character_id="user",
        action_type="speak",
        raw_text=user_text,
        target_id=target_char,
        session_id=session_id
    )

    # Find sensory consequence for target character
    sensory_feed = user_text
    gating_level = "direct"
    consequences = world_res.get("consequences", [])
    for c in consequences:
        recip = c.get("recipient_id", "").lower()
        if recip == target_char or target_char == "default" or len(consequences) == 1:
            sensory_feed = c.get("sensory_feed", user_text)
            gating_level = c.get("gating_level", "direct")
            break

    # --- Step 2: Observer Inference Gating & Bypass Protocol ---
    system_prompt = next((m.content for m in request.messages if m.role == "system"), "You are Luna.")
    location_name = _extract_location_from_messages(request.messages, system_prompt)
    if not consequences:
        location_name = "[Unmapped - Awaiting GM_ACTION: CREATE_ROOM]"

    telemetry = get_telemetry_collector()
    telemetry.record_request(session_id=session_id, location_name=location_name, gating_level=gating_level, latency=(time.time() - t0) * 1000)

    if gating_level.lower() in ["null", "blackout"]:
        logger.info(f"[SPMProxy] Character {target_char} turn bypassed (gating={gating_level}). Zero inference cost.")
        async def empty_generator():
            chunk = {
                "id": "chatcmpl-spm-bypass",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"*{target_char.replace('_', ' ').title()} hears only muffled sounds from elsewhere...*"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    # --- Step 3: Token Decoupling & Cognitive prompt assembly ---
    frontend_max_tokens = request.max_tokens or 300
    settings = get_settings_manager().get_settings()
    backend_max_tokens = settings.get("backend_max_tokens", 2048)

    system_prompt = _assemble_system_prompt(request.messages, settings)

    # RAG Memory Retrieval (filters out query_text to eliminate regeneration bleed)
    retrieved_memories = []
    if _db_pool and user_text:
        try:
            retriever = EpisodicRAGRetriever(_db_pool)
            query_emb = await embedder.generate_embedding(user_text)
            retrieved_memories = await retriever.retrieve_memories(
                character_id=target_char,
                query_embedding=query_emb,
                top_k=3,
                session_id=session_id,
                query_text=user_text,
            )
        except Exception as e:
            logger.warning(f"[SPMProxy] Memory retrieval skipped: {e}")

    # RAG Lore Retrieval
    retrieved_lore = {"invariants": [], "triggers": []}
    if _db_pool and user_text:
        try:
            retrieved_lore = await retriever.retrieve_lore_rules(
                character_id=target_char,
                query_embedding=query_emb,
            )
        except Exception as e:
            logger.warning(f"[SPMProxy] Lore retrieval skipped: {e}")

    csa_messages = prompt_builder.build_csa_messages(
        system_prompt=system_prompt,
        sensory_feed=sensory_feed,
        retrieved_memories=retrieved_memories,
        chat_history=[{"role": m.role, "content": m.content} for m in request.messages],
        spatial_context=f"Location: {location_name}",
        frontend_max_tokens=frontend_max_tokens,
    )

    # Ensure closing tags are NOT in LLM stop sequence list
    raw_stop = request.stop or ["\nUser:", "\nHuman:", "\n<system>"]
    if isinstance(raw_stop, list):
        stop = [s for s in raw_stop if s not in CLOSE_TAGS]
    else:
        stop = raw_stop
        
    # --- Inject Active Lore ---
    lore_text_parts = []
    for inv in retrieved_lore.get("invariants", []):
        lore_text_parts.append(f"- [Invariant] {inv['rule_text']}")
    for trig in retrieved_lore.get("triggers", []):
        lore_text_parts.append(f"- [Trigger] {trig['rule_text']}")
    
    active_lore_str = ""
    if lore_text_parts:
        active_lore_str = "\n\n[ACTIVE LORE]\n" + "\n".join(lore_text_parts)

    # --- Inject Assistant Prefill (Only if Monologue is enabled) ---
    if prompt_builder.config.inner_monologue_enabled:
        directive = f"""{active_lore_str}

SYSTEM DIRECTIVE: You are the GAME MASTER. You MUST write your internal thoughts strictly inside <think>...</think> tags. Cross-reference the user's input against the ACTIVE LORE.
- If the user violates an Invariant (e.g. hallucinating), note it in your scratchpad.
- If the user violates a Trigger/Game Over rule, note the [RULE VIOLATION] in your scratchpad and issue a [GM WARNING: ...]
- If ANY character (including the user) moves to a new location, you MUST output [GM_ACTION: {{"type": "MOVE", "entity": "...", "room_id": "..."}}] inside your <think> block.
- If a described location doesn't exist, output [GM_ACTION: {{"type": "CREATE_ROOM", "room_id": "...", "name": "...", "desc": "..."}}] inside your <think> block.
- Evaluate if your planned response puppets the user. You MUST NOT describe the user's actions, feelings, or dialogue.

CRITICAL FORMATTING RULE:
After completing your GM scratchpad and GM actions, YOU MUST CLOSE THE TAG AND SEPARATE YOUR DIALOGUE. Output exactly:
</think>

---

After the horizontal rule, switch to the CHARACTER'S PERSPECTIVE.
DO NOT write any more plans, analysis, or 'I need to' notes outside of the <think> tags.
The text after </think> must ONLY be narrative and dialogue.
- For Lore Violations: Forcefully reject the hallucination in your public dialogue.
- For Rule Violations: React appropriately to enforce the rule. Do NOT write the GM Warning in your public dialogue.
- Anti-Puppeting: NEVER act, speak, or think for the user's character. Only describe your own character's actions and the environment."""
        if csa_messages and csa_messages[-1]["role"] == "user":
            csa_messages[-1]["content"] += directive
        else:
            csa_messages.append({"role": "user", "content": directive.strip()})
        csa_messages.append({"role": "assistant", "content": "<think>\n"})
        init_state = 0
    else:
        init_state = 1
    
    logger.info(f"[SPMIntoBackendLog] Sending to Backend (model={request.model}): {csa_messages}")

    # ---- Non-streaming path ----
    if request.stream is False:
        parser = MonologueStreamParser(max_public_tokens=frontend_max_tokens, initial_state=init_state)
        raw_stream = lemonade_client.generate_stream(
            messages=csa_messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=backend_max_tokens,
            stop=stop,
        )
        try:
            async for chunk in parser.process_token_stream(raw_stream):
                pass
        except LLMBackendError as e:
            logger.error(f"[SPMProxy] Backend failure for {target_char}; turn not saved: {e}")
            return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "llm_backend_error"}})
        inner_monologue, public_resp = parser.get_final_buffers()

        logger.info(f"[BackendReturnSPMLog] Monologue: {inner_monologue} | Public: {public_resp}")

        # Dispatch any GM actions found in the monologue sequentially in background
        task = asyncio.create_task(_dispatch_gm_actions(parser, session_id, target_char,
                                                turn_index=sum(1 for m in request.messages if m.role == 'user')))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        # Dispatch Lore Extraction in background
        _dispatch_lore_extraction(request, session_id, target_char, inner_monologue, public_resp)

        if inner_monologue:
            telemetry.push_thought_event(session_id, {
                "session_id": session_id,
                "character": target_char,
                "thought": inner_monologue,
            })

        if _db_pool and public_resp:
            try:
                table_name = f"csa_memory_{safe_char_id(target_char)}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(target_char))
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, user_text
                    )
                    emb = await embedder.generate_embedding(user_text)
                    emb_str = None if emb is None else "[" + ",".join(map(str, emb)) + "]"
                    await conn.execute(
                        f"""
                        INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, public_response, episodic_embedding)
                        VALUES ($1, $2, $3, $4, $5::vector);
                        """,
                        session_id, user_text, inner_monologue, public_resp, emb_str
                    )
            except Exception as e:
                logger.warning(f"[SPMProxy] Failed to persist turn memory: {e}")

        logger.info(f"[SPMReturnSillyLog] Returning to SillyTavern: {public_resp}")
        return JSONResponse(content={
            "id": "chatcmpl-spm-turn",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": public_resp},
                "finish_reason": "stop"
            }]
        })

    # ---- Streaming path ----
    async def sse_event_generator():
        parser = MonologueStreamParser(max_public_tokens=frontend_max_tokens, initial_state=init_state)
        raw_stream = lemonade_client.generate_stream(
            messages=csa_messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=backend_max_tokens,
            stop=stop,
        )

        def _chunk(content):
            return "data: " + json.dumps({
                "id": "chatcmpl-spm-turn",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }) + "\n\n"

        try:
            async for public_chunk in parser.process_token_stream(raw_stream):
                yield _chunk(public_chunk)
        except LLMBackendError as e:
            # Tell the user instead of inventing a reply; skip persistence, GM actions and lore extraction.
            logger.error(f"[SPMProxy] Backend failure for {target_char}; turn not saved: {e}")
            yield _chunk(f"*[SPM: the LLM backend is unavailable ({e}). This turn was not saved.]*")
            yield "data: " + json.dumps({"id": "chatcmpl-spm-turn", "object": "chat.completion.chunk",
                                         "created": int(time.time()), "model": request.model,
                                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        final_chunk = {
            "id": "chatcmpl-spm-turn",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        telemetry.record_request(session_id=session_id, gating_level=gating_level, latency=(time.time() - t0) * 1000)
        inner_monologue, public_resp = parser.get_final_buffers()

        logger.info(f"[BackendReturnSPMLog] Monologue: {inner_monologue} | Public: {public_resp}")
        logger.info(f"[SPMReturnSillyLog] Sent streaming chunks to SillyTavern. Final public response: {public_resp}")

        # Dispatch any GM actions found in the monologue sequentially in background
        task = asyncio.create_task(_dispatch_gm_actions(parser, session_id, target_char,
                                                turn_index=sum(1 for m in request.messages if m.role == 'user')))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        # Dispatch Lore Extraction in background
        _dispatch_lore_extraction(request, session_id, target_char, inner_monologue, public_resp)

        # Push inner monologue to Thought Monitor SSE stream
        if inner_monologue:
            telemetry.push_thought_event(session_id, {
                "session_id": session_id,
                "character": target_char,
                "thought": inner_monologue,
            })

        telemetry.record_turn_trace(session_id, {
            "gating_level": gating_level,
            "target_char": target_char,
            "monologue_len": len(inner_monologue),
            "public_len": len(public_resp),
        })

        # Persist finalized turn without duplicate bleed on regeneration
        if _db_pool and public_resp:
            try:
                table_name = f"csa_memory_{safe_char_id(target_char)}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(target_char))
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, user_text
                    )
                    emb = await embedder.generate_embedding(user_text)
                    emb_str = None if emb is None else "[" + ",".join(map(str, emb)) + "]"
                    await conn.execute(
                        f"""
                        INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, public_response, episodic_embedding)
                        VALUES ($1, $2, $3, $4, $5::vector);
                        """,
                        session_id, user_text, inner_monologue, public_resp, emb_str
                    )
            except Exception as e:
                logger.warning(f"[SPMProxy] Failed to persist turn memory: {e}")

        logger.info(
            f"[SPMProxy] Turn finished for {target_char}. "
            f"Monologue chars: {len(inner_monologue)}, Public chars: {len(public_resp)}"
        )

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")


# ======================================================================
# FR-003: Tiered Data Lifecycle & Cold Storage
# ======================================================================

@router.post("/v1/memories/archive")
async def archive_memories(request: dict):
    """
    Archive old volatile memories to cold .jsonl.gz storage.
    Accepts JSON body with: character_id, session_id, max_records, max_age_days.
    Core memories (is_core_memory = TRUE) are never archived.
    """
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": strings.get("api.errors.db_not_configured")}
        )

    character_id = request.get("character_id", "")
    session_id = request.get("session_id", "default_session")
    max_records = request.get("max_records", 500)
    max_age_days = request.get("max_age_days", 30)

    if not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": strings.get("api.errors.missing_char_id")}
        )

    manager = MemoryTierManager(_db_pool)
    result = await manager.archive_old_memories(
        character_id=character_id,
        session_id=session_id,
        max_records=max_records,
        max_age_days=max_age_days,
    )

    return JSONResponse(content=result)


@router.post("/v1/memories/reconstitute")
async def reconstitute_memory(request: dict):
    """
    Reconstitute a cold archive back into hot memory storage.
    Accepts JSON body with: archive_id, character_id.
    """
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": strings.get("api.errors.db_not_configured")}
        )

    archive_id = request.get("archive_id")
    character_id = request.get("character_id")

    if not archive_id or not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": strings.get("api.errors.missing_archive_args")}
        )

    manager = MemoryTierManager(_db_pool)
    result = await manager.reconstitute_cold_archive(
        archive_id=archive_id,
        character_id=character_id,
    )

    return JSONResponse(content=result)


@router.get("/v1/memories/stats")
async def memory_stats(
    character_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Return tiered memory statistics (hot, warm, cold counts).
    Query params: character_id (required), session_id (optional).
    """
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": strings.get("api.errors.db_not_configured")}
        )

    if not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": strings.get("api.errors.missing_char_id")}
        )

    manager = MemoryTierManager(_db_pool)
    result = await manager.get_tier_stats(
        character_id=character_id,
        session_id=session_id,
    )
    return JSONResponse(content=result)

def _log_task_done(task):
    try:
        task.result()
    except Exception as e:
        logger.error(f"[LoreExtraction] Background task failed: {e}")

_MAIN_PROMPT_NAMES = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*[.\n]", re.IGNORECASE)


def _strip_user_persona(messages: List[ChatCompletionMessage], target_char: str) -> List[ChatCompletionMessage]:
    """Drop SillyTavern's user-persona system message so lore extraction only sees NPC/world context (BUG-008).

    In chat-completion mode ST sends the persona as its own system message ("[Vardus is a tall human...]").
    The user's name comes from ST's main prompt ("...a fictional chat between Arvenia and Vardus.").
    If the names can't be determined, the messages are returned unchanged.
    """
    user_name = None
    for m in messages:
        if m.role == "system" and m.content:
            match = _MAIN_PROMPT_NAMES.search(m.content)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                if safe_char_id(a) == target_char:
                    user_name = b
                elif safe_char_id(b) == target_char:
                    user_name = a
                break
    if not user_name:
        return messages
    opener = re.compile(r"^\s*\[?\s*" + re.escape(user_name) + r"(?:'s)?\b", re.IGNORECASE)
    return [m for m in messages if not (m.role == "system" and m.content and opener.match(m.content))]


def _dispatch_lore_extraction(request, session_id: str, target_char: str, inner_monologue: str, public_resp: str):
    if not _db_pool:
        return
    settings = get_settings_manager().get_settings()
    cadence = int(settings.get("periodic_review_cadence", 3))
    use_alt = settings.get("use_alternate_extraction_model", False)
    alt_model = settings.get("alternate_extraction_model_name", "")
    ext_model = alt_model if (use_alt and alt_model) else request.model
    
    extractor = LoreExtractionWorker(_db_pool)
    actual_messages = [m for m in request.messages if m.content and m.content.strip() not in ("<think>", "</think>")]
    actual_messages = _strip_user_persona(actual_messages, target_char)
    user_messages = [m for m in actual_messages if getattr(m, 'role', '') == 'user']
    user_msg_count = len(user_messages)
    
    assistant_turn = {"role": "assistant", "content": f"<think>\n{inner_monologue}\n</think>\n{public_resp}"}
    full_turn_history = [m.model_dump() for m in actual_messages] + [assistant_turn]
    
    if user_msg_count <= 1:
        task = asyncio.create_task(
            extractor.extract_initial_rules(session_id, target_char, "\n".join(f"{m['role']}: {m.get('content', '')}" for m in full_turn_history), model=ext_model)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_task_done)
    elif user_msg_count > 1 and (user_msg_count - 1) % cadence == 0:
        recent = full_turn_history[-(cadence * 2):]
        task = asyncio.create_task(
            extractor.periodic_review_rules(session_id, target_char, recent, model=ext_model)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_task_done)

def _gm_action_key(session_id: str, turn_index: int, idx: int, action: dict) -> str:
    """Deterministic per (session, turn, action): a regenerate re-derives the same key, so the
    engine can drop the duplicate. The old uuid4 keys made every regenerate re-apply its actions."""
    payload = json.dumps(action, sort_keys=True)
    return "gm-" + hashlib.sha1(f"{session_id}|{turn_index}|{idx}|{payload}".encode()).hexdigest()[:32]


async def _dispatch_gm_actions(parser: MonologueStreamParser, session_id: str, target_char: str,
                               turn_index: int = 0):
    """Extracts GM actions from the parser and dispatches them sequentially in the background."""
    actions = parser.extract_gm_actions()

    for idx, action in enumerate(actions):
        action_type = action.get("type")
        logger.info(f"[GMAction] Dispatching GM Action: {action}")
        try:
            if action_type == "MOVE":
                await evennia_client.move_character(
                    character_id=action.get("entity", target_char),
                    room_id=action.get("room_id", ""),
                    session_id=session_id,
                    idempotency_key=_gm_action_key(session_id, turn_index, idx, action)
                )
            elif action_type == "CREATE_ROOM":
                await evennia_client.create_room(
                    room_id=action.get("room_id", ""),
                    name=action.get("name", "New Room"),
                    desc=action.get("desc", ""),
                    session_id=session_id,
                    idempotency_key=_gm_action_key(session_id, turn_index, idx, action)
                )
            else:
                logger.warning(f"[GMAction] Unrecognized GM action type: {action_type}")
        except Exception as e:
            logger.error(f"[GMAction] Task '{action_type}' failed for session {session_id}: {e}")


@router.get("/v1/imports/status/{session_id}")
async def get_import_status(session_id: str):
    """Check the status of a bulk import job for a session (FR-002)."""
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": strings.get("api.errors.db_not_configured")}
        )
    worker = BulkImportWorker(_db_pool)
    status = await worker.check_import_status(session_id)
    if status is None:
        return JSONResponse(
            status_code=404,
            content={"error": strings.get("api.errors.no_import_job", session_id=session_id)}
        )
    return JSONResponse(content=status)


@router.get("/v1/imports")
async def list_all_imports():
    """List all bulk import jobs (FR-002)."""
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": strings.get("api.errors.db_not_configured")}
        )
    worker = BulkImportWorker(_db_pool)
    imports = await worker.get_all_imports()
    return JSONResponse(content={"imports": imports, "total": len(imports)})
