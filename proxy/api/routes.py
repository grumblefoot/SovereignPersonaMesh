"""
OpenAI-Compatible API Routes for SPM Proxy (Port 5050).
Emulates /v1/chat/completions endpoint for SillyTavern, handling spatial routing,
sensory gating bypass, RAG retrieval, and real-time monologue stripping over SSE.
"""

import json
import time
import asyncio
import logging
import uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from config.hardware_tiers import get_hardware_config, HardwareTierEnum
from config.manager import get_settings_manager
from proxy.core.fifo_queue import InferenceFIFOQueue
from proxy.core.stream_parser import MonologueStreamParser
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.rag.retriever import EpisodicRAGRetriever
from proxy.rag.import_worker import BulkImportWorker, get_import_worker, _compute_dynamic_batch_size, BULK_IMPORT_THRESHOLD
from proxy.rag.tier_manager import MemoryTierManager
from proxy.backend_client.lemonade_client import LemonadeLLMClient
from proxy.backend_client.evennia_client import EvenniaWorldClient
from scripts.onnx_embedder import CPUEmbeddingEngine

from proxy.core.telemetry import get_telemetry_collector

logger = logging.getLogger(__name__)

router = APIRouter()

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

def _dispatch_gm_actions(parser: MonologueStreamParser, session_id: str, target_char: str):
    """Extracts GM actions from the parser and dispatches them asynchronously."""
    actions = parser.extract_gm_actions()
    for action in actions:
        action_type = action.get("type")
        logger.info(f"[GMAction] Dispatching GM Action: {action}")
        if action_type == "MOVE":
            asyncio.create_task(
                evennia_client.move_character(
                    character_id=action.get("entity", target_char),
                    room_id=action.get("room_id", ""),
                    session_id=session_id,
                    idempotency_key=str(uuid.uuid4())
                )
            )
        elif action_type == "CREATE_ROOM":
            asyncio.create_task(
                evennia_client.create_room(
                    room_id=action.get("room_id", ""),
                    name=action.get("name", "New Room"),
                    desc=action.get("desc", ""),
                    session_id=session_id,
                    idempotency_key=str(uuid.uuid4())
                )
            )


class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "google/gemma-4-26B-A4B-it"
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 128000
    stream: Optional[bool] = True
    stop: Optional[List[str]] = None


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "spm-sovereign-mesh", "object": "model", "owned_by": "spm"},
            {"id": "google/gemma-4-26B-A4B-it", "object": "model", "owned_by": "spm"}
        ]
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
                return match.group(1).strip().lower()
            # Pattern 2: [Character: CharName] or Character: CharName
            match = re.search(r"(?:\[Character:\s*|Character:\s*)([A-Za-z0-9_\-\s]+)(?:\]|\n|$)", content, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
            # Pattern 3: [<CharName>:] or [<CharName>'s ...]
            match = re.search(r"\[([A-Za-z0-9_\-\s]+)(?:'s|:)", content)
            if match:
                char_name = match.group(1).strip().lower()
                if char_name not in ("scenario", "system", "user", "assistant", "context"):
                    return char_name
        elif msg.name:
            return msg.name.strip().lower()
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


def _extract_session_id(request: Request, body: dict) -> str:
    """
    Extract session_id from request with precedence:
    X-Session-ID header > body session_id > X-Chat-ID > user/character pair.
    Implements FR-001 session-bound context isolation.
    """
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
        asyncio.create_task(
            worker.process_bulk_import_background(
                session_id=session_id,
                character_id=target_char,
                messages=[m.model_dump() for m in request.messages],
                skip_registration=True,
            )
        )
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
    logger.info(f"[SillyIntoSPMLog] Request payload: {request.model_dump()}")

    # Extract session ID for FR-001 session isolation
    session_id = _extract_session_id(req, request.model_dump())

    # Extract target character identifier
    target_char = _extract_target_char(request.messages)
    last_msg = request.messages[-1] if request.messages else ChatCompletionMessage(role="user", content="")
    user_text = last_msg.content

    # --- FR-002: Bulk Import Detection ---
    await _check_bulk_import(request, session_id, _db_pool)

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
                    "delta": {"content": "*Luna hears muffled sounds from another room...*"},
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

    system_prompt = "You are Luna, an intelligent character inside the Sovereign Persona Mesh."
    for m in request.messages:
        if m.role == "system":
            system_prompt = m.content
            break

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
        spatial_context="Location: The Cellar",
        frontend_max_tokens=frontend_max_tokens,
    )

    # Ensure </thinking> is NOT in LLM stop sequence list
    raw_stop = request.stop or ["\nUser:", "\nHuman:", "\n<system>"]
    if isinstance(raw_stop, list):
        stop = [s for s in raw_stop if s != "</thinking>"]
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
- If the user violates a Trigger/Game Over rule (e.g. attacking), note the [RULE VIOLATION] in your scratchpad. If a warning is required, you MUST write a warning addressed to the player INSIDE your scratchpad using exactly this format: [GM WARNING: your warning message to the player here]

CRITICAL FORMATTING RULE:
After completing your GM scratchpad, YOU MUST CLOSE THE TAG AND SEPARATE YOUR DIALOGUE. Output exactly:
</think>

---

After the horizontal rule, switch to the CHARACTER'S PERSPECTIVE.
- For Lore Violations: Forcefully reject the hallucination in your public dialogue.
- For Rule Violations: React appropriately to enforce the rule. Do NOT write the GM Warning in your public dialogue; the system will extract it from your scratchpad automatically."""
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
        async for chunk in parser.process_token_stream(raw_stream):
            pass
        inner_monologue, public_resp = parser.get_final_buffers()

        logger.info(f"[BackendReturnSPMLog] Monologue: {inner_monologue} | Public: {public_resp}")

        # Dispatch any GM actions found in the monologue
        _dispatch_gm_actions(parser, session_id, target_char)

        if inner_monologue:
            telemetry.push_thought_event(session_id, {
                "session_id": session_id,
                "character": target_char,
                "thought": inner_monologue,
            })

        if _db_pool and public_resp:
            try:
                table_name = f"csa_memory_{target_char.lower()}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", target_char.lower())
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, user_text
                    )
                    emb = await embedder.generate_embedding(user_text)
                    emb_str = "[" + ",".join(map(str, emb)) + "]"
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

        async for public_chunk in parser.process_token_stream(raw_stream):
            chunk_data = {
                "id": "chatcmpl-spm-turn",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": public_chunk}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

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

        # Dispatch any GM actions found in the monologue
        _dispatch_gm_actions(parser, session_id, target_char)

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
                table_name = f"csa_memory_{target_char.lower()}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", target_char.lower())
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, user_text
                    )
                    emb = await embedder.generate_embedding(user_text)
                    emb_str = "[" + ",".join(map(str, emb)) + "]"
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
            content={"error": "Database pool not configured"}
        )

    character_id = request.get("character_id", "")
    session_id = request.get("session_id", "default_session")
    max_records = request.get("max_records", 500)
    max_age_days = request.get("max_age_days", 30)

    if not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": "character_id is required"}
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
            content={"error": "Database pool not configured"}
        )

    archive_id = request.get("archive_id")
    character_id = request.get("character_id")

    if not archive_id or not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": "archive_id and character_id are required"}
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
            content={"error": "Database pool not configured"}
        )

    if not character_id:
        return JSONResponse(
            status_code=400,
            content={"error": "character_id is required"}
        )

    manager = MemoryTierManager(_db_pool)
    result = await manager.get_tier_stats(
        character_id=character_id,
        session_id=session_id,
    )
    return JSONResponse(content=result)

def _dispatch_gm_actions(parser: MonologueStreamParser, session_id: str, target_char: str):
    """Extracts GM actions from the parser and dispatches them asynchronously."""
    actions = parser.extract_gm_actions()
    
    async def safe_execute(coro, action_type):
        try:
            await coro
        except Exception as e:
            logger.error(f"[GMAction] Task '{action_type}' failed for session {session_id}: {e}")

    for action in actions:
        action_type = action.get("type")
        logger.info(f"[GMAction] Dispatching GM Action: {action}")
        if action_type == "MOVE":
            asyncio.create_task(
                safe_execute(
                    evennia_client.move_character(
                        character_id=action.get("entity", target_char),
                        room_id=action.get("room_id", ""),
                        session_id=session_id,
                        idempotency_key=str(uuid.uuid4())
                    ),
                    "MOVE"
                )
            )
        elif action_type == "CREATE_ROOM":
            asyncio.create_task(
                safe_execute(
                    evennia_client.create_room(
                        room_id=action.get("room_id", ""),
                        name=action.get("name", "New Room"),
                        desc=action.get("desc", ""),
                        session_id=session_id,
                        idempotency_key=str(uuid.uuid4())
                    ),
                    "CREATE_ROOM"
                )
            )
        else:
            logger.warning(f"[GMAction] Unrecognized GM action type: {action_type}")


@router.get("/v1/imports/status/{session_id}")
async def get_import_status(session_id: str):
    """Check the status of a bulk import job for a session (FR-002)."""
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database pool not configured"}
        )
    worker = BulkImportWorker(_db_pool)
    status = await worker.check_import_status(session_id)
    if status is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No import job found for session={session_id}"}
        )
    return JSONResponse(content=status)


@router.get("/v1/imports")
async def list_all_imports():
    """List all bulk import jobs (FR-002)."""
    if _db_pool is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database pool not configured"}
        )
    worker = BulkImportWorker(_db_pool)
    imports = await worker.get_all_imports()
    return JSONResponse(content={"imports": imports, "total": len(imports)})
