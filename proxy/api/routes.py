"""
OpenAI-Compatible API Routes for SPM Proxy (Port 5050).
Emulates /v1/chat/completions endpoint for SillyTavern, handling spatial routing,
sensory gating bypass, RAG retrieval, and real-time monologue stripping over SSE.
"""

import json
import time
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from config.hardware_tiers import get_hardware_config, HardwareTierEnum
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


def set_db_pool(pool):
    """Inject a db_pool into the routes module (used by tests)."""
    global _db_pool
    _db_pool = pool

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
    model: str = "google/gemma-4-26B-A4B-it"
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
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


def _extract_target_char(messages: List[ChatCompletionMessage]) -> str:
    """Extract the target character identifier from messages."""
    for msg in reversed(messages):
        if msg.role == "system" and "Character:" in msg.content:
            target_char = msg.content.split("Character:")[1].split("\n")[0].strip().lower()
            return target_char
        elif msg.name:
            return msg.name.lower()
    return "luna"


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
    # Fallback to user + target character identifier for SillyTavern stability
    user_name = body.get("user") or "user"
    messages = body.get("messages", [])
    target_char = "default"
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content", "")
            if "You are " in content:
                parts = content.split("You are ")
                if len(parts) > 1:
                    target_char = parts[1].split(".")[0].split(",")[0].strip().lower().replace(" ", "_")
                    break
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
    if db_pool is None:
        return False

    # Only check on first request to a new session
    worker = BulkImportWorker(db_pool)
    existing = await worker.check_import_status(session_id)
    if existing:
        return False  # Already being imported or completed

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


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completions endpoint intercepted by SPM Proxy.
    Supports session-bound context isolation via _extract_session_id().
    Triggers async bulk import when > 10 messages detected for a new session.
    """
    t0 = time.time()

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
        if c.get("recipient_id") == target_char:
            sensory_feed = c.get("sensory_feed", user_text)
            gating_level = c.get("gating_level", "direct")
            break

    # --- Step 2: Observer Inference Gating & Bypass Protocol ---
    telemetry = get_telemetry_collector()
    telemetry.record_request(session_id=session_id, gating_level=gating_level, latency=(time.time() - t0) * 1000)

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

    # --- Step 3: Build cognitive prompt ---
    system_prompt = "You are Luna, an intelligent character inside the Sovereign Persona Mesh."
    for m in request.messages:
        if m.role == "system":
            system_prompt = m.content
            break

    formatted_prompt = prompt_builder.build_csa_prompt(
        system_prompt=system_prompt,
        sensory_feed=sensory_feed,
        retrieved_memories=[],
        chat_history=[{"role": m.role, "content": m.content} for m in request.messages],
        spatial_context="Location: The Cellar"
    )

    stop = request.stop or ["</ctrl94>", "\nUser:"]

    # --- Step 4: Routing by stream flag ---
    if not request.stream:
        public_resp = await fifo_queue.enqueue_and_execute(
            _gather_public_response,
            prompt=formatted_prompt,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096,
            stop=stop,
        )
        telemetry.record_request(session_id=session_id, gating_level=gating_level, latency=(time.time() - t0) * 1000)
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
        parser = MonologueStreamParser()
        raw_stream = lemonade_client.generate_stream(
            prompt=formatted_prompt,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096,
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
