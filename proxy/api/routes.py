"""
OpenAI-Compatible API Routes for SPM Proxy (Port 5050).
Emulates /v1/chat/completions endpoint for SillyTavern, handling spatial routing,
sensory gating bypass, RAG retrieval, and real-time monologue stripping over SSE.
"""

import json
import time
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from config.hardware_tiers import get_hardware_config, HardwareTierEnum
from proxy.core.fifo_queue import InferenceFIFOQueue
from proxy.core.stream_parser import MonologueStreamParser
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.rag.retriever import EpisodicRAGRetriever
from proxy.backend_client.lemonade_client import LemonadeLLMClient
from proxy.backend_client.evennia_client import EvenniaWorldClient
from scripts.onnx_embedder import CPUEmbeddingEngine

logger = logging.getLogger(__name__)

router = APIRouter()

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
    model: str
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


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completions endpoint intercepted by SPM Proxy.
    """
    logger.info(f"[SPMProxy] Received chat completion request ({len(request.messages)} messages)...")

    # Extract target character identifier
    last_msg = request.messages[-1] if request.messages else ChatCompletionMessage(role="user", content="")
    user_text = last_msg.content
    target_char = "luna"  # Default fallback character

    # Check last user message for target character mention or system prompt identity
    for msg in reversed(request.messages):
        if msg.role == "system" and "Character:" in msg.content:
            target_char = msg.content.split("Character:")[1].split("\n")[0].strip().lower()
            break
        elif msg.name:
            target_char = msg.name.lower()
            break

    session_id = "session_abc123"

    # Step 1: Submit action to Evennia World Engine
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

    # Check Observer Inference Gating & Bypass Protocol (Null/Blackout)
    if gating_level.lower() in ["null", "blackout"]:
        logger.info(f"[SPMProxy] Character {target_char} turn bypassed (gating={gating_level}). Zero inference cost.")
        # Return empty streaming response or ambient status chunk
        async def empty_generator():
            chunk = {
                "id": "chatcmpl-spm-bypass",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": "*Luna hears muffled sounds from another room...*"}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    # Step 2: Generate query vector on CPU threads (AVX-512 offloading)
    query_vector = await embedder.generate_embedding(user_text)

    # Step 3: Build Cognitive Prompt
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

    # Step 4: Stream turn execution via FIFO Queue & Monologue Parser
    async def sse_event_generator():
        parser = MonologueStreamParser()
        raw_stream = lemonade_client.generate_stream(
            prompt=formatted_prompt,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096
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

        inner_monologue, public_resp = parser.get_final_buffers()
        logger.info(f"[SPMProxy] Turn finished for {target_char}. Monologue chars: {len(inner_monologue)}, Public chars: {len(public_resp)}")

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
