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
from config.manager import get_settings_manager, resolve_gm_actions_mode
from proxy.core.stream_parser import MonologueStreamParser, CLOSE_TAGS
from proxy.core.sensory_filter import ObserverInferenceGatingFilter
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.rag.retriever import EpisodicRAGRetriever
from proxy.rag.import_worker import BulkImportWorker, get_import_worker, _compute_dynamic_batch_size, BULK_IMPORT_THRESHOLD
from proxy.rag.tier_manager import MemoryTierManager
from proxy.rag.lore_extractor import LoreExtractionWorker
from proxy.backend_client.lemonade_client import LemonadeLLMClient, LLMBackendError, DEFAULT_CHAT_MODEL, SPM_VIRTUAL_MODEL_ID
from proxy.core.llm_scheduler import get_scheduled_client, QueueFull, QueueWaitTimeout, TurnSuperseded
from proxy.core import gm_actions as gm_validation
from proxy.rag import budget
from proxy.gating.action_parser import parse_user_message, parse_message, CONF_TAG
from proxy.gating.world_seed import propose_world_seed
from proxy.gating.perception import gated_history, record_turn_perceptions, render_history_rows
from proxy.backend_client.evennia_client import EvenniaWorldClient
from proxy.embeddings import get_embedding_service, space_id_for
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

# Service components. The LLM client is the SHARED scheduled client (OPEN-004):
# every generation acquires a backend slot; metadata calls pass through.
prompt_builder = CognitivePromptBuilder()
lemonade_client = get_scheduled_client()
evennia_client = EvenniaWorldClient()
embedder = get_embedding_service()



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


_NAME = r"[^\[\]\n:'’]"          # a name: any script, no brackets/newline/colon/apostrophe
_CONTROL_TAGS = ("scenario", "system", "user", "assistant", "context", "scene", "move",
                 "whisper", "shout", "gm_action", "character", "charactername")


def _looks_like_title(text: str) -> bool:
    """SillyTavern's own "Write X's next reply" line is structured, so X may be a longer
    title: scenario cards are named like "My Hero Academia RPG World" (5 words), which
    the 3-word name guard rejected, filing the whole scenario under the shared 'default'
    character (QA F24, 2026-10-06). Still not a sentence: no clause punctuation and no
    "is/are/was" (the persona-description failure mode from F1)."""
    words = text.split()
    if not 0 < len(words) <= 8:
        return False
    if re.search(r"[.;:!?]", text):
        return False
    return not any(w.lower() in ("is", "are", "was", "were") for w in words)


def _looks_like_name(text: str) -> bool:
    """Names are short. A bracketed persona/card DESCRIPTION ("[Vardus is a tall and fit
    human male in his late 20's ...") is not a name (QA 2026-10-05)."""
    return 0 < len(text.split()) <= 3


def _extract_target_char(messages: List[ChatCompletionMessage]) -> str:
    """Extract the target character identifier from system prompts or message metadata.

    SillyTavern's own instruction ("Write <char>'s next reply ...") is authoritative and
    is searched in EVERY system message first: the old reversed scan hit the persona
    block before it and named the character after the user's persona description.
    Names may be in any script ("美 Mei")."""
    if not messages:
        return "default"
    for msg in messages:
        if msg.role == "system" and msg.content:
            match = re.search(r"Write\s+(.+?)['’]s\s+next\s+reply", msg.content, re.IGNORECASE)
            if match and _looks_like_title(match.group(1)):
                return safe_char_id(match.group(1))
    for msg in reversed(messages):
        if msg.role == "system" and msg.content:
            content = msg.content
            # Pattern 1: [CharName's Personality=...]
            match = re.search(rf"\[({_NAME}+?)['’]s\s+Personality=", content, re.IGNORECASE)
            if match and _looks_like_name(match.group(1)):
                return safe_char_id(match.group(1))
            # Pattern 2: [Character: CharName] or Character: CharName
            match = re.search(rf"(?:\[Character:\s*|Character:\s*)({_NAME}+?)(?:\]|\n|$)", content, re.IGNORECASE)
            if match and _looks_like_name(match.group(1)):
                return safe_char_id(match.group(1))
            # Pattern 3: [<CharName>:] or [<CharName>'s ...] — last resort. SPM's own
            # control tags are excluded ("[scene:KEY]" once named a character 'scene'),
            # and so is anything longer than a name.
            match = re.search(rf"\[({_NAME}+?)(?:['’]s|:)", content)
            if match and _looks_like_name(match.group(1)):
                char_name = safe_char_id(match.group(1))
                if char_name not in _CONTROL_TAGS:
                    return char_name
        elif msg.name:
            return safe_char_id(msg.name)
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
            # Pattern 3: "Write X's next reply" — SillyTavern's standard chat
            # instruction. Checked BEFORE the generic bracket pattern: brackets in
            # the same message are often control tags, not names.
            match = re.search(r"Write\s+([A-Za-z0-9_\-\s]+?)'s\s+next\s+reply", content, re.IGNORECASE)
            if match:
                return safe_char_id(match.group(1))
            # Pattern 4: [<CharName>:] or [<CharName>'s ...] — last resort, with SPM's
            # own control tags excluded (a bare "[scene:KEY]" prompt once made every
            # row land under a character literally named 'scene').
            match = re.search(r"\[([A-Za-z0-9_\-\s]+)(?:'s|:)", content)
            if match:
                char_name = safe_char_id(match.group(1))
                if char_name not in ("scenario", "system", "user", "assistant", "context",
                                     "scene", "move", "whisper", "shout", "gm_action",
                                     "character", "charactername"):
                    return char_name
        elif msg.name:
            return safe_char_id(msg.name)
    return "default"



def _extract_persona_display(messages: List[ChatCompletionMessage]) -> str:
    """The persona's name as written ("Vardus"), for labelling perceived lines."""
    for msg in messages:
        if msg.role == "system" and msg.content:
            m = re.search(r"chat\s+between\s+.+?\s+and\s+([^\n.,]+)", msg.content, re.IGNORECASE)
            if m and _looks_like_name(m.group(1)):
                return m.group(1).strip()
    return ""


def _label_persona(text: str, persona: str) -> str:
    """Replace the engine's 'User' LABEL (first occurrence only, never words inside
    the speech) with the persona's name."""
    if not persona or not text:
        return text
    return re.sub(r"\bUser\b", persona, text, count=1)


def _extract_persona_name(messages: List[ChatCompletionMessage]) -> str:
    """The USER-side name from ST's "chat between <char> and <persona>" line.
    GM MOVE proposals name the persona ("Tom"), never the engine id "user";
    the validator maps this alias onto "user" (gm plan A.3)."""
    for msg in messages:
        if msg.role == "system" and msg.content:
            # Any script on either side ("between 美 Mei and Vardus"); the persona is
            # the last name, up to the sentence end.
            m = re.search(r"chat\s+between\s+.+?\s+and\s+([^\n.,]+)", msg.content, re.IGNORECASE)
            if m and _looks_like_name(m.group(1)):
                return m.group(1).strip().lower()
    return ""


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


_ST_CHAT_MARKER_REGEX = re.compile(r"^\[Start a new (?:group )?chat\b[^\]]*\]$", re.IGNORECASE)


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
        if _ST_CHAT_MARKER_REGEX.match(m.content.strip()):
            # SillyTavern's chat-start markers are UI scaffolding, not story. Read
            # literally, "[Start a new group chat. Group members: ...]" made the GM
            # create rooms called "Family Chat" and "Chat Interface" (QA F13).
            continue
        parts.append(m.content)
    if not parts:
        return "You are a character inside the Sovereign Persona Mesh."
    return "\n\n".join(parts)


class _EphemeralTurn(Exception):
    """Internal: short-circuits world writes for quiet/impersonate generations."""


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
                # Gating phase 5 (side channels): imported memories must not carry the
                # user's private 'quoted thoughts' (decision 11).
                # Only what was SAID: system messages (character cards, persona,
                # SillyTavern's instructions) are not memories (QA F11).
                messages=[{**m.model_dump(),
                           "content": _redact_private_spans(m.content)
                           if m.role == "user" and m.content else m.content}
                          for m in request.messages if m.role in ("user", "assistant")],
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
    # The user's turn is the last USER message. In group chats SillyTavern appends its
    # nudge ("[Write the next reply only as X.]") as the final SYSTEM message; taking
    # messages[-1] made characters "hear" the user speak that instruction (QA F16).
    last_msg = next((m for m in reversed(request.messages) if m.role == "user"), None) \
        or ChatCompletionMessage(role="user", content="")
    user_text = last_msg.content

    # B5 (gm_actions_and_lore_scope.md): ST background generations must not touch
    # world state. The extension sends X-SPM-Gen-Type; 'quiet' and 'impersonate'
    # turns still ANSWER (with gated context) but WRITE nothing — no tick, no
    # perception rows, no reply action, no memories, no lore, no GM actions.
    # An unsubstituted "{{spmGenType}}" macro (extension missing) counts as absent.
    gen_type = (req.headers.get("X-SPM-Gen-Type") or "").strip().lower()
    ephemeral = gen_type in ("quiet", "impersonate")
    if ephemeral:
        logger.info(f"[SPMProxy] Ephemeral generation ({gen_type}): nothing will be persisted.")

    # Quiet/impersonate are SillyTavern's own utility generations (summaries, writing
    # the USER's next message). They get NO character framing: wrapping them in the
    # target character's gated prompt + GM scratchpad directive made impersonate
    # return only "[CHARACTER PERSPECTIVE: Vardus]" (QA step A5, 2026-10-05).
    # Passed through as SillyTavern built them; <think> reasoning is still stripped.
    if ephemeral:
        return await _ephemeral_passthrough(request, session_id)

    # --- FR-002: Bulk Import Detection ---
    # Only for a session SPM has never seen (a chat that predates SPM). Once a session
    # has perception rows, the gated log is the authority: a group member joining a
    # long chat used to get the ENTIRE raw transcript imported as her memories,
    # including scenes she never witnessed (QA F11, 2026-10-05).
    _pre_count = sum(1 for m in request.messages if m.role == "user")
    is_new_session = await _session_is_new(session_id, _pre_count)
    is_bulk = await _check_bulk_import(request, session_id, _db_pool) if is_new_session else False

    # --- Step 1: spatial routing via Evennia ---
    # Decision 10 on the live path: the turn id is the user-message count, so a regenerate
    # re-sends the same turn_id and the engine's per-session tick does not advance.
    user_msg_count = sum(1 for m in request.messages if m.role == "user")
    turn_id = f"{session_id}:{user_msg_count}"

    # Sprint 2 chunk 4: deterministic action parsing (decision 11, fail-open). The user's
    # 'quoted thoughts' are PRIVATE: they never reach the engine, other characters'
    # memories, or this character's sensory feed (DESIGN-001's deterministic half).
    parsed_msg = parse_message(user_text)
    parsed_actions = parsed_msg.actions
    # Out-of-character directions (**bold**, ((...)), [OOC: ...]) are the user's
    # instructions to the AI: never perceived by any character, never stored, but
    # delivered to the model for THIS turn as an author's direction (QA F17).
    ooc_directions = parsed_msg.ooc_texts
    observable_actions = [a for a in parsed_actions if a.action_type != "thought"]
    # Narrated movement ("*Lian enters the room, a tea set in hand*") is visible to
    # whoever is present; only explicit [move:PLACE] tags (content = a room name)
    # are hidden. Excluding every move dropped such narration entirely (QA F17).
    _visible = [a for a in observable_actions
                if not (a.action_type == "move" and a.confidence >= CONF_TAG)]
    _speechlike = {"speak", "whisper", "shout"}
    if _visible and all(a.action_type in _speechlike for a in _visible):
        _joined = " ".join(a.content for a in _visible)          # pure speech: as before
    else:
        # Mixed speech and action keeps the distinction: "Yes." *waggles fingers*.
        # Flattening made actions part of what the user SAID (QA F18).
        _joined = " ".join(f'"{a.content}"' if a.action_type in _speechlike else f"*{a.content}*"
                           for a in _visible)
    observable_text = _joined.strip() or (
        user_text if not parsed_actions and not parsed_msg.ooc else "")
    # A message that is ONLY a direction has no in-world action this turn.
    direction_only = not observable_text and bool(ooc_directions)
    whispers = [a for a in observable_actions if a.action_type == "whisper"]
    primary_action = ("whisper" if whispers
                      else "shout" if any(a.action_type == "shout" for a in observable_actions)
                      else "speak")
    move_requests = [a for a in observable_actions if a.action_type == "move" and a.target]

    # Sprint 2 chunk 2 wiring: on a session's FIRST turn, seed the world deterministically
    # ([scene:KEY] tag > keyword template match > generic_void) and place the player and
    # the target character together, so consequences and perception rows flow from turn 1.
    # "First turn" = the first time SPM sees this SESSION, not "one user message": a
    # branch (or a chat that predates SPM) arrives carrying many user messages and was
    # never seeded, so nobody was placed (QA step A4, 2026-10-05). Ephemeral turns
    # (quiet/impersonate) never seed: they write nothing.
    if is_new_session and not ephemeral:
        try:
            seed = propose_world_seed(
                "\n".join(m.content for m in request.messages if m.role == "system" and m.content),
                user_text, characters=["user", target_char])
            await evennia_client.configure_world(
                template_key=seed.template_key, session_id=session_id,
                placements=[{"character_id": pl.character_id, "room_id": pl.room_id}
                            for pl in seed.placements],
                origin="system")
            logger.info(f"[SPMProxy] Seeded world '{seed.template_key}' ({seed.source}) for {session_id}.")
        except Exception as e:
            logger.warning(f"[SPMProxy] World seeding skipped: {e}")
    # Group chats: a character who speaks for the first time AFTER the session was
    # seeded (Mei's mother joining) was never placed in any room, so she perceived
    # nothing (QA 2026-10-05). Place a missing target in the player's room before the
    # player's action, so she hears it; GM actions can move her afterwards.
    if not is_new_session:
        await _ensure_target_placed(session_id, target_char, turn_id)

    # OPEN-010: if the world engine is down, degrade to an ungated turn instead of a raw 500.
    # Roleplay must survive an engine outage; the turn simply has no spatial consequences.
    try:
        if ephemeral or direction_only:
            # No in-world action: ephemeral turns write nothing; a direction-only
            # message is the author talking to the AI, not the user's character acting.
            raise _EphemeralTurn()
        world_res = await evennia_client.submit_action(
            character_id="user",
            action_type=primary_action,
            raw_text=observable_text,
            target_id=(whispers[0].target.lower() if whispers and whispers[0].target
                       else target_char),
            session_id=session_id,
            turn_id=turn_id,
        )
        # Explicit user moves ([move:PLACE] or parsed movement) are applied as mutations;
        # unknown destinations 404 at the engine and are ignored (fail-open).
        for i, mv in enumerate(move_requests):
            room_slug = re.sub(r"[^a-z0-9_]+", "_", mv.target.lower()).strip("_")
            try:
                await evennia_client.move_character(
                    character_id="user", room_id=room_slug, session_id=session_id,
                    idempotency_key=f"user-move:{turn_id}:{i}", origin="user")
            except Exception as mv_exc:
                logger.info(f"[SPMProxy] User move to '{room_slug}' not applied: {mv_exc}")
    except _EphemeralTurn:
        world_res = {"consequences": []}
    except Exception as e:
        logger.error(f"[SPMProxy] World engine unavailable; continuing ungated for session {session_id}: {e}")
        world_res = {"consequences": [], "engine_unavailable": True}

    # Characters perceive the player by persona name ("Vardus"), not the engine id
    # ("User: ..."), in their history and this turn's feed (QA F18).
    _persona = _extract_persona_display(request.messages)
    for _c in world_res.get("consequences", []):
        _c["sensory_feed"] = _label_persona(_c.get("sensory_feed") or "", _persona)

    # Sprint 2 chunk 3: record what every recipient perceived (post-gating) — the
    # per-character gated history is rebuilt from these rows, not the raw transcript.
    if _db_pool and world_res.get("consequences"):
        try:
            async with _db_pool.acquire() as conn:
                await record_turn_perceptions(
                    conn, session_id=session_id,
                    tick=int(world_res.get("action_tick", 0)),
                    turn_id=turn_id, actor_id="user", action_type=primary_action,
                    consequences=world_res["consequences"],
                )
        except Exception as e:
            logger.warning(f"[SPMProxy] Perception recording skipped: {e}")

    # Find sensory consequence for target character
    sensory_feed = observable_text
    gating_level = "direct"
    consequences = world_res.get("consequences", [])
    for c in consequences:
        recip = c.get("recipient_id", "").lower()
        # Only the TARGET's own row counts. The old 'len(consequences) == 1' fallback
        # judged a character by someone else's position: Mei (kitchen) blacked out,
        # so Lian, who had no row at all, inherited Mei's blackout (QA 2026-10-05).
        if recip == target_char or target_char == "default":
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
    settings = get_settings_manager().get_settings()
    # public cap: ST's max_tokens, else the configured default (plan §5).
    frontend_max_tokens = request.max_tokens or int(settings.get("public_output_default", 400) or 400)

    system_prompt = _assemble_system_prompt(request.messages, settings)

    # RAG Memory Retrieval (filters out query_text to eliminate regeneration bleed)
    retrieved_memories = []
    query_emb = None          # defined up front: lore retrieval and the RAG trace read it
    query_space_id = None     # even when the memory block below bails out early
    if _db_pool and user_text:
        try:
            retriever = EpisodicRAGRetriever(_db_pool)
            query_emb = await embedder.generate_embedding(observable_text)
            query_space_id = None
            if query_emb is not None:
                async with _db_pool.acquire() as conn:
                    query_space_id = await space_id_for(embedder, conn)
            retrieved_memories = await retriever.retrieve_memories(
                character_id=target_char,
                query_embedding=query_emb,
                top_k=3,
                max_cosine_distance=_embed_threshold(settings),
                session_id=session_id,
                query_text=user_text,
                embedding_space_id=query_space_id,
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
                max_cosine_distance=_embed_threshold(settings),
                embedding_space_id=query_space_id,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning(f"[SPMProxy] Lore retrieval skipped: {e}")

    # Admin RAG inspector: record what retrieval ACTUALLY returned this turn (it used
    # to display Math.random() numbers). Real distances, scores and the threshold.
    try:
        telemetry.record_turn_trace(session_id, {
            "kind": "rag",
            "character": target_char,
            "turn_id": turn_id,
            "threshold": _embed_threshold(settings),
            "vector_search": query_emb is not None,
            "memories": [{
                "text": str(m.get("sensory_input") or "")[:160],
                "cosine_distance": (round(float(m["cosine_distance"]), 4)
                                    if m.get("cosine_distance") is not None else None),
                "rag_score": (round(float(m["rag_score"]), 4)
                              if m.get("rag_score") is not None else None),
                "core": bool(m.get("is_core_memory")),
            } for m in retrieved_memories],
            "lore_invariants": len(retrieved_lore.get("invariants", [])),
            "lore_triggers": len(retrieved_lore.get("triggers", [])),
        })
    except Exception as e:
        logger.debug(f"[SPMProxy] RAG trace skipped: {e}")

    # Sprint 2 chunk 4: the character's history is what THEY perceived (spm_perception),
    # never the raw transcript — the omniscience fix. Falls back to the raw messages ONLY
    # when the SESSION has no perception rows at all (fresh chat, engine down) or the
    # setting is off. The character's own row count must never drive the fallback: a
    # character behind a closed door has few rows BECAUSE they heard nothing, and handing
    # them the raw transcript at that moment is the exact leak this layer exists to stop
    # (leak suite s2/s8/s11). Raw-transcript fallback still redacts the user's private
    # thoughts (decision 11): without this, turn 1 leaked 'quoted thoughts' through it.
    chat_history = [{"role": m.role, "content": _redact_private_spans(m.content) if m.role == "user" else m.content}
                    for m in request.messages]
    if _db_pool and settings.get("gated_history_enabled", True):
        try:
            async with _db_pool.acquire() as conn:
                percep_rows = await gated_history(conn, session_id=session_id,
                                                  recipient_id=target_char, limit=30)
                session_has_rows = bool(percep_rows) or bool(await conn.fetchval(
                    "SELECT 1 FROM spm_perception WHERE session_id = $1 LIMIT 1",
                    session_id))
            if session_has_rows:
                chat_history = render_history_rows(percep_rows, target_char)
                logger.info(f"[SPMProxy] Gated history in use for {target_char}: "
                            f"{len(chat_history)} perceived turns (raw transcript withheld).")
        except Exception as e:
            logger.warning(f"[SPMProxy] Gated history unavailable, using raw transcript: {e}")

    # Token budget P1 (OPEN-008): fit memories, lore and history into the window.
    # The card and the latest user turn are never trimmed; if even they don't fit,
    # refuse visibly rather than silently truncating the card (decision 16).
    retrieved_memories, retrieved_lore, chat_history, budget_report = budget.allocate(
        settings=settings, hw_config=prompt_builder.config, model=str(request.model),
        system_text=system_prompt, memories=retrieved_memories, lore=retrieved_lore,
        history=chat_history)
    if budget_report.trimmed.get("history") or budget_report.trimmed.get("memories") \
            or budget_report.trimmed.get("lore"):
        logger.info(f"[TokenBudget] {budget_report.trimmed} trimmed; used={budget_report.used}")
    if budget_report.refused:
        logger.error(f"[TokenBudget] turn refused: {budget_report.refusal_reason}")
        notice = f"*[SPM: {budget_report.refusal_reason} This turn was not sent.]*"
        if request.stream is False:
            return JSONResponse(status_code=413, content={
                "error": {"message": budget_report.refusal_reason, "type": "context_overflow"}})

        async def overflow_stream():
            yield "data: " + json.dumps({"id": "chatcmpl-spm-overflow",
                "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request.model, "choices": [{"index": 0,
                "delta": {"content": notice}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(overflow_stream(), media_type="text/event-stream")

    gm_mode = resolve_gm_actions_mode(settings)   # resolves "auto" by backend locality
    csa_messages = prompt_builder.build_csa_messages(
        system_prompt=system_prompt,
        sensory_feed=sensory_feed,
        retrieved_memories=retrieved_memories,
        chat_history=chat_history,
        spatial_context=f"Location: {location_name}",
        frontend_max_tokens=frontend_max_tokens,
        gm_mode=gm_mode,
        max_history=None,     # the allocator above already budgeted the history
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

    # Decision D1 (owner, option A): an author direction goes only to the FIRST
    # character who replies to it in this turn (regenerating that character still gets
    # it). Later responders learn what happened through perception, like anyone else;
    # sending it to all of them leaked the direction to characters elsewhere.
    if ooc_directions and await _turn_answered_by_other(session_id, turn_id, target_char):
        logger.info(f"[SPMProxy] Author direction withheld from {target_char}: "
                    f"another character already answered turn {turn_id}.")
        ooc_directions = []

    # --- Inject Assistant Prefill (Only if Monologue is enabled) ---
    if prompt_builder.config.inner_monologue_enabled:
        # A2 (gm_actions_and_lore_scope.md): the GM_ACTION bullets are assembled by
        # mode, so 'off' spends zero prompt tokens on them and 'move_only' never
        # advertises CREATE_ROOM. The lore/anti-puppeting parts are mode-independent.
        gm_move = ('\n- If ANY character (including the user) moves to a new location, you MUST '
                   'output [GM_ACTION: {"type": "MOVE", "entity": "...", "room_id": "..."}] '
                   'inside your <think> block.')
        gm_create = ('\n- If a described location doesn\'t exist, output [GM_ACTION: '
                     '{"type": "CREATE_ROOM", "room_id": "...", "name": "...", "desc": "..."}] '
                     'inside your <think> block.')
        gm_action_parts = {"off": "", "move_only": gm_move, "full": gm_move + gm_create}[gm_mode]
        if gm_mode != "off":
            gm_action_parts += await _known_rooms_block(session_id, target_char, gm_mode)
        directive = f"""{active_lore_str}

SYSTEM DIRECTIVE: You are the GAME MASTER. You MUST write your internal thoughts strictly inside <think>...</think> tags. Cross-reference the user's input against the ACTIVE LORE.
- If the user violates an Invariant (e.g. hallucinating), note it in your scratchpad.
- If the user violates a Trigger/Game Over rule, note the [RULE VIOLATION] in your scratchpad and issue a [GM WARNING: ...]{gm_action_parts}
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
        directive = _author_direction_block(ooc_directions) + directive
        if csa_messages and csa_messages[-1]["role"] == "user":
            csa_messages[-1]["content"] += directive
        else:
            csa_messages.append({"role": "user", "content": directive.strip()})
        csa_messages.append({"role": "assistant", "content": "<think>\n"})
        init_state = 0
    else:
        block = _author_direction_block(ooc_directions)
        if block:
            if csa_messages and csa_messages[-1]["role"] == "user":
                csa_messages[-1]["content"] += block
            else:
                csa_messages.append({"role": "user", "content": block.strip()})
        init_state = 1
    
    # Token budget P0 (OPEN-008): never ask the backend for more room than the
    # window has left. backend_max_tokens is now a CEILING, not a flat value.
    backend_max_tokens = budget.clamp_max_tokens(
        csa_messages, settings, prompt_builder.config, model=str(request.model))

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
            job_kind="chat",
            session_id=session_id,
        )
        try:
            async for chunk in parser.process_token_stream(raw_stream):
                pass
        except (LLMBackendError, QueueWaitTimeout) as e:
            logger.error(f"[SPMProxy] Backend failure for {target_char}; turn not saved: {e}")
            return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "llm_backend_error"}})
        except QueueFull as e:
            logger.error(f"[SPMProxy] Backend busy for {target_char}; turn not saved: {e}")
            return JSONResponse(status_code=503, headers={"Retry-After": "10"},
                                content={"error": {"message": str(e), "type": "llm_backend_busy"}})
        except TurnSuperseded as e:
            # A newer request for this session replaced this turn (regenerate). The
            # client has abandoned this response; end it quietly without saving.
            logger.info(f"[SPMProxy] Turn superseded for {target_char}: {e}")
            return JSONResponse(status_code=409, content={"error": {"message": str(e), "type": "turn_superseded"}})
        inner_monologue, public_resp = parser.get_final_buffers()

        logger.info(f"[BackendReturnSPMLog] Monologue: {inner_monologue} | Public: {public_resp}")

        if not ephemeral:
            # Dispatch any GM actions found in the monologue sequentially in background
            # GM moves FIRST, then the reply as a world event: the reply narrates the
            # scene after those moves ("Mei re-enters; Lian asks her..."). Run in
            # parallel, the reply was recorded while Mei was still in the kitchen, so
            # she never heard the question (QA F19, 2026-10-05).
            task = asyncio.create_task(_gm_then_reply(
                parser, session_id, target_char, public_resp, turn_id,
                turn_index=user_msg_count, persona=_extract_persona_name(request.messages)))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

            # Dispatch Lore Extraction in background
            if not parser.is_tainted:   # a withheld/tainted reply is never mined for lore
                _dispatch_lore_extraction(request, session_id, target_char, inner_monologue, public_resp,
                                      is_new_session=is_new_session)

        if inner_monologue:
            telemetry.push_thought_event(session_id, {
                "session_id": session_id,
                "character": target_char,
                "thought": inner_monologue,
            })

        if _db_pool and public_resp and not ephemeral and not parser.is_tainted:
            try:
                table_name = f"csa_memory_{safe_char_id(target_char)}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(target_char))
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, observable_text
                    )
                    emb = await embedder.generate_embedding(observable_text)
                    emb_str = None if emb is None else "[" + ",".join(map(str, emb)) + "]"
                    space_id = await space_id_for(embedder, conn) if emb is not None else None
                    await conn.execute(
                        f"""
                        INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, public_response, episodic_embedding, embedding_space_id)
                        VALUES ($1, $2, $3, $4, $5::vector, $6);
                        """,
                        session_id, observable_text, inner_monologue, public_resp, emb_str, space_id
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
            job_kind="chat",
            session_id=session_id,
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
        except TurnSuperseded as e:
            # A newer request for this session replaced this queued turn (regenerate);
            # the client has already abandoned this stream. End it quietly, save nothing.
            logger.info(f"[SPMProxy] Turn superseded for {target_char}: {e}")
            yield "data: [DONE]\n\n"
            return
        except (LLMBackendError, QueueFull, QueueWaitTimeout) as e:
            # Tell the user instead of inventing a reply; skip persistence, GM actions and lore extraction.
            logger.error(f"[SPMProxy] Backend failure/busy for {target_char}; turn not saved: {e}")
            notice = "backend is busy" if isinstance(e, (QueueFull, QueueWaitTimeout)) else f"LLM backend is unavailable ({e})"
            yield _chunk(f"*[SPM: the {notice}. This turn was not saved.]*")
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

        if not ephemeral:
            # Dispatch any GM actions found in the monologue sequentially in background
            # GM moves FIRST, then the reply as a world event: the reply narrates the
            # scene after those moves ("Mei re-enters; Lian asks her..."). Run in
            # parallel, the reply was recorded while Mei was still in the kitchen, so
            # she never heard the question (QA F19, 2026-10-05).
            task = asyncio.create_task(_gm_then_reply(
                parser, session_id, target_char, public_resp, turn_id,
                turn_index=user_msg_count, persona=_extract_persona_name(request.messages)))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

            # Dispatch Lore Extraction in background
            if not parser.is_tainted:   # a withheld/tainted reply is never mined for lore
                _dispatch_lore_extraction(request, session_id, target_char, inner_monologue, public_resp,
                                      is_new_session=is_new_session)

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
        if _db_pool and public_resp and not ephemeral and not parser.is_tainted:
            try:
                table_name = f"csa_memory_{safe_char_id(target_char)}"
                async with _db_pool.acquire() as conn:
                    await conn.execute("SELECT create_csa_memory_table($1);", safe_char_id(target_char))
                    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS public_response TEXT;")
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE session_id = $1 AND LOWER(sensory_input) = LOWER($2);",
                        session_id, observable_text
                    )
                    emb = await embedder.generate_embedding(observable_text)
                    emb_str = None if emb is None else "[" + ",".join(map(str, emb)) + "]"
                    space_id = await space_id_for(embedder, conn) if emb is not None else None
                    await conn.execute(
                        f"""
                        INSERT INTO {table_name} (session_id, sensory_input, inner_monologue, public_response, episodic_embedding, embedding_space_id)
                        VALUES ($1, $2, $3, $4, $5::vector, $6);
                        """,
                        session_id, observable_text, inner_monologue, public_resp, emb_str, space_id
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
    # A cancelled task is expected: the scheduler preempts background lore when chat
    # needs the backend (decision 13). task.result() on it raises CancelledError, a
    # BaseException that escaped this handler as a logged asyncio ERROR (QA F22).
    if task.cancelled():
        logger.info("[LoreExtraction] Background task cancelled (preempted by chat); it re-runs on a later turn.")
        return
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


def _dispatch_lore_extraction(request, session_id: str, target_char: str, inner_monologue: str, public_resp: str,
                              is_new_session: Optional[bool] = None):
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
    # Gating phase 5 (side channels): the extractor must never see the user's private
    # 'quoted thoughts' (decision 11 — private EVERYWHERE, including background jobs).
    # The character's own monologue stays: this extractor runs as that character.
    full_turn_history = [
        {**m.model_dump(),
         "content": _redact_private_spans(m.content) if m.role == "user" else m.content}
        for m in actual_messages
    ] + [assistant_turn]
    
    # Initial extraction on the session's first turn as SPM sees it (a branch carries
    # history, so 'one user message' missed it); periodic review otherwise.
    first_turn = is_new_session if is_new_session is not None else user_msg_count <= 1
    if first_turn:
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

async def _known_rooms_block(session_id: str, target_char: str, gm_mode: str = "full") -> str:
    """Existing rooms for the GM, so it reuses ids instead of inventing duplicates
    ('family_home_kitchen' beside 'kitchen' split one place into two and broke
    gating, QA F15). Only the occupants of the CHARACTER'S OWN room are listed:
    naming everyone's location would tell her where others are without her having
    seen it, which is the omniscience leak again."""
    try:
        snap = await evennia_client.get_snapshot(session_id=session_id, template_key="")
    except Exception:
        return ""
    rooms = snap.get("rooms", [])
    if not rooms:
        return ""
    where = {}
    for o in snap.get("occupants", []):
        where.setdefault(o["room_id"], []).append(o["entity_id"])
    own = next((r for r, who in where.items() if target_char in [w.lower() for w in who]), None)
    lines = []
    for r in rooms:
        label = f"- {r['room_id']} ({r.get('name') or r['room_id']})"
        if r["room_id"] == own:
            present = [w for w in where.get(own, []) if w.lower() != target_char]
            label += " <- you are here" + (f", with: {', '.join(present)}" if present else "")
        lines.append(label)
    rule = ("use these exact room_id values; CREATE_ROOM only for a place not listed, "
            "and never re-create one under a new name" if gm_mode == "full"
            else "use these exact room_id values; never invent new ones")
    return (f"\n- KNOWN ROOMS ({rule}; 'user' is the player):\n" + "\n".join(lines))


# (session, turn) -> the character a direction belongs to: the first to answer it.
# Remembered so that regenerating the first responder still gets the direction even
# after another character has replied in the same turn.
_direction_owner: Dict[tuple, str] = {}


async def _turn_answered_by_other(session_id: str, turn_id: str, target_char: str) -> bool:
    """Decision D1 / option A: True when this turn's direction belongs to a DIFFERENT
    character, i.e. someone else was the first to answer it."""
    key = (session_id, turn_id)
    owner = _direction_owner.get(key)
    if owner is None and _db_pool is not None:
        try:
            async with _db_pool.acquire() as conn:
                owner = await conn.fetchval(
                    "SELECT actor_id FROM spm_perception WHERE session_id = $1 "
                    "AND turn_id LIKE $2 ORDER BY created_at, id LIMIT 1",
                    session_id, f"{turn_id}#reply:%")
        except Exception as e:
            logger.warning(f"[SPMProxy] direction-recipient check failed, delivering: {e}")
    if owner is None:
        owner = target_char            # nobody has answered yet: this character is first
    _direction_owner[key] = owner
    return owner != target_char


def _author_direction_block(directions: List[str]) -> str:
    """The user's out-of-character directions, delivered to the model for this turn."""
    if not directions:
        return ""
    lines = "\n".join(f"- {d}" for d in directions)
    return ("\n\n[AUTHOR'S DIRECTION for this reply: written by the user, out of character. "
            "Follow it when writing the scene. No character said or heard it; never quote it.]\n"
            + lines)


async def _ephemeral_passthrough(request: "ChatCompletionRequest", session_id: str):
    """Quiet/impersonate: SillyTavern's prompt as-is to the backend; nothing persisted.
    Still scheduled (P0 chat lane), still budget-clamped, and reasoning is stripped."""
    settings = get_settings_manager().get_settings()
    messages = [{"role": m.role, "content": m.content} for m in request.messages if m.content]
    # SillyTavern's Response Length is the ceiling here (it is the user's own
    # utility call); the window clamp still applies on top.
    ceiling = request.max_tokens or settings.get("backend_max_tokens", 2048)
    max_tokens = budget.clamp_max_tokens(messages, settings, prompt_builder.config,
                                         model=str(request.model), configured_ceiling=ceiling)

    def _stream():
        return lemonade_client.generate_stream(
            messages=messages, model=request.model,
            temperature=request.temperature or 0.7, max_tokens=max_tokens,
            stop=None, job_kind="chat", session_id=session_id)

    if request.stream is False:
        parser = MonologueStreamParser(max_public_tokens=request.max_tokens or 10_000, initial_state=1)
        try:
            async for _ in parser.process_token_stream(_stream()):
                pass
        except (LLMBackendError, QueueFull, QueueWaitTimeout) as e:
            return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "llm_backend_error"}})
        except TurnSuperseded as e:
            return JSONResponse(status_code=409, content={"error": {"message": str(e), "type": "turn_superseded"}})
        _, public = parser.get_final_buffers()
        logger.info(f"[SPMReturnSillyLog] Ephemeral reply ({len(public)} chars): {public[:500]}")
        return JSONResponse(content={
            "id": "chatcmpl-spm-ephemeral", "object": "chat.completion", "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": public},
                         "finish_reason": "stop"}]})

    async def gen():
        parser = MonologueStreamParser(max_public_tokens=request.max_tokens or 10_000, initial_state=1)

        def chunk(content, finish=None):
            return "data: " + json.dumps({
                "id": "chatcmpl-spm-ephemeral", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "delta": {"content": content} if content else {},
                             "finish_reason": finish}]}) + "\n\n"
        try:
            async for piece in parser.process_token_stream(_stream()):
                yield chunk(piece)
        except TurnSuperseded:
            yield "data: [DONE]\n\n"
            return
        except (LLMBackendError, QueueFull, QueueWaitTimeout) as e:
            yield chunk(f"*[SPM: the LLM backend is unavailable ({e}).]*")
        _, public = parser.get_final_buffers()
        logger.info(f"[SPMReturnSillyLog] Ephemeral reply ({len(public)} chars): {public[:500]}")
        yield chunk("", finish="stop")
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


async def _ensure_target_placed(session_id: str, target_char: str, turn_id: str) -> None:
    """Place the target character in the player's room if the world doesn't have them.
    Idempotent per (session, turn, character); never moves a character already placed."""
    if not target_char or target_char == "default":
        return
    try:
        snap = await evennia_client.get_snapshot(session_id=session_id, template_key="")
        where = {o["entity_id"].lower(): o["room_id"] for o in snap.get("occupants", [])}
        if target_char.lower() in where:
            return
        room = where.get("user") or next((r["room_id"] for r in snap.get("rooms", [])), None)
        if not room:
            return
        await evennia_client.move_character(
            character_id=target_char, room_id=room, session_id=session_id,
            idempotency_key=f"join:{session_id}:{target_char}", origin="system",
            template_key="")
        logger.info(f"[SPMProxy] {target_char} joined session {session_id}: placed in '{room}' with the player.")
    except Exception as e:
        logger.warning(f"[SPMProxy] Could not place joining character {target_char}: {e}")


async def _session_is_new(session_id: str, user_msg_count: int) -> bool:
    """True the first time SPM sees this session: it has no perception rows yet.
    Falls back to the old 'single user message' rule if the DB is unavailable."""
    if _db_pool is None:
        return user_msg_count <= 1
    try:
        async with _db_pool.acquire() as conn:
            seen = await conn.fetchval(
                "SELECT 1 FROM spm_perception WHERE session_id = $1 LIMIT 1", session_id)
        return seen is None
    except Exception as e:
        logger.warning(f"[SPMProxy] new-session check failed, using message count: {e}")
        return user_msg_count <= 1


def _embed_threshold(settings) -> float:
    """Decision 6: per-model cosine-distance cut, calibrated via the bake-off."""
    try:
        return float(settings.get("EMBEDDING_MAX_COSINE_DISTANCE", 0.35))
    except (TypeError, ValueError):
        return 0.35


def _redact_private_spans(text: str) -> str:
    """Strip thought spans from a user message (fallback-history redaction)."""
    if not text:
        return text
    try:
        actions = parse_user_message(text)
    except Exception:
        return text
    observable = [a.content for a in actions if a.action_type != "thought"]
    return " ".join(x for x in observable if x).strip() or "..."


def _gm_action_key(session_id: str, turn_index: int, idx: int, action: dict) -> str:
    """Deterministic per (session, turn, action): a regenerate re-derives the same key, so the
    engine can drop the duplicate. The old uuid4 keys made every regenerate re-apply its actions."""
    payload = json.dumps(action, sort_keys=True)
    return "gm-" + hashlib.sha1(f"{session_id}|{turn_index}|{idx}|{payload}".encode()).hexdigest()[:32]


async def _record_reply_action(session_id: str, target_char: str, public_resp: str,
                               turn_id: str, tainted: bool = False) -> None:
    """The character's public reply is a world action too: other characters perceive it
    per gating, and the character remembers saying it (a 'self' perception row)."""
    if not public_resp:
        return
    if tainted:
        # The stream parser's failsafe fired: the text may contain leaked planning
        # prose. It was already shown to the user (nothing to do there), but it must
        # NOT become world state or the character's memory of what they said —
        # gated history would re-inject it into every later prompt.
        logger.warning(f"[SPMProxy] Reply for {target_char} tainted by monologue bleed; "
                       f"not recorded as a world action (turn {turn_id}).")
        return
    consequences = []
    tick = 0
    try:
        # Same engine turn_id as the user's message: the reply happens in the SAME
        # turn, so it reuses that turn's tick (decision 10: the tick counts user
        # turns). The '#reply:<char>' key below only separates its perception rows; it
        # is PER CHARACTER, because the recorder replaces all rows of a key: with one
        # shared '#reply' key, Mei's reply erased Lian's reply in the same turn (QA F21).
        # With
        # '#reply' here, every reply advanced the clock — tick 3 after 2 messages
        # (QA step A2, 2026-10-05).
        res = await evennia_client.submit_action(
            character_id=target_char, action_type="speak", raw_text=public_resp,
            target_id=None, session_id=session_id, turn_id=turn_id)
        consequences = list(res.get("consequences", []))
        tick = int(res.get("action_tick", 0))
        # A reply is prose with its own dialogue ('Lian paused... "Mei," she said').
        # The engine wraps speech in quotes, which presented the whole paragraph as
        # spoken words in others' histories (F18, reply side). Unwrap when the reply
        # already carries its own quotes or *action* markup.
        if any(ch in public_resp for ch in ('"', '“', '*')):
            for c in consequences:
                feed = c.get("sensory_feed") or ""
                label, sep, rest = feed.partition(': "')
                if sep and rest == public_resp + '"':
                    c["sensory_feed"] = f"{label}: {public_resp}"
    except Exception as e:
        logger.warning(f"[SPMProxy] Reply action not routed (engine?): {e}")
    consequences.append({"recipient_id": target_char, "sensory_feed": public_resp,
                         "gating_level": "self", "distance_ft": 0.0, "barriers": []})
    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                await record_turn_perceptions(conn, session_id=session_id, tick=tick,
                                              turn_id=f"{turn_id}#reply:{target_char}",
                                              actor_id=target_char, action_type="speak",
                                              consequences=consequences)
        except Exception as e:
            logger.warning(f"[SPMProxy] Reply perception recording skipped: {e}")


async def _gm_then_reply(parser: MonologueStreamParser, session_id: str, target_char: str,
                         public_resp: str, turn_id: str, turn_index: int, persona: str) -> None:
    """Apply the turn's GM actions, then record the reply as a world event, in order."""
    try:
        await _dispatch_gm_actions(parser, session_id, target_char,
                                   turn_index=turn_index, persona=persona)
    except Exception as e:  # never let a GM failure lose the reply
        logger.error(f"[GMAction] dispatch failed for {session_id}: {e}")
    await _record_reply_action(session_id, target_char, public_resp, turn_id,
                               tainted=parser.is_tainted)


async def _dispatch_gm_actions(parser: MonologueStreamParser, session_id: str, target_char: str,
                               turn_index: int = 0, persona: str = ""):
    """LLM-proposed world actions, gated and validated before anything hits the engine
    (OPEN-005 / SD-01). Mode off: nothing is dispatched (the stream parser already
    strips stray [GM_ACTION...] text either way). Otherwise the deterministic validator
    in proxy/core/gm_actions.py decides: schema, semantics against a single world
    snapshot, CREATE-then-MOVE ordering, caps, reason-coded rejections."""
    actions = parser.extract_gm_actions()
    if not actions:
        return

    settings = get_settings_manager().get_settings()
    mode = resolve_gm_actions_mode(settings)      # resolves "auto" by backend locality
    if mode == "off":
        logger.debug(f"[GMAction] ignored {len(actions)} action(s): gm_actions_mode=off")
        return

    known_rooms, known_entities, room_count = set(), set(), 0
    try:
        snap = await evennia_client.get_snapshot(session_id=session_id, template_key="")
        known_rooms = {r["room_id"] for r in snap.get("rooms", [])}
        known_entities = {o["entity_id"] for o in snap.get("occupants", [])}
        room_count = len(known_rooms)
    except Exception as e:
        logger.error(f"[GMAction] world snapshot unavailable; dropping {len(actions)} action(s): {e}")
        return

    batch = gm_validation.validate_batch(
        actions,
        mode=mode,
        known_rooms=known_rooms,
        known_entities=known_entities,
        target_char=target_char,
        user_aliases=frozenset({persona} if persona else ()),
        max_per_turn=int(settings.get("gm_actions_max_per_turn", 4)),
        rooms_in_session=room_count,
        max_rooms_per_session=int(settings.get("gm_actions_max_rooms_per_session", 40)),
    )
    for rej in batch.rejections:
        logger.warning(f"[GMAction] rejected ({rej.reason}): {rej.detail} — {rej.action}")

    turn_anchor = str(turn_index)
    for action in batch.accepted:          # CREATE first, then MOVE (validator order)
        key = gm_validation.idempotency_key(session_id, turn_anchor, action)
        logger.info(f"[GMAction] Dispatching validated GM Action: {action.model_dump()}")
        try:
            if isinstance(action, gm_validation.MoveAction):
                await evennia_client.move_character(
                    character_id=action.entity, room_id=action.room_id,
                    session_id=session_id, idempotency_key=key, origin="gm",
                    template_key="")       # "" = the session's live world
            else:
                await evennia_client.create_room(
                    room_id=action.room_id, name=action.name or "New Room",
                    desc=action.desc, session_id=session_id,
                    idempotency_key=key, origin="gm", template_key="")
        except Exception as e:
            logger.error(f"[GMAction] Task '{action.type}' failed for session {session_id}: {e}")


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
