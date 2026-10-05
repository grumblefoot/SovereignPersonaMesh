"""
FastAPI Server for Evennia World State Engine Liaison Interface (Port 4005).
Provides headless endpoints for action evaluation, spatial state queries, and tick lock management.
Uses HybridWorldBuilder for dynamic world state and SpatialConstraintsMatrix for deterministic gating.
"""

import time
import json
import asyncio
import logging
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import asyncpg

from .models import (
    ActionPayload, ActionResponse, SensoryConsequence, GatingLevel,
    WorldStateQuery, CharacterWorldState, SessionLockPayload, RoomMetadata,
    ActionType, BarrierType, TickRequest, TickResponse
)
from .spatial_matrix import SpatialConstraintsMatrix
from .session_lock import SessionLockManager, LockError
from .hybrid_builder import HybridWorldBuilder
from core.resource_manager import strings
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(_app):
    # startup_event / shutdown_event are defined below; looked up when the app starts.
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(title="Evennia World State Engine Liaison API", version="0.2.0", lifespan=_lifespan)

# ── Internal state ──────────────────────────────────────────────────────
lock_manager = SessionLockManager()
world_builder = HybridWorldBuilder()
class AppState:
    def __init__(self):
        # Legacy global tick. Kept only so older callers/tests that poke the
        # attribute keep working; it is NO LONGER the source of truth
        # (decision 10): the authoritative clock is per-session below.
        self.action_tick_counter: int = 1420
        # Decision 10: per-session tick and turn_id -> tick map, mirroring
        # proxy/engine/in_process.py's _SessionState. A new turn_id advances
        # the session tick by 1; a repeated turn_id (regeneration) reuses it.
        self.session_ticks: Dict[str, int] = {}
        self.session_turn_ticks: Dict[str, Dict[str, int]] = {}
        self._db_pool = None
        self.current_world: Dict[str, RoomMetadata] = {}
        self.room_to_template: Dict[str, str] = {}
        self.session_worlds: Dict[str, Dict[str, Dict[str, RoomMetadata]]] = {}
        # B5: per-session FIFO of seen idempotency keys -> {digest, message}.
        self.idempotency_seen: Dict[str, "OrderedDict[str, Dict[str, str]]"] = {}
        # A2: per-session stateful edges (doors etc.): (a,b) sorted tuple -> edge dict.
        self.session_edges: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        # A2: per-session mutation audit (origin tracking), FIFO-capped.
        self.mutation_log: Dict[str, List[Dict[str, Any]]] = {}

app_state = AppState()

IDEMPOTENCY_KEY_CAP = 512
MUTATION_LOG_CAP = 256


def _log_mutation(session_id: str, kind: str, origin: str, detail: Dict[str, Any]) -> None:
    log = app_state.mutation_log.setdefault(session_id, [])
    log.append({"kind": kind, "origin": origin, **detail})
    while len(log) > MUTATION_LOG_CAP:
        log.pop(0)


def _edge_key(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


def _session_edge(session_id: str, a: str, b: str) -> Optional[Dict[str, Any]]:
    return app_state.session_edges.get(session_id, {}).get(_edge_key(a, b))


def _session_tick(session_id: str) -> int:
    """Current tick value for a session (0 before its first advance)."""
    return app_state.session_ticks.get(session_id, 0)


def _advance_session_tick(session_id: str) -> int:
    """Advance a session's clock by one and return the new tick."""
    tick = app_state.session_ticks.get(session_id, 0) + 1
    app_state.session_ticks[session_id] = tick
    return tick


def _tick_for_turn(session_id: str, turn_id: str) -> Tuple[int, bool]:
    """Resolve the tick for a turn_id (decision 10).

    Returns (tick, advanced). A new turn_id advances the session tick by 1;
    a repeated turn_id returns the same tick with advanced=False, so
    regenerations never move the clock.
    """
    turns = app_state.session_turn_ticks.setdefault(session_id, {})
    if turn_id in turns:
        return turns[turn_id], False
    tick = _advance_session_tick(session_id)
    turns[turn_id] = tick
    return tick, True


def _resolve_action_tick(session_id: str, turn_id: Optional[str]) -> int:
    """Tick an action should carry.

    With turn_id: idempotent per turn (regenerations reuse the tick).
    Without: legacy behavior — advance once per call — so old callers that
    never send turn ids keep seeing a strictly incrementing tick.
    """
    if turn_id is None:
        return _advance_session_tick(session_id)
    tick, _advanced = _tick_for_turn(session_id, turn_id)
    return tick


def _payload_digest(payload: BaseModel) -> str:
    import hashlib, json as _json
    body = payload.model_dump(exclude={"idempotency_key"})
    return hashlib.sha1(_json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _check_duplicate(session_id: str, key: Optional[str], digest: Optional[str] = None) -> Optional[str]:
    """Return the stored message if this idempotency key was already applied.

    A repeated key with a DIFFERENT payload digest is a conflict (HTTP 409), matching
    the engine contract's MutationConflict semantics."""
    if not key:
        return None
    seen = app_state.idempotency_seen.get(session_id, {}).get(key)
    if seen is None:
        return None
    if digest is not None and seen.get("digest") not in (None, digest):
        raise HTTPException(status_code=409, detail=f"idempotency_key '{key}' was already used for a different mutation")
    return seen.get("message")


_last_digest: Dict[str, Optional[str]] = {"v": None}


def _record_idempotency(session_id: str, key: Optional[str], message: str) -> None:
    """Remember an applied idempotency key, FIFO-capped per session."""
    if not key:
        return
    seen = app_state.idempotency_seen.get(session_id)
    if seen is None:
        seen = app_state.idempotency_seen[session_id] = OrderedDict()
    seen[key] = {"digest": _last_digest.get("v"), "message": message}
    seen.move_to_end(key)
    while len(seen) > IDEMPOTENCY_KEY_CAP:
        seen.popitem(last=False)

import os
# Database config
DB_CONFIG = {
    "user": os.environ.get("SPM_DB_USER", "spm_user"),
    "password": os.environ.get("SPM_DB_PASSWORD", "spm_secure_password"),
    "database": os.environ.get("SPM_DB_NAME", "litellm_postgres"),
    "host": os.environ.get("SPM_DB_HOST", "localhost"),
    "port": int(os.environ.get("SPM_DB_PORT", 5432))
}


def _ensure_world(template_key: str = "dynamic", session_id: str = "default_session") -> Dict[str, RoomMetadata]:
    """Ensure session-scoped world state matches the requested template. Returns the world dict for the session."""
    if session_id not in app_state.session_worlds:
        app_state.session_worlds[session_id] = {}
    if template_key not in app_state.session_worlds[session_id]:
        # Deep copy so sessions never share RoomMetadata objects (B3: a move in
        # one session must not mutate another session's world).
        fresh = world_builder.instantiate_world(template_key)
        app_state.session_worlds[session_id][template_key] = {
            rid: RoomMetadata(**r.model_dump()) for rid, r in fresh.items()
        }
    # Also sync the legacy app_state.current_world for backward compatibility
    
    if not app_state.current_world:
        app_state.current_world = app_state.session_worlds[session_id][template_key]
    return app_state.session_worlds[session_id][template_key]


def _get_session_world(session_id: str, template_key: str = "dynamic") -> Optional[Dict[str, RoomMetadata]]:
    """Get the world dict for a session. Returns None if the session doesn't exist (caller should call _ensure_world)."""
    if session_id not in app_state.session_worlds:
        return None
    return app_state.session_worlds[session_id].get(template_key)


# ── Database Persistence Helpers ────────────────────────────────────────

async def _persist_room(session_id: str, template_key: str, room_id: str, room: RoomMetadata):
    """Persist a single room state to PostgreSQL."""
    if not app_state._db_pool:
        return
    try:
        async with app_state._db_pool.acquire() as conn:
            await conn.execute(strings.get("sql.persist_room"), session_id, template_key, room_id, room.model_dump_json(), app_state.action_tick_counter)
    except Exception as e:
        logging.error(f"Failed to persist room {room_id}: {e}")

async def _log_objective_action(session_id: str, action_tick: int, actor_id: str, location_id: str, action_type: str, raw_event: str):
    """Log an objective action to PostgreSQL."""
    if not app_state._db_pool:
        return
    try:
        async with app_state._db_pool.acquire() as conn:
            await conn.execute(strings.get("sql.log_objective_action"), session_id, action_tick, actor_id, location_id, action_type, raw_event)
    except Exception as e:
        logging.error(f"Failed to log objective action: {e}")

# ── Health check ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Quick readiness probe."""
    return {
        "status": "ok",
        # Decision 10: report the default_session's session tick so existing
        # readiness checks keep seeing a tick field.
        "tick": _session_tick("default_session"),
        "template": list(app_state.current_world.keys()) if app_state.current_world else "none",
        "uptime_seconds": round(time.time() - app.state.start_time, 1),
        "active_sessions": len(app_state.session_worlds),
    }


# ── Clock ───────────────────────────────────────────────────────────────

@app.post("/api/v1/world/tick", response_model=TickResponse)
async def advance_tick(payload: TickRequest):
    """
    Advance the session clock once per NEW turn_id (decision 10).

    A repeated turn_id (regeneration, swipe, retry) returns the same tick with
    advanced=false, so callers can drive the clock idempotently.
    """
    tick, advanced = _tick_for_turn(payload.session_id, payload.turn_id)
    return TickResponse(tick=tick, turn_id=payload.turn_id, advanced=advanced)


# ── Action evaluation ───────────────────────────────────────────────────

@app.post("/api/v1/world/action", response_model=ActionResponse)
async def submit_action(payload: ActionPayload, background_tasks: BackgroundTasks):
    """
    Evaluates physical intentions (speak|whisper|shout|move|manipulate).
    Returns action_tick and sensory feeds for recipient characters based on
    real room positions, distances, and the SpatialConstraintsMatrix.
    Supports session-scoped world state for FR-001 isolation.
    With turn_id, the tick advances at most once per turn (decision 10).
    """
    # Per-session clock (decision 10). turn_id present -> idempotent per turn;
    # absent -> legacy advance-per-call so old callers keep working.
    action_tick = _resolve_action_tick(payload.session_id, payload.turn_id)

    # Ensure session-scoped world is loaded
    template_key = getattr(payload, "template_key", "dynamic")
    _ensure_world(template_key, payload.session_id)
    world = _get_session_world(payload.session_id, template_key)
    if not world:
        _ensure_world()
        # NOTE: legacy fallback — only reachable when the session world could
        # not be created; reads the global read-alias world.
        world = app_state.current_world

    # Find which room the actor is in
    actor_room = _find_actor_room(payload.character_id, payload.session_id, template_key)
    loc_id = actor_room if actor_room else "unknown"

    background_tasks.add_task(
        _log_objective_action,
        payload.session_id,
        action_tick,
        payload.character_id,
        loc_id,
        payload.action_type.value,
        payload.raw_text
    )

    consequences: list[SensoryConsequence] = []
    seen_ids: set[str] = set()

    for room_id, room in world.items():
        for char_id in room.present_characters:
            if char_id in seen_ids:
                continue
            if char_id == payload.character_id:
                continue  # the actor does not perceive their own action as a consequence
            seen_ids.add(char_id)
            # Case-insensitive: clients send display names ("Mira"), rooms hold ids ("mira").
            is_target = (payload.target_id is not None
                         and char_id.lower() == payload.target_id.lower())

            # Determine if recipient is in the same room as the actor
            same_room = (actor_room is not None and
                         room_id == actor_room)
            dist, barriers = _compute_distance_and_barriers(
                actor_room, room_id, payload.action_type, world=world,
                session_id=payload.session_id,
            )

            gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
                distance_ft=dist,
                barriers=barriers,
                action_type=payload.action_type,
                raw_text=payload.raw_text,
                actor_id=payload.character_id,
                recipient_id=char_id,
                is_target=is_target,
                session_id=payload.session_id,
                action_tick=action_tick,
                shout=(payload.action_type == ActionType.SHOUT),
            )

            if True:  # BLACKOUT recipients are included with an empty feed (PRD 4.1.1: the proxy's zero-inference bypass needs to see them)
                if gating == GatingLevel.BLACKOUT:
                    feed = ""
                consequences.append(SensoryConsequence(
                    recipient_id=char_id,
                    sensory_feed=feed,
                    gating_level=gating,
                    distance_ft=dist,
                    barriers=barriers,
                ))

    # NOTE: The former "always include Seamus / upstairs / tavern characters with a
    # fabricated DEGRADED muffled feed" block (B7) has been removed. Sensory
    # consequences now come only from real spatial evaluation; distance propagation
    # across rooms is Sprint 2 work.

    return ActionResponse(
        success=True,
        action_tick=action_tick,
        consequences=consequences,
    )


# ── World state query ───────────────────────────────────────────────────

@app.get("/api/v1/world/state", response_model=CharacterWorldState)
async def query_world_state(character_id: str, session_id: str = "default_session", template_key: str = "dynamic"):
    """
    Queries local room metadata for any character (lighting, exits, nearby entities, distances).
    Session-scoped state per FR-001.
    """
    char_id_lower = character_id.lower()
    template_key = template_key

    world = _get_session_world(session_id, template_key)
    if not world:
        _ensure_world(template_key, session_id)
        world = _get_session_world(session_id, template_key)

    # Find the character's current room in the session-scoped world
    char_room = _find_actor_room(char_id_lower, session_id, template_key)
    if char_room is None:
        # Character not in any tracked room — return a default state
        default_room = RoomMetadata(
            room_id="unknown",
            room_name="Unknown Location",
            description="No room assigned.",
            lighting="normal",
            exits=[],
            present_characters=[],
            nearby_objects=[],
        )
        return CharacterWorldState(
            character_id=char_id_lower,
            current_room=default_room,
            gating_level=GatingLevel.BLACKOUT,
            sensory_feed=strings.get("app.no_room", char_id_lower=char_id_lower),
            distances={},
        )

    # Look up the room in session-scoped world first, then fall back to legacy app_state.current_world
    room = world.get(char_room)
    if room is None:
        room = app_state.current_world.get(char_room)  # NOTE: global read alias — room not in the session's world.
    if room is None:
        room = RoomMetadata(
            room_id="unknown", room_name="Unknown Location",
            description="No room assigned.", lighting="normal",
            exits=[], present_characters=[], nearby_objects=[],
        )
    distances = _compute_all_distances(char_id_lower, session_id, template_key)

    return CharacterWorldState(
        character_id=char_id_lower,
        current_room=room,
        gating_level=GatingLevel.DIRECT,
        sensory_feed=strings.get("app.inside_room", room_name=room.room_name, description=room.description),
        distances=distances,
        flavor_text=room.flavor_text,
    )


# ── Session lock ────────────────────────────────────────────────────────

@app.post("/api/v1/world/lock")
async def handle_session_lock(payload: SessionLockPayload):
    """Session & Tick Lock endpoint to prevent race conditions during turn generation."""
    if payload.lock_action == "acquire":
        try:
            token = await lock_manager.acquire_lock(payload.session_id)
            return {"success": True, "lock_token": token}
        except LockError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif payload.lock_action == "release":
        if not payload.lock_token:
            raise HTTPException(status_code=400, detail="lock_token required for release")
        released = lock_manager.release_lock(payload.session_id, payload.lock_token)
        return {"success": released}
    else:
        raise HTTPException(status_code=400, detail="Invalid lock_action")


# ── Dynamic Room Creation ─────────────────────────────────────────────

class CreateRoomPayload(BaseModel):
    """Payload to create a new room on the fly."""
    room_id: str
    room_name: str
    description: str
    template_key: str = "dynamic"
    session_id: str = "default_session"
    idempotency_key: Optional[str] = None
    origin: str = "system"  # gm | system | user (A2: mutation provenance)

@app.post("/api/v1/world/rooms")
async def create_room(payload: CreateRoomPayload, background_tasks: BackgroundTasks):
    """Create a new room dynamically and add it to the session world."""
    # B5/A2: honor idempotency keys — replay returns the original result; a different
    # payload under the same key is a 409 conflict.
    digest = _payload_digest(payload)
    dup = _check_duplicate(payload.session_id, payload.idempotency_key, digest)
    if dup is not None:
        return {
            "success": True,
            "duplicate": True,
            "message": f"Duplicate request (idempotency key already applied): {dup}",
            "room_id": payload.room_id,
        }

    _ensure_world(payload.template_key, payload.session_id)
    world = _get_session_world(payload.session_id, payload.template_key)
    
    if payload.room_id in world:
        raise HTTPException(status_code=409, detail=f"Room '{payload.room_id}' already exists.")
        
    new_room = RoomMetadata(
        room_id=payload.room_id,
        room_name=payload.room_name,
        description=payload.description,
        lighting="normal",
        exits=[],
        present_characters=[],
        nearby_objects=[]
    )
    world[payload.room_id] = new_room

    room_msg = f"Room '{payload.room_name}' created successfully."
    _last_digest["v"] = digest
    _record_idempotency(payload.session_id, payload.idempotency_key, room_msg)
    _log_mutation(payload.session_id, "CREATE_ROOM", payload.origin, {"room_id": payload.room_id})

    background_tasks.add_task(_persist_room, payload.session_id, payload.template_key, payload.room_id, new_room)
    
    return {
        "success": True,
        "message": room_msg,
        "room_id": payload.room_id
    }


# ── Character management ────────────────────────────────────────────────

class CharacterMovePayload(BaseModel):
    """Move a character to a room within the active template."""
    character_id: str
    room_id: str
    template_key: str = "dynamic"
    session_id: str = "default_session"
    idempotency_key: Optional[str] = None
    origin: str = "system"  # gm | system | user (A2: mutation provenance)


class CharacterResponse(BaseModel):
    """Generic response for character operations."""
    success: bool
    message: str
    duplicate: bool = False          # A2: idempotent replay marker
    origin: Optional[str] = None     # A2: echoed mutation provenance
    character_id: Optional[str] = None
    room_id: Optional[str] = None


@app.post("/api/v1/world/characters", response_model=CharacterResponse)
async def add_character_to_world(payload: CharacterMovePayload, background_tasks: BackgroundTasks):
    """Add a character to a specific room in the world template."""
    # Ensure the template exists
    if payload.template_key not in world_builder.templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{payload.template_key}' not found",
        )

    room = world_builder.get_room(payload.template_key, payload.room_id)
    if room is None:
        raise HTTPException(
            status_code=404,
            detail=f"Room '{payload.room_id}' not found in template '{payload.template_key}'",
        )

    added = world_builder.add_character_to_room(
        payload.template_key, payload.room_id, payload.character_id,
    )

    # Also update the active world if the room is in app_state.current_world
    # NOTE: global read alias — current_world mirrors the first-loaded world for
    # legacy callers; the session world is kept in sync below via _persist_room.
    if app_state.current_world and payload.room_id in app_state.current_world:
        _remove_character_from_all_rooms(payload.character_id, payload.session_id)
        if payload.character_id not in app_state.current_world[payload.room_id].present_characters:
            app_state.current_world[payload.room_id].present_characters.append(payload.character_id)

    session_id = payload.session_id
    background_tasks.add_task(_persist_room, session_id, payload.template_key, payload.room_id, app_state.current_world[payload.room_id] if payload.room_id in app_state.current_world else room)

    return CharacterResponse(
        success=True,
        message=f"Character '{payload.character_id}' added to room '{payload.room_id}'",
        character_id=payload.character_id,
        room_id=payload.room_id,
    )


@app.delete("/api/v1/world/characters/{character_id}", response_model=CharacterResponse)
async def remove_character_from_world(character_id: str, background_tasks: BackgroundTasks, template_key: str = "dynamic", session_id: str = "default_session"):
    """Remove a character from all rooms in a template."""
    modified_rooms = set()
    for room_id in world_builder.templates.get(template_key, {}):
        if character_id in world_builder.templates[template_key][room_id].present_characters:
            world_builder.remove_character_from_room(template_key, room_id, character_id)
            modified_rooms.add((template_key, room_id))

    target_worlds = [app_state.current_world]  # NOTE: global read alias (legacy mirror), session worlds appended.
    if session_id in app_state.session_worlds:
        for tmpl_dict in app_state.session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for r_id, r in list(w.items()):
            if character_id in r.present_characters:
                r.present_characters = [c for c in r.present_characters if c != character_id]
                modified_rooms.add((template_key, r_id))
                
    for tmpl_key, r_id in modified_rooms:
        room_obj = None
        if session_id in app_state.session_worlds and tmpl_key in app_state.session_worlds[session_id] and r_id in app_state.session_worlds[session_id][tmpl_key]:
            room_obj = app_state.session_worlds[session_id][tmpl_key][r_id]
        elif r_id in app_state.current_world:  # NOTE: global read alias (legacy mirror).
            room_obj = app_state.current_world[r_id]  # NOTE: global read alias (legacy mirror).
        if room_obj:
            background_tasks.add_task(_persist_room, session_id, tmpl_key, r_id, room_obj)

    return CharacterResponse(
        success=True,
        message=f"Character '{character_id}' removed from all rooms in '{template_key}'",
        character_id=character_id,
    )


@app.get("/api/v1/world/characters")
async def list_characters(template_key: str = "dynamic", session_id: Optional[str] = None):
    """List all characters in a template with their current rooms."""
    if template_key not in world_builder.templates and template_key != "dynamic":
        raise HTTPException(status_code=404, detail=f"Template '{template_key}' not found")

    world = None
    if session_id:
        world = _get_session_world(session_id, template_key)
        
    if world is None:
        world = world_builder.templates.get(template_key, {})

    result: List[Dict[str, Any]] = []
    for room_id, room in world.items():
        for char_id in room.present_characters:
            result.append({
                "character_id": char_id,
                "room_id": room_id,
                "room_name": room.room_name,
            })
    return result


@app.post("/api/v1/world/move", response_model=CharacterResponse)
async def move_character(payload: CharacterMovePayload, background_tasks: BackgroundTasks):
    """Move an existing character from their current room to a new one."""
    session_id = payload.session_id

    # B5/A2: honor idempotency keys; conflicting reuse is a 409.
    digest = _payload_digest(payload)
    dup = _check_duplicate(session_id, payload.idempotency_key, digest)
    if dup is not None:
        return CharacterResponse(
            success=True,
            duplicate=True,
            message=f"Duplicate request (idempotency key already applied): {dup}",
            character_id=payload.character_id,
            room_id=payload.room_id,
        )
    
    # A2: the session world is authoritative — destination must exist there.
    world = _ensure_world(payload.template_key, session_id)
    if payload.room_id not in world:
        raise HTTPException(status_code=404, detail=f"Room '{payload.room_id}' not found in session '{session_id}'")

    # Track which rooms were modified to persist them
    modified_rooms = set()
    # Remove from all rooms first, then add to destination
    for tmpl_key in world_builder.templates:
        for room_id in list(world_builder.templates[tmpl_key].keys()):
            if payload.character_id in world_builder.templates[tmpl_key][room_id].present_characters:
                world_builder.remove_character_from_room(tmpl_key, room_id, payload.character_id)
                modified_rooms.add((tmpl_key, room_id))
            
    # Remove from session worlds
    target_worlds = [app_state.current_world]  # NOTE: global read alias (legacy mirror), session worlds appended.
    if session_id in app_state.session_worlds:
        for tmpl_dict in app_state.session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for r_id, r in list(w.items()):
            if payload.character_id in r.present_characters:
                r.present_characters.remove(payload.character_id)
                modified_rooms.add((payload.template_key, r_id))

    # A2: write the SESSION world (authoritative); never the builder templates —
    # template writes leaked placements across sessions (B6's sibling).
    if payload.character_id not in world[payload.room_id].present_characters:
        world[payload.room_id].present_characters.append(payload.character_id)
    if app_state.current_world and payload.room_id in app_state.current_world:  # NOTE: global read alias (legacy mirror).
        if payload.character_id not in app_state.current_world[payload.room_id].present_characters:  # NOTE: global read alias (legacy mirror).
            app_state.current_world[payload.room_id].present_characters.append(payload.character_id)  # NOTE: global read alias (legacy mirror).
            
    modified_rooms.add((payload.template_key, payload.room_id))
    
    for tmpl_key, r_id in modified_rooms:
        room_obj = None
        if session_id in app_state.session_worlds and tmpl_key in app_state.session_worlds[session_id] and r_id in app_state.session_worlds[session_id][tmpl_key]:
            room_obj = app_state.session_worlds[session_id][tmpl_key][r_id]
        elif r_id in app_state.current_world:  # NOTE: global read alias (legacy mirror).
            room_obj = app_state.current_world[r_id]  # NOTE: global read alias (legacy mirror).
        if room_obj:
            background_tasks.add_task(_persist_room, session_id, tmpl_key, r_id, room_obj)

    move_msg = f"Character '{payload.character_id}' moved to room '{payload.room_id}'"
    _last_digest["v"] = digest
    _record_idempotency(session_id, payload.idempotency_key, move_msg)
    _log_mutation(session_id, "MOVE", payload.origin, {"character_id": payload.character_id, "room_id": payload.room_id})

    return CharacterResponse(
        success=True,
        message=move_msg,
        character_id=payload.character_id,
        room_id=payload.room_id,
    )


# ── World configuration ─────────────────────────────────────────────────

class PlacementPayload(BaseModel):
    character_id: str
    room_id: str


class WorldConfigPayload(BaseModel):
    """Switch the active world to a different template."""
    template_key: str
    flavor_text: Optional[str] = None
    session_id: str = "default_session"
    placements: List[PlacementPayload] = []
    origin: str = "system"


@app.post("/api/v1/world/configure", response_model=CharacterResponse)
async def configure_world(payload: WorldConfigPayload):
    """Load a different room template as the active world."""
    if payload.template_key not in world_builder.templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{payload.template_key}' not found",
        )
    # B4: key the world by the real session id, not the template key.
    session_id_for_config = payload.session_id
    already_configured = payload.template_key in app_state.session_worlds.get(
        session_id_for_config, {})
    if already_configured:
        # Re-configure of a live session world is NON-destructive: keep every room,
        # occupant and runtime edge exactly as they are. (The proxy re-sends configure
        # on what it thinks is turn 1 — after a proxy restart, or when a world was
        # seeded directly on the engine — and rebuilding here silently un-placed every
        # character the proxy didn't know about: leak suite s3/s6/s7/s10.)
        world_inst = app_state.session_worlds[session_id_for_config][payload.template_key]
        if payload.flavor_text:
            # New flavor text on a re-configure is an explicit request — apply it
            # (review C1.2a); rooms, occupants and edges still stay untouched.
            for rm in world_inst.values():
                rm.flavor_text = payload.flavor_text
    else:
        # Same per-session deep copy as _ensure_world: instantiate_world's copies share
        # present_characters lists with the template (shallow model_copy).
        world_inst = {rid: RoomMetadata(**r.model_dump())
                      for rid, r in world_builder.instantiate_world(payload.template_key).items()}
        if payload.flavor_text:
            for rm in world_inst.values():
                rm.flavor_text = payload.flavor_text
        app_state.session_worlds.setdefault(session_id_for_config, {})[payload.template_key] = world_inst

        # Try to load existing rooms for this session from database
        if app_state._db_pool:
            try:
                async with app_state._db_pool.acquire() as conn:
                    rows = await conn.fetch("SELECT room_id, room_data FROM world_state_sessions WHERE session_id = $1 AND template_key = $2", session_id_for_config, payload.template_key)
                    for row in rows:
                        room_id = row['room_id']
                        room_data = json.loads(row['room_data'])
                        app_state.session_worlds[session_id_for_config][payload.template_key][room_id] = RoomMetadata(**room_data)
            except Exception as e:
                logging.error(f"Failed to load existing world state: {e}")

    app_state.current_world = app_state.session_worlds[session_id_for_config][payload.template_key]

    # A2: optional seed placements — validate every room first, apply only if all valid.
    world_now = app_state.session_worlds[session_id_for_config][payload.template_key]
    bad = [pl.room_id for pl in payload.placements if pl.room_id not in world_now]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown room(s) in placements: {bad}; nothing was placed")
    # Case-insensitive like whisper targets (review C1.1): the live world holds "mira",
    # and a placement for "Mira" must not create a second occupant.
    placed_anywhere = {c.lower() for rm in world_now.values() for c in rm.present_characters}
    for pl in payload.placements:
        if pl.character_id.lower() in placed_anywhere:
            # Already in the world (possibly another room): a configure placement is a
            # spawn point, never a teleport — moving is /world/move's job.
            continue
        world_now[pl.room_id].present_characters.append(pl.character_id)
        placed_anywhere.add(pl.character_id.lower())
        _log_mutation(session_id_for_config, "PLACE", payload.origin, {"character_id": pl.character_id, "room_id": pl.room_id})

    return CharacterResponse(
        success=True,
        message=f"World loaded: template '{payload.template_key}'" + (f" with {len(payload.placements)} placement(s)" if payload.placements else ""),
        origin=payload.origin,
    )


@app.get("/api/v1/world/templates")
async def list_templates():
    """List all available world templates."""
    return {"templates": world_builder.list_templates()}


@app.delete("/api/v1/world/admin/reset")
async def reset_world_state(session_id: Optional[str] = None):
    """Factory reset. With ?session_id=X, clear ONLY that session (A2, adapter gap 9);
    without, clear everything as before."""
    if session_id is not None:
        for store in (app_state.session_worlds, app_state.idempotency_seen,
                      app_state.session_ticks, app_state.session_turn_ticks,
                      app_state.session_edges, app_state.mutation_log):
            store.pop(session_id, None)
        return {"status": "success", "message": f"Session '{session_id}' world state cleared"}
    app_state.session_worlds.clear()
    app_state.idempotency_seen.clear()
    app_state.session_ticks.clear()
    app_state.session_turn_ticks.clear()
    app_state.session_edges.clear()
    app_state.mutation_log.clear()

    app_state.current_world = {}
    return {"status": "success", "message": "In-memory world state cache cleared"}

# ── A2: snapshot + stateful barriers ────────────────────────────────────

@app.get("/api/v1/world/snapshot")
async def world_snapshot(session_id: str = "default_session", template_key: str = "dynamic"):
    """Full session graph for the proxy's perception layer (adapter gap 1): rooms,
    edges (explicit stateful edges first, exit-derived defaults for the rest),
    occupants and the per-session tick. Create-on-read via _ensure_world.

    template_key="" means "whatever this session is actually running" — callers like
    the GM-action validator don't track the template and must not create a parallel
    'dynamic' world beside a configured one."""
    if not template_key:
        template_key = next(iter(app_state.session_worlds.get(session_id, {})), "dynamic")
    world = _ensure_world(template_key, session_id)
    rooms, occupants, edges, seen = [], [], [], set()
    for rid, room in world.items():
        rooms.append({
            "room_id": rid, "name": room.room_name, "description": room.description,
            "flavor_text": getattr(room, "flavor_text", "") or "",
            "present_characters": list(room.present_characters),
            "nearby_objects": list(room.nearby_objects),
            "exits": list(room.exits),
        })
        for ch in room.present_characters:
            occupants.append({"entity_id": ch, "room_id": rid, "kind": "character", "posture": []})
    for key, edge in app_state.session_edges.get(session_id, {}).items():
        a, b = key
        seen.add(key)
        edges.append({"a": a, "b": b, "barrier": edge.get("barrier", "none"),
                      "state": edge.get("state", "closed"),
                      "distance_ft": float(edge.get("distance_ft", 15.0))})
    for rid, room in world.items():
        for ex in room.exits:
            key = _edge_key(rid, ex)
            if key in seen or ex not in world:
                continue
            seen.add(key)
            # Exit-derived default mirrors _compute_distance_and_barriers' heuristic.
            edges.append({"a": key[0], "b": key[1], "barrier": "closed_door",
                          "state": "closed", "distance_ft": 15.0})
    return {"session_id": session_id, "tick": _session_tick(session_id),
            "rooms": rooms, "edges": edges, "occupants": occupants}


class BarrierPayload(BaseModel):
    """Create or update a stateful edge between two rooms (adapter gap 4)."""
    session_id: str = "default_session"
    template_key: str = "dynamic"
    a: str
    b: str
    barrier: str = "closed_door"   # closed_door | open_door | metal_partition | solid_wall | drywall | none
    state: str = "closed"          # open | closed
    distance_ft: float = 15.0
    idempotency_key: Optional[str] = None
    origin: str = "system"


@app.post("/api/v1/world/barrier")
async def set_barrier(payload: BarrierPayload):
    """Set a door/partition's type and state on the edge (a, b), both directions.
    The spatial evaluation reads this before any heuristic, so closing a door
    blacks the pair out and opening it restores line of sound."""
    digest = _payload_digest(payload)
    dup = _check_duplicate(payload.session_id, payload.idempotency_key, digest)
    if dup is not None:
        return {"success": True, "duplicate": True, "message": dup}
    world = _ensure_world(payload.template_key, payload.session_id)
    missing = [r for r in (payload.a, payload.b) if r not in world]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown room(s): {missing}")
    if payload.state not in ("open", "closed"):
        raise HTTPException(status_code=400, detail="state must be 'open' or 'closed'")
    _str_to_barrier(payload.barrier)  # validates the name
    edges = app_state.session_edges.setdefault(payload.session_id, {})
    edges[_edge_key(payload.a, payload.b)] = {
        "barrier": payload.barrier, "state": payload.state, "distance_ft": payload.distance_ft,
    }
    # Keep exits consistent so adjacency exists for pathing/snapshots.
    for x, y in ((payload.a, payload.b), (payload.b, payload.a)):
        if y not in world[x].exits:
            world[x].exits.append(y)
    msg = f"Edge ({payload.a}, {payload.b}) set: {payload.barrier}/{payload.state}"
    _last_digest["v"] = digest
    _record_idempotency(payload.session_id, payload.idempotency_key, msg)
    _log_mutation(payload.session_id, "SET_BARRIER", payload.origin,
                  {"a": payload.a, "b": payload.b, "barrier": payload.barrier, "state": payload.state})
    return {"success": True, "duplicate": False, "message": msg}


# ── Lock info ───────────────────────────────────────────────────────────

@app.get("/api/v1/world/lock/{session_id}")
async def get_lock_info(session_id: str):
    """Return lock metadata for a session, including TTL expiry."""
    info = lock_manager.get_lock_info(session_id)
    if info is None:
        return {"locked": False, "message": f"No active lock for session={session_id}"}
    return {"locked": True, "info": info}


# ── Internal helpers ────────────────────────────────────────────────────

def _find_actor_room(character_id: str, session_id: str = "default_session", template_key: str = "dynamic") -> Optional[str]:
    """Find the room_id where character_id is present in the active world for a session."""
    world = _get_session_world(session_id, template_key)
    if not world:
        # NOTE: global read alias — the session has no world yet.
        world = app_state.current_world
    for room_id, room in world.items():
        if character_id in room.present_characters:
            return room_id
    return None


def _str_to_barrier(barrier_str: str) -> BarrierType:
    """Convert string barrier name to BarrierType enum."""
    mapping = {
        "closed_door": BarrierType.CLOSED_DOOR,
        "solid_wall": BarrierType.SOLID_WALL,
        "open_door": BarrierType.OPEN_DOOR,
        "drywall": BarrierType.DRYWALL,
        "metal_partition": BarrierType.METAL_PARTITION,
    }
    return mapping.get(barrier_str, BarrierType.CLOSED_DOOR)


def _compute_distance_and_barriers(
    actor_room: Optional[str],
    target_room_id: str,
    action_type: ActionType,
    world: Optional[Dict[str, RoomMetadata]] = None,
    session_id: Optional[str] = None,
) -> tuple[float, List[BarrierType]]:
    """Compute distance (ft) and barriers between actor room and target room.

    Same room → 0-5 ft, no barriers.
    Adjacent room → ~15 ft, closed door barrier.
    Other room → 45 ft, closed door + solid wall.

    Adjacency is read from the ACTING session's world when ``world`` is passed
    (the /world/action path). The global ``app_state.current_world`` is only a
    read alias for legacy callers that have no session in hand (e.g. the void
    graph helpers); NOTE each such use at its call site.
    """
    if actor_room is None:
        return (45.0, [_str_to_barrier("closed_door"), _str_to_barrier("solid_wall")])

    if actor_room == target_room_id:
        return (3.0, [])

    # A2: an explicit stateful edge (door set by /world/barrier) overrides heuristics.
    if session_id is not None:
        edge = _session_edge(session_id, actor_room, target_room_id)
        if edge is not None:
            if edge.get("state") == "open" or edge.get("barrier") in (None, "none", "open_door"):
                return (float(edge.get("distance_ft", 15.0)), [])
            return (float(edge.get("distance_ft", 15.0)), [_str_to_barrier(edge["barrier"])])

    # Check adjacency via exit lists — in the session's own topology.
    lookup = world if world is not None else app_state.current_world  # NOTE: global read alias (legacy callers)
    actor_room_obj = lookup.get(actor_room)
    if actor_room_obj and target_room_id in actor_room_obj.exits:
        if actor_room_obj.lighting == "abstract" or actor_room in ("central_nexus", "node_alpha", "node_beta"):
            return (20.0, [_str_to_barrier("solid_wall")])
        return (15.0, [_str_to_barrier("closed_door")])

    # Not adjacent — distant
    return (45.0, [_str_to_barrier("closed_door"), _str_to_barrier("solid_wall")])


def _compute_all_distances(character_id: str, session_id: str = "default_session", template_key: str = "dynamic") -> Dict[str, float]:
    """Compute distances from character_id to every other character in the world for a session."""
    distances: Dict[str, float] = {}
    world = _get_session_world(session_id, template_key)
    if not world:
        # NOTE: global read alias — session has no world for this template.
        world = app_state.current_world
    char_room = _find_actor_room(character_id, session_id, template_key)

    for room_id, room in world.items():
        for other_id in room.present_characters:
            if other_id == character_id:
                distances[other_id] = 0.0
            elif room_id == char_room:
                # Same room: pick a representative distance
                distances[other_id] = min(3.0, 4.0)
            else:
                dist, _ = _compute_distance_and_barriers(char_room, room_id, ActionType.SPEAK, world=world)
                distances[other_id] = dist
    return distances


def _remove_character_from_all_rooms(character_id: str, session_id: str = "default_session") -> None:
    """Remove a character from every room in the active world for a session."""
    target_worlds = [app_state.current_world]  # NOTE: global read alias (legacy mirror), session worlds appended.
    if session_id in app_state.session_worlds:
        for tmpl_dict in app_state.session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for room_id, room in list(w.items()):
            room.present_characters = [c for c in room.present_characters if c != character_id]


# ── Startup ─────────────────────────────────────────────────────────────

async def startup_event():
    """Load the default world template on startup and init db pool."""
    app.state.start_time = time.time()

    # SPM_WORLD_DB=0 runs the engine memory-only: no pool, so every persistence helper
    # no-ops. The leak-suite rig needs this — it drives this app from TWO event loops
    # (its own TestClient and the proxy's ASGITransport), and an asyncpg pool created on
    # one loop breaks when acquired from the other.
    if os.getenv("SPM_WORLD_DB", "1") == "0":
        app_state._db_pool = None
        logging.info("Evennia World State Engine running memory-only (SPM_WORLD_DB=0)")
        _ensure_world("dynamic")   # the default world load still happens (review C1.3)
        return

    try:
        app_state._db_pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=5,
            max_size=20,
            command_timeout=30,
        )
        logging.info("Evennia World State Engine connected to PostgreSQL")
    except Exception as e:
        logging.error(f"Failed to create asyncpg pool: {e}")

    # A2: rebuild session worlds from Postgres so a restart keeps placements.
    # Runs ONCE per process (TestClient re-entries must not resurrect state), and
    # SPM_WORLD_RELOAD=0 disables it entirely (the test suite sets this).
    # Ticks resume from each session's max persisted action_tick (objective_world_log);
    # stateful edges are runtime-only for now (persistence joins Sprint 2 seeding).
    if app_state._db_pool and not getattr(app_state, "_reload_done", False) and os.getenv("SPM_WORLD_RELOAD", "1") != "0":
        app_state._reload_done = True
        try:
            async with app_state._db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT session_id, template_key, room_id, room_data FROM world_state_sessions")
                for row in rows:
                    try:
                        room = RoomMetadata(**json.loads(row["room_data"]))
                    except Exception as exc:
                        logging.error(f"Skipping unloadable room {row['room_id']}: {exc}")
                        continue
                    app_state.session_worlds.setdefault(row["session_id"], {}).setdefault(row["template_key"], {})[row["room_id"]] = room
                ticks = await conn.fetch("SELECT session_id, MAX(action_tick) AS t FROM objective_world_log GROUP BY session_id")
                for row in ticks:
                    app_state.session_ticks[row["session_id"]] = int(row["t"] or 0)
                if rows:
                    logging.info(f"Reloaded {len(rows)} room(s) across {len(app_state.session_worlds)} session(s) from Postgres")
        except Exception as e:
            logging.error(f"World state reload skipped: {e}")

    _ensure_world("dynamic")

async def shutdown_event():
    
    if app_state._db_pool:
        await app_state._db_pool.close()
        logging.info("Closed asyncpg pool")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("evennia_world.app:app", host="0.0.0.0", port=4005)

