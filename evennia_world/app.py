"""
FastAPI Server for Evennia World State Engine Liaison Interface (Port 4005).
Provides headless endpoints for action evaluation, spatial state queries, and tick lock management.
Uses HybridWorldBuilder for dynamic world state and SpatialConstraintsMatrix for deterministic gating.
"""

import time
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncpg

from .models import (
    ActionPayload, ActionResponse, SensoryConsequence, GatingLevel,
    WorldStateQuery, CharacterWorldState, SessionLockPayload, RoomMetadata,
    ActionType, BarrierType
)
from .spatial_matrix import SpatialConstraintsMatrix
from .session_lock import SessionLockManager, LockError
from .hybrid_builder import HybridWorldBuilder
from core.resource_manager import strings

app = FastAPI(title="Evennia World State Engine Liaison API", version="0.2.0")

# ── Internal state ──────────────────────────────────────────────────────
lock_manager = SessionLockManager()
world_builder = HybridWorldBuilder()
class AppState:
    def __init__(self):
        self.action_tick_counter: int = 1420
        self._db_pool = None
        self.current_world: Dict[str, RoomMetadata] = {}
        self.room_to_template: Dict[str, str] = {}
        self.session_worlds: Dict[str, Dict[str, Dict[str, RoomMetadata]]] = {}

app_state = AppState()

import os
# Database config
DB_CONFIG = {
    "user": os.environ.get("SPM_DB_USER", "spm_user"),
    "password": os.environ.get("SPM_DB_PASSWORD", "spm_secure_password"),
    "database": os.environ.get("SPM_DB_NAME", "litellm_postgres"),
    "host": os.environ.get("SPM_DB_HOST", "localhost"),
    "port": int(os.environ.get("SPM_DB_PORT", 5432))
}


def _ensure_world(template_key: str = "dungeon_cellar", session_id: str = "default_session") -> Dict[str, RoomMetadata]:
    """Ensure session-scoped world state matches the requested template. Returns the world dict for the session."""
    if session_id not in app_state.session_worlds:
        app_state.session_worlds[session_id] = {}
    if template_key not in app_state.session_worlds[session_id]:
        app_state.session_worlds[session_id][template_key] = world_builder.instantiate_world(template_key)
    # Also sync the legacy app_state.current_world for backward compatibility
    
    if not app_state.current_world:
        app_state.current_world = app_state.session_worlds[session_id].get(template_key, {})
    return app_state.session_worlds[session_id][template_key]


def _get_session_world(session_id: str, template_key: str = "dungeon_cellar") -> Optional[Dict[str, RoomMetadata]]:
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
            query = """
                INSERT INTO world_state_sessions (session_id, template_key, room_id, room_data, action_tick)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id, template_key, room_id)
                DO UPDATE SET 
                    room_data = EXCLUDED.room_data, 
                    action_tick = EXCLUDED.action_tick, 
                    updated_at = CURRENT_TIMESTAMP
            """
            await conn.execute(query, session_id, template_key, room_id, room.model_dump_json(), app_state.action_tick_counter)
    except Exception as e:
        logging.error(f"Failed to persist room {room_id}: {e}")

async def _log_objective_action(session_id: str, action_tick: int, actor_id: str, location_id: str, action_type: str, raw_event: str):
    """Log an objective action to PostgreSQL."""
    if not app_state._db_pool:
        return
    try:
        async with app_state._db_pool.acquire() as conn:
            query = """
                INSERT INTO objective_world_log (session_id, action_tick, actor_id, location_id, action_type, raw_event)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            await conn.execute(query, session_id, action_tick, actor_id, location_id, action_type, raw_event)
    except Exception as e:
        logging.error(f"Failed to log objective action: {e}")

# ── Health check ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Quick readiness probe."""
    return {
        "status": "ok",
        "tick": app_state.action_tick_counter,
        "template": list(app_state.current_world.keys()) if app_state.current_world else "none",
        "uptime_seconds": round(time.time() - app.state.start_time, 1),
    }


# ── Action evaluation ───────────────────────────────────────────────────

@app.post("/api/v1/world/action", response_model=ActionResponse)
async def submit_action(payload: ActionPayload, background_tasks: BackgroundTasks):
    """
    Evaluates physical intentions (speak|whisper|move|manipulate).
    Returns action_tick and sensory feeds for recipient characters based on
    real room positions, distances, and the SpatialConstraintsMatrix.
    Supports session-scoped world state for FR-001 isolation.
    """
    
    app_state.action_tick_counter += 1

    # Ensure session-scoped world is loaded
    template_key = getattr(payload, "template_key", "dungeon_cellar")
    _ensure_world(template_key, payload.session_id)
    world = _get_session_world(payload.session_id, template_key)
    if not world:
        _ensure_world()
        world = app_state.current_world

    # Find which room the actor is in
    actor_room = _find_actor_room(payload.character_id, payload.session_id)
    loc_id = actor_room if actor_room else "unknown"

    background_tasks.add_task(
        _log_objective_action,
        payload.session_id,
        app_state.action_tick_counter,
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
            seen_ids.add(char_id)
            is_target = (char_id == payload.target_id)

            # Determine if recipient is in the same room as the actor
            same_room = (actor_room is not None and
                         room_id == actor_room)
            dist, barriers = _compute_distance_and_barriers(
                actor_room, room_id, payload.action_type
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
                action_tick=app_state.action_tick_counter,
            )

            if gating != GatingLevel.BLACKOUT:
                consequences.append(SensoryConsequence(
                    recipient_id=char_id,
                    sensory_feed=feed,
                    gating_level=gating,
                    distance_ft=dist,
                    barriers=barriers,
                ))

    # Always include Seamus (or any "upstairs" / distant character) if they exist
    for room_id, room in world.items():
        if "upstairs" in room_id or "tavern" in room_id:
            for char_id in room.present_characters:
                if char_id not in seen_ids:
                    seen_ids.add(char_id)
                    gating = GatingLevel.DEGRADED
                    feed = strings.get("app.muffled_sounds", room_id=room_id)
                    consequences.append(SensoryConsequence(
                        recipient_id=char_id,
                        sensory_feed=feed,
                        gating_level=gating,
                        distance_ft=45.0,
                        barriers=["closed_door", "solid_wall"],
                    ))

    return ActionResponse(
        success=True,
        action_tick=app_state.action_tick_counter,
        consequences=consequences,
    )


# ── World state query ───────────────────────────────────────────────────

@app.get("/api/v1/world/state", response_model=CharacterWorldState)
async def query_world_state(character_id: str, session_id: str = "default_session", template_key: str = "dungeon_cellar"):
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
    char_room = _find_actor_room(char_id_lower, session_id)
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
        room = app_state.current_world.get(char_room)
    if room is None:
        room = RoomMetadata(
            room_id="unknown", room_name="Unknown Location",
            description="No room assigned.", lighting="normal",
            exits=[], present_characters=[], nearby_objects=[],
        )
    distances = _compute_all_distances(char_id_lower, session_id)

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
    template_key: str = "dungeon_cellar"
    session_id: str = "default_session"

@app.post("/api/v1/world/rooms")
async def create_room(payload: CreateRoomPayload, background_tasks: BackgroundTasks):
    """Create a new room dynamically and add it to the session world."""
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
    
    background_tasks.add_task(_persist_room, payload.session_id, payload.template_key, payload.room_id, new_room)
    
    return {
        "success": True,
        "message": f"Room '{payload.room_name}' created successfully.",
        "room_id": payload.room_id
    }


# ── Character management ────────────────────────────────────────────────

class CharacterMovePayload(BaseModel):
    """Move a character to a room within the active template."""
    character_id: str
    room_id: str
    template_key: str = "dungeon_cellar"


class CharacterResponse(BaseModel):
    """Generic response for character operations."""
    success: bool
    message: str
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
    if app_state.current_world and payload.room_id in app_state.current_world:
        _remove_character_from_all_rooms(payload.character_id, payload.session_id if hasattr(payload, 'session_id') else "default_session")
        if payload.character_id not in app_state.current_world[payload.room_id].present_characters:
            app_state.current_world[payload.room_id].present_characters.append(payload.character_id)

    session_id = payload.session_id if hasattr(payload, 'session_id') else "default_session"
    background_tasks.add_task(_persist_room, session_id, payload.template_key, payload.room_id, app_state.current_world[payload.room_id] if payload.room_id in app_state.current_world else room)

    return CharacterResponse(
        success=True,
        message=f"Character '{payload.character_id}' added to room '{payload.room_id}'",
        character_id=payload.character_id,
        room_id=payload.room_id,
    )


@app.delete("/api/v1/world/characters/{character_id}", response_model=CharacterResponse)
async def remove_character_from_world(character_id: str, background_tasks: BackgroundTasks, template_key: str = "dungeon_cellar", session_id: str = "default_session"):
    """Remove a character from all rooms in a template."""
    modified_rooms = set()
    for room_id in world_builder.templates.get(template_key, {}):
        if character_id in world_builder.templates[template_key][room_id].present_characters:
            world_builder.remove_character_from_room(template_key, room_id, character_id)
            modified_rooms.add((template_key, room_id))

    target_worlds = [app_state.current_world]
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
        elif r_id in app_state.current_world:
            room_obj = app_state.current_world[r_id]
        if room_obj:
            background_tasks.add_task(_persist_room, session_id, tmpl_key, r_id, room_obj)

    return CharacterResponse(
        success=True,
        message=f"Character '{character_id}' removed from all rooms in '{template_key}'",
        character_id=character_id,
    )


@app.get("/api/v1/world/characters")
async def list_characters(template_key: str = "dungeon_cellar"):
    """List all characters in a template with their current rooms."""
    if template_key not in world_builder.templates:
        raise HTTPException(status_code=404, detail=f"Template '{template_key}' not found")

    result: List[Dict[str, Any]] = []
    for room_id, room in world_builder.templates[template_key].items():
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
    session_id = payload.session_id if hasattr(payload, 'session_id') else "default_session"
    
    # Track which rooms were modified to persist them
    modified_rooms = set()
    # Remove from all rooms first, then add to destination
    for tmpl_key in world_builder.templates:
        for room_id in list(world_builder.templates[tmpl_key].keys()):
            if payload.character_id in world_builder.templates[tmpl_key][room_id].present_characters:
                world_builder.remove_character_from_room(tmpl_key, room_id, payload.character_id)
                modified_rooms.add((tmpl_key, room_id))
            
    # Remove from session worlds
    target_worlds = [app_state.current_world]
    if session_id in app_state.session_worlds:
        for tmpl_dict in app_state.session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for r_id, r in list(w.items()):
            if payload.character_id in r.present_characters:
                r.present_characters.remove(payload.character_id)
                modified_rooms.add((payload.template_key, r_id))

    world_builder.add_character_to_room(
        payload.template_key, payload.room_id, payload.character_id,
    )
    
    if app_state.current_world and payload.room_id in app_state.current_world:
        if payload.character_id not in app_state.current_world[payload.room_id].present_characters:
            app_state.current_world[payload.room_id].present_characters.append(payload.character_id)
    elif session_id in app_state.session_worlds and payload.template_key in app_state.session_worlds[session_id]:
        room_obj = app_state.session_worlds[session_id][payload.template_key].get(payload.room_id)
        if room_obj and payload.character_id not in room_obj.present_characters:
            room_obj.present_characters.append(payload.character_id)
            
    modified_rooms.add((payload.template_key, payload.room_id))
    
    for tmpl_key, r_id in modified_rooms:
        room_obj = None
        if session_id in app_state.session_worlds and tmpl_key in app_state.session_worlds[session_id] and r_id in app_state.session_worlds[session_id][tmpl_key]:
            room_obj = app_state.session_worlds[session_id][tmpl_key][r_id]
        elif r_id in app_state.current_world:
            room_obj = app_state.current_world[r_id]
        if room_obj:
            background_tasks.add_task(_persist_room, session_id, tmpl_key, r_id, room_obj)

    return CharacterResponse(
        success=True,
        message=f"Character '{payload.character_id}' moved to room '{payload.room_id}'",
        character_id=payload.character_id,
        room_id=payload.room_id,
    )


# ── World configuration ─────────────────────────────────────────────────

class WorldConfigPayload(BaseModel):
    """Switch the active world to a different template."""
    template_key: str
    flavor_text: Optional[str] = None


@app.post("/api/v1/world/configure", response_model=CharacterResponse)
async def configure_world(payload: WorldConfigPayload):
    """Load a different room template as the active world."""
    if payload.template_key not in world_builder.templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{payload.template_key}' not found",
        )
    session_id_for_config = payload.template_key
    world_inst = world_builder.instantiate_world(payload.template_key)
    if payload.flavor_text:
        for rm in world_inst.values():
            rm.flavor_text = payload.flavor_text
    app_state.session_worlds[session_id_for_config] = {payload.template_key: world_inst}
    
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
    return CharacterResponse(
        success=True,
        message=f"World loaded: template '{payload.template_key}'",
    )


@app.get("/api/v1/world/templates")
async def list_templates():
    """List all available world templates."""
    return {"templates": world_builder.list_templates()}


@app.delete("/api/v1/world/admin/reset")
async def reset_world_state():
    """Clear in-memory cache of world state for factory reset."""
    app_state.session_worlds.clear()
    
    app_state.current_world = {}
    return {"status": "success", "message": "In-memory world state cache cleared"}


# ── Lock info ───────────────────────────────────────────────────────────

@app.get("/api/v1/world/lock/{session_id}")
async def get_lock_info(session_id: str):
    """Return lock metadata for a session, including TTL expiry."""
    info = lock_manager.get_lock_info(session_id)
    if info is None:
        return {"locked": False, "message": f"No active lock for session={session_id}"}
    return {"locked": True, "info": info}


# ── Internal helpers ────────────────────────────────────────────────────

def _find_actor_room(character_id: str, session_id: str = "default_session") -> Optional[str]:
    """Find the room_id where character_id is present in the active world for a session."""
    world = _get_session_world(session_id)
    if not world:
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
) -> tuple[float, List[BarrierType]]:
    """Compute distance (ft) and barriers between actor room and target room.

    Same room → 0-5 ft, no barriers.
    Adjacent room → ~15 ft, closed door barrier.
    Other room → 45 ft, closed door + solid wall.
    """
    if actor_room is None:
        return (45.0, [_str_to_barrier("closed_door"), _str_to_barrier("solid_wall")])

    if actor_room == target_room_id:
        return (3.0, [])

    # Check adjacency via exit lists
    actor_room_obj = app_state.current_world.get(actor_room)
    if actor_room_obj and target_room_id in actor_room_obj.exits:
        if actor_room_obj.lighting == "abstract" or actor_room in ("central_nexus", "node_alpha", "node_beta"):
            return (20.0, [_str_to_barrier("solid_wall")])
        return (15.0, [_str_to_barrier("closed_door")])

    # Not adjacent — distant
    return (45.0, [_str_to_barrier("closed_door"), _str_to_barrier("solid_wall")])


def _compute_all_distances(character_id: str, session_id: str = "default_session") -> Dict[str, float]:
    """Compute distances from character_id to every other character in the world for a session."""
    distances: Dict[str, float] = {}
    world = _get_session_world(session_id)
    if not world:
        world = app_state.current_world
    char_room = _find_actor_room(character_id, session_id)

    for room_id, room in world.items():
        for other_id in room.present_characters:
            if other_id == character_id:
                distances[other_id] = 0.0
            elif room_id == char_room:
                # Same room: pick a representative distance
                distances[other_id] = min(3.0, 4.0)
            else:
                dist, _ = _compute_distance_and_barriers(char_room, room_id, ActionType.SPEAK)
                distances[other_id] = dist
    return distances


def _remove_character_from_all_rooms(character_id: str, session_id: str = "default_session") -> None:
    """Remove a character from every room in the active world for a session."""
    target_worlds = [app_state.current_world]
    if session_id in app_state.session_worlds:
        for tmpl_dict in app_state.session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for room_id, room in list(w.items()):
            room.present_characters = [c for c in room.present_characters if c != character_id]


# ── Startup ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load the default world template on startup and init db pool."""
    app.state.start_time = time.time()
    
    
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
        
    _ensure_world("dungeon_cellar")

@app.on_event("shutdown")
async def shutdown_event():
    
    if app_state._db_pool:
        await app_state._db_pool.close()
        logging.info("Closed asyncpg pool")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("evennia_world.app:app", host="0.0.0.0", port=4005)

