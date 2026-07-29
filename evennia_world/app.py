"""
FastAPI Server for Evennia World State Engine Liaison Interface (Port 4005).
Provides headless endpoints for action evaluation, spatial state queries, and tick lock management.
Uses HybridWorldBuilder for dynamic world state and SpatialConstraintsMatrix for deterministic gating.
"""

import time
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from .models import (
    ActionPayload, ActionResponse, SensoryConsequence, GatingLevel,
    WorldStateQuery, CharacterWorldState, SessionLockPayload, RoomMetadata,
    ActionType, BarrierType
)
from .spatial_matrix import SpatialConstraintsMatrix
from .session_lock import SessionLockManager, LockError
from .hybrid_builder import HybridWorldBuilder

app = FastAPI(title="Evennia World State Engine Liaison API", version="1.1.0")

# ── Internal state ──────────────────────────────────────────────────────
lock_manager = SessionLockManager()
world_builder = HybridWorldBuilder()
action_tick_counter: int = 1420

# Room-to-template mapping for the active world
current_world: Dict[str, RoomMetadata] = {}
room_to_template: Dict[str, str] = {}

# Session-keyed world state maps for FR-001 session isolation
# session_worlds: Dict[session_id, Dict[template_key, Dict[room_id, RoomMetadata]]]
session_worlds: Dict[str, Dict[str, Dict[str, RoomMetadata]]] = {}


def _ensure_world(template_key: str = "dungeon_cellar", session_id: str = "default_session") -> Dict[str, RoomMetadata]:
    """Ensure session-scoped world state matches the requested template. Returns the world dict for the session."""
    if session_id not in session_worlds:
        session_worlds[session_id] = {}
    if template_key not in session_worlds[session_id]:
        session_worlds[session_id][template_key] = world_builder.instantiate_world(template_key)
    # Also sync the legacy current_world for backward compatibility
    global current_world
    if not current_world:
        current_world = session_worlds[session_id].get(template_key, {})
    return session_worlds[session_id][template_key]


def _get_session_world(session_id: str, template_key: str = "dungeon_cellar") -> Dict[str, RoomMetadata]:
    """Get the world dict for a session, falling back to legacy current_world."""
    return session_worlds.get(session_id, {}).get(template_key, current_world)


# ── Health check ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Quick readiness probe."""
    return {
        "status": "ok",
        "tick": action_tick_counter,
        "template": list(current_world.keys()) if current_world else "none",
        "uptime_seconds": round(time.time() - app.state.start_time, 1),
    }


# ── Action evaluation ───────────────────────────────────────────────────

@app.post("/api/v1/world/action", response_model=ActionResponse)
async def submit_action(payload: ActionPayload):
    """
    Evaluates physical intentions (speak|whisper|move|manipulate).
    Returns action_tick and sensory feeds for recipient characters based on
    real room positions, distances, and the SpatialConstraintsMatrix.
    Supports session-scoped world state for FR-001 isolation.
    """
    global action_tick_counter
    action_tick_counter += 1

    # Ensure session-scoped world is loaded
    template_key = "dungeon_cellar"
    _ensure_world(template_key, payload.session_id)
    world = _get_session_world(payload.session_id, template_key)
    if not world:
        _ensure_world()
        world = current_world

    # Find which room the actor is in
    actor_room = _find_actor_room(payload.character_id, payload.session_id)

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
                    feed = f"You hear muffled sounds from {room_id}."
                    consequences.append(SensoryConsequence(
                        recipient_id=char_id,
                        sensory_feed=feed,
                        gating_level=gating,
                        distance_ft=45.0,
                        barriers=["closed_door", "solid_wall"],
                    ))

    return ActionResponse(
        success=True,
        action_tick=action_tick_counter,
        consequences=consequences,
    )


# ── World state query ───────────────────────────────────────────────────

@app.get("/api/v1/world/state", response_model=CharacterWorldState)
async def query_world_state(character_id: str, session_id: str = "default_session"):
    """
    Queries local room metadata for any character (lighting, exits, nearby entities, distances).
    Session-scoped state per FR-001.
    """
    char_id_lower = character_id.lower()
    template_key = "dungeon_cellar"

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
            sensory_feed=f"Character {char_id_lower} is not assigned to a room.",
            distances={},
        )

    # Look up the room in session-scoped world first, then fall back to legacy current_world
    room = world.get(char_room)
    if room is None:
        room = current_world.get(char_room)
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
        sensory_feed=f"You are inside {room.room_name}. {room.description}",
        distances=distances,
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
async def add_character_to_world(payload: CharacterMovePayload):
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

    # Also update the active world if the room is in current_world
    if current_world and payload.room_id in current_world:
        _remove_character_from_all_rooms(payload.character_id, payload.session_id if hasattr(payload, 'session_id') else "default_session")
        if payload.character_id not in current_world[payload.room_id].present_characters:
            current_world[payload.room_id].present_characters.append(payload.character_id)

    return CharacterResponse(
        success=True,
        message=f"Character '{payload.character_id}' added to room '{payload.room_id}'",
        character_id=payload.character_id,
        room_id=payload.room_id,
    )


@app.delete("/api/v1/world/characters/{character_id}", response_model=CharacterResponse)
async def remove_character_from_world(character_id: str, template_key: str = "dungeon_cellar"):
    """Remove a character from all rooms in a template."""
    for room_id in world_builder.templates.get(template_key, {}):
        world_builder.remove_character_from_room(template_key, room_id, character_id)

    _remove_character_from_all_rooms(character_id)

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
async def move_character(payload: CharacterMovePayload):
    """Move an existing character from their current room to a new one."""
    # Remove from all rooms first, then add to destination
    for tmpl_key in world_builder.templates:
        for room_id in list(world_builder.templates[tmpl_key].keys()):
            world_builder.remove_character_from_room(tmpl_key, room_id, payload.character_id)

    world_builder.add_character_to_room(
        payload.template_key, payload.room_id, payload.character_id,
    )

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


@app.post("/api/v1/world/configure", response_model=CharacterResponse)
async def configure_world(payload: WorldConfigPayload):
    """Load a different room template as the active world."""
    if payload.template_key not in world_builder.templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{payload.template_key}' not found",
        )
    session_id_for_config = payload.template_key
    session_worlds[session_id_for_config] = {payload.template_key: world_builder.instantiate_world(payload.template_key)}
    global current_world
    current_world = session_worlds[session_id_for_config][payload.template_key]
    return CharacterResponse(
        success=True,
        message=f"World loaded: template '{payload.template_key}'",
    )


@app.get("/api/v1/world/templates")
async def list_templates():
    """List all available world templates."""
    return {"templates": world_builder.list_templates()}


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
        world = current_world
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
    actor_room_obj = current_world.get(actor_room)
    if actor_room_obj and target_room_id in actor_room_obj.exits:
        return (15.0, [_str_to_barrier("closed_door")])

    # Not adjacent — distant
    return (45.0, [_str_to_barrier("closed_door"), _str_to_barrier("solid_wall")])


def _compute_all_distances(character_id: str, session_id: str = "default_session") -> Dict[str, float]:
    """Compute distances from character_id to every other character in the world for a session."""
    distances: Dict[str, float] = {}
    world = _get_session_world(session_id)
    if not world:
        world = current_world
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
    target_worlds = [current_world]
    if session_id in session_worlds:
        for tmpl_dict in session_worlds[session_id].values():
            target_worlds.append(tmpl_dict)
    for w in target_worlds:
        for room_id, room in list(w.items()):
            if character_id in room.present_characters:
                room.present_characters.remove(character_id)


# ── Startup ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load the default world template on startup."""
    app.state.start_time = time.time()
    _ensure_world("dungeon_cellar")
