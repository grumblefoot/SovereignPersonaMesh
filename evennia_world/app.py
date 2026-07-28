"""
FastAPI Server for Evennia World State Engine Liaison Interface (Port 4005).
Provides headless endpoints for action evaluation, spatial state queries, and tick lock management.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from .models import (
    ActionPayload, ActionResponse, SensoryConsequence, GatingLevel,
    WorldStateQuery, CharacterWorldState, SessionLockPayload
)
from .spatial_matrix import SpatialConstraintsMatrix
from .session_lock import SessionLockManager
from .hybrid_builder import HybridWorldBuilder

app = FastAPI(title="Evennia World State Engine Liaison API", version="1.0.0")

# Internal state tracking
lock_manager = SessionLockManager()
world_builder = HybridWorldBuilder()
current_world = world_builder.instantiate_world("dungeon_cellar")
action_tick_counter = 1420


@app.post("/api/v1/world/action", response_model=ActionResponse)
async def submit_action(payload: ActionPayload):
    """
    Evaluates physical intentions (speak|whisper|move|manipulate).
    Returns action_tick and sensory feeds for recipient characters.
    """
    global action_tick_counter
    action_tick_counter += 1

    consequences = []
    # Mock character distance setup for cellar vs tavern
    cellar_chars = ["rowan", "domino", "luna"]

    for char_id in cellar_chars:
        is_target = (char_id == payload.target_id)
        dist = 3.0 if (char_id in ["rowan", "domino"]) else 15.0
        barriers = []

        gating, feed = SpatialConstraintsMatrix.evaluate_sensory_feed(
            distance_ft=dist,
            barriers=barriers,
            action_type=payload.action_type,
            raw_text=payload.raw_text,
            actor_id=payload.character_id,
            recipient_id=char_id,
            is_target=is_target
        )

        if gating != GatingLevel.BLACKOUT:
            consequences.append(SensoryConsequence(
                recipient_id=char_id,
                sensory_feed=feed,
                gating_level=gating,
                distance_ft=dist,
                barriers=barriers
            ))

    # Add Seamus (in tavern upstairs, behind closed cellar door)
    consequences.append(SensoryConsequence(
        recipient_id="seamus",
        sensory_feed="You hear muffled sounds from downstairs.",
        gating_level=GatingLevel.DEGRADED,
        distance_ft=45.0,
        barriers=["closed_door", "solid_wall"]
    ))

    return ActionResponse(
        success=True,
        action_tick=action_tick_counter,
        consequences=consequences
    )


@app.get("/api/v1/world/state", response_model=CharacterWorldState)
async def query_world_state(character_id: str, session_id: str = "default_session"):
    """
    Queries local room metadata for any character (lighting, exits, nearby entities, distances).
    """
    char_id_lower = character_id.lower()
    if char_id_lower == "seamus":
        room = current_world["tavern_upstairs"]
        gating = GatingLevel.DEGRADED
        feed = "You are resting in the tavern upstairs."
    else:
        room = current_world["cellar"]
        gating = GatingLevel.DIRECT
        feed = f"You are inside {room.room_name}."

    return CharacterWorldState(
        character_id=char_id_lower,
        current_room=room,
        gating_level=gating,
        sensory_feed=feed,
        distances={"rowan": 0.0, "domino": 4.0, "luna": 15.0, "seamus": 45.0}
    )


@app.post("/api/v1/world/lock")
async def handle_session_lock(payload: SessionLockPayload):
    """
    Session & Tick Lock endpoint to prevent race conditions during turn generation.
    """
    if payload.lock_action == "acquire":
        token = await lock_manager.acquire_lock(payload.session_id)
        return {"success": True, "lock_token": token}
    elif payload.lock_action == "release":
        if not payload.lock_token:
            raise HTTPException(status_code=400, detail="lock_token required for release")
        released = lock_manager.release_lock(payload.session_id, payload.lock_token)
        return {"success": released}
    else:
        raise HTTPException(status_code=400, detail="Invalid lock_action")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4005)
