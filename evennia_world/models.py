"""
Pydantic Models for Evennia World State Engine Liaison API (Port 4005).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    SPEAK = "speak"
    WHISPER = "whisper"
    MOVE = "move"
    MANIPULATE = "manipulate"


class GatingLevel(str, Enum):
    DIRECT = "direct"
    DEGRADED = "degraded"
    BLACKOUT = "blackout"


class BarrierType(str, Enum):
    NONE = "none"
    OPEN_DOOR = "open_door"
    CLOSED_DOOR = "closed_door"
    DRYWALL = "drywall"
    SOLID_WALL = "solid_wall"
    METAL_PARTITION = "metal_partition"


class ActionPayload(BaseModel):
    character_id: str
    action_type: ActionType
    target_id: Optional[str] = None
    raw_text: str
    session_id: str = "default_session"


class SensoryConsequence(BaseModel):
    recipient_id: str
    sensory_feed: str
    gating_level: GatingLevel
    distance_ft: float
    barriers: List[BarrierType] = Field(default_factory=list)


class ActionResponse(BaseModel):
    success: bool
    action_tick: int
    consequences: List[SensoryConsequence]


class WorldStateQuery(BaseModel):
    character_id: str
    session_id: str = "default_session"


class RoomMetadata(BaseModel):
    room_id: str
    room_name: str
    description: str
    lighting: str = "normal"
    exits: List[str] = Field(default_factory=list)
    present_characters: List[str] = Field(default_factory=list)
    nearby_objects: List[str] = Field(default_factory=list)


class CharacterWorldState(BaseModel):
    character_id: str
    current_room: RoomMetadata
    gating_level: GatingLevel
    sensory_feed: str
    distances: Dict[str, float] = Field(default_factory=dict)


class SessionLockPayload(BaseModel):
    session_id: str
    lock_action: str # "acquire" or "release"
    lock_token: Optional[str] = None
