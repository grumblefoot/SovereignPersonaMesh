"""
Deterministic Spatial & Acoustic Constraints Matrix for Evennia World Engine.
Evaluates physical proximity and environmental barriers to output sensory feeds and gating levels.
"""

from typing import Tuple, List
from .models import GatingLevel, BarrierType, ActionType


class SpatialConstraintsMatrix:
    """
    Evaluates sensory feeds based on distance, acoustic barriers, and visual obstructions.
    Rules:
    - 0-5ft: Direct visibility & acoustic clarity (Direct).
    - 5-15ft: Slight volume drop / speech-to-text degradation (Degraded).
    - >15ft: Drywall / closed door / solid wall replaces dialogue with sound descriptions (Blackout/Degraded).
    - Any distance + Solid Wall / Metal Partition: Complete sensory blackout.
    """

    @staticmethod
    def evaluate_sensory_feed(
        distance_ft: float,
        barriers: List[BarrierType],
        action_type: ActionType,
        raw_text: str,
        actor_id: str,
        recipient_id: str,
        is_target: bool = False
    ) -> Tuple[GatingLevel, str]:

        # Check hard blackout barriers
        if BarrierType.SOLID_WALL in barriers or BarrierType.METAL_PARTITION in barriers:
            return GatingLevel.BLACKOUT, ""

        if actor_id == recipient_id:
            # Self feed
            if action_type == ActionType.WHISPER and is_target:
                return GatingLevel.DIRECT, f'You whisper to {recipient_id}: "{raw_text}"'
            return GatingLevel.DIRECT, raw_text

        # Whisper gating rules
        if action_type == ActionType.WHISPER:
            if is_target and distance_ft <= 5.0 and BarrierType.CLOSED_DOOR not in barriers:
                return GatingLevel.DIRECT, f'{actor_id.capitalize()} whispers to you: "{raw_text}"'
            elif distance_ft <= 15.0 and BarrierType.CLOSED_DOOR not in barriers:
                return GatingLevel.DEGRADED, f'You hear {actor_id.capitalize()} murmur quietly, but cannot make out the words.'
            else:
                return GatingLevel.BLACKOUT, ""

        # Normal speech gating rules
        if BarrierType.CLOSED_DOOR in barriers:
            return GatingLevel.DEGRADED, f'You hear muffled voices from behind the closed door.'

        if distance_ft <= 5.0:
            return GatingLevel.DIRECT, f'{actor_id.capitalize()}: "{raw_text}"'
        elif distance_ft <= 15.0:
            return GatingLevel.DEGRADED, f'You hear {actor_id.capitalize()} speaking nearby: "{raw_text}"'
        else:
            return GatingLevel.BLACKOUT, f'You hear distant thuds or muffled sounds.'
