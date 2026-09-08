"""
Deterministic Spatial & Acoustic Constraints Matrix for Evennia World Engine.
Evaluates physical proximity and environmental barriers to output sensory feeds and gating levels.
"""

from typing import Tuple, List, Dict
from collections import defaultdict
from .models import GatingLevel, BarrierType, ActionType


class SpatialConstraintsMatrix:
    """
    Evaluates sensory feeds based on a strict state machine and authoritative world state.

    Session-scoped gating history: _gating_history maps session_id -> (actor_id, recipient_id)
    -> (GatingLevel, action_tick).  Each session keeps its own cooldown windows so data
    never bleeds across sessions.
    """
    _gating_history: Dict[str, Dict[Tuple[str, str], Tuple[GatingLevel, int]]] = defaultdict(dict)

    # Number of ticks a gating transition is suppressed during its cooldown window.
    HYSTERESIS_TICKS: int = 2

    @classmethod
    def evaluate_sensory_feed(
        cls,
        distance_ft: float,
        barriers: List[BarrierType],
        action_type: ActionType,
        raw_text: str,
        actor_id: str,
        recipient_id: str,
        is_target: bool = False,
        session_id: str = "default",
        action_tick: int = 0
    ) -> Tuple[GatingLevel, str]:

        # 1. Omniscient/World bypass
        if recipient_id.lower() in ["scenario", "system", "world", "narrator", "context"]:
            return GatingLevel.DIRECT, raw_text

        # 2. Strict State Determination
        if BarrierType.SOLID_WALL in barriers:
            target_gating = GatingLevel.BLACKOUT
        elif distance_ft <= 5.0:
            target_gating = GatingLevel.DIRECT
        elif distance_ft <= 20.0:
            target_gating = GatingLevel.DEGRADED
        else:
            target_gating = GatingLevel.BLACKOUT

        # 3. Hysteresis Cooldown (bidirectional)
        hist_key = (actor_id, recipient_id)
        session_hist = cls._gating_history[session_id]
        last_gating, last_tick = session_hist.get(hist_key, (GatingLevel.DIRECT, 0))

        # Preserve the raw physical-state gating for history recording
        # (before any hysteresis suppression).
        raw_gating = target_gating

        if (target_gating == GatingLevel.BLACKOUT and last_gating == GatingLevel.DIRECT
                and (action_tick - last_tick) < cls.HYSTERESIS_TICKS):
            target_gating = GatingLevel.DEGRADED

        if (target_gating == GatingLevel.DIRECT and last_gating == GatingLevel.BLACKOUT
                and (action_tick - last_tick) < cls.HYSTERESIS_TICKS):
            target_gating = GatingLevel.DEGRADED

        # Always record the RAW physical-state (before suppression) so cooldown
        # timers track actual state transitions, not smoothed output levels.
        session_hist[hist_key] = (raw_gating, action_tick)

        # 4. Feed Formatting
        if target_gating == GatingLevel.BLACKOUT:
            return GatingLevel.BLACKOUT, ""

        if actor_id == recipient_id:
            if action_type == ActionType.WHISPER and is_target:
                return GatingLevel.DIRECT, f'You whisper to {recipient_id}: "{raw_text}"'
            return GatingLevel.DIRECT, raw_text

        if action_type == ActionType.WHISPER:
            if target_gating == GatingLevel.DIRECT and is_target:
                return GatingLevel.DIRECT, f'{actor_id.capitalize()} whispers to you: "{raw_text}"'
            elif target_gating == GatingLevel.DEGRADED:
                return GatingLevel.DEGRADED, f'You hear {actor_id.capitalize()} murmur quietly.'
            else:
                return GatingLevel.BLACKOUT, ""

        if target_gating == GatingLevel.DIRECT:
            return GatingLevel.DIRECT, f'{actor_id.capitalize()}: "{raw_text}"'
        else:
            return GatingLevel.DEGRADED, f'You hear muffled voices or sounds from nearby.'
