"""
Deterministic Spatial & Acoustic Constraints Matrix for Evennia World Engine.
Evaluates physical proximity and environmental barriers to output sensory feeds and gating levels.

Semantics (SRD 3.3.2, PRD 4.2, SPRINT_PLAN.md §6 decision 9 — Sprint 1 Track A):

- Distance bands for SPEAK (and anything that is not a shout):
    <= 5 ft            DIRECT
    > 5 ft and <= 15 ft  DEGRADED
    > 15 ft            BLACKOUT
- Barrier blackouts beat distance: CLOSED_DOOR, METAL_PARTITION and SOLID_WALL
  always produce BLACKOUT regardless of proximity. An OPEN_DOOR does not gate.
- DRYWALL is DEGRADED-at-best: it never produces DIRECT (sound is always muffled
  through it), but it does not blackout either — the SRD's barrier table lists
  drywall under "muffled/degraded", not under the blackout row. Distance still
  applies on top: drywall beyond the 15 ft speech band is BLACKOUT because the
  distance class alone carries nothing.
- SHOUT (decision 11) lands one gating level better than speak at the same
  geometry, via the explicit ``shout=True`` parameter:
    where speak would be DEGRADED (5–15 ft), a shout is DIRECT;
    where speak would be distance-BLACKOUT (>15 ft but <=30 ft), a shout is DEGRADED;
    beyond 30 ft a shout is BLACKOUT too.
  Barrier blackouts are never lifted by volume: closed door, metal partition and
  solid wall stay BLACKOUT for shouts as well (PRD 4.2). Drywall keeps its
  DEGRADED-at-best cap for shouts.
- NO hysteresis. Gating is a pure function of the current geometry; the former
  2-tick cooldown that smoothed DIRECT<->BLACKOUT transitions into DEGRADED was
  removed by decision 9 (it let a character who just left a room keep hearing it).
  ``session_id`` and ``action_tick`` are accepted for wire/API compatibility and
  are deliberately unused.
"""

from typing import Tuple, List

from .models import GatingLevel, BarrierType, ActionType


# Distance thresholds (ft).
DIRECT_WITHIN_FT: float = 5.0
SPEAK_REACH_FT: float = 15.0    # end of the DEGRADED band for speak-class actions
SHOUT_REACH_FT: float = 30.0    # a shout covers one extra level of distance

# Barriers that block sound completely, whatever the distance or volume.
HARD_BARRIERS = frozenset(
    {BarrierType.SOLID_WALL, BarrierType.CLOSED_DOOR, BarrierType.METAL_PARTITION}
)


class SpatialConstraintsMatrix:
    """
    Evaluates sensory feeds from the current authoritative world state only.
    Pure function of (distance, barriers, action type): no cross-tick state.
    """

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
        action_tick: int = 0,
        shout: bool = False,
    ) -> Tuple[GatingLevel, str]:
        # NOTE: session_id/action_tick are kept for call-site compatibility
        # (HTTP payloads and the in-process engine adapter). Hysteresis was
        # removed by decision 9, so they no longer influence gating.

        # 1. Omniscient/World bypass
        if recipient_id.lower() in ["scenario", "system", "world", "narrator", "context"]:
            return GatingLevel.DIRECT, raw_text

        # 2. Strict State Determination (SRD 3.3.2)
        if any(b in HARD_BARRIERS for b in barriers):
            # Barrier blackout beats distance and beats volume: shouts stay out too.
            target_gating = GatingLevel.BLACKOUT
        elif shout:
            if distance_ft <= SPEAK_REACH_FT:
                target_gating = GatingLevel.DIRECT
            elif distance_ft <= SHOUT_REACH_FT:
                target_gating = GatingLevel.DEGRADED
            else:
                target_gating = GatingLevel.BLACKOUT
        else:
            if distance_ft <= DIRECT_WITHIN_FT:
                target_gating = GatingLevel.DIRECT
            elif distance_ft <= SPEAK_REACH_FT:
                target_gating = GatingLevel.DEGRADED
            else:
                target_gating = GatingLevel.BLACKOUT

        # Drywall never yields a clean signal: cap DIRECT down to DEGRADED.
        if BarrierType.DRYWALL in barriers and target_gating == GatingLevel.DIRECT:
            target_gating = GatingLevel.DEGRADED

        # 3. Feed Formatting (a shout is formatted like speech; only gating differs)
        if target_gating == GatingLevel.BLACKOUT:
            return GatingLevel.BLACKOUT, ""

        if actor_id == recipient_id:
            if action_type == ActionType.WHISPER and is_target:
                return GatingLevel.DIRECT, f'You whisper to {recipient_id}: "{raw_text}"'
            return GatingLevel.DIRECT, raw_text

        if action_type == ActionType.WHISPER:
            if target_gating == GatingLevel.DIRECT and is_target:
                if "*" in raw_text or '"' in raw_text:
                    # Text with its own markup is not re-quoted (F18, whisper side).
                    return GatingLevel.DIRECT, f'{actor_id.capitalize()} whispers to you: {raw_text}'
                return GatingLevel.DIRECT, f'{actor_id.capitalize()} whispers to you: "{raw_text}"'
            # A bystander within earshot NOTICES the whispering but never gets the words
            # (decision 11). A same-room bystander used to fall through to BLACKOUT, so a
            # character standing right there was skipped as "out of earshot" (QA F23).
            elif target_gating in (GatingLevel.DIRECT, GatingLevel.DEGRADED):
                return GatingLevel.DEGRADED, f'You hear {actor_id.capitalize()} murmur quietly.'
            else:
                return GatingLevel.BLACKOUT, ""

        if target_gating == GatingLevel.DIRECT:
            if "*" in raw_text:
                # Mixed speech + action keeps its own markup ('"Hi." *waves*'): wrapping
                # it in quotes turned actions into speech (QA F18, 2026-10-05).
                return GatingLevel.DIRECT, f'{actor_id.capitalize()}: {raw_text}'
            return GatingLevel.DIRECT, f'{actor_id.capitalize()}: "{raw_text}"'
        else:
            return GatingLevel.DEGRADED, f'You hear muffled voices or sounds from nearby.'
