"""
Hybrid Semantic-Template World Construction Engine.
Matches SillyTavern Lorebooks/character cards to rigid, pre-tested Evennia room network templates.
Supports dynamic character management, score-based template matching, and room creation.
"""

import copy
import logging
from typing import Dict, Any, List, Optional, Set
from .models import RoomMetadata

logger = logging.getLogger(__name__)

from core.resource_manager import strings

# Score-based keyword weighting for template matching
_KEYWORD_SCORES: Dict[str, Dict[str, float]] = {
    "dungeon_cellar": {"dungeon": 3.0, "cellar": 3.0, "basement": 2.0, "prison": 2.0, "jail": 2.0, "tavern": 1.5, "ale": 1.0},
    "forest_camp": {"forest": 3.0, "camp": 2.0, "woods": 2.5, "tree": 1.5, "wild": 2.0, "nature": 1.5},
    "castle_exterior": {"castle": 3.0, "fortress": 2.5, "courtyard": 2.0, "throne": 2.0, "knight": 1.5, "armory": 2.0, "stables": 1.5},
    "tavern_common": {"tavern": 3.0, "inn": 2.5, "bar": 2.0, "pub": 2.0, "ale": 1.5, "common": 1.0},
}


class HybridWorldBuilder:
    """Builds and manages world state from room templates, with dynamic character and room management."""

    def __init__(self, templates: Dict[str, Dict[str, RoomMetadata]] | None = None):
        # Deep copy to prevent shared state mutations across instances
        if templates is not None:
            self.templates: Dict[str, Dict[str, RoomMetadata]] = copy.deepcopy(templates)
        else:
            raw_templates = strings.get_raw("templates")
            self.templates = {}
            if isinstance(raw_templates, dict) or hasattr(raw_templates, "items"):
                for t_key, t_dict in raw_templates.items():
                    self.templates[t_key] = {}
                    for r_key, r_dict in t_dict.items():
                        self.templates[t_key][r_key] = RoomMetadata(**r_dict)

    # ── Template matching ─────────────────────────────────────────────

    def match_template(self, keywords: List[str], min_score: float = 0.65) -> str:
        """Score each template against the given keywords and return the highest-scoring key.

        If no template exceeds min_score (default 0.65), returns "generic_void".
        """
        keywords_lower = [k.lower() for k in keywords]
        best_key: str = "generic_void"
        best_score: float = -1.0

        for template_key, kw_map in _KEYWORD_SCORES.items():
            score = 0.0
            for kw in keywords_lower:
                score += kw_map.get(kw, 0.0)
            if score > best_score:
                best_score = score
                best_key = template_key

        if best_score < min_score:
            logger.info(
                f"[HybridWorldBuilder] No template matched keywords={keywords} "
                f"(best_score={best_score:.2f} < min_score={min_score}), defaulting to generic_void"
            )
            return "generic_void"

        logger.info(
            f"[HybridWorldBuilder] Matched template={best_key} "
            f"score={best_score:.2f} for keywords={keywords}"
        )
        return best_key

    def match_template_fuzzy(self, text: str, min_score: float = 0.5) -> str:
        """Convenience wrapper: extract single-word tokens from free text and match."""
        tokens = text.lower().split()
        # Strip punctuation from tokens
        stripped = [t.strip(".,!?;:\"'()[]{}") for t in tokens if t.strip(".,!?;:\"'()[]{}")]
        return self.match_template(stripped, min_score)

    # ── Instantiation ──────────────────────────────────────────────────

    def instantiate_world(self, template_key: str) -> Dict[str, RoomMetadata]:
        """Return a deep copy of the room dict for the given template key."""
        if template_key in self.templates:
            result = {rid: RoomMetadata(**r.model_dump()) for rid, r in self.templates[template_key].items()}
            logger.info(f"[HybridWorldBuilder] Instantiated {len(result)} rooms for template={template_key}")
            return result
        logger.warning(f"[HybridWorldBuilder] Unknown template={template_key}, falling back to default_meeting_room")
        return self.templates["default_meeting_room"]

    # ── Dynamic room management ────────────────────────────────────────

    def add_room(self, template_key: str, room: RoomMetadata) -> None:
        """Add a room to an existing template (or create the template if missing)."""
        if template_key not in self.templates:
            self.templates[template_key] = {}
        self.templates[template_key][room.room_id] = room
        logger.info(f"[HybridWorldBuilder] Added room={room.room_id} to template={template_key}")

    def remove_room(self, template_key: str, room_id: str) -> bool:
        """Remove a room from a template. Returns True if the room existed."""
        if template_key not in self.templates:
            return False
        if room_id in self.templates[template_key]:
            del self.templates[template_key][room_id]
            logger.info(f"[HybridWorldBuilder] Removed room={room_id} from template={template_key}")
            return True
        return False

    def get_room(self, template_key: str, room_id: str) -> Optional[RoomMetadata]:
        """Retrieve a specific room from a template."""
        if template_key in self.templates:
            return self.templates[template_key].get(room_id)
        return None

    # ── Dynamic character management ───────────────────────────────────

    def add_character_to_room(self, template_key: str, room_id: str, character_id: str) -> bool:
        """Add a character to a room's present_characters list (no duplicates)."""
        room = self.get_room(template_key, room_id)
        if room is None:
            return False
        if character_id not in room.present_characters:
            room.present_characters.append(character_id)
            logger.info(f"[HybridWorldBuilder] Added character={character_id} to {room_id}")
            return True
        return False

    def remove_character_from_room(self, template_key: str, room_id: str, character_id: str) -> bool:
        """Remove a character from a room's present_characters list."""
        room = self.get_room(template_key, room_id)
        if room is None:
            return False
        if character_id in room.present_characters:
            room.present_characters.remove(character_id)
            logger.info(f"[HybridWorldBuilder] Removed character={character_id} from {room_id}")
            return True
        return False

    def get_nearby_characters(self, template_key: str, room_id: str, radius_ft: float = 15.0) -> List[str]:
        """Return characters in the given room (within a conceptual proximity radius).

        For template-based worlds without real distance tracking, this simply returns
        all characters present in the room.  A future real Evennia integration could
        filter by actual distance.
        """
        room = self.get_room(template_key, room_id)
        if room is None:
            return []
        return list(room.present_characters)

    # ── World utilities ────────────────────────────────────────────────

    def list_templates(self) -> List[str]:
        """Return available template keys."""
        return list(self.templates.keys())

    def get_all_characters_in_world(self, template_key: str) -> Set[str]:
        """Collect every unique character present across all rooms in a template."""
        chars: Set[str] = set()
        if template_key not in self.templates:
            return chars
        for room in self.templates[template_key].values():
            chars.update(room.present_characters)
        return chars
