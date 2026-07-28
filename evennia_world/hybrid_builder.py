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

# ── Default room templates ──────────────────────────────────────────────

DEFAULT_ROOM_TEMPLATES: Dict[str, Dict[str, RoomMetadata]] = {
    "dungeon_cellar": {
        "cellar": RoomMetadata(
            room_id="cellar",
            room_name="The Dungeon Cellar",
            description="A damp, stone-walled cellar filled with wooden crates and rusting iron chains.",
            lighting="dim",
            exits=["tavern_upstairs"],
            present_characters=["rowan", "domino", "luna"],
            nearby_objects=["wooden_crate", "iron_partition"],
        ),
        "tavern_upstairs": RoomMetadata(
            room_id="tavern_upstairs",
            room_name="The Medieval Tavern Main Hall",
            description="A warm tavern hall smelling of roasted meats and ale.",
            lighting="bright",
            exits=["cellar"],
            present_characters=["seamus"],
            nearby_objects=["oak_table", "hearth"],
        ),
    },
    "default_meeting_room": {
        "meeting_room": RoomMetadata(
            room_id="meeting_room",
            room_name="The Meeting Room",
            description="A plain, quiet chamber with a large stone table in the center.",
            lighting="normal",
            exits=[],
            present_characters=["luna"],
            nearby_objects=["stone_table"],
        ),
    },
    "forest_camp": {
        "forest_clearing": RoomMetadata(
            room_id="forest_clearing",
            room_name="Forest Clearing",
            description="An open patch of land surrounded by dense trees, with a small campfire smoldering in the center.",
            lighting="dim",
            exits=["forest_path_north", "forest_path_south", "thickets"],
            present_characters=[],
            nearby_objects=["campfire", "tent", "woodpile"],
        ),
        "forest_path_north": RoomMetadata(
            room_id="forest_path_north",
            room_name="Northern Forest Path",
            description="A narrow dirt trail winding through tall pines. The canopy blocks most sunlight.",
            lighting="dim",
            exits=["forest_clearing"],
            present_characters=[],
            nearby_objects=["fallen_log", "mossy_rocks"],
        ),
        "forest_path_south": RoomMetadata(
            room_id="forest_path_south",
            room_name="Southern Forest Path",
            description="A wider trail leading south toward a distant river. Wildflowers line the edges.",
            lighting="normal",
            exits=["forest_clearing"],
            present_characters=[],
            nearby_objects=["river_stone", "wildflowers"],
        ),
        "thickets": RoomMetadata(
            room_id="thickets",
            room_name="Dense Thickets",
            description="Impassable-looking undergrowth with thorny brambles and tangled roots.",
            lighting="dim",
            exits=["forest_clearing"],
            present_characters=[],
            nearby_objects=["bramble", "vine_tangle"],
        ),
    },
    "castle_exterior": {
        "courtyard": RoomMetadata(
            room_id="courtyard",
            room_name="Castle Courtyard",
            description="A wide stone courtyard surrounded by high castle walls. Guards patrol the perimeter.",
            lighting="bright",
            exits=["great_hall", "armory", "stables", "gate_house"],
            present_characters=[],
            nearby_objects=["stone_bench", "flag_pole", "well"],
        ),
        "great_hall": RoomMetadata(
            room_id="great_hall",
            room_name="The Great Hall",
            description="A towering hall with a vaulted ceiling, long banquet tables, and a raised dais.",
            lighting="bright",
            exits=["courtyard", "throne_room"],
            present_characters=[],
            nearby_objects=["long_table", "tapestry", "guard_railing"],
        ),
        "throne_room": RoomMetadata(
            room_id="throne_room",
            room_name="The Throne Room",
            description="An ornate chamber dominated by a massive stone throne. Royal banners hang from the walls.",
            lighting="bright",
            exits=["great_hall"],
            present_characters=[],
            nearby_objects=["throne", "royal_banners", "marble_floor"],
        ),
        "armory": RoomMetadata(
            room_id="armory",
            room_name="The Armory",
            description="A cold, organized hall lined with weapons racks, armor stands, and weapon crates.",
            lighting="normal",
            exits=["courtyard"],
            present_characters=[],
            nearby_objects=["sword_rack", "shield_wall", "quiver"],
        ),
        "stables": RoomMetadata(
            room_id="stables",
            room_name="The Stables",
            description="A long barn-like structure housing horses and riding equipment. Hay and straw cover the floor.",
            lighting="normal",
            exits=["courtyard"],
            present_characters=[],
            nearby_objects=["saddle_rack", "hay_bale", "water_trough"],
        ),
        "gate_house": RoomMetadata(
            room_id="gate_house",
            room_name="The Gate House",
            description="A fortified entry point with a heavy wooden portcullis and arrow slits.",
            lighting="normal",
            exits=["courtyard"],
            present_characters=[],
            nearby_objects=["portcullis", "gate_key", "sentry_post"],
        ),
    },
    "tavern_common": {
        "tavern_common_room": RoomMetadata(
            room_id="tavern_common_room",
            room_name="The Tavern Common Room",
            description="A bustling common room with wooden tables, a large hearth, and a well-stocked bar.",
            lighting="bright",
            exits=["tavern_kitchen", "tavern_private", "alley"],
            present_characters=[],
            nearby_objects=["bar_counter", "hearth", "board_games"],
        ),
        "tavern_kitchen": RoomMetadata(
            room_id="tavern_kitchen",
            room_name="The Tavern Kitchen",
            description="A steamy kitchen with copper pots hanging from the ceiling and a massive oven.",
            lighting="dim",
            exits=["tavern_common_room"],
            present_characters=[],
            nearby_objects=["copper_pots", "wooden_oven", "spice_racks"],
        ),
        "tavern_private": RoomMetadata(
            room_id="tavern_private",
            room_name="Tavern Private Room",
            description="A small, curtained-off alcove for private conversations and secret meetings.",
            lighting="dim",
            exits=["tavern_common_room"],
            present_characters=[],
            nearby_objects=["velvet_curtain", "small_table", "candlestick"],
        ),
        "alley": RoomMetadata(
            room_id="alley",
            room_name="The Back Alley",
            description="A narrow, cobblestone alley behind the tavern. Garbage bins line the wall.",
            lighting="dim",
            exits=["tavern_common_room"],
            present_characters=[],
            nearby_objects=["garbage_bin", "drainpipe", "cat"],
        ),
    },
}

# Score-based keyword weighting for template matching
_KEYWORD_SCORES: Dict[str, Dict[str, float]] = {
    "dungeon_cellar": {"dungeon": 3.0, "cellar": 3.0, "basement": 2.0, "prison": 2.0, "jail": 2.0, "tavern": 1.5, "ale": 1.0},
    "forest_camp": {"forest": 3.0, "camp": 2.0, "woods": 2.5, "tree": 1.5, "wild": 2.0, "nature": 1.5, "nature": 1.5},
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
            self.templates: Dict[str, Dict[str, RoomMetadata]] = copy.deepcopy(DEFAULT_ROOM_TEMPLATES)

    # ── Template matching ─────────────────────────────────────────────

    def match_template(self, keywords: List[str], min_score: float = 0.5) -> str:
        """Score each template against the given keywords and return the highest-scoring key.

        If no template exceeds min_score, returns "default_meeting_room".
        """
        keywords_lower = [k.lower() for k in keywords]
        best_key: str = "default_meeting_room"
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
                f"(best_score={best_score:.1f} < min_score={min_score}), defaulting"
            )
            return "default_meeting_room"

        logger.info(
            f"[HybridWorldBuilder] Matched template={best_key} "
            f"score={best_score:.1f} for keywords={keywords}"
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
