"""
Hybrid Semantic-Template World Construction Engine.
Matches SillyTavern Lorebooks/character cards to rigid, pre-tested Evennia room network templates.
"""

import logging
from typing import Dict, Any, List
from .models import RoomMetadata

logger = logging.getLogger(__name__)

PREDEFINED_ROOM_TEMPLATES = {
    "dungeon_cellar": {
        "cellar": RoomMetadata(
            room_id="cellar",
            room_name="The Dungeon Cellar",
            description="A damp, stone-walled cellar filled with wooden crates and rusting iron chains.",
            lighting="dim",
            exits=["tavern_upstairs"],
            present_characters=["rowan", "domino", "luna"],
            nearby_objects=["wooden_crate", "iron_partition"]
        ),
        "tavern_upstairs": RoomMetadata(
            room_id="tavern_upstairs",
            room_name="The Medieval Tavern Main Hall",
            description="A warm tavern hall smelling of roasted meats and ale.",
            lighting="bright",
            exits=["cellar"],
            present_characters=["seamus"],
            nearby_objects=["oak_table", "hearth"]
        )
    },
    "default_meeting_room": {
        "meeting_room": RoomMetadata(
            room_id="meeting_room",
            room_name="The Meeting Room",
            description="A plain, quiet chamber with a large stone table in the center.",
            lighting="normal",
            exits=[],
            present_characters=["luna"],
            nearby_objects=["stone_table"]
        )
    }
}


class HybridWorldBuilder:
    def __init__(self):
        self.templates = PREDEFINED_ROOM_TEMPLATES

    def match_template(self, keywords: List[str]) -> str:
        """Match lorebook/character keywords to closest room template."""
        keywords_lower = [k.lower() for k in keywords]
        for kw in keywords_lower:
            if "dungeon" in kw or "cellar" in kw or "basement" in kw:
                return "dungeon_cellar"
        return "default_meeting_room"

    def instantiate_world(self, template_key: str) -> Dict[str, RoomMetadata]:
        """Instantiate pre-tested room network."""
        logger.info(f"[HybridWorldBuilder] Instantiating room template: {template_key}")
        return self.templates.get(template_key, self.templates["default_meeting_room"])
