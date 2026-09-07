"""
Declarative GM Action Registry.
Defines available Game Master actions that the LLM can output inside its <thinking> monologue.
Each action defines strict typing and a static prompt fragment for CognitivePromptBuilder injection.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class GMActionDefinition:
    id: str
    description: str
    schema_example: str
    prompt_fragment: str
    tags: List[str] = field(default_factory=list)
    priority: int = 0


class GMActionRegistry:
    def __init__(self):
        self._actions: Dict[str, GMActionDefinition] = {}
        self._register_default_actions()

    def register(self, action: GMActionDefinition) -> None:
        self._actions[action.id] = action

    def get(self, action_id: str) -> Optional[GMActionDefinition]:
        return self._actions.get(action_id)

    def get_all_active(self, include_tags: Optional[List[str]] = None) -> List[GMActionDefinition]:
        """Returns all actions, optionally filtered by tags, sorted by priority descending."""
        if include_tags is None:
            actions = list(self._actions.values())
        else:
            actions = [a for a in self._actions.values() if any(t in include_tags for t in a.tags)]
        
        return sorted(actions, key=lambda a: a.priority, reverse=True)

    def _register_default_actions(self):
        self.register(GMActionDefinition(
            id="MOVE",
            description="Move a character or entity to a different room",
            schema_example='{"type": "MOVE", "entity": "user", "room_id": "new_room"}',
            prompt_fragment='To move an entity: [GM_ACTION: {"type": "MOVE", "entity": "character_name", "room_id": "snake_case_room_id"}]',
            priority=100
        ))
        
        self.register(GMActionDefinition(
            id="CREATE_ROOM",
            description="Create a new room when a new location is discovered",
            schema_example='{"type": "CREATE_ROOM", "room_id": "new_room", "name": "Room Name", "desc": "Room description"}',
            prompt_fragment='To create a new room: [GM_ACTION: {"type": "CREATE_ROOM", "room_id": "snake_case_room_id", "name": "Room Name", "desc": "Room description"}]',
            priority=90
        ))


# Global default instance (can be overridden or mocked in tests)
default_gm_registry = GMActionRegistry()
