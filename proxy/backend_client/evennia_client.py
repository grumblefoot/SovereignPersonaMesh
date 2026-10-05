"""
Async HTTP Client for Evennia World State Engine (Port 4005).
Handles action evaluation dispatches and spatial state queries.
"""

import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EvenniaWorldClient:
    def __init__(self, base_url: str = "http://localhost:4005/api/v1"):
        self.base_url = base_url.rstrip("/")
        self._client = None


    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        await self.client.aclose()

    async def submit_action(
        self,
        character_id: str,
        action_type: str,
        raw_text: str,
        target_id: Optional[str] = None,
        session_id: str = "default_session",
        turn_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatches physical intention to Evennia World Engine (POST /api/v1/world/action)."""
        payload = {
            "character_id": character_id,
            "action_type": action_type,
            "target_id": target_id,
            "raw_text": raw_text,
            "session_id": session_id,
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
        endpoint = f"{self.base_url}/world/action"
        
        resp = await self.client.post(endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_character_state(self, character_id: str, session_id: str = "default_session") -> Dict[str, Any]:
        """Queries current spatial state for a character (GET /api/v1/world/state)."""
        endpoint = f"{self.base_url}/world/state?character_id={character_id}&session_id={session_id}"
        
        resp = await self.client.get(endpoint)
        resp.raise_for_status()
        return resp.json()

    async def move_character(self, character_id: str, room_id: str, session_id: str = "default_session", idempotency_key: Optional[str] = None, origin: str = "system") -> Dict[str, Any]:
        """Moves a character to a room."""
        payload = {
            "character_id": character_id,
            "room_id": room_id,
            "session_id": session_id,
            "template_key": "dynamic",
            "origin": origin,
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key  # body: the server ignores the header form
        endpoint = f"{self.base_url}/world/move"
        headers = {"X-Idempotency-Key": idempotency_key} if idempotency_key else {}
        
        resp = await self.client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


    async def configure_world(self, template_key: str, session_id: str = "default_session",
                              placements: list = None, origin: str = "system") -> Dict[str, Any]:
        """Instantiate a template into a session, optionally seeding character placements."""
        payload = {"template_key": template_key, "session_id": session_id,
                   "placements": placements or [], "origin": origin}
        resp = await self.client.post(f"{self.base_url}/world/configure", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def set_barrier(self, a: str, b: str, barrier: str = "closed_door", state: str = "closed",
                          distance_ft: float = 15.0, session_id: str = "default_session",
                          template_key: str = "dynamic", idempotency_key: Optional[str] = None,
                          origin: str = "system") -> Dict[str, Any]:
        """Create/update a stateful edge (door, partition) between two rooms (A2)."""
        payload = {"a": a, "b": b, "barrier": barrier, "state": state, "distance_ft": distance_ft,
                   "session_id": session_id, "template_key": template_key, "origin": origin}
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        resp = await self.client.post(f"{self.base_url}/world/barrier", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def create_room(self, room_id: str, name: str, desc: str, session_id: str = "default_session", idempotency_key: Optional[str] = None, origin: str = "system") -> Dict[str, Any]:
        """Creates a new dynamic room."""
        payload = {
            "room_id": room_id,
            "room_name": name,
            "description": desc,
            "session_id": session_id,
            "template_key": "dynamic",
            "origin": origin,
        }
        # The server reads idempotency_key from the BODY; the old header-only form was
        # silently ignored, so GM-action replays were never deduped over the wire.
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        endpoint = f"{self.base_url}/world/rooms"
        headers = {"X-Idempotency-Key": idempotency_key} if idempotency_key else {}
        
        resp = await self.client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
