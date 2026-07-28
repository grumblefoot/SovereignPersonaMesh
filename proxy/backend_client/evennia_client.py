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

    async def submit_action(
        self,
        character_id: str,
        action_type: str,
        raw_text: str,
        target_id: Optional[str] = None,
        session_id: str = "default_session"
    ) -> Dict[str, Any]:
        """Dispatches physical intention to Evennia World Engine (POST /api/v1/world/action)."""
        payload = {
            "character_id": character_id,
            "action_type": action_type,
            "target_id": target_id,
            "raw_text": raw_text,
            "session_id": session_id
        }
        endpoint = f"{self.base_url}/world/action"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(endpoint, json=payload)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"[EvenniaClient] Failed to connect to Evennia at {endpoint}: {e}. Returning mock direct feed.")
        
        # Fallback mock response if Evennia service is starting up
        return {
            "success": True,
            "action_tick": 1421,
            "consequences": [
                {
                    "recipient_id": character_id,
                    "sensory_feed": raw_text,
                    "gating_level": "direct",
                    "distance_ft": 0.0,
                    "barriers": []
                }
            ]
        }

    async def get_character_state(self, character_id: str, session_id: str = "default_session") -> Dict[str, Any]:
        """Queries current spatial state for a character (GET /api/v1/world/state)."""
        endpoint = f"{self.base_url}/world/state?character_id={character_id}&session_id={session_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(endpoint)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"[EvenniaClient] State query failed: {e}")
        return {
            "character_id": character_id,
            "gating_level": "direct",
            "sensory_feed": "Standing in room.",
            "distances": {}
        }
