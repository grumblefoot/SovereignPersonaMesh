"""
Session & Tick Lock Manager for Evennia Engine.
Freezes world state evolution and action_tick until active Character Subagents complete turn generation.
"""

import asyncio
import uuid
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SessionLockManager:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tokens: Dict[str, str] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def acquire_lock(self, session_id: str) -> str:
        """Acquire tick lock for session, returning a unique lock token."""
        lock = self.get_lock(session_id)
        await lock.acquire()
        token = str(uuid.uuid4())
        self._tokens[session_id] = token
        logger.info(f"[SessionLock] Lock acquired for session={session_id}, token={token}")
        return token

    def release_lock(self, session_id: str, lock_token: str) -> bool:
        """Release tick lock for session if token matches."""
        if session_id in self._tokens and self._tokens[session_id] == lock_token:
            lock = self._locks.get(session_id)
            if lock and lock.locked():
                lock.release()
                del self._tokens[session_id]
                logger.info(f"[SessionLock] Lock released for session={session_id}, token={lock_token}")
                return True
        logger.warning(f"[SessionLock] Failed release lock attempt for session={session_id}, invalid token.")
        return False
