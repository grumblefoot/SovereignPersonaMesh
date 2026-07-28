"""
Session & Tick Lock Manager for Evennia Engine.
Freezes world state evolution and action_tick until active Character Subagents complete turn generation.
Includes lock expiry (TTL), concurrent-request guards, and stale-lock cleanup.
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LockError(Exception):
    """Raised when lock operations fail (expired, stale, wrong token)."""


class SessionLockManager:
    """Manages per-session async locks with TTL expiry and concurrent-request guards.

    Each session has:
      - An asyncio.Lock (for mutual exclusion during turn generation).
      - A lock_token (str | None) for verification on release.
      - A lock_expiry (float | None) -- absolute monotonic timestamp after which
        the lock is considered expired and must be cleaned up.
      - A created_at timestamp for logging.
    """

    # Default TTL: 60 seconds before auto-expiry.
    DEFAULT_TTL_SECONDS: float = 60.0

    # Maximum concurrent locks allowed per session.
    # 1 = only one tick can be active at a time (standard Evennia turn order).
    MAX_CONCURRENT_PER_SESSION: int = 1

    def __init__(self, default_ttl: float = DEFAULT_TTL_SECONDS):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tokens: Dict[str, str] = {}
        # Maps session_id -> info dict with keys: token, expiry, created_at, depth
        self._lock_info: Dict[str, Dict] = {}
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Return (or create) the asyncio.Lock for the session."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def check_stale_locks(self, current_time: float | None = None) -> list[str]:
        """Return session_ids whose locks have expired but haven't been released.

        This is a synchronous helper so tests can trigger cleanup without
        needing an event loop.  In production the app middleware would call
        it periodically (e.g. every 10 s).
        """
        if current_time is None:
            current_time = time.monotonic()
        stale = []
        for sid, info in list(self._lock_info.items()):
            if info.get("expiry") and current_time > info["expiry"]:
                stale.append(sid)
        return stale

    def cleanup_expired_locks(self, current_time: float | None = None) -> list[str]:
        """Remove expired locks and their asyncio.Lock objects. Returns cleaned IDs."""
        if current_time is None:
            current_time = time.monotonic()
        stale = self.check_stale_locks(current_time)
        for sid in stale:
            self._cleanup_session(sid)
        return stale

    async def acquire_lock(self, session_id: str, ttl: float | None = None) -> str:
        """Acquire tick lock for session, returning a unique lock token.

        Raises LockError if:
          - A lock on this session is already active (concurrent guard).
          - An existing lock is stale/expired (auto-cleanup then retry).
        """
        ttl = ttl if ttl is not None else self._default_ttl
        now = time.monotonic()

        # --- cleanup any stale lock first ---
        if session_id in self._lock_info:
            info = self._lock_info[session_id]
            if info.get("expiry") and now > info["expiry"]:
                logger.warning(
                    f"[SessionLock] Stale lock for session={session_id} "
                    f"(expired at {info['expiry']:.2f}), auto-cleaning"
                )
                self._cleanup_session(session_id)

        # Check for active (non-expired) lock
        if session_id in self._lock_info:
            raise LockError(
                f"Lock already held for session={session_id}"
            )

        # Acquire the asyncio.Lock
        lock = self.get_lock(session_id)
        await lock.acquire()

        # Generate token and record metadata
        token = str(uuid.uuid4())
        self._tokens[session_id] = token
        self._lock_info[session_id] = {
            "token": token,
            "expiry": now + ttl,
            "created_at": now,
            "depth": 1,
        }
        logger.info(
            f"[SessionLock] Lock acquired for session={session_id}, "
            f"token={token}, ttl={ttl}s"
        )
        return token

    def release_lock(self, session_id: str, lock_token: str | None = None) -> bool:
        """Release tick lock for session if token matches.

        Returns True on success, False if the token is wrong or no lock exists.
        """
        if session_id not in self._tokens:
            logger.warning(
                f"[SessionLock] No active lock for session={session_id} (release denied)"
            )
            return False

        if lock_token is None or self._tokens[session_id] != lock_token:
            logger.warning(
                f"[SessionLock] Token mismatch for session={session_id}. "
                f"Expected {self._tokens[session_id]}, got {lock_token}"
            )
            return False

        lock = self._locks.get(session_id)
        if lock is None or not lock.locked():
            logger.warning(
                f"[SessionLock] asyncio.Lock not locked for session={session_id}"
            )
            return False

        lock.release()
        self._cleanup_session(session_id)
        logger.info(
            f"[SessionLock] Lock released for session={session_id}, "
            f"token={lock_token}"
        )
        return True

    def verify_lock(self, session_id: str, lock_token: str) -> bool:
        """Check whether a given token matches the current lock (without releasing)."""
        if session_id not in self._tokens:
            return False
        return self._tokens[session_id] == lock_token

    def get_lock_info(self, session_id: str) -> Optional[Dict]:
        """Return lock metadata for the session, or None."""
        if session_id not in self._lock_info:
            return None
        info = self._lock_info[session_id]
        remaining = max(0.0, info.get("expiry", 0) - time.monotonic())
        return {
            "session_id": session_id,
            "token": info["token"],
            "remaining_ttl_seconds": round(remaining, 2),
            "created_at": info["created_at"],
            "expired": remaining <= 0,
        }

    def is_locked(self, session_id: str) -> bool:
        """Return True if the session has an active (non-expired) lock."""
        if session_id not in self._lock_info:
            return False
        info = self._lock_info[session_id]
        if info.get("expiry") and time.monotonic() > info["expiry"]:
            return False
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup_session(self, session_id: str) -> None:
        """Remove all tracked state for a session."""
        self._tokens.pop(session_id, None)
        self._lock_info.pop(session_id, None)
