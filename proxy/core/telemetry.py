"""
Telemetry collector for SPM Proxy (Port 5050).

Singleton class that tracks uptime, total requests, average turn latency,
spatial gating breakdown, RAG retrieval hits, active sessions, and a
circular log buffer (last 100 entries).
"""

import time
import threading
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional


class TelemetryCollector:
    """Thread-safe singleton telemetry collector."""

    _instance: Optional["TelemetryCollector"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_buffer_size: int = 100):
        if self._initialized:
            return
        self._initialized = True
        self._start_time = time.time()
        self._total_requests = 0
        self._total_latency = 0.0
        self._gating_counts = defaultdict(int)  # "direct", "muffled", "blackout", "null"
        self._rag_hits = 0
        self._active_sessions = set()
        self._memory_tiers = {"hot": 0, "warm": 0, "cold": 0}
        self._lock = threading.Lock()
        # Circular buffer: last N log entries
        self._log_buffer: deque = deque(maxlen=log_buffer_size)

    def record_request(
        self,
        session_id: str = "default_session",
        gating_level: str = "direct",
        latency: float = 0.0,
        rag_count: int = 0,
        status_code: int = 200,
    ) -> None:
        """Record a completed request with its latency and gating classification."""
        with self._lock:
            self._total_requests += 1
            self._total_latency += latency
            self._gating_counts[gating_level] += 1
            self._rag_hits += rag_count
            self._active_sessions.add(session_id)
            self._log_buffer.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "level": "INFO" if status_code < 400 else "ERROR",
                "session_id": session_id,
                "gating_level": gating_level,
                "latency_ms": round(latency, 2),
                "rag_count": rag_count,
                "status_code": status_code,
            })

    def record_rag_hit(self) -> None:
        """Record a successful RAG memory retrieval."""
        with self._lock:
            self._rag_hits += 1

    def record_session(self, session_id: str) -> None:
        """Mark a session as active."""
        with self._lock:
            self._active_sessions.add(session_id)

    def dismiss_session(self, session_id: str) -> None:
        """Mark a session as inactive."""
        with self._lock:
            self._active_sessions.discard(session_id)

    def set_memory_tiers(self, hot: int = 0, warm: int = 0, cold: int = 0) -> None:
        """Set current memory tier counts."""
        with self._lock:
            self._memory_tiers = {"hot": hot, "warm": warm, "cold": cold}

    def append_log(self, level: str, message: str, session_id: Optional[str] = None,
                   source: Optional[str] = None) -> None:
        """Append a log entry to the circular buffer."""
        with self._lock:
            self._log_buffer.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "level": level,
                "message": message,
                "session_id": session_id,
                "source": source,
            })

    # -- Query --

    def get_stats(self, db_pool: Optional[Any] = None) -> Dict[str, Any]:
        """Return a snapshot of all telemetry data as a dict."""
        with self._lock:
            uptime = time.time() - self._start_time
            avg_latency = (self._total_latency / self._total_requests
                           if self._total_requests else 0.0)
            return {
                "uptime_seconds": round(uptime, 2),
                "uptime_human": self._humanize_uptime(uptime),
                "total_requests": self._total_requests,
                "average_turn_latency_ms": round(avg_latency * 1000, 4),
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "gating_breakdown": dict(self._gating_counts),
                "spatial_gating_counts": dict(self._gating_counts),
                "rag_retrieval_hits": self._rag_hits,
                "active_sessions": len(self._active_sessions),
                "active_sessions_count": len(self._active_sessions),
                "memory_tiers": dict(self._memory_tiers),
                "log_buffer_size": len(self._log_buffer),
            }

    get_metrics = get_stats

    def get_logs(self, session_id: Optional[str] = None,
                 level: Optional[str] = None,
                 limit: int = 50) -> List[Dict[str, Any]]:
        """Return filtered log entries from the circular buffer."""
        with self._lock:
            entries = list(self._log_buffer)
        # Filter by session_id
        if session_id is not None:
            entries = [e for e in entries if e.get("session_id") == session_id]
        # Filter by level
        if level is not None:
            entries = [e for e in entries if e.get("level") == level.upper()]
        # Limit
        return entries[-limit:] if limit else entries

    @staticmethod
    def _humanize_uptime(seconds: float) -> str:
        """Convert seconds to human-readable uptime string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)


def get_telemetry_collector() -> TelemetryCollector:
    """Get the singleton TelemetryCollector instance."""
    return TelemetryCollector()
