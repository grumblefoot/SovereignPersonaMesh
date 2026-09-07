import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.core.telemetry import get_telemetry_collector

@pytest.fixture(autouse=True)
def reset_telemetry():
    tel = get_telemetry_collector()
    tel.reset()
    yield
    tel.reset()

def test_telemetry_singleton():
    t1 = get_telemetry_collector()
    t2 = get_telemetry_collector()
    assert t1 is t2

def test_record_turn_trace():
    tel = get_telemetry_collector()
    tel.record_turn_trace("s1", {"action": "think"})
    trace = tel.get_session_trace("s1")
    assert len(trace) == 1
    assert trace[0]["action"] == "think"
    assert "timestamp" in trace[0]

def test_subscribe_thoughts():
    tel = get_telemetry_collector()
    
    class DummyQueue:
        def put_nowait(self, item):
            self.item = item
            
    q = DummyQueue()
    tel.subscribe_thoughts(q)
    
    tel.push_thought_event("s1", {"thought": "test"})
    assert hasattr(q, "item")
    assert q.item["thought"] == "test"
    assert q.item["session_id"] == "s1"
    
    recent = tel.get_recent_thoughts()
    assert len(recent) == 1
    assert recent[0]["thought"] == "test"
    
    tel.unsubscribe_thoughts(q)
    tel.push_thought_event("s1", {"thought": "test2"})
    assert q.item["thought"] == "test"  # Didn't get updated

def test_record_request():
    tel = get_telemetry_collector()
    tel.record_request(
        session_id="s2",
        gating_level="muffled",
        latency=1.5,
        rag_count=2,
        status_code=400,
        location_name="Tavern"
    )
    
    stats = tel.get_metrics()
    assert stats["total_requests"] == 1
    assert stats["average_turn_latency_ms"] == 1.5
    assert stats["gating_breakdown"]["muffled"] == 1
    assert stats["rag_retrieval_hits"] == 2
    
    logs = tel.get_logs()
    assert len(logs) == 1
    assert logs[0]["level"] == "ERROR"
    assert logs[0]["location_name"] == "Tavern"

def test_record_rag_hit():
    tel = get_telemetry_collector()
    tel.record_rag_hit()
    stats = tel.get_metrics()
    assert stats["rag_retrieval_hits"] == 1

def test_record_dismiss_session():
    tel = get_telemetry_collector()
    tel.record_session("s3")
    assert "s3" in tel._active_sessions
    
    tel.dismiss_session("s3")
    assert "s3" not in tel._active_sessions

def test_set_memory_tiers():
    tel = get_telemetry_collector()
    tel.set_memory_tiers(hot=10, warm=5, cold=2)
    stats = tel.get_metrics()
    assert stats["memory_tiers"]["hot"] == 10
    assert stats["memory_tiers"]["warm"] == 5
    assert stats["memory_tiers"]["cold"] == 2

def test_append_log():
    tel = get_telemetry_collector()
    tel.append_log("INFO", "msg", "s4", "source1")
    logs = tel.get_logs(session_id="s4", level="info")
    assert len(logs) == 1
    assert logs[0]["message"] == "msg"

@pytest.mark.asyncio
async def test_hydrate_from_db_empty():
    tel = get_telemetry_collector()
    await tel.hydrate_from_db(None)
    
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetch.return_value = []
    
    await tel.hydrate_from_db(mock_pool)
    stats = tel.get_metrics()
    assert stats["active_sessions"] == 0

@pytest.mark.asyncio
async def test_hydrate_from_db_success():
    tel = get_telemetry_collector()
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    
    # First fetch for tables
    # Second fetch for records
    mock_conn.fetch.side_effect = [
        [{"table_name": "csa_memory_char1"}],
        [
            {
                "character": "char1",
                "session_id": "sess_1",
                "timestamp": datetime(2026, 1, 1),
                "sensory_input": "in",
                "inner_monologue": "mono",
                "public_response": "out"
            }
        ]
    ]
    
    await tel.hydrate_from_db(mock_pool)
    
    assert "sess_1" in tel._active_sessions
    recent = tel.get_recent_thoughts()
    assert len(recent) == 1
    assert recent[0]["character"] == "char1"
    
    trace = tel.get_session_trace("sess_1")
    assert len(trace) == 1
    assert trace[0]["sensory_input"] == "in"

@pytest.mark.asyncio
async def test_hydrate_from_db_error():
    tel = get_telemetry_collector()
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetch.side_effect = Exception("DB error")
    
    # Should safely catch exception
    await tel.hydrate_from_db(mock_pool)
