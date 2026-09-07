import pytest
import os
import asyncio
from unittest.mock import patch, AsyncMock, mock_open, MagicMock

from proxy.rag.import_worker import (
    _detect_hardware_tier,
    _compute_dynamic_batch_size,
    HardwareTierEnum,
    BulkImportWorker,
    get_import_worker
)

def test_detect_hardware_tier_exceptions():
    with patch("builtins.open", side_effect=FileNotFoundError):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.EXPERIMENTAL

    # Test PERFORMANCE tier branch
    mock_mem = "MemTotal:       17000000 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_mem)):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.PERFORMANCE

def test_compute_dynamic_batch_size_exceptions():
    with patch("os.cpu_count", side_effect=Exception):
        with patch("builtins.open", side_effect=FileNotFoundError):
            batch_size = _compute_dynamic_batch_size()
            assert batch_size >= 2

@pytest.mark.asyncio
async def test_get_all_imports():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_pool.acquire.return_value = PoolMock()
    
    from datetime import datetime
    mock_conn.fetch.return_value = [
        {
            "import_id": "test_id_1",
            "session_id": "sess1",
            "character_id": "char1",
            "status": "completed",
            "total_messages": 10,
            "processed_messages": 10,
            "error_log": None,
            "created_at": datetime(2026, 1, 1),
            "updated_at": datetime(2026, 1, 1)
        },
        {
            "import_id": "test_id_2",
            "session_id": "sess2",
            "character_id": "char2",
            "status": "pending",
            "total_messages": 20,
            "processed_messages": 0,
            "error_log": "error",
            "created_at": None,
            "updated_at": None
        }
    ]
    
    worker = BulkImportWorker(db_pool=mock_pool)
    result = await worker.get_all_imports()
    
    assert len(result) == 2
    assert result[0]["import_id"] == "test_id_1"
    assert result[0]["created_at"] == "2026-01-01T00:00:00"
    assert result[1]["created_at"] is None

def test_get_import_worker():
    mock_pool1 = AsyncMock()
    mock_pool2 = AsyncMock()
    
    # First call initializes
    worker1 = get_import_worker(mock_pool1)
    assert worker1.db_pool is mock_pool1
    
    # Second call with same pool returns exact same worker
    worker2 = get_import_worker(mock_pool1)
    assert worker1 is worker2
    
    # Third call with new pool creates new worker
    worker3 = get_import_worker(mock_pool2)
    assert worker3 is not worker1
    assert worker3.db_pool is mock_pool2

@pytest.mark.asyncio
async def test_import_telemetry_and_regex_fallbacks():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_pool.acquire.return_value = PoolMock()
    worker = BulkImportWorker(db_pool=mock_pool)
    
    # Mock embedder
    worker.embedder = AsyncMock()
    worker.embedder.batch_generate_embeddings.return_value = [[0.1] for _ in range(4)]
    
    messages = [
        {"content": "No location here"},
        {"content": "A prison door opens"},
        {"content": "Going to the hall"},
        {"content": "Area: Secret Base"}
    ]
    
    with patch("proxy.core.telemetry.get_telemetry_collector") as mock_telemetry:
        # First test: telemetry succeeds
        mock_collector = MagicMock()
        mock_telemetry.return_value = mock_collector
        
        await worker.process_bulk_import_background("sess", "char", messages, skip_registration=True)
        
        assert mock_collector.record_request.call_count >= 2
        
        # Second test: telemetry throws exception
        mock_telemetry.side_effect = Exception("Telemetry failed")
        # Should catch exception and pass
        await worker.process_bulk_import_background("sess2", "char2", messages, skip_registration=True)
