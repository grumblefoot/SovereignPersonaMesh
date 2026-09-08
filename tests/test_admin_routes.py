import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient
from proxy.main import app
from proxy.api.admin_routes import (
    set_admin_db_pool,
    get_admin_db_pool,
    ensure_db_pool
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_admin_pool():
    set_admin_db_pool(None)
    import proxy.api.routes
    proxy.api.routes._db_pool = None
    yield
    set_admin_db_pool(None)
    proxy.api.routes._db_pool = None

@pytest.mark.asyncio
async def test_ensure_db_pool():
    with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = "mock_pool"
        pool = await ensure_db_pool()
        assert pool == "mock_pool"
        assert get_admin_db_pool() == "mock_pool"
        
        # Test second call uses cached pool
        pool2 = await ensure_db_pool()
        assert pool2 == "mock_pool"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_ensure_db_pool_failure():
    with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = Exception("DB error")
        pool = await ensure_db_pool()
        assert pool is None

def test_get_admin_stats_with_db():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetchval.return_value = 10485760  # 10 MB
    
    set_admin_db_pool(mock_pool)
    
    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        mock_tel.return_value.get_metrics.return_value = {"active_sessions": 2}
        response = client.get("/admin/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["active_sessions"] == 2
        assert data["db_size_mb"] == 10.0

def test_get_admin_stats_db_error():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetchval.side_effect = Exception("error")
    
    set_admin_db_pool(mock_pool)
    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        mock_tel.return_value.get_metrics.return_value = {"active_sessions": 2}
        response = client.get("/admin/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["db_size_mb"] == 0.0

def test_get_config():
    with patch('proxy.api.admin_routes.get_settings_manager') as mock_mgr:
        mock_mgr.return_value.get_settings.return_value = {"setting_1": "value"}
        response = client.get("/admin/api/v1/config")
        assert response.status_code == 200
        assert response.json() == {"setting_1": "value"}

def test_post_config():
    with patch('proxy.api.admin_routes.get_settings_manager') as mock_mgr:
        mock_mgr.return_value.write_settings.return_value = {"setting_1": "new"}
        response = client.post("/admin/api/v1/config", json={"setting_1": "new"})
        assert response.status_code == 200
        assert response.json() == {"status": "success", "settings": {"setting_1": "new"}}

def test_get_logs():
    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        mock_tel.return_value.get_logs.return_value = ["log1"]
        response = client.get("/admin/api/v1/logs?session_id=s1")
        assert response.status_code == 200
        assert response.json() == {"logs": ["log1"]}
        mock_tel.return_value.get_logs.assert_called_with(session_id="s1", level=None, limit=50)

def test_get_session_trace():
    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        mock_tel.return_value.get_session_trace.return_value = ["trace1"]
        response = client.get("/admin/api/v1/sessions/s1/trace")
        assert response.status_code == 200
        assert response.json() == {"session_id": "s1", "trace": ["trace1"]}

def test_delete_session():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetch.return_value = [{"table_name": "csa_memory_char1"}]
    mock_conn.execute.return_value = "DELETE 5"
    
    set_admin_db_pool(mock_pool)
    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        response = client.delete("/admin/api/v1/sessions/s1")
        assert response.status_code == 200
        assert response.json()["deleted_records"] == 5
        mock_tel.return_value.dismiss_session.assert_called_with("s1")

def test_delete_session_no_db():
    set_admin_db_pool(None)
    with patch('proxy.api.admin_routes.get_admin_db_pool', return_value=None):
        response = client.delete("/admin/api/v1/sessions/s1")
        assert response.status_code == 503

def test_delete_session_error():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.execute.side_effect = Exception("DB Error")
    
    set_admin_db_pool(mock_pool)
    response = client.delete("/admin/api/v1/sessions/s1")
    assert response.status_code == 500

def test_factory_reset():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.fetch.return_value = [{"table_name": "csa_memory_char1"}]
    
    set_admin_db_pool(mock_pool)
    
    # Mock httpx context manager for httpx.AsyncClient delete
    mock_client = AsyncMock()
    class HttpxMockContext:
        async def __aenter__(self):
            return mock_client
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch('proxy.api.admin_routes.get_telemetry_collector') as mock_tel:
        with patch('httpx.AsyncClient', return_value=HttpxMockContext()):
            response = client.delete("/admin/api/v1/factory_reset")
            assert response.status_code == 200
            assert "Factory reset complete" in response.json()["message"]
            mock_tel.return_value.reset.assert_called_once()
            mock_client.delete.assert_called_once()

def test_factory_reset_no_db():
    set_admin_db_pool(None)
    with patch('proxy.api.admin_routes.ensure_db_pool', new_callable=AsyncMock) as mock_ensure:
        mock_ensure.return_value = None
        response = client.delete("/admin/api/v1/factory_reset")
        assert response.status_code == 503

def test_factory_reset_error():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    class PoolMock:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_pool.acquire.return_value = PoolMock()
    mock_conn.execute.side_effect = Exception("DB Error")
    
    set_admin_db_pool(mock_pool)
    response = client.delete("/admin/api/v1/factory_reset")
    assert response.status_code == 500

def test_shutdown_system():
    with patch('subprocess.run') as mock_sub:
        with patch('asyncio.create_task') as mock_task:
            response = client.post("/admin/api/v1/shutdown")
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            mock_task.assert_called_once()
