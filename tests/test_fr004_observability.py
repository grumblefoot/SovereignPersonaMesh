"""
FR-004: Observability, Real-Time Logging & Built-in Admin Dashboard Tests.

Verifies:
  1. SettingsManager (reading, writing, updating config/config.json)
  2. TelemetryCollector (recording requests, spatial gating breakdown, log buffer)
  3. Admin REST API (/admin/api/v1/stats, /admin/api/v1/config, /admin/api/v1/sessions/{id}, /admin/api/v1/factory_reset)
  4. Web Admin Dashboard HTML serving (GET /admin)
"""

import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from config.manager import SettingsManager
from proxy.core.telemetry import TelemetryCollector
from proxy.main import app as proxy_app


class TestFR004Observability:
    """Test suite for FR-004 Observability & Admin Dashboard."""

    def test_settings_manager_read_write(self, tmp_path):
        """SettingsManager reads, merges defaults, and writes config.json without data corruption."""
        config_file = str(tmp_path / "test_config.json")
        mgr = SettingsManager(config_file)

        settings = mgr.get_settings()
        assert settings["SPM_PROXY_PORT"] == 5050
        assert settings["BACKEND_LLM_URL"] == "http://localhost:8000/v1"

        updated = mgr.write_settings({"SPM_PROXY_PORT": 9090, "CUSTOM_KEY": "test_val"})
        assert updated["SPM_PROXY_PORT"] == 9090
        assert updated["CUSTOM_KEY"] == "test_val"

        # Re-read from disk
        mgr2 = SettingsManager(config_file)
        settings2 = mgr2.get_settings()
        assert settings2["SPM_PROXY_PORT"] == 9090

    def test_telemetry_collector_metrics(self):
        """TelemetryCollector tracks requests, latencies, spatial gating counts, and logs."""
        collector = TelemetryCollector(log_buffer_size=10)
        collector.record_request("sess_1", "direct", 12.5, rag_count=3)
        collector.record_request("sess_1", "muffled", 18.2, rag_count=1)
        collector.record_request("sess_2", "blackout", 1.2, rag_count=0)

        metrics = collector.get_metrics()
        assert metrics["total_requests"] == 3
        assert metrics["spatial_gating_counts"]["direct"] == 1
        assert metrics["spatial_gating_counts"]["muffled"] == 1
        assert metrics["spatial_gating_counts"]["blackout"] == 1
        assert metrics["rag_retrieval_hits"] == 4
        assert metrics["active_sessions_count"] == 2

        logs = collector.get_logs()
        assert len(logs) == 3
        assert logs[0]["gating_level"] == "direct"

    def test_admin_api_endpoints(self):
        """Test GET /admin/api/v1/stats and GET/POST /admin/api/v1/config."""
        with TestClient(proxy_app) as client:
            # Test /stats
            resp_stats = client.get("/admin/api/v1/stats")
            assert resp_stats.status_code == 200
            data_stats = resp_stats.json()
            assert "total_requests" in data_stats
            assert "spatial_gating_counts" in data_stats

            # Test GET /config
            resp_cfg = client.get("/admin/api/v1/config")
            assert resp_cfg.status_code == 200
            assert "SPM_PROXY_PORT" in resp_cfg.json()

            # Test POST /config
            resp_update = client.post(
                "/admin/api/v1/config",
                json={"BACKEND_LLM_URL": "http://127.0.0.1:9999/v1"}
            )
            assert resp_update.status_code == 200
            assert resp_update.json()["settings"]["BACKEND_LLM_URL"] == "http://127.0.0.1:9999/v1"

    def test_admin_dashboard_ui_serving(self):
        """GET /admin serves the HTML Admin Dashboard."""
        with TestClient(proxy_app) as client:
            resp = client.get("/admin")
            assert resp.status_code == 200
            assert "html" in resp.headers["content-type"]
            assert "Sovereign Persona Mesh (SPM) Admin" in resp.text

    @pytest.mark.asyncio
    async def test_session_deletion_and_factory_reset_mock(self):
        """DELETE /admin/api/v1/sessions/{id} and DELETE /admin/api/v1/factory_reset."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"table_name": "csa_memory_seraphina"}]
        mock_conn.execute.return_value = "DELETE 5"

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_cm

        with patch("proxy.api.admin_routes._admin_db_pool", mock_pool):
            with TestClient(proxy_app) as client:
                # Test session deletion
                resp_del = client.delete("/admin/api/v1/sessions/test_sess_99")
                assert resp_del.status_code == 200
                assert resp_del.json()["deleted_records"] == 5

                # Test factory reset
                resp_reset = client.delete("/admin/api/v1/factory_reset")
                assert resp_reset.status_code == 200
                assert resp_reset.json()["status"] == "success"
