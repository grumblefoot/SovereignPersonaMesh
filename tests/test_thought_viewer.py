"""
Unit tests for FR-007 / Issue #6: Real-Time Observability & "Thought Viewer" Logging in Admin Dashboard.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from proxy.core.telemetry import get_telemetry_collector
from proxy.api.admin_routes import router as admin_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(admin_router)
client = TestClient(app)


class TestTelemetryCollectorTraces:
    def test_record_and_get_session_trace(self):
        telemetry = get_telemetry_collector()
        telemetry.reset()

        trace_data = {
            "gating_level": "direct",
            "rag_scores": [0.85, 0.62],
            "monologue_chars": 120,
            "public_chars": 80,
        }
        telemetry.record_turn_trace("session_abc", trace_data)

        traces = telemetry.get_session_trace("session_abc")
        assert len(traces) == 1
        assert traces[0]["gating_level"] == "direct"
        assert traces[0]["rag_scores"] == [0.85, 0.62]

    def test_subscribe_and_push_thought_event(self):
        telemetry = get_telemetry_collector()
        telemetry.reset()

        q = asyncio.Queue()
        telemetry.subscribe_thoughts(q)

        telemetry.push_thought_event("session_xyz", {"chunk": "<ctrl94>Planning turn..."})

        event = q.get_nowait()
        assert event["session_id"] == "session_xyz"
        assert event["chunk"] == "<ctrl94>Planning turn..."

        telemetry.unsubscribe_thoughts(q)


class TestAdminTraceEndpoint:
    def test_get_session_trace_endpoint(self):
        telemetry = get_telemetry_collector()
        telemetry.reset()
        telemetry.record_turn_trace("session_test", {"action": "speak", "latency": 12.5})

        res = client.get("/admin/api/v1/sessions/session_test/trace")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == "session_test"
        assert len(data["trace"]) == 1
        assert data["trace"][0]["action"] == "speak"
