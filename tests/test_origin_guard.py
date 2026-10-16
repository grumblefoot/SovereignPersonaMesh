"""Brief FX-B: the admin API is closed to other web pages (review finding 3).

CORSMiddleware(allow_origins=["*"]) is gone: any page open in the user's
browser could drive /admin/api/v1 (private full_logs report + download,
shutdown). The origin_guard middleware refuses cross-site requests to /admin;
/report full_logs, /shutdown and PRIVATE bundle downloads have their own
endpoint-level rules.

Safety: in-process TestClient only (peer address set via client=), os.kill and
build_report patched — no live services, no live database.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from proxy.main import app
from spm import report as rp
from spm.diagnostics import Check

EVIL_ORIGIN = "https://evil.example"


def _client(host: str = "127.0.0.1") -> TestClient:
    """TestClient with a chosen peer address (starlette supports client=;
    see the guard in test_shutdown.py)."""
    return TestClient(app, client=(host, 54321))


lb = _client("127.0.0.1")
remote = _client("10.0.0.9")


# ── CORS is gone ─────────────────────────────────────────────────────────────

def test_no_cors_headers_on_responses():
    for path in ("/health", "/admin/api/v1/stats"):
        res = lb.get(path, headers={"Origin": EVIL_ORIGIN})
        lowered = {k.lower() for k in res.headers}
        assert "access-control-allow-origin" not in lowered, path
        assert "access-control-allow-credentials" not in lowered, path


# ── shutdown: the guard, then the loopback rule ──────────────────────────────

def test_shutdown_foreign_origin_403_and_nothing_scheduled():
    with patch("asyncio.create_task") as mock_task, patch("os.kill") as mock_kill:
        res = lb.post("/admin/api/v1/shutdown", headers={"Origin": EVIL_ORIGIN})
    assert res.status_code == 403
    assert res.json() == {"error": "cross-site request refused"}
    mock_task.assert_not_called()
    mock_kill.assert_not_called()


def test_shutdown_same_loopback_origin_allowed():
    with patch("asyncio.create_task") as mock_task, patch("os.kill"):
        res = lb.post("/admin/api/v1/shutdown",
                      headers={"Origin": "http://127.0.0.1:5050",
                               "Host": "127.0.0.1:5050"})
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    mock_task.assert_called_once()


def test_shutdown_localhost_origin_matches_127_host():
    with patch("asyncio.create_task") as mock_task, patch("os.kill"):
        res = lb.post("/admin/api/v1/shutdown",
                      headers={"Origin": "http://localhost:5050",
                               "Host": "127.0.0.1:5050"})
    assert res.status_code == 200
    mock_task.assert_called_once()


def test_shutdown_no_origin_no_referer_allowed():
    """curl, scripts, the CLI: no Origin, no Referer — allowed from loopback."""
    with patch("asyncio.create_task") as mock_task, patch("os.kill"):
        res = lb.post("/admin/api/v1/shutdown")
    assert res.status_code == 200
    mock_task.assert_called_once()


def test_shutdown_non_loopback_client_403():
    with patch("asyncio.create_task") as mock_task, patch("os.kill") as mock_kill:
        res = remote.post("/admin/api/v1/shutdown")
    assert res.status_code == 403
    mock_task.assert_not_called()
    mock_kill.assert_not_called()


def test_unsafe_method_foreign_referer_without_origin_403():
    res = lb.post("/admin/api/v1/shutdown",
                  headers={"Referer": "https://evil.example/page.html"})
    assert res.status_code == 403
    assert res.json() == {"error": "cross-site request refused"}


def test_unsafe_method_same_site_referer_allowed():
    with patch("asyncio.create_task") as mock_task, patch("os.kill"):
        res = lb.post("/admin/api/v1/shutdown",
                      headers={"Referer": "http://127.0.0.1:5050/admin",
                               "Host": "127.0.0.1:5050"})
    assert res.status_code == 200
    mock_task.assert_called_once()


# ── report: guard + loopback rule for full_logs ──────────────────────────────

@pytest.fixture()
def patched_build(monkeypatch):
    """The seam from test_report_api.py: build_report never runs for real."""
    calls = []

    async def fake_build(env, **kwargs):
        calls.append(kwargs)
        private = bool(kwargs.get("full_logs"))
        return rp.Report(
            markdown="# SPM bug report\n\ndetails",
            checks=[Check("install", "ok", "SPM 0.3.0", {"version": "0.3.0"})],
            path=Path(f"/tmp/nowhere/spm-report-{'PRIVATE-' if private else ''}"
                      "20261014-000000.zip"),
            issue_url=None, private=private)

    monkeypatch.setattr(rp, "build_report", fake_build)
    monkeypatch.setattr("spm.run.child_env", lambda root: {})
    monkeypatch.setattr(rp, "_backend_url",
                        lambda env, repo_root=None: "http://127.0.0.1:13305/v1")
    return calls


def test_report_foreign_origin_403(patched_build):
    res = lb.post("/admin/api/v1/report",
                  json={"description": "d", "full_logs": True},
                  headers={"Origin": EVIL_ORIGIN})
    assert res.status_code == 403
    assert res.json() == {"error": "cross-site request refused"}
    assert not patched_build


def test_report_full_logs_non_loopback_client_403(patched_build):
    res = remote.post("/admin/api/v1/report",
                      json={"description": "d", "full_logs": True})
    assert res.status_code == 403
    assert not patched_build


def test_report_full_logs_loopback_allowed(patched_build):
    res = lb.post("/admin/api/v1/report",
                  json={"description": "d", "full_logs": True})
    assert res.status_code == 200
    assert res.json()["private"] is True
    assert len(patched_build) == 1


def test_report_without_full_logs_non_loopback_allowed(patched_build):
    """Only full_logs is peer-restricted (the guard still blocks cross-site)."""
    res = remote.post("/admin/api/v1/report", json={"description": "d"})
    assert res.status_code == 200
    assert len(patched_build) == 1


# ── PRIVATE bundles are never served; public ones still are ──────────────────

@pytest.fixture()
def reports_dir(monkeypatch, tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "spm-report-20261014-101112.zip").write_bytes(b"PKfakezip")
    (d / "spm-report-PRIVATE-20261014-101113.zip").write_bytes(b"PKprivate")
    monkeypatch.setattr("proxy.api.admin_routes._reports_dir", lambda: d)
    return d


def test_private_bundle_403_with_path_message(reports_dir):
    name = "spm-report-PRIVATE-20261014-101113.zip"
    res = lb.get(f"/admin/api/v1/report/{name}")
    assert res.status_code == 403
    error = res.json()["error"]
    assert "Private reports are not served over the network." in error
    # Names the file inside the SPM folder; never the absolute path (second review, item 3).
    assert f"reports/{name}" in error
    assert str(reports_dir) not in error
    # A foreign browser could not read it either.
    res = remote.get(f"/admin/api/v1/report/{name}")
    assert res.status_code == 403


def test_public_bundle_still_downloads(reports_dir):
    res = lb.get("/admin/api/v1/report/spm-report-20261014-101112.zip")
    assert res.status_code == 200
    assert res.content == b"PKfakezip"


# ── /v1 is not affected by the guard ─────────────────────────────────────────

def test_chat_completions_foreign_origin_not_guarded():
    """/v1 must not 403 from the guard (SillyTavern's server may send Origin;
    any failure here comes from the handler, not the guard)."""
    res = lb.post("/v1/chat/completions",
                  json={"model": "m", "messages": [{"role": "user",
                                                    "content": "hi"}]},
                  headers={"Origin": EVIL_ORIGIN})
    # Under the suite's unreachable test backend the handler answers 200 with an
    # in-band "backend unavailable" message; the guard never produced a 403.
    assert res.status_code == 200
    assert "cross-site request refused" not in res.text


def test_dns_rebinding_host_is_refused_on_a_loopback_bind(monkeypatch):
    """A rebinding page reaches 127.0.0.1 with Host: evil.example (same-origin to itself)."""
    from fastapi.testclient import TestClient
    from proxy.main import app
    monkeypatch.delenv("SPM_HOST", raising=False)
    c = TestClient(app)
    r = c.get("/admin/api/v1/stats", headers={"Host": "evil.example:5050",
                                              "Origin": "http://evil.example:5050"})
    assert r.status_code == 403
    assert c.get("/admin/api/v1/stats", headers={"Host": "127.0.0.1:5050"}).status_code != 403
    monkeypatch.setenv("SPM_HOST", "0.0.0.0")          # deliberately opened to the LAN
    assert c.get("/admin/api/v1/stats", headers={"Host": "192.168.1.20:5050"}).status_code != 403


def test_container_keeps_the_loopback_host_rule(monkeypatch):
    from fastapi.testclient import TestClient
    from proxy.main import app
    monkeypatch.setenv("SPM_HOST", "0.0.0.0")
    monkeypatch.setenv("SPM_IN_CONTAINER", "1")
    c = TestClient(app)
    assert c.get("/admin/api/v1/stats", headers={"Host": "evil.example:5050"}).status_code == 403
    assert c.get("/admin/api/v1/stats", headers={"Host": "127.0.0.1:5050"}).status_code != 403
