"""Tests for the R4 admin report endpoints (proxy/api/admin_routes.py).

Safety (brief R4): every test patches `spm.report.build_report` — no checks,
no bundle, no model server, no live services. The remote-consent check uses a
fake backend URL string; nothing connects anywhere. Downloads read from a
patched temp `reports/` directory.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from proxy.main import app
from spm import report as rp
from spm.diagnostics import Check

client = TestClient(app)
# Brief FX-B: /report with full_logs is loopback-only, so the suite's positive
# report tests use a loopback peer (the default peer 'testclient' is refused).
lb_client = TestClient(app, client=("127.0.0.1", 54321))

REMOTE_URL = "https://api.example-llm.com/v1"
LOCAL_URL = "http://127.0.0.1:13305/v1"


def _fake_report(*, private=False, enhanced_error=None, issue_url=None):
    markdown = "# SPM bug report\n\ndetails"
    if enhanced_error:
        markdown += f"\n\n## Enhanced analysis\n\nEnhanced analysis unavailable: {enhanced_error}"
    path = Path(f"/tmp/nowhere/spm-report-{'PRIVATE-' if private else ''}20261014-000000.zip")
    return rp.Report(markdown=markdown,
                     checks=[Check("install", "ok", "SPM 0.3.0", {"version": "0.3.0"}),
                             Check("model_server", "fail", "unreachable", {"error": "x"})],
                     path=path, issue_url=issue_url, private=private)


@pytest.fixture()
def patched_build(monkeypatch):
    """The seam the brief demands: build_report never runs for real."""
    calls = []

    async def fake_build(env, **kwargs):
        calls.append(kwargs)
        private = bool(kwargs.get("full_logs"))
        err = "the model server call failed" if kwargs.get("enhanced") is True else None
        return _fake_report(private=private, enhanced_error=err,
                            issue_url=None if private else "https://github.com/x/issues/new?y=1")

    monkeypatch.setattr(rp, "build_report", fake_build)
    monkeypatch.setattr("spm.run.child_env", lambda root: {})
    # No dependency on this machine's config: default to a local backend.
    monkeypatch.setattr(rp, "_backend_url", lambda env, repo_root=None: LOCAL_URL)
    return calls


def test_post_report_returns_fields(patched_build):
    res = client.post("/admin/api/v1/report",
                      json={"description": "it broke", "system_info": True})
    assert res.status_code == 200
    data = res.json()
    assert data["markdown"].startswith("# SPM bug report")
    assert [c["name"] for c in data["checks"]] == ["install", "model_server"]
    assert data["checks"][0]["status"] == "ok"
    assert data["file"].startswith("spm-report-") and data["file"].endswith(".zip")
    assert data["private"] is False
    assert data["issue_url"] and "issues/new" in data["issue_url"]
    assert "enhanced_error" not in data
    assert patched_build[0]["description"] == "it broke"
    assert patched_build[0]["system_info"] is True


def test_post_report_private_has_no_issue_url(patched_build):
    res = lb_client.post("/admin/api/v1/report",
                         json={"description": "look at my logs", "full_logs": True})
    assert res.status_code == 200
    data = res.json()
    assert data["private"] is True
    assert data["issue_url"] is None
    assert data["file"].startswith("spm-report-PRIVATE-")


def test_post_report_enhanced_error_surfaces(patched_build):
    res = client.post("/admin/api/v1/report",
                      json={"description": "d", "enhanced": True})
    assert res.status_code == 200
    assert res.json()["enhanced_error"] == "the model server call failed"


def test_post_report_remote_enhanced_needs_confirmation(patched_build, monkeypatch):
    monkeypatch.setattr(rp, "_backend_url", lambda env, repo_root=None: REMOTE_URL)
    res = client.post("/admin/api/v1/report",
                      json={"description": "d", "enhanced": True})
    assert res.status_code == 409
    body = res.json()
    assert body["needs_confirmation"] is True
    assert "api.example-llm.com" in body["message"]
    assert not patched_build  # no report was built without consent


def test_post_report_remote_enhanced_confirmed_builds(patched_build, monkeypatch):
    monkeypatch.setattr(rp, "_backend_url", lambda env, repo_root=None: REMOTE_URL)
    res = client.post("/admin/api/v1/report",
                      json={"description": "d", "enhanced": True,
                            "confirm_remote": True})
    assert res.status_code == 200
    assert len(patched_build) == 1


def test_post_report_local_enhanced_no_confirmation(patched_build, monkeypatch):
    monkeypatch.setattr(rp, "_backend_url", lambda env, repo_root=None: LOCAL_URL)
    res = client.post("/admin/api/v1/report",
                      json={"description": "d", "enhanced": True})
    assert res.status_code == 200
    assert len(patched_build) == 1


# ── download ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def reports_dir(monkeypatch, tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "spm-report-20261014-101112.zip").write_bytes(b"PKfakezip")
    (d / "spm-report-PRIVATE-20261014-101113.zip").write_bytes(b"PKprivate")
    (d / "not-a-report.zip").write_bytes(b"nope")
    monkeypatch.setattr("proxy.api.admin_routes._reports_dir", lambda: d)
    return d


def test_download_serves_bundle(reports_dir):
    res = client.get("/admin/api/v1/report/spm-report-20261014-101112.zip")
    assert res.status_code == 200
    assert res.content == b"PKfakezip"


def test_download_refuses_private_bundle(reports_dir):
    # Brief FX-B: a PRIVATE bundle is never served over the network.
    name = "spm-report-PRIVATE-20261014-101113.zip"
    res = client.get(f"/admin/api/v1/report/{name}")
    assert res.status_code == 403
    body = res.json()["error"]
    assert "Private reports are not served over the network." in body
    assert name in body
    # Second review, item 14: the message names the file, never the absolute path.
    assert str(reports_dir) not in body


def test_download_refuses_private_bundle_case_insensitively(reports_dir):
    # Second review, item 3: on Windows/macOS `spm-report-private-x.zip` opens the real
    # PRIVATE file, so the check must not depend on case.
    real = "spm-report-PRIVATE-20261014-101113.zip"
    (reports_dir / real).write_bytes(b"PK\x05\x06" + b"\0" * 18)
    for name in (real.lower(), real.replace("PRIVATE", "Private")):
        res = client.get(f"/admin/api/v1/report/{name}")
        assert res.status_code == 403, name
        assert "Private reports" in res.json()["error"]


def test_download_rejects_traversal_and_bad_names(reports_dir):
    import urllib.parse
    bad = ["../x", "not-a-report.zip", "spm-report-x.txt", "spm-report-.zip",
           "SPM-REPORT-20261014-101112.zip", "spm-report-a/b.zip", "x.zip"]
    for name in bad:
        res = client.get("/admin/api/v1/report/"
                         + urllib.parse.quote(name, safe=""))
        assert res.status_code == 404, name
    # Path traversal: the client normalizes ../x to x — either way, 404.
    res = client.get("/admin/api/v1/report/../spm-report-20261014-101112.zip")
    assert res.status_code in (404, 405)


def test_download_missing_file_is_404(reports_dir):
    res = client.get("/admin/api/v1/report/spm-report-19990101-000000.zip")
    assert res.status_code == 404
