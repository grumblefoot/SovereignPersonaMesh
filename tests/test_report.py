"""Tests for spm/report.py (brief R2).

Safety (brief R2): nothing here runs the real checks — `run_checks` and
`name_table` are patched — so no socket, no database, no live services. All
files live in tmp directories.
"""
import json
import re
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from spm import report as rp
from spm.diagnostics import Check

# ── fixtures / helpers ────────────────────────────────────────────────────────

SECRET = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
PASSWORD = "hunter2-supersecret"
PLAYER_NAME = "Morgath"
PROSE = ("Zelda whispered that the old bridge would not hold three riders "
         "after the flood")


def _fake_checks():
    return [
        Check("install", "ok", "SPM 0.3.0 on Python 3.13", {"version": "0.3.0"}),
        Check("model_server", "fail", "model server unreachable at http://127.0.0.1:9/v1",
              {"error": "ConnectError"}),
    ]


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """No live services: checks and the name table are fakes; logs dir in tmp."""
    root = tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    (root / "VERSION").write_text("0.3.0", encoding="utf-8")
    log = (root / "logs" / "spm_proxy.out")
    log.write_text(
        "\n".join([
            f"2026-10-14 10:00:01 [INFO] proxy.core: POSTGRES_PASSWORD={PASSWORD} "
            f"loaded from env",
            f"2026-10-14 10:00:02 [INFO] proxy.core: using key {SECRET} for the backend",
            f"2026-10-14 10:00:03 [INFO] proxy.routes: chat for {PLAYER_NAME} begins "
            "and the whole party rides out at dawn together",
            PROSE,  # continuation line (no timestamp): public mode drops it
            "2026-10-14 10:00:04 [INFO] proxy.routes: [SillyIntoSPMLog] "
            "Request payload with message content",
            "2026-10-14 10:00:05 [ERROR] proxy.routes: Embedding probe failed",
        ]) + "\n",
        encoding="utf-8")

    async def fake_run_checks(env, **kwargs):
        return _fake_checks()

    async def fake_name_table(env, *_):
        return {PLAYER_NAME: "player#1"}

    monkeypatch.setattr(rp, "run_checks", fake_run_checks)
    monkeypatch.setattr(rp, "name_table", fake_name_table)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    return root


def _zip_texts(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        assert set(zf.namelist()) == {"report.md", "checks.json", "logs-redacted.txt"} \
            or set(zf.namelist()) == {"report.md", "checks.json", "logs-full-PRIVATE.txt"}
        return {n: zf.read(n).decode("utf-8") for n in zf.namelist()}


# ── build_report: public ──────────────────────────────────────────────────────

async def test_public_bundle_redacts_everything(patched):
    report = await rp.build_report(
        {}, description=f"Key was {SECRET} and POSTGRES_PASSWORD={PASSWORD}",
        repo_root=patched)
    assert report.private is False
    assert report.path is not None and report.path.name.startswith("spm-report-")
    assert not report.path.name.startswith("spm-report-PRIVATE")
    texts = _zip_texts(report.path)
    all_text = "\n".join(texts.values()) + report.markdown
    for planted in (SECRET, PASSWORD, PLAYER_NAME, PROSE):
        assert planted not in all_text, f"planted string leaked: {planted!r}"
    # The structure is still there.
    assert "# SPM bug report" in texts["report.md"]
    assert "## Checks" in texts["report.md"]
    assert "## Check: model_server" in texts["report.md"]
    checks = json.loads(texts["checks.json"])
    assert [c["name"] for c in checks] == ["install", "model_server"]


async def test_public_issue_url_present_and_mentions_bundle(patched):
    report = await rp.build_report({}, description="model server unreachable",
                                   repo_root=patched)
    assert report.issue_url and report.issue_url.startswith(rp.ISSUE_URL + "?")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(report.issue_url).query)
    assert query["title"][0].startswith("[bug] model server unreachable")
    assert f"Please attach {report.path.name}" in query["body"][0]
    # Checks table rides along in the body.
    assert "model_server" in query["body"][0]


# ── build_report: private (--full-logs) ───────────────────────────────────────

async def test_full_logs_keeps_prose_not_secrets(patched):
    report = await rp.build_report({}, description="it broke badly",
                                   full_logs=True, repo_root=patched)
    assert report.private is True
    assert report.issue_url is None
    assert report.path.name.startswith("spm-report-PRIVATE-")
    texts = _zip_texts(report.path)
    logs = texts["logs-full-PRIVATE.txt"]
    # Prose kept (content-line filter off, continuation lines kept)...
    assert PROSE in logs
    # ...but secrets are still gone.
    for planted in (SECRET, PASSWORD):
        assert planted not in logs, f"secret leaked into private logs: {planted!r}"
    assert texts["report.md"].startswith(
        "PRIVATE: contains chat text. Send it only to the developers, "
        "never post it publicly.")


# ── issue_url cap ─────────────────────────────────────────────────────────────

def test_issue_url_cap_long_chinese_description():
    body = "模型服务器无法连接，回复为空。" * 900  # >1-byte code points when encoded
    url = rp.issue_url("[bug] 测试", body)
    assert len(url) <= rp.MAX_ENCODED_URL
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query,
                                  keep_blank_values=True)
    decoded = query["body"][0]
    assert body.startswith(decoded.rstrip())
    assert len(decoded) > 0


def test_issue_url_short_body_untouched():
    url = rp.issue_url("[bug] hi", "short body")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["body"][0] == "short body"
    assert query["title"][0] == "[bug] hi"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_report_yes_writes_bundle_and_prints_path(patched, monkeypatch, capsys):
    from spm.__main__ import main

    opened = []
    monkeypatch.setattr(rp.webbrowser, "open", lambda url: opened.append(url))
    rc = main(["report", "--yes", "--description", "cli broke"], repo_root=patched)
    assert rc == 0
    out = capsys.readouterr().out
    bundles = list((patched / "reports").glob("spm-report-*.zip"))
    assert len(bundles) == 1
    assert str(bundles[0]) in out
    assert "model_server" in out  # the check table was printed
    assert opened == []  # --yes never opens a browser
    # Exit 0 even though a check failed — that is the point of a report.


def test_cli_enhanced_runs_the_analysis(patched, monkeypatch, capsys):
    from spm.__main__ import main

    class FakeClient:
        base_url = "http://127.0.0.1:1/v1"

        def __init__(self):
            self.calls = []

        async def generate_stream(self, **kw):
            self.calls.append(kw)
            yield json.dumps({
                "summary": "backend down", "likely_causes": [],
                "what_to_try": ["start the server"], "severity": "high"})

    monkeypatch.setenv("BACKEND_LLM_URL", "http://127.0.0.1:1/v1")

    async def fake_window(*a, **k):
        return 8192

    monkeypatch.setattr(rp, "_window_tokens", fake_window)
    client = FakeClient()
    rc = main(["report", "--yes", "--enhanced", "--description", "x"],
              repo_root=patched, enhanced_client=client)
    assert rc == 0
    assert len(client.calls) == 1
    out = capsys.readouterr().out
    assert "Report bundle:" in out
    bundles = sorted((patched / "reports").glob("spm-report-*.zip"))
    with zipfile.ZipFile(bundles[-1]) as zf:
        assert "## Enhanced analysis" in zf.read("report.md").decode("utf-8")


def test_full_logs_enhanced_prompt_has_no_names(patched, monkeypatch):
    """--full-logs --enhanced (FX-G 4): the description goes through user_text
    with the name table in both modes, so the prompt and the private report.md
    keep an 8+ word symptom readable, with the name tokenised and paths gone
    (text() wiped the prose; the prompt was re-redacted without the names)."""
    from spm.__main__ import main

    class FakeClient:
        base_url = "http://127.0.0.1:1/v1"

        def __init__(self):
            self.calls = []

        async def generate_stream(self, **kw):
            self.calls.append(kw)
            yield json.dumps({"summary": "s", "likely_causes": [],
                              "what_to_try": [], "severity": "low"})

    monkeypatch.setenv("BACKEND_LLM_URL", "http://127.0.0.1:1/v1")

    async def fake_window(*a, **k):
        return None

    monkeypatch.setattr(rp, "_window_tokens", fake_window)
    client = FakeClient()
    symptom = "loses the thread of the story after every second reply since the update"
    rc = main(["report", "--yes", "--enhanced", "--full-logs", "--description",
               f"{PLAYER_NAME} {symptom} in /home/someoneelse/x"],
              repo_root=patched, enhanced_client=client)
    assert rc == 0 and len(client.calls) == 1
    sent = json.dumps(client.calls[0]["messages"])
    assert PLAYER_NAME not in sent and "player#1" in sent
    assert "/home/someoneelse" not in sent
    assert symptom in sent  # the symptom itself is kept (no "<redacted N words>")
    bundles = sorted((patched / "reports").glob("spm-report-PRIVATE-*.zip"))
    with zipfile.ZipFile(bundles[-1]) as zf:
        md = zf.read("report.md").decode("utf-8")
    assert f"player#1 {symptom} in ~user/x" in md


def test_reports_dir_is_gitignored():
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    entries = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    assert "reports/" in entries
