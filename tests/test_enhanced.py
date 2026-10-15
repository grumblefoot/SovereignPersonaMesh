"""Tests for the R3 Enhanced analysis (spm/report.py: enhanced_analysis).

Safety (brief R3): every test here uses a FAKE model client — nothing in this
file ever opens a socket to a model server. Checks and the name table are
fakes too; all files live in tmp directories.
"""
import asyncio
import json
import zipfile
from pathlib import Path

import pytest

from spm import report as rp
from spm.diagnostics import Check

# ── helpers / fixtures ────────────────────────────────────────────────────────

LOCAL_URL = "http://127.0.0.1:13305/v1"
REMOTE_URL = "https://api.example-llm.com/v1"


def _payload(**over):
    data = {
        "summary": "the model server is not answering",
        "likely_causes": [{"cause": "server down", "evidence": "connect refused",
                           "fix": "start the server"}],
        "what_to_try": ["spm start", "curl the health endpoint"],
        "severity": "high",
    }
    data.update(over)
    return data


def _text(**over):
    return json.dumps(_payload(**over))


class FakeClient:
    """Stands in for LemonadeLLMClient. Yields the queued responses (a BaseException
    entry is raised mid-stream) and records every call's kwargs."""

    def __init__(self, responses, delay=0.0):
        self.responses = list(responses)
        self.calls = []
        self.delay = delay
        self.base_url = LOCAL_URL

    async def generate_stream(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.responses:
            raise AssertionError("FakeClient called more times than scripted")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        yield item


def _checks():
    return [
        Check("install", "ok", "SPM 0.3.0", {"version": "0.3.0"}),
        Check("model_server", "fail", "unreachable", {"error": "ConnectError"}),
    ]


@pytest.fixture()
def no_window(monkeypatch):
    """Deterministic budget: pretend detect_window says 'unknown' (-> 6,000)."""
    async def fake_window(base_url, model, client):
        return None
    monkeypatch.setattr(rp, "_window_tokens", fake_window)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A tmp repo root whose playbook is big enough to exercise truncation."""
    r = tmp_path / "repo"
    docs = r / "docs"
    docs.mkdir(parents=True)
    (r / "VERSION").write_text("0.3.0", encoding="utf-8")
    playbook = "# Troubleshooting\n\nIntro paragraph.\n\n" + "".join(
        f"## Section {i}\n\n" + ("filler words for the playbook body. " * 120) + "\n\n"
        for i in range(1, 13))
    (docs / "troubleshooting.md").write_text(playbook, encoding="utf-8")
    monkeypatch.setattr(rp, "default_repo_root", lambda: r)
    return r


def _est_msgs(msgs, model=""):
    from proxy.rag.budget import estimate_tokens
    total = 0
    for m in msgs:
        total += estimate_tokens(m.get("content", ""), model)
    return total


# ── happy paths ───────────────────────────────────────────────────────────────

async def test_valid_json_first_time(no_window):
    client = FakeClient([_text()])
    result = await rp.enhanced_analysis({}, _checks(), "server won't start",
                                        client=client)
    assert result["summary"] == "the model server is not answering"
    assert result["severity"] == "high"
    assert result["likely_causes"][0]["fix"] == "start the server"
    assert "error" not in result and "need_checks" not in result
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["max_tokens"] == 800
    assert call["temperature"] == 0.2
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]


async def test_fenced_json_is_parsed(no_window):
    fenced = "Here you go:\n```json\n" + _text() + "\n```\nHope that helps!"
    client = FakeClient([fenced])
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client)
    assert result.get("summary") and "error" not in result
    assert len(client.calls) == 1


async def test_invalid_then_valid_retry(no_window):
    client = FakeClient(["not json at all", _text()])
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client)
    assert "error" not in result
    assert len(client.calls) == 2
    # The retry appends the hint to the same conversation.
    second = client.calls[1]["messages"]
    assert second[-1]["content"] == "Reply with the JSON object only."


# ── failure modes ─────────────────────────────────────────────────────────────

async def test_invalid_twice_is_error_and_report_still_builds(no_window, root, monkeypatch):
    async def fake_run_checks(env, **kw):
        return _checks()

    async def fake_name_table(env):
        return {}

    monkeypatch.setattr(rp, "run_checks", fake_run_checks)
    monkeypatch.setattr(rp, "name_table", fake_name_table)
    client = FakeClient(["garbage one", "still garbage",
                         "garbage three", "still garbage four"])
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client)
    assert "error" in result
    report = await rp.build_report({}, description="x", repo_root=root,
                                   enhanced=True, enhanced_client=client)
    assert report.path is not None  # the Traditional report is still delivered
    with zipfile.ZipFile(report.path) as zf:
        md = zf.read("report.md").decode("utf-8")
    assert "Enhanced analysis unavailable: " in md
    assert "## Checks" in md


async def test_timeout_returns_error(no_window, monkeypatch):
    monkeypatch.setattr(rp, "_MODEL_TIMEOUT_S", 0.05)
    client = FakeClient(["late"], delay=1.0)
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client)
    assert "timed out" in result["error"]


async def test_connection_failure_returns_error(no_window):
    client = FakeClient([ConnectionError("connection refused")])
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client)
    assert "error" in result and "ConnectionError" in result["error"]


# ── need_checks ───────────────────────────────────────────────────────────────

async def test_need_checks_one_extra_round(no_window):
    first = _text(need_checks=["chat_overview", "totally_made_up"])
    second = _text(need_checks=["backend_models_full"])  # must be ignored
    client = FakeClient([first, second])

    seen_names = []

    async def fake_extra(env, names=None, **kw):
        seen_names.append(list(names or []))
        return [Check("chat_overview", "ok", "nothing", {})]

    result = await rp.enhanced_analysis({}, _checks(), "x", client=client,
                                        run_extra=fake_extra)
    assert "error" not in result
    assert len(client.calls) == 2                      # exactly one extra round
    assert seen_names == [["chat_overview"]]           # unknown name dropped
    assert result["need_checks"] == ["chat_overview"]
    assert "chat_overview" in client.calls[1]["messages"][-1]["content"]


async def test_only_unknown_need_checks_no_extra_round(no_window):
    async def explode(*a, **k):
        raise AssertionError("run_extra must not be called")

    client = FakeClient([_text(need_checks=["nope", "also_nope"])])
    result = await rp.enhanced_analysis({}, _checks(), "x", client=client,
                                        run_extra=explode)
    assert "error" not in result
    assert "need_checks" not in result
    assert len(client.calls) == 1


# ── prompt budget ─────────────────────────────────────────────────────────────

async def test_window_8192_truncates_playbook_under_55_pct(monkeypatch, root):
    from proxy.rag.budget import estimate_tokens

    async def fake_window(base_url, model, client):
        return 8192
    monkeypatch.setattr(rp, "_window_tokens", fake_window)

    client = FakeClient([_text()])
    await rp.enhanced_analysis({}, _checks(), "x", client=client)
    call = client.calls[0]
    msgs = call["messages"]
    budget = int(8192 * 0.55)
    assert _est_msgs(msgs, call["model"]) <= budget
    # Truncation cut from the END: the intro and early sections survive...
    system = msgs[0]["content"]
    assert "Intro paragraph" in system and "## Section 1" in system
    assert "## Section 12" not in system
    # ...and the playbook really did have to shrink (it would not fit whole).
    playbook = (root / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    assert estimate_tokens(playbook, "") > budget


# ── privacy ───────────────────────────────────────────────────────────────────

async def test_prompt_contains_no_planted_secret(no_window):
    secret = "sk-" + "A" * 24
    client = FakeClient([_text()])
    await rp.enhanced_analysis({}, _checks(),
                               f"it broke, my key is {secret} and "
                               "POSTGRES_PASSWORD=hunter2-supersecret",
                               client=client)
    sent = json.dumps(client.calls[0]["messages"])
    assert secret not in sent
    assert "hunter2-supersecret" not in sent
    assert "***" in sent


# ── CLI consent ───────────────────────────────────────────────────────────────

def _cli_patched(monkeypatch, tmp_root):
    async def fake_run_checks(env, **kw):
        return _checks()

    async def fake_name_table(env):
        return {}

    monkeypatch.setattr(rp, "run_checks", fake_run_checks)
    monkeypatch.setattr(rp, "name_table", fake_name_table)


def test_remote_without_yes_falls_back_to_traditional(monkeypatch, root, capsys):
    """Sync test: cmd_report calls asyncio.run, which needs no outer loop."""
    from spm.__main__ import main
    monkeypatch.setenv("BACKEND_LLM_URL", REMOTE_URL)
    monkeypatch.setattr(rp.webbrowser, "open", lambda url: None)
    _cli_patched(monkeypatch, root)
    client = FakeClient([_text()])
    rc = main(["report", "--enhanced", "--description", "x"],
              repo_root=root, enhanced_client=client,
              input_fn=lambda prompt: "n")
    assert rc == 0
    out = capsys.readouterr().out
    assert "a remote service (this may cost tokens)" in out
    assert "Falling back to the Traditional report." in out
    assert client.calls == []  # no consent -> no model call at all
    bundles = list((root / "reports").glob("spm-report-*.zip"))
    assert len(bundles) == 1
    with zipfile.ZipFile(bundles[0]) as zf:
        assert "## Enhanced analysis" not in zf.read("report.md").decode("utf-8")


def test_remote_with_yes_proceeds(monkeypatch, root, capsys):
    """Sync test: cmd_report calls asyncio.run, which needs no outer loop."""
    from spm.__main__ import main
    monkeypatch.setenv("BACKEND_LLM_URL", REMOTE_URL)
    _cli_patched(monkeypatch, root)
    async def fake_window(base_url, model, client):
        return None
    monkeypatch.setattr(rp, "_window_tokens", fake_window)
    client = FakeClient([_text()])
    rc = main(["report", "--yes", "--enhanced", "--description", "x"],
              repo_root=root, enhanced_client=client)
    assert rc == 0
    assert "a remote service (this may cost tokens)" in capsys.readouterr().out
    assert len(client.calls) == 1  # --yes is the consent


def test_backend_url_follows_the_settings_file(tmp_path):
    """The Enhanced analysis must use the server SPM uses (file over env)."""
    import json as _json
    from spm import report
    cfg = tmp_path / "config.local.json"
    cfg.write_text(_json.dumps({"BACKEND_LLM_URL": "http://10.0.0.5:1234/v1"}), encoding="utf-8")
    env = {"SPM_CONFIG_PATH": str(cfg), "BACKEND_LLM_URL": "http://127.0.0.1:13305/v1"}
    assert report._backend_url(env, repo_root=tmp_path) == "http://10.0.0.5:1234/v1"
