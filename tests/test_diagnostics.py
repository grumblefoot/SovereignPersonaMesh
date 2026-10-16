"""Tests for spm/diagnostics.py (brief R1b).

Safety (brief R1b): nothing here calls the live services (5050/4005), the live
database, or the LM Studio box. HTTP goes through httpx.MockTransport, the
database is either a fake asyncpg connection or the throwaway test DB from
tests/_testdb.py, and subprocesses are faked.
"""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from spm import diagnostics as dx
from spm.redact import Redactor
from tests._testdb import TEST_DB_CONFIG

# ── helpers ───────────────────────────────────────────────────────────────────


def _json(obj, status=200):
    return httpx.Response(status, json=obj)


def _patch_http(monkeypatch, handler):
    """Route every diagnostics HTTP call through a MockTransport handler."""
    real_factory = httpx.AsyncClient

    async def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_factory(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(dx, "_make_http", factory)


def _recording_handler(routes, default=(404, {"detail": "not found"})):
    """Handler recording (method, url) and replying from `routes`.

    `routes` maps path -> httpx.Response | (status, body) | callable(request).
    A response object is returned once; tuples/callables every time.
    """
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        for key, value in routes.items():
            if request.url.path == key:
                if callable(value):
                    return value(request)
                if isinstance(value, httpx.Response):
                    return value
                status, body = value
                return _json(body, status)
        status, body = default
        return _json(body, status)

    return handler, calls


def _models(ids):
    return {"data": [{"id": i, "object": "model"} for i in ids]}


def _dump(check):
    return json.dumps({"summary": check.summary, "details": check.details})


class FakeConn:
    """Fake asyncpg connection: SQL is routed by substring; values are copied
    per connection (asyncpg cursors are consumed once, like our fakes)."""

    def __init__(self, router):
        self.router = router
        self.closed = False
        self.queries = []

    async def fetch(self, sql):
        self.queries.append(sql)
        for needle, value in self.router.items():
            if needle in sql:
                value = value(sql) if callable(value) else value
                return list(value) if isinstance(value, list) else value
        raise AssertionError(f"unexpected SQL: {sql!r}")

    async def fetchval(self, sql):
        rows = await self.fetch(sql)
        first = rows[0] if rows else None
        if isinstance(first, dict):
            return next(iter(first.values()))
        return first

    async def close(self):
        self.closed = True


def _ctx(root=None, redactor=None):
    # An empty temp root by default: the repo root holds the developer's real
    # config/config.local.json, which the effective-settings rule would read.
    import tempfile
    return dx._Ctx(root=Path(root) if root else Path(tempfile.mkdtemp()),
                   redactor=redactor or Redactor())


def _snapshot(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob("*"))


def _string_keys(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.append(str(k))
            _string_keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _string_keys(v, acc)


# ── install ───────────────────────────────────────────────────────────────────

async def test_install_healthy(tmp_path, monkeypatch):
    (tmp_path / "VERSION").write_text("1.2.3\n")

    def fake_run(args, timeout):
        assert "rev-parse" in args and timeout <= 3.0
        return type("R", (), {"returncode": 0, "stdout": "abc1234\n", "stderr": ""})()

    monkeypatch.setattr(dx, "_subprocess_run", fake_run)
    c = await dx.check_install({"SPM_IN_CONTAINER": "1"}, _ctx(tmp_path))
    assert c.status == "ok"
    assert c.details["version"] == "1.2.3"
    assert c.details["git_commit"] == "abc1234"
    assert c.details["in_container"] is True
    assert set(c.details["packages"]) == {"fastapi", "uvicorn", "pydantic", "asyncpg",
                                          "httpx", "python-dotenv"}


async def test_install_git_timeout_is_optional(tmp_path, monkeypatch):
    def fake_run(args, timeout):
        raise subprocess_timeout()

    monkeypatch.setattr(dx, "_subprocess_run", fake_run)
    c = await dx.check_install({}, _ctx(tmp_path))
    assert c.status == "ok"
    assert c.details["git_commit"] is None
    assert c.details["version"] == "unknown"


def subprocess_timeout():
    import subprocess
    return subprocess.TimeoutExpired(["git"], 3)


# ── services ──────────────────────────────────────────────────────────────────

async def test_services_healthy(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "spm.pids").write_text(json.dumps({"engine": os.getpid(),
                                                           "proxy": 999999}))
    handler, calls = _recording_handler({
        "/health": (200, {"status": "ok"}),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_services({}, _ctx(tmp_path))
    assert c.status == "ok"
    assert c.details["pids"]["engine"]["alive"] is True
    assert c.details["pids"]["proxy"]["alive"] is False
    assert any(u.endswith("/health") for _, u in calls)


async def test_services_both_down(monkeypatch):
    async def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_http(monkeypatch, refuse)
    c = await dx.check_services({}, _ctx())
    assert c.status == "fail"
    assert "engine down" in c.summary and "proxy down" in c.summary


async def test_services_one_up_is_warn(monkeypatch):
    async def handler(request):
        if request.url.port == 5050:
            return _json({"status": "ok"})
        raise httpx.ConnectError("refused", request=request)

    _patch_http(monkeypatch, handler)
    c = await dx.check_services({"SPM_PORT": "6555", "SPM_ENGINE_PORT": "5050"}, _ctx())
    assert c.status == "warn"


# ── config ────────────────────────────────────────────────────────────────────

async def test_config_secret_keys_removed(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"reasoning_mode": "prefill",
                               "gm_actions_mode": "on",
                               "context_window_override": 8192,
                               "BACKEND_API_KEY": "zebrakey",
                               "db_password": "zebrapw"}))
    c = await dx.check_config({"SPM_CONFIG_PATH": str(cfg),
                               "BACKEND_LLM_URL": "http://example.invalid:9/v1"},
                              _ctx(tmp_path))
    assert c.status == "ok"
    assert c.details["reasoning_mode"] == "prefill"
    assert c.details["gm_actions_mode"] == "on"
    assert c.details["context_window_override"] == 8192
    keys = []
    _string_keys(c.details, keys)
    assert not any(dx._SECRET_KEY_RE.search(k) for k in keys)
    assert "zebra" not in _dump(c)          # secret VALUES are gone entirely
    # env override wins over file/defaults
    assert c.details["settings"]["BACKEND_LLM_URL"] == "http://example.invalid:9/v1"


async def test_config_missing_file_warn(tmp_path):
    c = await dx.check_config({}, _ctx(tmp_path))
    assert c.status == "warn"


async def test_config_malformed_warn(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{not json at all")
    c = await dx.check_config({"SPM_CONFIG_PATH": str(cfg)}, _ctx(tmp_path))
    assert c.status == "warn"
    cfg.write_text(json.dumps(["a list", "not an object"]))
    c = await dx.check_config({"SPM_CONFIG_PATH": str(cfg)}, _ctx(tmp_path))
    assert c.status == "warn"


# ── model_server ──────────────────────────────────────────────────────────────

_ENV_SERVER = {"BACKEND_LLM_URL": "http://modelserver.invalid:9/v1",
               "SPM_DEFAULT_MODEL": "Gemma-test"}


async def test_model_server_healthy_never_generates(monkeypatch):
    handler, calls = _recording_handler({
        "/v1/models": (200, _models(["Gemma-test-26B", "other-model"])),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_model_server(_ENV_SERVER, _ctx())
    assert c.status == "ok"
    assert c.details["matched"] == "Gemma-test-26B"
    assert "Gemma-test-26B" in c.details["model_ids"]
    assert all(m == "GET" for m, _ in calls)           # never a generation POST
    assert not any("completion" in u or "generate" in u for _, u in calls)


async def test_model_server_unreachable(monkeypatch):
    async def refuse(request):
        raise httpx.ConnectError("no route to host", request=request)

    _patch_http(monkeypatch, refuse)
    c = await dx.check_model_server(_ENV_SERVER, _ctx())
    assert c.status == "fail"
    assert "unreachable" in c.summary


async def test_model_server_malformed(monkeypatch):
    handler, _ = _recording_handler({"/v1/models": (200, {"nope": True})})
    _patch_http(monkeypatch, handler)
    c = await dx.check_model_server(_ENV_SERVER, _ctx())
    assert c.status == "warn"
    assert c.details["matched"] is None


async def test_model_server_http500(monkeypatch):
    handler, _ = _recording_handler({"/v1/models": (500, {"detail": "boom"})})
    _patch_http(monkeypatch, handler)
    c = await dx.check_model_server(_ENV_SERVER, _ctx())
    assert c.status == "warn"


async def test_model_server_model_not_listed(monkeypatch):
    handler, _ = _recording_handler({"/v1/models": (200, _models(["some-other"]))})
    _patch_http(monkeypatch, handler)
    c = await dx.check_model_server(_ENV_SERVER, _ctx())
    assert c.status == "warn"
    assert c.details["matched"] is None


async def test_model_server_without_settings_uses_the_built_in_default(monkeypatch, tmp_path):
    """No env and no settings file: SPM itself falls back to its default model server
    (config/manager.py, Lemonade on 13305), so the check probes that same URL."""
    from config.manager import _DEFAULT_VALUES
    assert "13305" in str(_DEFAULT_VALUES["BACKEND_LLM_URL"])   # not SillyTavern's 8000
    monkeypatch.setattr(dx, "_make_http", lambda **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    [c] = await dx.run_checks({}, names=("model_server",), repo_root=tmp_path)
    assert c.status in ("fail", "warn")


# ── embeddings ────────────────────────────────────────────────────────────────

async def test_embeddings_none_provider_skipped(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("provider 'none' must not make requests")

    monkeypatch.setattr(dx, "_make_http", boom)
    c = await dx.check_embeddings({"EMBEDDING_PROVIDER": "none"}, _ctx())
    assert c.status == "skipped"


async def test_embeddings_healthy(monkeypatch):
    handler, calls = _recording_handler({
        "/v1/models": (200, _models(["bge-m3-embed", "Gemma-chat"])),
        "/v1/embeddings": (200, {"data": [{"embedding": [0.1] * 7}]}),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_embeddings({"EMBEDDING_URL": "http://embed.invalid:9/v1"}, _ctx())
    assert c.status == "ok"
    assert c.details["dimension"] == 7
    assert ("POST", "http://embed.invalid:9/v1/embeddings") in calls


async def test_embeddings_never_posts_without_embed_id(monkeypatch):
    handler, calls = _recording_handler({
        "/v1/models": (200, _models(["Gemma-chat", "vision-model"])),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_embeddings({"EMBEDDING_URL": "http://embed.invalid:9/v1"}, _ctx())
    assert c.status == "warn"
    assert calls and all(m == "GET" for m, _ in calls)   # never POST /embeddings


async def test_embeddings_probe_failure(monkeypatch):
    handler, _ = _recording_handler({
        "/v1/models": (200, _models(["embed-x"])),
        "/v1/embeddings": (500, {"detail": "nope"}),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_embeddings({"EMBEDDING_URL": "http://embed.invalid:9/v1"}, _ctx())
    assert c.status == "fail"


async def test_embeddings_server_down(monkeypatch):
    async def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_http(monkeypatch, refuse)
    c = await dx.check_embeddings({"EMBEDDING_URL": "http://embed.invalid:9/v1"}, _ctx())
    assert c.status == "fail"


async def test_embeddings_timeout(monkeypatch):
    async def timeout_handler(request):
        raise httpx.ConnectTimeout("timed out connecting", request=request)

    _patch_http(monkeypatch, timeout_handler)
    c = await dx.check_embeddings({"EMBEDDING_URL": "http://embed.invalid:9/v1"}, _ctx())
    assert c.status == "fail"
    assert "error" in c.details


# ── database (fake connection) ────────────────────────────────────────────────

_FAKE_ENV = {"POSTGRES_DB": "fakedb", "POSTGRES_USER": "fakeuser",
             "POSTGRES_PASSWORD": "fakepw"}


async def test_database_fake_ok(tmp_path, monkeypatch):
    conn = FakeConn({
        "SELECT version()": [{"version": "PostgreSQL 16.4 on x86_64\nmore"}],
        "pg_extension": [{"present": 1}],
        "spm_schema_migrations": [{"filename": "001_a.sql"}, {"filename": "002_b.sql"}],
        "information_schema.tables": [{"table_name": "spm_perception"},
                                      {"table_name": "world_state_sessions"}],
        "count(*)": lambda sql: [{"count": 3 if "spm_perception" in sql else 1}],
    })

    async def fake_connect(cfg):
        return conn

    monkeypatch.setattr(dx, "_connect", fake_connect)
    mig = tmp_path / "scripts" / "migrations"
    mig.mkdir(parents=True)
    (mig / "001_a.sql").write_text("--")
    (mig / "002_b.sql").write_text("--")
    (mig / "003_pending.sql").write_text("--")
    (mig / "008_drop_char_lore_tables.sql").write_text("--")

    c = await dx.check_database(_FAKE_ENV, _ctx(tmp_path))
    assert c.status == "warn"                       # pending migration
    assert c.details["vector_extension"] is True
    assert c.details["ledger"]["pending"] == ["003_pending.sql"]
    assert c.details["ledger"]["manual_only"] == ["008_drop_char_lore_tables.sql"]
    assert c.details["row_counts"] == {"spm_perception": 3, "world_state_sessions": 1}
    assert conn.closed is True


async def test_database_no_db_name(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("must not connect without POSTGRES_DB")

    monkeypatch.setattr(dx, "_connect", boom)
    c = await dx.check_database({}, _ctx())
    assert c.status == "fail"
    assert "POSTGRES_DB" in c.summary


async def test_database_connect_error_redacted(monkeypatch):
    async def refuse(cfg):
        raise RuntimeError(f"password authentication failed for user "
                           f"'{cfg['user']}' using password '{cfg['password']}'")

    monkeypatch.setattr(dx, "_connect", refuse)
    c = await dx.check_database(_FAKE_ENV, _ctx())
    assert c.status == "fail"
    dumped = _dump(c)
    assert "fakepw" not in dumped
    assert "***" in dumped


# ── database (throwaway test DB) ─────────────────────────────────────────────

def _real_db_env():
    return {"POSTGRES_HOST": TEST_DB_CONFIG["host"],
            "POSTGRES_PORT": str(TEST_DB_CONFIG["port"]),
            "POSTGRES_USER": TEST_DB_CONFIG["user"],
            "POSTGRES_PASSWORD": TEST_DB_CONFIG["password"],
            "POSTGRES_DB": TEST_DB_CONFIG["database"]}


@pytest.mark.parametrize("password", ["definitely-not-the-password", "Sup3rSecretValue"])
async def test_database_wrong_password_fails_without_leaking(password):
    env = _real_db_env()
    env["POSTGRES_PASSWORD"] = password
    c = await dx.check_database(env, _ctx())
    assert c.status == "fail"
    assert password not in _dump(c)


async def test_database_real_throwaway_db():
    c = await dx.check_database(_real_db_env(), _ctx())
    assert c.status in ("ok", "warn"), c.summary
    assert c.details["server_version"].startswith("PostgreSQL")
    assert c.details["vector_extension"] is True
    counts = c.details["row_counts"]
    assert isinstance(counts, dict) and counts.get("spm_perception", 0) >= 0
    # every migration file is accounted for: applied, pending, or manual-only
    from spm.run import default_repo_root
    files, manual = dx._migration_files(_ctx(default_repo_root()))
    ledger = c.details["ledger"]
    assert set(files) <= set(ledger["applied"]) | set(ledger["pending"]) | set(manual)
    assert not (set(ledger["pending"]) & set(manual))


# ── chat_overview + name_table ────────────────────────────────────────────────

async def _insert_overview_rows():
    import asyncpg
    from tests._testdb import TEST_DB_CONFIG
    uniq = str(int(time.time() * 1000))
    sess = f"diag_ov_sess_{uniq}"
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        tables = {r["table_name"] for r in await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'")}
        for table, sql in (
            ("spm_ingress_log",
             "INSERT INTO spm_ingress_log (session_id, action_tick, actor_id, "
             "location_id, action_type) VALUES ($1, 1, 'user', 'loc1', 'move')"),
            ("spm_perception",
             "INSERT INTO spm_perception (session_id, tick, actor_id, recipient_id, "
             "action_type, gating_level, perceived_text) "
             "VALUES ($1, 1, 'user', $2, 'say', 'direct', 'hello')"),
            ("spm_chat_rules",
             "INSERT INTO spm_chat_rules (session_id, level, rule_text, rule_type) "
             "VALUES ($1, 'world', 'the rule', 'invariant')"),
            ("spm_chat_settings",
             "INSERT INTO spm_chat_settings (session_id, player_name) "
             "VALUES ($1, $2)"),
            ("world_state_sessions",
             "INSERT INTO world_state_sessions (session_id, template_key, room_id, "
             "room_data) VALUES ($1, $2, $3, '{}'::jsonb)"),
        ):
            if table not in tables:
                continue
            if table == "spm_chat_settings":
                await conn.execute(sql, sess, f"DiagPlayer_{uniq}")
            elif table == "world_state_sessions":
                await conn.execute(sql, sess, "tpl_a", f"room_{uniq}_a")
                await conn.execute(sql, sess, "tpl_b", f"room_{uniq}_b")
            elif table == "spm_perception":
                await conn.execute(sql, sess, f"DiagChar_{uniq}")
            else:
                await conn.execute(sql, sess)
        return sess
    finally:
        await conn.close()


async def test_chat_overview_real_db_counts_only():
    sess = await _insert_overview_rows()
    c = await dx.check_chat_overview(_real_db_env(), _ctx())
    assert c.status == "ok", c.summary
    row = next(s for s in c.details["sessions"] if s["session"] == sess)
    assert row["turns"] == 1 and row["rules"] == 1 and row["rooms"] == 2
    # counts only — the rule text we inserted never appears
    assert "the rule" not in _dump(c)


async def test_chat_overview_query_errors(tmp_path, monkeypatch):
    async def bad_fetch(self, sql):
        raise RuntimeError("permission denied")

    conn = FakeConn({})
    monkeypatch.setattr(FakeConn, "fetch", bad_fetch)

    async def fake_connect(cfg):
        return conn

    monkeypatch.setattr(dx, "_connect", fake_connect)
    c = await dx.check_chat_overview(_FAKE_ENV, _ctx(tmp_path))
    assert c.status == "fail"
    assert conn.closed is True


async def test_name_table_real_db():
    sess = await _insert_overview_rows()
    import asyncpg
    from tests._testdb import TEST_DB_CONFIG
    conn = await asyncpg.connect(**TEST_DB_CONFIG)
    try:
        player = await conn.fetchval(
            "SELECT player_name FROM spm_chat_settings WHERE session_id = $1", sess)
    finally:
        await conn.close()
    table = await dx.name_table(_real_db_env())
    assert table.get(sess, "").startswith("chat#")
    # the perception recipient id is the character name (actor 'user' is skipped)
    uniq = sess.split("_")[-1]
    assert table.get(f"DiagChar_{uniq}", "").startswith("char#")
    assert table.get(player, "").startswith("player#")
    assert "user" not in table


async def test_name_table_failure_returns_empty(monkeypatch):
    async def refuse(cfg):
        raise RuntimeError("cannot connect")

    monkeypatch.setattr(dx, "_connect", refuse)
    assert await dx.name_table(_FAKE_ENV) == {}
    assert await dx.name_table({}) == {}


# ── recent_errors ─────────────────────────────────────────────────────────────

_PLANTED = [
    ("2026-10-14 03:04:05 [ERROR] proxy.api.routes: "
     f"backend auth used {'pass'}word=Sup3rSecretValue while retrying", 2),
    ("2026-10-14 03:04:06 [ERROR] proxy.api.routes: "
     f"dsn postgres://usr:Sup3rSecretValue@198.51.100.7:5432/db session_id=st_chat_{1234567890}", 1),
    ("2026-10-14 03:04:07 [ERROR] proxy.api.routes: "
     f"home path /home/{'realuser'}name appeared in a traceback for module", 1),
    ("2026-10-14 03:04:08 [WARNING] proxy.core.reasoning: "
     "the model answered with a long stretch of prose that should not survive review", 1),
    ("Traceback (most recent call last):", 1),
    ("2026-10-14 03:04:09 [INFO] nothing: quiet line", 1),
]


def _plant_logs(root: Path):
    logs = root / "logs"
    logs.mkdir()
    lines = []
    for line, count in _PLANTED:
        lines.extend([line] * count)
    (logs / "spm_engine.out").write_text("\n".join(lines) + "\n")
    (logs / "notes.txt").write_text("ERROR: not a log file, must be ignored\n")


async def test_recent_errors_redacts_planted_secrets(tmp_path):
    _plant_logs(tmp_path)
    c = await dx.check_recent_errors({}, _ctx(tmp_path))
    assert c.status == "warn"
    dumped = _dump(c)
    assert "Sup3rSecretValue" not in dumped
    assert "st_chat_" not in dumped
    assert "/home/" not in dumped
    assert "realusername" not in dumped
    assert "198.51.100.7" not in dumped
    assert "prose that should not survive review" not in dumped
    # grouped with counts (the duplicated ERROR line counts twice in one group)
    assert any(g["count"] >= 2 for g in c.details["groups"])
    assert c.details["lines_matched"] >= 6


async def test_recent_errors_clean_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "spm_engine.out").write_text("2026-10-14 03:04:05 [INFO] x: fine\n")
    c = await dx.check_recent_errors({}, _ctx(tmp_path))
    assert c.status == "ok"


async def test_recent_errors_no_logs_dir(tmp_path):
    c = await dx.check_recent_errors({}, _ctx(tmp_path))
    assert c.status == "ok"


async def test_recent_errors_reads_only_the_tail_window(tmp_path):
    """FX-G 5a: a huge log is not read whole; only the last 512 KiB count."""
    logs = tmp_path / "logs"
    logs.mkdir()
    early = "2026-10-14 03:04:05 [ERROR] x: early error outside the window"
    late = "2026-10-14 03:04:06 [ERROR] x: late error inside the window"
    filler = "2026-10-14 03:04:05 [INFO] x: " + ("f" * 90) + "\n"
    body = early + "\n" + filler * (dx._SCAN_WINDOW_BYTES // len(filler) + 50) + late + "\n"
    (logs / "spm_proxy.out").write_text(body, encoding="utf-8")
    assert len(body) > dx._SCAN_WINDOW_BYTES + 2000
    raw = dx._scan_logs(tmp_path, 50)
    assert raw == [late]
    c = await dx.check_recent_errors({}, _ctx(tmp_path))
    assert c.details["lines_matched"] == 1


async def test_blocking_subprocess_is_preempted_by_the_check_timeout(tmp_path, monkeypatch):
    """FX-G 5b: subprocess helpers run off the event loop, so a hung git/GPU
    probe cannot outlive the per-check cap."""
    def hung_run(args, timeout):
        time.sleep(1.0)
        return type("R", (), {"returncode": 0, "stdout": "abc1234\n", "stderr": ""})()

    monkeypatch.setattr(dx, "_subprocess_run", hung_run)
    t0 = time.perf_counter()
    results = await dx.run_checks({}, names=("install", "system"), system_info=True,
                                  repo_root=tmp_path, per_check_timeout=0.1)
    elapsed = time.perf_counter() - t0
    assert [r.status for r in results] == ["fail", "fail"], results
    assert all("timed out" in r.summary for r in results)
    assert elapsed < 0.8, elapsed


def test_gpu_names_fall_back_to_powershell_when_wmic_is_missing(monkeypatch):
    """FX-G 5c: wmic is gone on Windows 11 24H2."""
    seen = []

    def fake_run(args, timeout):
        seen.append((list(args), timeout))
        if args[0] == "wmic":
            raise FileNotFoundError("wmic")
        return type("R", (), {"returncode": 0, "stderr": "",
                              "stdout": "\nNVIDIA GeForce RTX 4070\n\n"})()

    monkeypatch.setattr(dx, "_subprocess_run", fake_run)
    monkeypatch.setattr(dx.os, "name", "nt")
    assert dx._gpu_names() == ["NVIDIA GeForce RTX 4070"]
    assert seen == [
        (["wmic", "path", "win32_VideoController", "get", "name"], 2.0),
        (["powershell", "-NoProfile", "-Command",
          "(Get-CimInstance Win32_VideoController).Name"], 2.0),
    ]


async def test_recent_errors_full_uses_larger_limit(tmp_path):
    _plant_logs(tmp_path)
    c50 = await dx.check_recent_errors({}, _ctx(tmp_path))
    cfull = await dx.check_recent_errors_full({}, _ctx(tmp_path))
    assert cfull.details["limit"] == 300
    assert cfull.details["lines_after_filter"] >= c50.details["lines_after_filter"]


# ── scheduler ─────────────────────────────────────────────────────────────────

async def test_scheduler_ok(monkeypatch):
    handler, calls = _recording_handler({
        "/health": (200, {"status": "ok"}),
        "/admin/api/v1/scheduler": (200, {"queue": [], "running": 0}),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_scheduler({}, _ctx())
    assert c.status == "ok"
    assert c.details["snapshot"]["running"] == 0


async def test_scheduler_skipped_when_proxy_down(monkeypatch):
    async def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_http(monkeypatch, refuse)
    c = await dx.check_scheduler({}, _ctx())
    assert c.status == "skipped"


async def test_scheduler_bad_endpoints(monkeypatch):
    handler, _ = _recording_handler({
        "/health": (200, {"status": "ok"}),
        "/admin/api/v1/scheduler": (404, {"detail": "gone"}),
    })
    _patch_http(monkeypatch, handler)
    c = await dx.check_scheduler({}, _ctx())
    assert c.status == "warn"


# ── system (opt-in) ───────────────────────────────────────────────────────────

async def test_system_skipped_unless_requested(monkeypatch):
    results = await dx.run_checks({}, names=("system",), repo_root=Path("."))
    assert results[0].status == "skipped"
    assert results[0].summary == "not requested"


async def test_system_when_requested(tmp_path, monkeypatch):
    def fake_run(args, timeout):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(dx, "_subprocess_run", fake_run)
    results = await dx.run_checks({}, names=("system",), system_info=True,
                                  repo_root=tmp_path)
    c = results[0]
    assert c.status == "ok"
    assert c.details["cpu_count"] == os.cpu_count()
    if os.name != "nt":
        assert c.details["memory_bytes"]["memtotal"] > 0
    assert c.details["gpu"] == []


# ── backend_models_full ───────────────────────────────────────────────────────

async def test_backend_models_full_redacts(monkeypatch, tmp_path):
    sneaky = "sk-" + "abcdefghijklmno" + "0123"
    handler, _ = _recording_handler({
        "/v1/models": (200, _models(["good-model", sneaky])),
    })
    _patch_http(monkeypatch, handler)
    results = await dx.run_checks({"BACKEND_LLM_URL": "http://modelserver.invalid:9/v1"},
                                  names=("backend_models_full",), repo_root=tmp_path)
    c = results[0]
    assert c.status == "ok"
    assert sneaky not in _dump(c)


async def test_backend_models_full_down(monkeypatch):
    async def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_http(monkeypatch, refuse)
    c = await dx.check_backend_models_full({"BACKEND_LLM_URL":
                                            "http://modelserver.invalid:9/v1"}, _ctx())
    assert c.status == "fail"


# ── run_checks orchestration ──────────────────────────────────────────────────

async def test_run_checks_one_check_per_name_even_when_everything_fails(tmp_path,
                                                                        monkeypatch):
    def broken_http(**kwargs):
        raise RuntimeError("sockets are down")

    async def broken_connect(cfg):
        raise RuntimeError("db is down")

    def broken_run(args, timeout):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(dx, "_make_http", broken_http)
    monkeypatch.setattr(dx, "_connect", broken_connect)
    monkeypatch.setattr(dx, "_subprocess_run", broken_run)

    names = dx.CHECKS + dx.EXTRA_CHECKS
    results = await dx.run_checks({"POSTGRES_DB": "x", "BACKEND_LLM_URL":
                                   "http://h:9/v1", "EMBEDDING_URL": "http://h:9/v1"},
                                  names=names, repo_root=tmp_path)
    assert [r.name for r in results] == list(names)
    for r in results:
        assert r.status in ("ok", "warn", "fail", "skipped"), r
        assert isinstance(r.details, dict)
    assert results[[r.name for r in results].index("system")].status == "skipped"


async def test_run_checks_timeout_becomes_fail(tmp_path, monkeypatch):
    async def slow_check(env, ctx):
        await asyncio.sleep(5)
        return dx.Check("install", "ok", "too late", {})

    monkeypatch.setitem(dx._REGISTRY, "install", slow_check)
    results = await dx.run_checks({}, names=("install",), repo_root=tmp_path,
                                  per_check_timeout=0.1)
    assert results[0].status == "fail"
    assert "timed out" in results[0].summary


async def test_run_checks_missing_dependency_skipped(tmp_path, monkeypatch):
    async def no_pkg(env, ctx):
        raise ModuleNotFoundError("No module named 'asyncpg'", name="asyncpg")

    monkeypatch.setitem(dx._REGISTRY, "database", no_pkg)
    results = await dx.run_checks({}, names=("database",), repo_root=tmp_path)
    assert results[0].status == "skipped"
    assert "asyncpg" in results[0].summary


async def test_run_checks_unknown_name(tmp_path):
    results = await dx.run_checks({}, names=("nope",), repo_root=tmp_path)
    assert results[0].status == "fail"
    assert "unknown" in results[0].summary


async def test_run_checks_redacts_crash_message(tmp_path, monkeypatch):
    async def crash(env, ctx):
        raise RuntimeError("boom at /home/r3aln4me/secret/dir with password=hunter2")

    monkeypatch.setitem(dx._REGISTRY, "install", crash)
    results = await dx.run_checks({}, names=("install",), repo_root=tmp_path)
    dumped = _dump(results[0])
    assert results[0].status == "fail"
    assert "hunter2" not in dumped
    assert "r3aln4me" not in dumped


async def test_run_checks_writes_nothing(tmp_path, monkeypatch):
    _plant_logs(tmp_path)
    before = _snapshot(tmp_path)

    def fake_run(args, timeout):
        raise FileNotFoundError(args[0])

    async def refuse(cfg):
        raise RuntimeError("no db")

    monkeypatch.setattr(dx, "_subprocess_run", fake_run)
    monkeypatch.setattr(dx, "_connect", refuse)
    _patch_http(monkeypatch, _recording_handler({
        "/health": (200, {}),
        "/v1/models": (200, _models(["embed-me", "chat-model"])),
        "/v1/embeddings": (200, {"data": [{"embedding": [0.0] * 4}]}),
        "/admin/api/v1/scheduler": (200, {}),
    })[0])

    await dx.run_checks({"BACKEND_LLM_URL": "http://h.invalid:9/v1",
                         "EMBEDDING_URL": "http://h.invalid:9/v1",
                         "POSTGRES_DB": "x"},
                        names=dx.CHECKS + dx.EXTRA_CHECKS, system_info=True,
                        repo_root=tmp_path)
    assert _snapshot(tmp_path) == before


def test_effective_settings_file_wins_over_env(tmp_path):
    """Real run 2026-10-14: the model-server check used .env's URL although the settings
    file (which SPM itself obeys) pointed elsewhere."""
    import json as _json
    from spm import diagnostics as d
    cfg = tmp_path / "config.local.json"
    cfg.write_text(_json.dumps({"BACKEND_LLM_URL": "http://10.0.0.5:1234/v1"}), encoding="utf-8")
    ctx = d._Ctx(root=tmp_path, redactor=d.Redactor()) if hasattr(d, "_Ctx") else None
    env = {"SPM_CONFIG_PATH": str(cfg), "BACKEND_LLM_URL": "http://127.0.0.1:13305/v1"}
    merged, path, err, exists = d._effective_settings(env, ctx)
    assert merged["BACKEND_LLM_URL"] == "http://10.0.0.5:1234/v1" and exists and err is None


def test_redact_tree_redacts_keys():
    from spm import diagnostics as d
    from spm.redact import Redactor
    out = d._redact_tree({"csa_memory_zelda": 3}, Redactor(names={"zelda": "char#1"}))
    assert out == {"csa_memory_char#1": 3}


# ── third review, item 4: names stay hidden when the database is down ────────

async def test_name_table_falls_back_to_the_last_good_names(monkeypatch, tmp_path):
    from spm.redact import Redactor

    async def db_up(env):
        return {"chat": ["st_chat_abc"], "char": ["mei", "lian"], "player": ["Brenda"]}

    async def db_down(env):
        raise OSError("connection refused")

    monkeypatch.setattr(dx, "_name_lists", db_up)
    first = await dx.name_table(_FAKE_ENV)
    assert first["mei"].startswith("char#") and first["Brenda"] == "player#1"
    assert (tmp_path / "run" / "names.json").is_file()

    monkeypatch.setattr(dx, "_name_lists", db_down)
    table = await dx.name_table(_FAKE_ENV)
    assert table == first
    line = ("2026-10-16 10:00:00,000 [INFO] proxy.gating: [Perception] turn 12: "
            "Mei (kitchen) -> Lian (garden) muffled")
    out = Redactor(names=table).text(line)
    assert "Mei" not in out and "Lian" not in out and "char#" in out


async def test_name_table_merges_new_names_into_the_cache(monkeypatch, tmp_path):
    async def first(env):
        return {"chat": [], "char": ["mei"], "player": []}

    async def later(env):
        return {"chat": [], "char": ["lian"], "player": []}

    monkeypatch.setattr(dx, "_name_lists", first)
    await dx.name_table(_FAKE_ENV)
    monkeypatch.setattr(dx, "_name_lists", later)
    table = await dx.name_table(_FAKE_ENV)
    assert set(table) == {"mei", "lian"}
