"""Tests for `python -m spm init` (brief P3b) and scripts/apply_migrations.py.

Each test uses a UNIQUE throwaway database (spm_test_init_<random>) on the
test server from tests/_testdb.py and drops it at the end. Never the live
database litellm_postgres.
"""

import asyncio
import pathlib
import uuid

import asyncpg
import pytest

import scripts.apply_migrations as apply_migrations_module
from spm.db import init_database, quote_ident
from tests._testdb import LIVE_DB_NAMES, TEST_DB_CONFIG


def _fresh_env() -> tuple[str, dict]:
    """Env dict for spm.init: test-server credentials, unique database name."""
    dbname = f"spm_test_init_{uuid.uuid4().hex[:10]}"
    assert dbname not in LIVE_DB_NAMES
    env = {
        "POSTGRES_HOST": TEST_DB_CONFIG["host"],
        "POSTGRES_PORT": str(TEST_DB_CONFIG["port"]),
        "POSTGRES_USER": TEST_DB_CONFIG["user"],
        "POSTGRES_PASSWORD": TEST_DB_CONFIG["password"],
        "POSTGRES_DB": dbname,
    }
    return dbname, env


async def _drop_db(dbname: str) -> None:
    admin = await asyncpg.connect(**{**TEST_DB_CONFIG, "database": "postgres"})
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {quote_ident(dbname)} WITH (FORCE);")
    finally:
        await admin.close()


@pytest.fixture
async def init_env():
    """Yield (dbname, env); guarantee the scratch database is dropped after."""
    dbname, env = _fresh_env()
    yield dbname, env
    await _drop_db(dbname)


async def _connect(dbname: str) -> asyncpg.Connection:
    return await asyncpg.connect(**{**TEST_DB_CONFIG, "database": dbname})


async def _table_exists(conn, name: str) -> bool:
    return await conn.fetchval(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=$1", name) is not None


async def _ledger(conn) -> list[str]:
    rows = await conn.fetch("SELECT filename FROM spm_schema_migrations ORDER BY filename")
    return [r["filename"] for r in rows]


def test_quote_ident_escapes_double_quotes():
    assert quote_ident('sp"m') == '"sp""m"'
    with pytest.raises(ValueError):
        quote_ident("")


async def test_missing_postgres_db_refuses(init_env, capsys):
    _, env = init_env
    env_no_db = {k: v for k, v in env.items() if k != "POSTGRES_DB"}
    rc = await init_database(env_no_db)
    assert rc == 1
    assert "POSTGRES_DB is not set" in capsys.readouterr().err


async def test_fresh_database_created_and_initialised(init_env):
    dbname, env = init_env
    rc = await init_database(env)
    assert rc == 0
    conn = await _connect(dbname)
    try:
        assert await _table_exists(conn, "spm_chat_rules")
        assert await _table_exists(conn, "spm_perception")
        ledger = await _ledger(conn)
        for n in range(1, 8):
            assert any(f"{n:03d}_" in f for f in ledger), f"migration {n:03d} missing"
        assert not any("008" in f for f in ledger), "manual-only 008 must not be applied"
    finally:
        await conn.close()


async def test_unrelated_tables_refuse_even_with_existing(init_env, capsys):
    """Brief FX-D: ANY table in schema public makes the database non-empty.
    Tables that are not SPM-shaped refuse init with AND without --existing."""
    dbname, env = init_env
    # Create the database ourselves with one unrelated table.
    admin = await asyncpg.connect(**{**TEST_DB_CONFIG, "database": "postgres"})
    try:
        await admin.execute(f"CREATE DATABASE {quote_ident(dbname)}")
    finally:
        await admin.close()
    conn = await _connect(dbname)
    try:
        await conn.execute("CREATE TABLE public.somebodys_data (id int)")
    finally:
        await conn.close()

    capsys.readouterr()
    assert await init_database(env) == 1
    err = capsys.readouterr().err
    assert "holds other data" in err and "POSTGRES_DB" in err

    capsys.readouterr()
    assert await init_database(env, existing=True) == 1
    err = capsys.readouterr().err
    assert "holds other data" in err

    # Untouched: the unrelated table is still the only one.
    conn = await _connect(dbname)
    try:
        n = await conn.fetchval(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
        assert n == 1
        assert not await _table_exists(conn, "spm_perception")
    finally:
        await conn.close()


def test_unreachable_host_exit_1_no_traceback(tmp_path, monkeypatch, capsys):
    """Brief FX-D: connection errors give one friendly line naming host and
    port and exit 1 — no traceback. Port 1 on localhost: refused instantly."""
    from spm.db import cmd_init
    # child_env merges os.environ with the .env WITHOUT overriding: a developer's
    # .env (in os.environ since collection) would otherwise win over the tmp .env
    # and this test would talk to a real server. POSTGRES_DB stays as conftest
    # sets it unless the tmp .env below is the only source.
    for var in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        "POSTGRES_HOST=127.0.0.1\nPOSTGRES_PORT=1\n"
        "POSTGRES_USER=nobody\nPOSTGRES_PASSWORD=nobody\n"
        "POSTGRES_DB=spm_test_unreachable\n", encoding="utf-8")
    capsys.readouterr()
    rc = cmd_init(tmp_path)
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot connect to PostgreSQL at 127.0.0.1:1" in err
    assert "Traceback" not in err


async def test_rerun_on_initialized_database_refuses(init_env, capsys):
    dbname, env = init_env
    assert await init_database(env) == 0
    capsys.readouterr()
    rc = await init_database(env)
    assert rc == 1
    err = capsys.readouterr().err
    assert "--existing" in err
    # Data was not touched: ledger still complete.
    conn = await _connect(dbname)
    try:
        assert len(await _ledger(conn)) >= 7
    finally:
        await conn.close()


async def test_existing_applies_migrations_and_keeps_rows(init_env):
    dbname, env = init_env
    assert await init_database(env) == 0

    conn = await _connect(dbname)
    try:
        await conn.execute(
            "INSERT INTO spm_chat_rules (session_id, level, rule_text, rule_type) "
            "VALUES ('seed-test', 'world', 'the seeded row survives', 'invariant')")
        # Make migration 007 look pending again: --existing must re-apply it
        # (it is idempotent) while leaving our seeded row alone.
        await conn.execute("DELETE FROM spm_schema_migrations "
                           "WHERE filename = '007_chat_rules.sql'")
    finally:
        await conn.close()

    rc = await init_database(env, existing=True)
    assert rc == 0

    conn = await _connect(dbname)
    try:
        row = await conn.fetchval(
            "SELECT rule_text FROM spm_chat_rules WHERE session_id = 'seed-test'")
        assert row == "the seeded row survives"
        ledger = await _ledger(conn)
        assert "007_chat_rules.sql" in ledger
        assert not any("008" in f for f in ledger)
    finally:
        await conn.close()


async def test_apply_migrations_reads_files_as_utf8(init_env, tmp_path, monkeypatch):
    """Regression: apply_migrations must pass encoding='utf-8' explicitly —
    on Windows Path.read_text() defaults to cp1252 and dies on non-ASCII."""
    dbname, env = init_env
    assert await init_database(env) == 0

    mdir = tmp_path / "migrations"
    mdir.mkdir()
    sql = mdir / "099_non_ascii.sql"
    sql.write_text("-- comentário não-ASCII: æøå 日本語\nSELECT 1;\n", encoding="utf-8")

    seen_encodings = []
    original_read_text = pathlib.Path.read_text

    def spy_read_text(self, *args, **kwargs):
        seen_encodings.append(kwargs.get("encoding"))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", spy_read_text)
    monkeypatch.setattr(apply_migrations_module, "MIGRATIONS_DIR", mdir)

    conn = await _connect(dbname)
    try:
        ran = await apply_migrations_module.apply_migrations(conn)
        assert ran == ["099_non_ascii.sql"]
        # The migration file was read with an explicit utf-8 encoding.
        assert "utf-8" in seen_encodings
    finally:
        await conn.close()


def test_connection_defaults_match_the_services():
    """Live check 2026-10-14: init defaulted to an empty password and could not connect
    where the proxy (default spm_secure_password) could."""
    from spm import db
    cfg = db.db_config_from_env({"POSTGRES_DB": "x"})
    assert cfg["password"] == "spm_secure_password" and cfg["user"] == "spm_user"
    assert cfg["host"] == "localhost" and cfg["port"] == 5432
