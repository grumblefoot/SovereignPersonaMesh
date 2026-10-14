"""`python -m spm init`: create and prepare the SPM database (brief P3b).

Connection comes from POSTGRES_HOST/PORT/USER/PASSWORD/DB (with the repo-root
.env merged in, same as `spm start`). POSTGRES_DB is REQUIRED on purpose:
silently defaulting to the live database name is exactly what this command
must never do.

Behaviour (see .sprint brief P3b):
  - missing database -> CREATE DATABASE via the `postgres` maintenance DB;
  - present and non-empty -> refuse unless --existing (then migrations only);
  - empty or new -> scripts/init_db.sql, then apply_migrations(conn);
  - CREATE EXTENSION vector failure -> pgvector install/rights hint, exit 1.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import asyncpg

from .run import child_env

_VECTOR_HINT = """\
ERROR: the pgvector extension could not be created.
  - Install pgvector on the PostgreSQL server first, e.g. on Fedora/RHEL:
        sudo dnf install postgresql16-serverPGDG pgvector_16
    on Debian/Ubuntu:
        sudo apt install postgresql-16-pgvector
  - Or let a superuser create it once in the target database:
        CREATE EXTENSION vector;
    then re-run this command."""


def quote_ident(name: str) -> str:
    """Quote `name` as a PostgreSQL identifier (double quotes, "" escaping).

    asyncpg has no placeholder syntax for identifiers, so CREATE/DROP DATABASE
    names must be formatted — quoting plus the NUL check makes injection via a
    hostile POSTGRES_DB value impossible.
    """
    if not name or "\x00" in name:
        raise ValueError(f"invalid database name: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def db_config_from_env(env: dict) -> dict | None:
    """POSTGRES_* connection settings from a merged environment dict.

    Returns None (with a printed reason) when POSTGRES_DB is unset — init must
    never fall back to a default database name.
    """
    database = env.get("POSTGRES_DB")
    if not database:
        print("ERROR: POSTGRES_DB is not set. Set POSTGRES_DB (and the other "
              "POSTGRES_* variables) in .env and re-run.", file=sys.stderr)
        return None
    try:
        port = int(env.get("POSTGRES_PORT") or 5432)
    except ValueError:
        print(f"ERROR: POSTGRES_PORT is not a number: {env.get('POSTGRES_PORT')!r}",
              file=sys.stderr)
        return None
    return {
        "host": env.get("POSTGRES_HOST", "localhost"),
        "port": port,
        "user": env.get("POSTGRES_USER", "spm_user"),
        # Same default as proxy/main.py and the engine (an empty default failed where the
        # services connected: Claude live check 2026-10-14).
        "password": env.get("POSTGRES_PASSWORD", "spm_secure_password"),
        "database": database,
    }


async def _database_exists(maint: asyncpg.Connection, database: str) -> bool:
    row = await maint.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
    return row is not None


async def _looks_initialized(conn: asyncpg.Connection) -> bool:
    """True if the database holds SPM data: the perception table or any dynamic
    csa_memory_* table (the same markers the brief names)."""
    row = await conn.fetchval(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public'
          AND (table_name = 'spm_perception' OR table_name LIKE 'csa\\_memory\\_%')
        LIMIT 1
        """
    )
    return row is not None


def _is_extension_error(err: Exception) -> bool:
    text = str(err).lower()
    return "vector" in text and "extension" in text


async def init_database(env: dict, *, existing: bool = False,
                        repo_root: Path | None = None) -> int:
    """Run the init flow; prints progress; returns the process exit code."""
    # Imported here so `import spm.db` stays light and scripts/ needs no
    # sys.path entry until init actually runs.
    import sys as _sys
    root_str = str((repo_root or Path(__file__).resolve().parent.parent))
    if root_str not in _sys.path:
        _sys.path.insert(0, root_str)
    from scripts.apply_migrations import apply_migrations

    cfg = db_config_from_env(env)
    if cfg is None:
        return 1
    database = cfg["database"]
    root = repo_root or Path(__file__).resolve().parent.parent
    init_sql_path = root / "scripts" / "init_db.sql"

    # 1. Ensure the database exists (maintenance connection).
    created = False
    maint = await asyncpg.connect(**{**cfg, "database": "postgres"})
    try:
        if await _database_exists(maint, database):
            print(f"Database {database!r} exists.")
        else:
            try:
                await maint.execute(f"CREATE DATABASE {quote_ident(database)};")
            except asyncpg.PostgresError as err:
                print(f"ERROR: cannot create database {database!r}: {err}",
                      file=sys.stderr)
                print("Fix: a superuser must either create it once —\n"
                      f'    CREATE DATABASE {quote_ident(database)};\n'
                      f"or grant the rights to the role {cfg['user']!r} —\n"
                      "    ALTER ROLE <user> CREATEDB;", file=sys.stderr)
                return 1
            created = True
            print(f"Created database {database!r}.")
    finally:
        await maint.close()

    conn = await asyncpg.connect(**cfg)
    try:
        # 2. Non-empty database: never touch existing data without --existing.
        if not created and await _looks_initialized(conn):
            if not existing:
                print(f"ERROR: database {database!r} already contains SPM tables "
                      "(spm_perception / csa_memory_*). Refusing to modify it.",
                      file=sys.stderr)
                print("If that is intended (upgrading an existing SPM database), "
                      "re-run with --existing.", file=sys.stderr)
                return 1
            ran = await apply_migrations(conn)
            print(f"Applied {len(ran)} pending migration(s) to {database!r}"
                  + (f": {', '.join(ran)}" if ran else " (nothing pending)."))
            print(f"Next step: python -m spm start")
            return 0

        # 3. Empty or new database: full schema, then complete the ledger.
        try:
            await conn.execute(init_sql_path.read_text(encoding="utf-8"))
        except asyncpg.PostgresError as err:
            if _is_extension_error(err):
                print(_VECTOR_HINT, file=sys.stderr)
            else:
                print(f"ERROR: {init_sql_path} failed: {err}", file=sys.stderr)
            return 1
        try:
            ran = await apply_migrations(conn)
        except asyncpg.PostgresError as err:
            if _is_extension_error(err):
                print(_VECTOR_HINT, file=sys.stderr)
                return 1
            raise

        # 5. Summary.
        n_tables = await conn.fetchval(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
        ledger = await conn.fetch(
            "SELECT filename FROM spm_schema_migrations ORDER BY filename")
        print(f"Database ready: {database!r}")
        print(f"  tables created:   {n_tables}")
        print(f"  migrations applied: {len(ran)}"
              + (f" ({', '.join(ran)})" if ran else "")
              + f"; ledger now lists {len(ledger)} file(s) "
                "(manual-only migrations like 008 are never applied).")
        print("Next step: python -m spm start")
        return 0
    finally:
        await conn.close()


def cmd_init(repo_root: Path, existing: bool = False) -> int:
    """Sync entry point for `python -m spm init [--existing]`."""
    env = child_env(repo_root)
    try:
        return asyncio.run(init_database(env, existing=existing, repo_root=repo_root))
    except asyncio.CancelledError:
        return 1
    except OSError as err:  # e.g. server unreachable
        print(f"ERROR: cannot reach PostgreSQL at "
              f"{env.get('POSTGRES_HOST', 'localhost')}: "
              f"{err}", file=sys.stderr)
        return 1
    except asyncpg.PostgresError as err:
        print(f"ERROR: PostgreSQL refused the connection: {err}", file=sys.stderr)
        return 1
