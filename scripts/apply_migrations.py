"""Apply the SQL migrations in scripts/migrations/ in filename order.

Applied files are tracked in spm_schema_migrations, so running this twice is
a no-op; the migration files themselves are also written to be idempotent, so
a crash between "execute" and "record" is safe to re-run.

Usage:
    python scripts/apply_migrations.py [--list]

Connection comes from the POSTGRES_* env vars (plus SPM_DB_NAME/POSTGRES_DB
for the database), defaulting to the live SPM database. The test suite calls
apply_migrations(conn) directly against the throwaway DB.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import List

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spm_schema_migrations (
    filename VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _db_config_from_env() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "spm_user"),
        "password": os.environ.get("POSTGRES_PASSWORD", "spm_secure_password"),
        "database": os.environ.get("SPM_DB_NAME")
        or os.environ.get("POSTGRES_DB")
        or "litellm_postgres",
    }


async def applied_migrations(conn: asyncpg.Connection) -> List[str]:
    await conn.execute(_TRACKING_TABLE_SQL)
    rows = await conn.fetch("SELECT filename FROM spm_schema_migrations ORDER BY filename;")
    return [r["filename"] for r in rows]


async def apply_migrations(conn: asyncpg.Connection) -> List[str]:
    """Apply every pending migration file on `conn`; return the names applied."""
    already = set(await applied_migrations(conn))
    ran: List[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in already:
            logger.info("[migrations] %s already applied, skipping.", path.name)
            continue
        logger.info("[migrations] applying %s ...", path.name)
        async with conn.transaction():
            await conn.execute(path.read_text())
            await conn.execute(
                "INSERT INTO spm_schema_migrations (filename) VALUES ($1);", path.name
            )
        ran.append(path.name)
        logger.info("[migrations] %s applied.", path.name)
    if not ran:
        logger.info("[migrations] nothing to apply.")
    return ran


async def _main_async(list_only: bool) -> None:
    cfg = _db_config_from_env()
    conn = await asyncpg.connect(**cfg)
    try:
        if list_only:
            done = set(await applied_migrations(conn))
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                mark = "applied" if path.name in done else "PENDING"
                print(f"{mark:8} {path.name}")
            return
        ran = await apply_migrations(conn)
        print(f"Applied {len(ran)} migration(s) to {cfg['database']}: {ran or 'none pending'}")
    finally:
        await conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show migration status, apply nothing")
    args = parser.parse_args()
    asyncio.run(_main_async(args.list))


if __name__ == "__main__":
    main()
