"""Resumable re-embed / backfill job (embeddings plan phase 2).

Walks every csa_memory_* and csa_lore_rules_* table and re-embeds rows whose
embedding is NULL or whose embedding_space_id differs from the active space,
in batches, stamping embedding_space_id as it goes. Each batch commits on its
own, so the job is safe to interrupt and re-run: a rerun simply continues
with whatever is still missing, and a completed run makes the next one a
no-op.

No-op when the provider is 'none' (nothing to embed vectors with) and when it
is 'fake' without --allow-fake (fake vectors in a real DB are only for tests).

Usage (from the repo root):
    python -m scripts.reembed [--batch-size 64] [--dry-run] [--allow-fake]

Connection comes from the POSTGRES_* env vars like scripts/apply_migrations.py.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg

if __package__ in (None, ""):  # direct `python scripts/reembed.py` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxy.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

_ZERO_UUID = "00000000-0000-0000-0000-000000000000"


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


def _vec_str(vec: List[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"


async def _tables_like(conn: asyncpg.Connection, pattern: str) -> List[str]:
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
        "AND tablename LIKE $1 ESCAPE '\\' ORDER BY tablename;",
        pattern,
    )
    return [r["tablename"] for r in rows]


async def _reembed_table(
    conn: asyncpg.Connection,
    service: EmbeddingService,
    table: str,
    text_sql: str,
    embedding_col: str,
    space_id: int,
    batch_size: int,
    dry_run: bool,
) -> Dict[str, int]:
    """Sweep one table with an id-keyset cursor. Rows whose text embeds to
    None (empty text, or a provider outage) are skipped this run and picked
    up by the next one, so a partial outage never stalls the sweep."""
    stats = {"updated": 0, "skipped": 0}
    cursor = _ZERO_UUID
    where = (
        f"({embedding_col} IS NULL OR embedding_space_id IS DISTINCT FROM $1)"
        " AND id > $2"
    )
    while True:
        rows = await conn.fetch(
            f"SELECT id, {text_sql} AS text FROM {table} "
            f"WHERE {where} ORDER BY id LIMIT $3;",
            space_id, cursor, batch_size,
        )
        if not rows:
            break
        cursor = str(rows[-1]["id"])
        if dry_run:
            stats["skipped"] += len(rows)
            continue
        vectors = await service.batch_generate_embeddings([r["text"] for r in rows])
        for row, vec in zip(rows, vectors):
            if vec is None:
                stats["skipped"] += 1
                continue
            await conn.execute(
                f"UPDATE {table} SET {embedding_col} = $1::vector, "
                f"embedding_space_id = $2 WHERE id = $3;",
                _vec_str(vec), space_id, row["id"],
            )
            stats["updated"] += 1
        logger.info(
            "[reembed] %s: %d embedded, %d skipped so far",
            table, stats["updated"], stats["skipped"],
        )
    return stats


async def reembed_all(
    conn: asyncpg.Connection,
    service: Optional[EmbeddingService] = None,
    batch_size: int = 64,
    allow_fake: bool = False,
    dry_run: bool = False,
    only_tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-embed all memory and lore rows that lack a vector in the active
    space. `only_tables` restricts the sweep to the named tables (tests, or
    a targeted backfill of one character).
    Returns {"updated", "skipped", "space_id", "tables"}."""
    service = service or get_embedding_service()
    totals: Dict[str, Any] = {"updated": 0, "skipped": 0, "space_id": None, "tables": 0}

    if not service.available:
        logger.info("[reembed] embedding provider is 'none': nothing to do.")
        return totals
    if service.provider_name == "fake" and not allow_fake:
        logger.warning(
            "[reembed] provider is 'fake' and --allow-fake was not given: "
            "refusing to write test vectors into the database."
        )
        return totals

    space_id = await service.ensure_space(conn)
    if space_id is None:
        logger.error(
            "[reembed] could not establish the active embedding space "
            "(provider down?); try again when the embedder is reachable."
        )
        return totals
    totals["space_id"] = space_id

    memory_tables = await _tables_like(conn, "csa\\_memory\\_%")
    lore_tables = await _tables_like(conn, "csa\\_lore\\_rules\\_%")
    if only_tables is not None:
        wanted = set(only_tables)
        memory_tables = [t for t in memory_tables if t in wanted]
        lore_tables = [t for t in lore_tables if t in wanted]
    logger.info(
        "[reembed] space id %d (%s/%s, dim %d); %d memory + %d lore tables%s",
        space_id, service.provider_name, service.model, service.space.dim,
        len(memory_tables), len(lore_tables), " [dry run]" if dry_run else "",
    )

    for table in memory_tables:
        # Make sure old tables are upgraded to the new shape before touching them.
        await conn.execute(
            "SELECT create_csa_memory_table($1);", table[len("csa_memory_"):]
        )
        stats = await _reembed_table(
            conn, service, table,
            "trim(coalesce(sensory_input, '') || ' ' || coalesce(public_response, ''))",
            "episodic_embedding", space_id, batch_size, dry_run,
        )
        totals["updated"] += stats["updated"]
        totals["skipped"] += stats["skipped"]
        totals["tables"] += 1

    for table in lore_tables:
        await conn.execute(
            "SELECT create_csa_lore_rules_table($1);", table[len("csa_lore_rules_"):]
        )
        stats = await _reembed_table(
            conn, service, table,
            "coalesce(rule_text, '')",
            "rule_embedding", space_id, batch_size, dry_run,
        )
        totals["updated"] += stats["updated"]
        totals["skipped"] += stats["skipped"]
        totals["tables"] += 1

    logger.info(
        "[reembed] done: %d rows embedded, %d skipped across %d tables.",
        totals["updated"], totals["skipped"], totals["tables"],
    )
    return totals


async def _main_async(args: argparse.Namespace) -> None:
    conn = await asyncpg.connect(**_db_config_from_env())
    try:
        totals = await reembed_all(
            conn,
            batch_size=args.batch_size,
            allow_fake=args.allow_fake,
            dry_run=args.dry_run,
        )
    finally:
        await conn.close()
    print(totals)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="count rows, write nothing")
    parser.add_argument(
        "--allow-fake", action="store_true",
        help="let the 'fake' test provider write vectors (tests only)",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
