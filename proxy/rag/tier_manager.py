"""
FR-003: Tiered Data Lifecycle & Cold Storage Reconstitution.

Provides a MemoryTierManager that:
  - Archives volatile (non-core) memory records into gzip-compressed .jsonl.gz
    cold storage files.
  - Reconstitutes cold archives back into the hot memory tables.
  - Reports tier statistics (hot / warm / cold record counts).

Core Memory Immunity: records with is_core_memory = TRUE are never archived
or deleted by any lifecycle operation.
"""

import gzip
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            pass
    return datetime.utcnow()


def _parse_uuid(val: Any) -> Optional[uuid.UUID]:
    if isinstance(val, uuid.UUID):
        return val
    if isinstance(val, str):
        try:
            return uuid.UUID(val)
        except ValueError:
            pass
    return None

# Where cold archive files are stored
COLD_ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "storage",
    "cold_archives",
)


class MemoryTierManager:
    """Manages the three-tier memory lifecycle (hot / warm / cold)."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    # ------------------------------------------------------------------
    # 1. Archive old volatile memories
    # ------------------------------------------------------------------

    async def archive_old_memories(
        self,
        character_id: str,
        session_id: str,
        max_records: int = 500,
        max_age_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Export volatile (is_core_memory = FALSE) records older than max_age_days
        to a .jsonl.gz file and register the archive in spm_cold_archives.

        Returns a summary dict with keys:
          archive_id, archive_path, record_count, deleted_count, char_id.
        """
        char_id = character_id.lower()
        table_name = f"csa_memory_{char_id}"

        # Ensure the character table and spm_cold_archives exist
        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT create_csa_memory_table($1);", char_id)
            await self._ensure_cold_archives_table(conn)

        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        # --- Read eligible rows (volatile only, ordered oldest first) ---
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, session_id, timestamp, sensory_input, inner_monologue,
                       episodic_embedding, importance_score, is_core_memory,
                       is_subjective, access_count, last_accessed_at
                FROM {table_name}
                WHERE session_id = $1
                  AND is_core_memory = FALSE
                  AND timestamp <= $2
                ORDER BY timestamp ASC
                LIMIT $3;
                """,
                session_id,
                cutoff,
                max_records,
            )

        if not rows:
            logger.info(
                f"[TierManager] No volatile records eligible for archival "
                f"(char={char_id}, session={session_id})."
            )
            return {
                "archive_id": None,
                "archive_path": None,
                "record_count": 0,
                "deleted_count": 0,
                "char_id": char_id,
            }

        # --- Build .jsonl.gz ---
        os.makedirs(COLD_ARCHIVE_DIR, exist_ok=True)
        os.makedirs(
            os.path.join(COLD_ARCHIVE_DIR, char_id),
            exist_ok=True,
        )
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        archive_path = os.path.join(
            COLD_ARCHIVE_DIR,
            char_id,
            f"{session_id}_{ts}.jsonl.gz",
        )

        with gzip.open(archive_path, "wt", encoding="utf-8") as gz_file:
            for row in rows:
                record = dict(row)
                # Serialize embedding vector as a plain list
                if record.get("episodic_embedding") is not None:
                    # asyncpg returns VECTOR as a numpy array-like; convert to list
                    record["episodic_embedding"] = _embedding_to_list(
                        record["episodic_embedding"]
                    )
                gz_file.write(json.dumps(record, default=str) + "\n")

        record_count = len(rows)
        record_ids = [str(r["id"]) for r in rows]

        # --- Register archive in spm_cold_archives ---
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO spm_cold_archives
                    (session_id, character_id, archive_path, record_count)
                VALUES ($1, $2, $3, $4)
                RETURNING archive_id, created_at;
                """,
                session_id,
                char_id,
                archive_path,
                record_count,
            )

        archive_id = str(result["archive_id"])

        # --- Delete archived rows (core memories are NOT touched) ---
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                f"""
                DELETE FROM {table_name}
                WHERE id = ANY($1::uuid[])
                  AND is_core_memory = FALSE;
                """,
                record_ids,
            )

        logger.info(
            f"[TierManager] Archived {record_count} volatile records from "
            f"{char_id} (session={session_id}) -> {archive_path}"
        )

        return {
            "archive_id": archive_id,
            "archive_path": archive_path,
            "record_count": record_count,
            "deleted_count": record_count,
            "char_id": char_id,
        }

    # ------------------------------------------------------------------
    # 2. Reconstitute a cold archive
    # ------------------------------------------------------------------

    async def reconstitute_cold_archive(
        self,
        archive_id: str,
        character_id: str,
    ) -> Dict[str, Any]:
        """
        Decompress a .jsonl.gz cold archive, re-insert records into the hot
        csa_memory_{character_id} table, then remove the archive record and file.

        Returns a summary dict with keys:
          reconstituted_count, archive_path, char_id, session_id.
        """
        char_id = character_id.lower()
        table_name = f"csa_memory_{char_id}"

        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT create_csa_memory_table($1);", char_id)
            await self._ensure_cold_archives_table(conn)

        # --- Fetch archive metadata ---
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT archive_id, session_id, character_id, archive_path,
                       record_count
                FROM spm_cold_archives
                WHERE archive_id = $1::uuid;
                """,
                archive_id,
            )

        if row is None:
            raise ValueError(f"Cold archive not found: {archive_id}")

        archive_path = row["archive_path"]
        session_id = row["session_id"]
        record_count = int(row["record_count"])

        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"Archive file missing: {archive_path}")

        # --- Decompress and re-insert ---
        reconstituted = 0
        errors = []

        async with self.db_pool.acquire() as conn:
            with gzip.open(archive_path, "rt", encoding="utf-8") as gz_file:
                for line in gz_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        await self._insert_memory_record(
                            conn, table_name, record, session_id
                        )
                        reconstituted += 1
                    except Exception as exc:
                        logger.error(f"[TierManager] Reconstitution insert error: {exc}")
                        errors.append(str(exc))

        # --- Remove archive DB row ---
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM spm_cold_archives WHERE archive_id = $1::uuid;",
                archive_id,
            )

        # --- Delete archive file ---
        try:
            os.remove(archive_path)
        except OSError:
            pass

        logger.info(
            f"[TierManager] Reconstituted {reconstituted} records from "
            f"{archive_path} into {char_id}."
        )

        return {
            "reconstituted_count": reconstituted,
            "error_count": len(errors),
            "archive_path": archive_path,
            "char_id": char_id,
            "session_id": session_id,
        }

    # ------------------------------------------------------------------
    # 3. Tier statistics
    # ------------------------------------------------------------------

    async def get_tier_stats(
        self,
        character_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return hot (volatile / recent), warm (core memory), and cold (archive)
        record counts for the given character and session.
        """
        char_id = character_id.lower()
        table_name = f"csa_memory_{char_id}"
        # Build WHERE clause: always starts with "WHERE", session filter is optional.
        where_session = ""
        params: list = []
        param_idx = 1
        if session_id is not None:
            where_session = f"WHERE session_id = ${param_idx}"
            params.append(session_id)
            param_idx += 1

        # --- Hot: non-core records in the memory table ---
        hot_where = "WHERE is_core_memory = FALSE"
        if session_id is not None:
            hot_where = f"WHERE session_id = ${param_idx} AND is_core_memory = FALSE"
            params = [session_id]
            param_idx = 2

        async with self.db_pool.acquire() as conn:
            await conn.execute("SELECT create_csa_memory_table($1);", char_id)
            await self._ensure_cold_archives_table(conn)
            hot = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {table_name}
                {hot_where};
                """,
                *params,
            )

        # --- Warm: core memory records ---
        warm_where = "WHERE is_core_memory = TRUE"
        if session_id is not None:
            warm_where = f"WHERE session_id = ${param_idx} AND is_core_memory = TRUE"
            warm_params = [session_id]

        async with self.db_pool.acquire() as conn:
            warm = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {table_name}
                {warm_where};
                """,
                *[session_id] if session_id is not None else [],
            )

        # --- Cold: registered archive records ---
        cold_where = "WHERE character_id = $1"
        cold_params: list = [char_id]
        if session_id is not None:
            cold_where += " AND session_id = $2"
            cold_params.append(session_id)

        async with self.db_pool.acquire() as conn:
            cold = await conn.fetchval(
                f"SELECT COALESCE(SUM(record_count), 0) FROM spm_cold_archives "
                f"{cold_where};",
                *cold_params,
            )

        return {
            "character_id": char_id,
            "session_id": session_id,
            "hot": int(hot),
            "warm": int(warm),
            "cold": int(cold),
            "total": int(hot + warm + cold),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _ensure_cold_archives_table(conn: asyncpg.Connection) -> None:
        """Ensure spm_cold_archives table exists."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS spm_cold_archives (
                archive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id VARCHAR(255) NOT NULL,
                character_id VARCHAR(255) NOT NULL,
                archive_path TEXT NOT NULL,
                record_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_cold_archives_session ON spm_cold_archives (session_id);
        """)

    @staticmethod
    async def _insert_memory_record(
        conn: asyncpg.Connection,
        table_name: str,
        record: Dict[str, Any],
        session_id: str,
    ) -> None:
        """Insert a single memory record dict into the character memory table."""
        rec_id = _parse_uuid(record.get("id"))
        timestamp = _parse_dt(record.get("timestamp"))
        last_accessed = _parse_dt(record.get("last_accessed_at"))

        embedding_str = None
        if record.get("episodic_embedding") is not None:
            embed_list = _embedding_to_list(record["episodic_embedding"])
            if embed_list is not None:
                embedding_str = "[" + ",".join(map(str, embed_list)) + "]"

        await conn.execute(
            f"""
            INSERT INTO {table_name}
                (id, session_id, timestamp, sensory_input, inner_monologue,
                 episodic_embedding, importance_score, is_core_memory,
                 is_subjective, access_count, last_accessed_at)
            VALUES (COALESCE($1, gen_random_uuid()), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11);
            """,
            rec_id,
            session_id,
            timestamp,
            record.get("sensory_input", ""),
            record.get("inner_monologue"),
            embedding_str,
            record.get("importance_score", 5),
            record.get("is_core_memory", False),
            record.get("is_subjective", True),
            record.get("access_count", 1),
            last_accessed,
        )


def _embedding_to_list(embedding: Any) -> Optional[List[float]]:
    """Convert an embedding (numpy array / bytes / list) to a plain Python list."""
    if embedding is None:
        return None
    # numpy arrays have __iter__
    try:
        return [float(v) for v in embedding]
    except (TypeError, ValueError):
        return None
