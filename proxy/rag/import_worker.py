"""
FR-002: Async Bulk Chat Import Worker with Dynamic Resource Allocation.

Detects chat sessions with > 10 historical messages, registers them in
spm_chat_imports, and processes message pairs in dynamic batches via a
background asyncio task. Batch size adapts to hardware tier, CPU cores,
and available memory.

Zero-LLM Core Rule: vectorization, batch sizing, and routing are all
pure Python — no LLM calls during import.
"""

import asyncio
import os
import logging
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.hardware_tiers import get_hardware_config, HardwareTierEnum
from scripts.onnx_embedder import CPUEmbeddingEngine

logger = logging.getLogger(__name__)

# Minimum / Maximum batch sizes for import processing
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 32

# Bulk import detection threshold (> 10 messages triggers import)
BULK_IMPORT_THRESHOLD = 10


def _detect_hardware_tier() -> HardwareTierEnum:
    """Auto-detect hardware tier from /proc on Linux."""
    try:
        total_mem_kb = 0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mem_kb = int(line.split()[1])
                    break
    except (FileNotFoundError, ValueError):
        total_mem_kb = 0

    total_mem_gb = total_mem_kb / (1024 * 1024)

    # Strix Halo 128GB GTT -> SOVEREIGN
    if total_mem_gb >= 64:
        return HardwareTierEnum.SOVEREIGN
    # >= 16GB -> EXPERIMENTAL minimum
    elif total_mem_gb >= 16:
        return HardwareTierEnum.PERFORMANCE
    else:
        return HardwareTierEnum.EXPERIMENTAL


def _compute_dynamic_batch_size() -> int:
    """
    Compute batch size based on hardware tier, CPU cores, and memory.

    - SOVEREIGN tier with many cores -> large batches (up to MAX_BATCH_SIZE)
    - EXPERIMENTAL tier -> small batches to avoid OOM
    - Scales logarithmically with core count
    """
    tier = _detect_hardware_tier()
    config = get_hardware_config(tier.value)

    # CPU core count
    try:
        core_count = os.cpu_count() or 1
    except Exception:
        core_count = 1

    # Memory-based scaling: more RAM = larger batches
    try:
        total_mem_kb = 0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mem_kb = int(line.split()[1])
                    break
        total_mem_gb = total_mem_kb / (1024 * 1024)
    except (FileNotFoundError, ValueError):
        total_mem_gb = 8.0

    # Core scaling factor (logarithmic to avoid oversized batches on many-core systems)
    core_factor = min(core_count, 16) / 4.0  # 1.0 at 4 cores, max 4.0 at 16+

    # Memory scaling factor: 0.5x at 8GB, 1.0x at 32GB, 2.0x at 64GB+
    mem_factor = min(total_mem_gb / 32.0, 2.0)

    # Tier base multiplier
    tier_multiplier = {
        HardwareTierEnum.SOVEREIGN: 2.0,
        HardwareTierEnum.PERFORMANCE: 1.5,
        HardwareTierEnum.EXPERIMENTAL: 0.75,
    }.get(tier, 1.0)

    batch_size = int(config.top_k_memories * core_factor * mem_factor * tier_multiplier)
    return max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))


class BulkImportWorker:
    """
    Background worker that processes bulk chat imports.

    Uses dynamic batch sizing to optimise throughput without overloading
    the host CPU or exhausting memory.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.embedder = CPUEmbeddingEngine()
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def register_import_job(
        self,
        session_id: str,
        character_id: str,
        total_messages: int,
    ) -> str:
        """Register a new import job and return the import_id."""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO spm_chat_imports
                    (session_id, character_id, status, total_messages, processed_messages)
                VALUES ($1, $2, 'pending', $3, 0)
                RETURNING import_id;
                """,
                session_id,
                character_id,
                total_messages,
            )
            import_id = str(result["import_id"])
            logger.info(
                f"[ImportWorker] Registered import job {import_id}: "
                f"session={session_id}, character={character_id}, "
                f"messages={total_messages}"
            )
            return import_id

    async def update_import_status(
        self,
        import_id: str,
        status: str,
        processed_messages: Optional[int] = None,
        error_log: Optional[str] = None,
    ):
        """Update import job status in the database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE spm_chat_imports
                SET status = $1,
                    processed_messages = COALESCE($2, processed_messages),
                    error_log = COALESCE($3, error_log),
                    updated_at = NOW()
                WHERE import_id = $4::uuid;
                """,
                status,
                processed_messages,
                error_log,
                import_id,
            )

    async def process_bulk_import_background(
        self,
        session_id: str,
        character_id: str,
        messages: List[Dict[str, Any]],
        skip_registration: bool = False,
    ):
        """
        Process a bulk import asynchronously in the background.

        This function is meant to be spawned via asyncio.create_task().
        It:
          1. Registers the job in spm_chat_imports (unless skip_registration=True).
          2. Processes message pairs in dynamic batches.
          3. Vectorizes embeddings via the ONNX CPU embedder.
          4. Inserts memories into csa_memory_{character_id}.
          5. Updates status to 'completed' or 'failed'.
        """
        import_id = None
        batch_size = _compute_dynamic_batch_size()
        logger.info(
            f"[ImportWorker] Starting bulk import for session={session_id}, "
            f"character={character_id}, messages={len(messages)}, "
            f"batch_size={batch_size}"
        )

        try:
            # Step 1: Register job (skip if _check_bulk_import already registered)
            if not skip_registration:
                import_id = await self.register_import_job(
                    session_id, character_id, len(messages)
                )
            else:
                # Read the existing import_id from the DB
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchval(
                        "SELECT import_id FROM spm_chat_imports WHERE session_id = $1;",
                        session_id,
                    )
                    import_id = str(row) if row else None

            # Step 2: Ensure the character memory table exists
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "SELECT create_csa_memory_table($1);", character_id.lower()
                )

            # Step 3: Process in dynamic batches
            total = len(messages)
            error_parts = []

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = messages[start:end]
                logger.info(
                    f"[ImportWorker] Processing batch {start // batch_size + 1}: "
                    f"messages {start + 1}-{end}/{total}"
                )

                try:
                    # Vectorize embeddings in parallel (non-LLM, pure CPU)
                    texts = [
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in batch
                    ]
                    embeddings = await self.embedder.batch_generate_embeddings(texts)

                    # Insert into character memory table
                    async with self.db_pool.acquire() as conn:
                        await conn.execute(
                            "SELECT create_csa_memory_table($1);", character_id.lower()
                        )
                        for i, msg in enumerate(batch):
                            await conn.execute(
                                f"""
                                INSERT INTO csa_memory_{character_id.lower()}
                                    (session_id, sensory_input, episodic_embedding,
                                     inner_monologue, importance_score, is_core_memory,
                                     is_subjective, access_count)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
                                """,
                                session_id,
                                msg.get("content", ""),
                                "[" + ",".join(map(str, embeddings[i])) + "]",
                                msg.get("inner_monologue", None),
                                msg.get("importance_score", 5),
                                msg.get("is_core_memory", False),
                                True,
                                1,
                            )

                    # Update progress
                    processed = min(end, total)
                    await self.update_import_status(
                        import_id, "processing", processed_messages=processed
                    )

                except Exception as e:
                    error_parts.append(
                        f"Batch {start}-{end}: {type(e).__name__}: {e}"
                    )
                    logger.error(
                        f"[ImportWorker] Batch failed: {'; '.join(error_parts[-3:])}"
                    )
                    # Continue processing remaining batches

            # Step 4: Finalize
            if error_parts:
                final_status = "completed_with_errors"
                final_error = "; ".join(error_parts[:10])  # cap error log
                logger.warning(
                    f"[ImportWorker] Import {import_id} completed with errors: {final_error}"
                )
            else:
                final_status = "completed"
                final_error = None

            await self.update_import_status(
                import_id, final_status,
                processed_messages=total,
                error_log=final_error,
            )
            logger.info(
                f"[ImportWorker] Import {import_id} finalized: status={final_status}, "
                f"processed={total}/{total}"
            )

        except Exception as e:
            logger.error(
                f"[ImportWorker] Fatal error in import for session={session_id}: {e}"
            )
            if import_id:
                await self.update_import_status(
                    import_id, "failed", error_log=f"{type(e).__name__}: {e}"
                )
            else:
                # Couldn't register; write to error log directly
                logger.error(f"[ImportWorker] Could not register import for {session_id}")

    async def check_import_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Query the current status of an import job for a session."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT import_id, session_id, character_id, status,
                       total_messages, processed_messages, error_log,
                       created_at, updated_at
                FROM spm_chat_imports
                WHERE session_id = $1;
                """,
                session_id,
            )
            if row:
                return {
                    "import_id": str(row["import_id"]),
                    "session_id": row["session_id"],
                    "character_id": row["character_id"],
                    "status": row["status"],
                    "total_messages": int(row["total_messages"]),
                    "processed_messages": int(row["processed_messages"]),
                    "error_log": row["error_log"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
            return None

    async def get_all_imports(self) -> List[Dict[str, Any]]:
        """Return all import jobs, ordered by created_at descending."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT import_id, session_id, character_id, status,
                       total_messages, processed_messages, error_log,
                       created_at, updated_at
                FROM spm_chat_imports
                ORDER BY created_at DESC;
                """
            )
            result = []
            for r in rows:
                result.append({
                    "import_id": str(r["import_id"]),
                    "session_id": r["session_id"],
                    "character_id": r["character_id"],
                    "status": r["status"],
                    "total_messages": int(r["total_messages"]),
                    "processed_messages": int(r["processed_messages"]),
                    "error_log": r["error_log"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                })
            return result


# Module-level singleton
_import_worker: Optional[BulkImportWorker] = None


def get_import_worker(db_pool: asyncpg.Pool) -> BulkImportWorker:
    """Get or create the module-level BulkImportWorker singleton."""
    global _import_worker
    if _import_worker is None:
        _import_worker = BulkImportWorker(db_pool)
    elif _import_worker.db_pool is not db_pool:
        _import_worker = BulkImportWorker(db_pool)
    return _import_worker
