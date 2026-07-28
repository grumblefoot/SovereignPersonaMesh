"""
Asyncio FIFO Request Queue for SPM FastAPI Proxy.
Enforces strict 100% GPU safety margin by queuing multi-agent completion requests and dispatching sequentially.
"""

import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class InferenceFIFOQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._processing_lock = asyncio.Lock()

    async def enqueue_and_execute(self, task_fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Enqueues an inference request and awaits turn execution under lock.
        """
        async with self._processing_lock:
            logger.info(f"[FIFOQueue] Executing sequential inference turn (queue depth: {self._queue.qsize()})...")
            return await task_fn(*args, **kwargs)

    def size(self) -> int:
        return self._queue.qsize()
