"""Deterministic fake LLM backend for scheduler tests (docs/plans/fifo_queue.md P1).

Every stream blocks on test-controlled asyncio Events; no sleeps, no wall clock.
Records the concurrency high-water mark and the dispatch order of models."""
import asyncio
from typing import List


class FakeLLMBackend:
    def __init__(self):
        self.active = 0
        self.high_water = 0
        self.dispatched_models: List[str] = []
        self.started: List[str] = []           # job tags in start order
        self._gates = {}

    def gate(self, tag: str) -> asyncio.Event:
        return self._gates.setdefault(tag, asyncio.Event())

    async def generate_stream(self, prompt=None, model="m", tag=None, chunks=("a", "b"), **kw):
        tag = tag or model
        self.active += 1
        self.high_water = max(self.high_water, self.active)
        self.dispatched_models.append(model)
        self.started.append(tag)
        try:
            for c in chunks:
                await self.gate(tag).wait()      # the test opens the gate
                yield c
        finally:
            self.active -= 1
