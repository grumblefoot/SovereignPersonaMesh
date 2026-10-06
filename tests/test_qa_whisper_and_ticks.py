"""QA 2026-10-05: whisper bystanders (F23) and persisted turn ticks (F20)."""
import asyncio

import pytest

from evennia_world.spatial_matrix import SpatialConstraintsMatrix
from evennia_world.models import ActionType, GatingLevel


def _whisper(recipient, is_target, distance=3.0, raw='"It was a nonsense word"'):
    return SpatialConstraintsMatrix.evaluate_sensory_feed(
        distance_ft=distance, barriers=[], action_type=ActionType.WHISPER, raw_text=raw,
        actor_id="user", recipient_id=recipient, is_target=is_target,
        session_id="s", action_tick=1, shout=False)


def test_same_room_bystander_notices_a_murmur_not_the_words():
    gating, feed = _whisper("lian", is_target=False)
    assert gating == GatingLevel.DEGRADED          # was BLACKOUT: "out of earshot" in the same room
    assert "murmur" in feed and "nonsense" not in feed


def test_whisper_target_gets_the_words_without_double_quotes():
    gating, feed = _whisper("mei", is_target=True,
                            raw='*leans forward* "It was a nonsense word"')
    assert gating == GatingLevel.DIRECT
    assert feed == 'User whispers to you: *leans forward* "It was a nonsense word"'


def test_far_bystander_still_hears_nothing():
    gating, _ = _whisper("someone", is_target=False, distance=60.0)
    assert gating == GatingLevel.BLACKOUT


@pytest.mark.asyncio
async def test_new_turn_ticks_are_persisted(monkeypatch):
    import evennia_world.app as world_app
    writes = []

    class Conn:
        async def execute(self, sql, *args):
            writes.append((sql, args))

    class Pool:
        def acquire(self):
            class Ctx:
                async def __aenter__(self_inner): return Conn()
                async def __aexit__(self_inner, *a): return False
            return Ctx()

    monkeypatch.setattr(world_app.app_state, "_db_pool", Pool())
    monkeypatch.setattr(world_app.app_state, "session_turn_ticks", {})
    monkeypatch.setattr(world_app.app_state, "session_ticks", {})
    tick, advanced = world_app._tick_for_turn("sess_t", "sess_t:1")
    again, advanced2 = world_app._tick_for_turn("sess_t", "sess_t:1")    # same turn: no new write
    await asyncio.sleep(0.01)
    assert (tick, advanced, again, advanced2) == (1, True, 1, False)
    assert [a for _, a in writes] == [("sess_t", "sess_t:1", 1)]
