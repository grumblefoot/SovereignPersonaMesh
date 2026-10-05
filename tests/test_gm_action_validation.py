"""GM_ACTION validation layer (gm_actions_and_lore_scope.md A4): pure unit tests.

The LLM proposes; this deterministic layer decides. Every rejection carries a
reason code, accepted batches come out CREATE-then-MOVE, and idempotency keys are
stable across regenerates of the same turn.
"""
import pytest

from proxy.core.gm_actions import (
    REASON_CAP, REASON_MODE, REASON_SCHEMA, REASON_UNKNOWN_ENTITY, REASON_UNKNOWN_ROOM,
    CreateRoomAction, MoveAction, idempotency_key, sanitize_text, slugify, validate_batch,
)

WORLD = dict(known_rooms={"cellar", "garden"}, known_entities={"mira"}, target_char="mira")


# ── slugs and sanitisation ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Wine Cellar", "wine_cellar"),
    ("wine-cellar", "wine_cellar"),
    ("  Grand   Hall  ", "grand_hall"),
    ("Café-Lounge", "cafe_lounge"),
    ("x" * 100, "x" * 48),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_sanitize_strips_tags_controls_and_nested_gm():
    dirty = "A <script>bad</script> room\x00 with [GM_ACTION: {}] inside"
    assert sanitize_text(dirty, 60) == "A bad room with  inside"


# ── schema ──────────────────────────────────────────────────────────────────

def test_unknown_type_and_extra_fields_rejected_with_schema():
    out = validate_batch([
        {"type": "DELETE_ROOM", "room_id": "cellar"},
        {"type": "MOVE", "entity": "mira", "room_id": "garden", "sneaky": True},
        "not even a dict",
    ], **WORLD)
    assert not out.accepted
    assert [r.reason for r in out.rejections] == [REASON_SCHEMA] * 3


def test_invalid_slug_rejected():
    out = validate_batch([{"type": "MOVE", "entity": "mira", "room_id": "!!!"}], **WORLD)
    assert [r.reason for r in out.rejections] == [REASON_SCHEMA]


# ── semantics ───────────────────────────────────────────────────────────────

def test_move_requires_known_entity_and_room():
    out = validate_batch([
        {"type": "MOVE", "entity": "stranger", "room_id": "cellar"},
        {"type": "MOVE", "entity": "mira", "room_id": "atlantis"},
        {"type": "MOVE", "entity": "USER", "room_id": "garden"},   # case-insensitive
    ], **WORLD)
    assert [r.reason for r in out.rejections] == [REASON_UNKNOWN_ENTITY, REASON_UNKNOWN_ROOM]
    assert [m.entity for m in out.moves] == ["USER"]


def test_create_then_move_into_new_room_in_one_batch():
    out = validate_batch([
        {"type": "MOVE", "entity": "mira", "room_id": "attic"},    # listed FIRST
        {"type": "CREATE_ROOM", "room_id": "Attic", "name": "The Attic", "desc": "Dusty."},
    ], **WORLD)
    assert not out.rejections
    assert [a.type for a in out.accepted] == ["CREATE_ROOM", "MOVE"]  # reordered
    assert out.moves[0].room_id == "attic"


def test_create_of_existing_room_is_a_noop_rejection():
    out = validate_batch([{"type": "CREATE_ROOM", "room_id": "cellar"}], **WORLD)
    assert not out.accepted
    assert out.rejections[0].reason == REASON_SCHEMA
    assert out.rejections[0].detail == "exists"


# ── caps ────────────────────────────────────────────────────────────────────

def test_per_turn_cap_counts_accepted_actions():
    acts = [{"type": "MOVE", "entity": "mira", "room_id": "garden"} for _ in range(6)]
    out = validate_batch(acts, max_per_turn=4, **WORLD)
    assert len(out.accepted) == 4
    assert [r.reason for r in out.rejections] == [REASON_CAP, REASON_CAP]


def test_room_cap_per_session():
    acts = [{"type": "CREATE_ROOM", "room_id": f"r{i}"} for i in range(3)]
    out = validate_batch(acts, rooms_in_session=39, max_rooms_per_session=40, **WORLD)
    assert len(out.creates) == 1
    assert [r.reason for r in out.rejections] == [REASON_CAP, REASON_CAP]


# ── modes ───────────────────────────────────────────────────────────────────

def test_mode_off_rejects_everything():
    out = validate_batch([
        {"type": "MOVE", "entity": "mira", "room_id": "garden"},
        {"type": "CREATE_ROOM", "room_id": "attic"},
    ], mode="off", **WORLD)
    assert not out.accepted
    assert {r.reason for r in out.rejections} == {REASON_MODE}


def test_mode_move_only_rejects_create_keeps_move():
    out = validate_batch([
        {"type": "MOVE", "entity": "mira", "room_id": "garden"},
        {"type": "CREATE_ROOM", "room_id": "attic"},
    ], mode="move_only", **WORLD)
    assert [m.room_id for m in out.moves] == ["garden"]
    assert [r.reason for r in out.rejections] == [REASON_MODE]


# ── idempotency ─────────────────────────────────────────────────────────────

def test_idempotency_key_stable_across_regenerates():
    a = MoveAction(type="MOVE", entity="mira", room_id="garden")
    b = MoveAction(type="MOVE", entity="mira", room_id="garden")
    assert idempotency_key("s1", "anchor", a) == idempotency_key("s1", "anchor", b)
    assert idempotency_key("s1", "anchor", a) != idempotency_key("s1", "other-turn", a)
    assert idempotency_key("s2", "anchor", a) != idempotency_key("s1", "anchor", a)
    c = CreateRoomAction(type="CREATE_ROOM", room_id="garden")
    assert idempotency_key("s1", "anchor", a) != idempotency_key("s1", "anchor", c)
