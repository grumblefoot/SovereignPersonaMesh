# Deliberate spec deviations

Deviations from the PRD/SRD that the owner approved, each with the decision that
authorised it. Spec amendments reference these ids.

## SD-01 — Optional LLM-assisted world actions (GM_ACTION)

**Approved:** decisions 13 (sprint plan) and D-A1/D-A2 (gm_actions_and_lore_scope.md),
2026-10-04. **Status:** live since Sprint 3 (`gm_actions_mode`: `off | move_only |
full`, default `full`; decision 14 presets `off` for paid APIs).

When on, the LLM proposes room creation and movement inside its `<think>` block.
The proposals reach the world only through the deterministic validation layer
(`proxy/core/gm_actions.py`): strict schema-validated JSON (discriminated union,
`extra="forbid"`), slug normalisation, name/desc sanitisation, semantic checks
against a world snapshot, CREATE-then-MOVE ordering, per-turn and per-session caps,
and sha256 idempotency keys stable across regenerates. Routing, gating and tag
parsing stay deterministic, so the Zero-LLM rule holds for the core path.

**The deviation from PRD 11.1.1** is that rooms may be created outside templates.
OPEN-005 is closed as "kept, constrained, optional".

## SD-02 — Per-model similarity threshold and native embedding dimensions

**Approved:** decision 6, 2026-10-04. The SRD's fixed cosine threshold (0.35) and
fixed 3584-dim vectors are replaced by: dimension-free vector storage locked to the
model's native size on first embed (`embedding_space_id` per provider/model/dim),
and a configurable `EMBEDDING_MAX_COSINE_DISTANCE` calibrated per model via
`scripts/embed_bakeoff.py` (0.45 for Qwen3-Embedding-0.6B; see
docs/plans/BAKEOFF_2026-10-05.md). HNSW indexes are optional, not assumed.

## SD-03 — Token budget split and refusal behaviour

**Approved:** decision 16 (split) and its Sprint-4 sub-decisions taken per plan
recommendation, 2026-10-05. The PRD 5.3 percent split governs (the SRD's split
leaves no output reserve); history is the elastic remainder with a 1024-token
floor. A character card that alone exceeds the window REFUSES the turn with a
visible notice — the card is never silently truncated. Rolling summaries (plan
P4) are off; trimmed turns are simply omitted. Hardware tiers remain as preset
aliases supplying the default window.
