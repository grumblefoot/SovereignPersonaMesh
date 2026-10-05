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
