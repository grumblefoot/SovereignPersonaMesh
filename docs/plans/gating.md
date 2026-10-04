# Plan: make sensory gating real (OPEN-003 + OPEN-006)

Status: planning only. Written 2026-10-03. Priority P1, and the product's reason to exist.

## Goal

A character's prompt contains only what that character could have perceived. Whispers reach only their target, speech through a closed door does not arrive, and a character in another room comes back without knowing what happened. All of this is decided in deterministic Python (SRD 3.1, Zero-LLM core rule). It must work on any single backend (Lemonade, KoboldCpp, OpenAI) with an unmodified SillyTavern (ST), and whether GM_ACTION is on or off. Success is measured by a canary-leak scenario suite that must report 0 leaks, plus a latency budget.

## Current state (evidence)

**The proxy never describes the action.**
- Every turn is sent as actor `"user"` with action `"speak"` (`proxy/api/routes.py:262-268`).
- The consequence for the target is picked loosely (`:274-279`). With an empty world this falls back to `direct`.
- Session identity falls back to `st_{user}_{target_char}` (`routes.py:154-157`). In a group chat each member therefore gets a **separate world**, which makes cross-character gating impossible as built.

**The prompt is ungated.**
- `build_csa_messages` appends the last 15 raw ST messages (`proxy/rag/prompt_builder.py:145`, also `:67`). `routes.py:350` passes `request.messages` unchanged.
- Memory rows store the raw `user_text` as `sensory_input` (`routes.py:449-459`), not what the character perceived (PRD 5.1: "what Luna physically witnessed").

**The bypass is mostly dead.**
- `ObserverInferenceGatingFilter` is imported (`routes.py:22`) and never called (`proxy/core/sensory_filter.py`).
- The engine drops BLACKOUT recipients (`evennia_world/app.py:180`), so the proxy's bypass branch (`routes.py:290-306`) only fires by accident.

**The engine (`evennia_world/app.py`) has these problems:**
- A hard-coded "upstairs/tavern" hack fabricates DEGRADED feeds (`:189-203`).
- Adjacency reads the legacy global `app_state.current_world` (`:612`). It hard-codes abstract room ids (`:614`). Same-room distance is a constant 3 ft.
- **New bug:** the local `CharacterMovePayload` (`:332-336`) has no `session_id`, and `hasattr` checks (`:370, :374, :447`) fall back to `default_session`. Every GM `MOVE` lands in `default_session`, not the chat's session.
- `move_character` prefers the global world over the session world (`:473-479`).
- `add_character` mutates the shared template (`:364`).
- `configure_world` uses the template key as the session id (`:516`).
- `_compute_all_distances` ignores `template_key` (`:625`).
- Session worlds are not reloaded at startup (`:672`).
- The templates in `evennia_world/res/strings.json` ship with demo occupants (rowan, domino, luna, seamus).
- The client hard-codes `template_key: "dynamic"` (`proxy/backend_client/evennia_client.py:64,80`).

**The spatial matrix** (`evennia_world/spatial_matrix.py:43-50`):
- DEGRADED extends to 20 ft; the spec says 15 (SRD 3.3.2).
- Only `SOLID_WALL` blacks out. `CLOSED_DOOR` and `METAL_PARTITION` should too (SRD 3.3.2, PRD 4.2).
- There is no shout tier.
- Hysteresis can smooth DIRECT↔BLACKOUT into DEGRADED for 2 ticks.

**The clock.** The tick is global, seeded at 1420 (`app.py:42`), and bumped on every `/action` (`:127`), including regenerations. It is shared across sessions, contrary to the playbook's user-message clock (OPEN-009). The session lock (`session_lock.py`) is never used by the proxy.

**Other leak channels outside the prompt:**
- Lore extraction receives the full raw transcript (`routes.py:720`).
- Bulk import writes the raw transcript into the target's memory (`proxy/rag/import_worker.py`).

**What ST actually sends.** See `tests/fixtures/test_chat_payload.json` and ST `public/scripts/openai.js:595-610`.
- User and assistant messages carry no `name` in solo chats.
- Conventions in use: `*action*`, `"speech"`, `'thought'`, and `**OOC directive**`.
- In groups, `names_behavior=DEFAULT` prefixes other speakers' content with `Name: `, `COMPLETION` sets `name`, and the group nudge `[Write the next reply only as {{char}}.]` names the target.
- ST sends no chat id, swipe id or generation type, but custom headers support macros (`{{char}}`, `{{group}}`, `{{lastMessageId}}`, `{{currentSwipeId}}`).
- The fixture's last turn ("Vardus enters the room alone, listening at the door as Arvenia's footsteps move away") is a ready-made gating scenario.

## Design

### 1. Split of responsibilities and the engine interface

Gating must not depend on Evennia vs the FastAPI stand-in. The engine is the **authority on world state**. The proxy computes **perception** with one shared pure function.

- `core/gating/matrix.py` holds the pure function:
  `perceive(event, recipient, topology) -> Perception(level, feed, distance_ft, barriers)`
  - `level` is one of `direct | degraded | blackout | null`. `null` means the recipient is not in the scene.
  - The engine's `/world/action` imports the same module, so its consequences stay spec-compatible (PRD 3.2.3).
- `proxy/world/engine.py` defines a `WorldEngine` protocol:
  - `resolve_session(session_id)`
  - `snapshot(session_id) -> Topology`: rooms, edges `{a, b, barrier, state, distance_ft}`, and occupants with optional `posture` flags such as `listening` or `asleep`
  - `apply(session_id, ops: list[WorldOp])` with `PLACE | MOVE | CREATE_ROOM | SET_BARRIER`, which is idempotent per `op_id`
  - `clock(session_id)` and `set_clock(session_id, turn_index)`
  - `HttpEngine` (stand-in on :4005) is the first adapter. An Evennia adapter must pass the same **contract test suite**.
- New endpoint `GET /api/v1/world/snapshot?session_id=`. Any engine can implement it.
- Gating a history of N messages needs **one** snapshot call per request, not N action calls.

### 2. Session identity and the user-message clock

**Session key** (fixes group chats). Resolve in this order:
1. `X-Session-ID` header. The ST setup doc gets a recipe for custom headers.
2. A **chat fingerprint**: a hash of the persona name plus the first non-system message.
3. When ST context trimming has dropped the first message, the earliest message hash already known to the ledger (section 4) maps back to its session.

The key never includes the target character, so all group members share one world.

**Clock.** `turn_index` is the absolute count of user-role chat messages. It is anchored through the ledger, so it survives ST truncation. It is per session and persisted, and it replaces the global counter. The engine's tick equals `turn_index`, and hysteresis (if kept) counts in turns.

**Regenerations, swipes, continue and edits:**
- Regenerate or swipe: the history up to the last user message matches the previous request's hashes. Result: same `turn_index`, no new world ops, and the memory row is replaced by key `(session, turn_index, target)` rather than today's `LOWER(sensory_input)` match.
- Continue: the last message is an assistant message by the target. No new event.
- Edit of an older message: the hash is unknown, so a new ledger entry is evaluated against the **position snapshot at that turn**. The orphaned entry is garbage-collected after N days.
- Moves stated in an edited message are not replayed into the current world; this is logged.

### 3. Deterministic actor and action detection (`proxy/core/action_parser.py`)

**Actor:**
- User messages: the persona name, taken from ST's main prompt `between A and B` (the regex already exists at `routes.py:683`), with a fallback to `user`.
- Assistant messages: `name` field, then a `Name: ` prefix, then the target in solo chats.
- ST narrator or `/sys` messages: actor `narrator`.

**Target extraction:** reorder to group nudge, then "Write X's next reply", then card patterns.

**Segments.** Each message is split into ordered segments:
- `"speech"` → SPEAK
- `*action*` → EMOTE (visual)
- `'thought'`, `> thought` and `<think>` → THOUGHT (actor only)
- `**…**`, `((…))` and `[OOC: …]` → OOC: passed to the target as an instruction, never a world event

The thought convention is configurable, because single quotes are ambiguous.

**Verb lexicon.** Applied to the surrounding action text:
- WHISPER: whisper, murmur, under her breath, leans in and says quietly
- SHOUT: shout, yell, scream, bellow, calls out; or all caps with `!`
- MOVE: enters, goes to/into, heads up/down the stairs, leaves, steps out, returns to; the destination is matched against room names, aliases and exits of the actor's room
- Target: "to X", "in X's ear", a vocative "X," at the start of a quote, or the request target

**Explicit tags** take precedence and are stripped before the LLM sees the text: `[whisper:Domino]`, `[shout]`, `[move:kitchen]`, `[move:Arvenia->hall]`. They are cheap to add with ST Quick Replies.

One message can produce several events in order, for example MOVE then SPEAK. The parser also runs on **assistant** text: "she leaves the room" moves the character. This is Python, so it is allowed under the Zero-LLM rule.

### 4. Perception ledger and gated history (the core)

**Table `spm_turn_ledger`:**
- Columns: `session_id, msg_hash, ordinal, turn_index, role, actor_id, events JSONB, actor_room, perceptions JSONB {character_id: {level, feed, segments_visible}}, source (live|inferred|spm_bypass), created_at`.
- Unique on `(session_id, msg_hash, ordinal)`.

It is computed **once per message** for every character known in the session. That is O(characters) per message and CPU only. It doubles as the cache.

**Per request:**
1. Normalise and hash the ST history messages (strip `Name: ` prefixes and whitespace).
2. Run one indexed `SELECT … WHERE msg_hash = ANY($2)`.
3. Evaluate only the misses against the turn's snapshot. Pre-existing or imported history gets `source=inferred`, using current positions as a best effort.
4. Keep a per-session in-process LRU of the last result so consecutive turns usually hit memory.

Target: p95 under 30 ms for a 300-message history (the SRD proxy cap is 150 ms total).

**`GatedHistoryBuilder.build(target, history)` rewrites per message:**

| Perception | What the target sees |
|---|---|
| Target's own assistant message | Kept verbatim |
| direct | Kept, minus segments the target could not get: other actors' THOUGHTs; whispers to someone else become `*X murmurs something to Y.*` |
| degraded | The deterministic feed ("You hear muffled voices from the hall."). Runs of consecutive degraded turns collapse into one line |
| blackout / null | Dropped. One gap marker per run (`[Time passes while you are elsewhere.]`), optional |

- Other characters' assistant messages are re-roled as `user` with a `Name: ` prefix, so the model does not think it wrote them.
- The current turn's events become the `Sensory Feed` block.
- `prompt_builder.build_csa_messages` takes the gated list (each item carries `turn_index` and `level`) instead of slicing raw history. Trimming is the token-budget plan's job.
- Memory rows store the target's **perceived** feed as `sensory_input`.

**Omniscient ST blocks** (Summary extension, Author's Note) are free text that cannot be redacted deterministically. Policy is configurable: `pass` or `strip_when_gating`. This is an owner decision. World Info lore passes through.

### 5. Spatial matrix fixes (OPEN-006)

**Speech** (SPEAK):
- Same room and ≤5 ft: direct.
- 5–15 ft or through an open door or drywall: degraded.
- Over 15 ft, or through a closed door, solid wall or metal partition: blackout.

**WHISPER:** only the target gets the transcript, and only within 5 ft. Others within 15 ft in the same room get "murmur". This satisfies PRD user story 1 without in-room coordinates.

**SHOUT:** one tier louder, so it can be degraded through a closed door or drywall. It never passes a solid wall or metal partition (PRD 4.2 "Any | Solid wall | Metal partition → blackout").

**EMOTE:** needs line of sight, meaning the same room and lighting not `dark`. Any closed barrier blocks it.

**Edges:**
- Exits become edge objects carrying barrier, open/closed state and distance.
- Adjacency reads the **session** topology.
- Blackout recipients are **returned**, not dropped.
- The upstairs hack, the hard-coded abstract room ids and the template demo occupants are deleted.
- The `MOVE` session bug and the template mutation are fixed.

**Activation hooks** (PRD 4.1.1):
- `listening` posture: a character "listening at the door" gets one tier up through a closed door.
- High-priority event keywords (explosion, scream) raise the tier.
- Both live in a small deterministic registry.

### 6. World seeding

**With GM_ACTION off** (the default if that toggle lands), a deterministic order:
1. Explicit tag in World Info or the card: `[SPM_WORLD: tavern_common] [SPM_START: tavern_common_room]`, or a choice in the admin UI.
2. `HybridWorldBuilder.match_template_fuzzy` over the card, scenario and greeting (PRD 11.1.1). The PRD's LLM extraction pass is deliberately skipped (Zero-LLM).
3. Fallback `default_meeting_room`, with everyone in one room. This is honest "no gating" and is shown in telemetry.

The user and all characters start in the start room, and new group members join the room of the actor who introduced them. A MOVE to an unknown destination creates a deterministic **`elsewhere:<slug>` room** joined to the origin by a closed door. Leaving the scene therefore gates correctly even with no topology.

**With GM_ACTION on**, `CREATE_ROOM` and `MOVE` become `WorldOp` proposals:
- They are schema-validated. `CREATE_ROOM` must name a connecting room and a barrier, defaulting to a closed door.
- They apply at the next turn boundary.
- On conflict, the parser wins for the user's own movements, and GM proposals win for NPC movements (owner decision).

Gating only ever reads engine state, so it is identical in both modes.

### 7. Bypass and ambient path

If the target's perception of the triggering event is blackout or null and no hook fires, ST still expects a reply. `blackout_policy` options:
- `status` (default): zero inference. The proxy streams a short italic line, as today, and calls `ObserverInferenceGatingFilter` to commit an ambient row. The bypass reply's hash is registered in the ledger as `source=spm_bypass`, so it never enters anyone else's history.
- `room_bound`: an LLM call on the gated prompt with "nothing reaches you; continue what you were doing" (PRD 11.3, turn 3). This goes through the FIFO queue.

Non-target characters who are present get ledger perceptions every turn. A **degraded** perception also commits an ambient memory row, with no inference; blackout commits nothing.

## Phases

Total sizing is about 18–25 focused days. Build the scenario harness first, mark its scenarios xfail, and flip them to passing as each phase lands.

**Phase 0: Matrix and engine correctness (S, 2 days).**
- *Scope:* section 5 changes, the per-session persisted tick, world reload at startup, and the snapshot endpoint.
- *Acceptance:*
  - The SRD/PRD table holds for every barrier and distance class.
  - A GM `MOVE` lands in the request's session.
  - Two sessions never share rooms or ticks.
  - Blackout consequences are returned.
- *Tests:*
  - Table-driven `test_spatial_matrix.py`. The current 20 ft test changes.
  - Engine tests through `httpx.ASGITransport` (in-process, no :4005).
  - A regression test for the `CharacterMovePayload` session bug.

**Phase 1: Interface, session and clock (M, 3–4 days).**
- *Scope:* the `WorldEngine` protocol, `HttpEngine`, the contract suite, session resolution, turn_index, and regeneration and continue detection. Also the scenario harness: YAML scenario → scripted ST payloads → assertions on the built prompt.
- *Acceptance:*
  - Three group members resolve to one session.
  - Truncating the oldest 20 messages keeps the session and the absolute turn_index.
  - Three regenerations leave the clock, the ledger and the memory rows unchanged.

**Phase 2: Action parser and seeding (M, 3–5 days).**
- *Scope:* section 3 and section 6 (GM-off path, `elsewhere` rooms).
- *Acceptance:* a labelled corpus of 150 or more segments, drawn from the fixture and synthetic group chats, reaches ≥95% correct action type and 100% correct explicit-tag parsing. No network or LLM is involved.
- *Tests:* parser unit tests, names_behavior modes 0/1/2, and template matching on the fixture card (expect `dungeon_cellar`).

**Phase 3: Ledger and gated history (L, 5–7 days).**
- *Scope:* migration (`scripts/init_db.sql`), the builder, prompt_builder integration, and perceived `sensory_input`.
- *Acceptance:*
  - Canary suite at 0 leaks across gated history, the sensory feed and the memory rows written.
  - p95 gating overhead under 30 ms at 300 messages (perf test against `spm_test`).
  - Unchanged behaviour in single-room sessions (golden prompt test).

**Phase 4: Bypass and hooks (S–M, 2–3 days).**
- *Scope:* both blackout policies, ambient commits and hook registry.
- *Acceptance:*
  - A blacked-out target in `status` mode makes 0 backend calls (asserted with a fake LLM client).
  - Exactly one ambient row is written.
  - The bypass message never appears in another character's prompt.

**Phase 5: Close the side channels and observability (M, 3 days).**
- *Scope:* lore extraction and bulk import consume gated or ledger-inferred input. Admin "perception view" (what X saw at turn N) and per-level counters in telemetry.
- *Acceptance:* the canary suite extends to lore-extractor input and imported memory rows, still at 0 leaks.

**Scenario suite** (all on `spm_test`, fake LLM, in-process engine):

| # | Scenario | Expected |
|---|---|---|
| S1 | Whisper in the same room | Bystander gets "murmur", not the canary |
| S2 | Whisper or speech across a closed door | Blackout, bypass |
| S3 | Speech through an open door at 10 ft | Degraded feed only |
| S4 | Shout through a closed door, then through a metal partition | Degraded, then blackout |
| S5 | Seamus upstairs during the cellar talk, then brought back | Only a gap marker |
| S6 | User moves mid-chat | Earlier speech stays visible to those present then |
| S7 | Regenerate or swipe ×3 | No changes (see Phase 1) |
| S8 | Edit an old message | Recomputed with that turn's positions |
| S9 | 3-member group, one request per speaker | Different per-speaker prompts, one session |
| S10 | GM off, unknown destination | `elsewhere` room |
| S11 | The captured fixture: Vardus alone behind the closed door | Arvenia's next prompt lacks his in-room actions |
| S12 | ST context truncation | Session and clock stable |
| S13 | Thoughts | Never reach others |
| S14 | Summary-block policy | Behaves as configured |

An opt-in live check (`RUN_LIVE_LLM_TESTS=1`) asks the real model about the canary. It is informational only, because the deterministic suite is the gate.

## Risks and open questions for the owner

1. **Blackout policy:** `status` line (zero cost, visible in ST) or `room_bound` LLM reaction (costs a turn)? Recommendation: `status` default, `room_bound` opt-in.
2. **Hysteresis:** remove it? It lets a character who just left still hear for 2 turns, which is a leak by design. Recommendation: remove.
3. **Fallback world** when nothing matches and GM is off: a single room (gating effectively off, flagged) or an `elsewhere`-only lazy topology? Recommendation: the single room plus a dashboard warning.
4. **Explicit tag syntax** (`[whisper:X]`, `[move:Y]`): acceptable? Should SPM ship ST Quick Replies for them?
5. **Omniscient ST blocks** (Summary extension, Author's Note): pass through or strip while gating is active?
6. **Session identity:** document the `X-Session-ID` header recipe as recommended, with the fingerprint as fallback?
7. **Spec deviations to confirm:**
   - same-room whisper bystanders get "murmur" at any distance up to 15 ft
   - drywall and open doors degrade
   - shout is one tier louder
   - emotes need line of sight
8. **Ambient memory rows for degraded non-targets:** these add embedding volume. Accept?
9. **Thought convention:** are `'single quotes'` always thoughts in your chats?
10. **Risks:**
    - Parser misclassification on free prose. Mitigated by explicit tags and a confidence fallback: an ambiguous verb is treated as SPEAK, which equals today's behaviour.
    - Inferred perceptions for imported history are only as good as current positions.
    - Edited old messages do not rewrite world history.

## Dependencies on sibling workstreams

- **Engine choice:** gating needs only the `WorldEngine` protocol plus `snapshot` and `apply`. Evennia must pass the same contract suite. The pure matrix lives in `core/gating`, outside either engine.
- **GM_ACTION toggle and session lore:**
  - GM output becomes validated `WorldOp`s and never perception.
  - `CREATE_ROOM` must carry a connection and a barrier.
  - Lore extraction must take the target's gated history (leak at `routes.py:720`) and reuse this plan's session resolver.
- **Token budget:**
  - Gating runs first and hands over an ordered list with `turn_index` and `level`.
  - Gap markers and collapsed degraded lines are small and droppable.
  - Any "summarize trimmed history" must be built **per target from gated history**, or it re-introduces omniscience.
- **Embeddings:** `sensory_input` becomes the perceived feed, and ambient rows add embed volume. Embedding must accept rows that have no LLM turn.
- **FIFO queue:** gating and bypass run before the queue and never enqueue. `room_bound` responses and lore enqueue as usual.
