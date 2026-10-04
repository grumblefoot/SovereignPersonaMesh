# Plan: World Engine (FastAPI stand-in vs. real Evennia)

Status: proposal for owner review, 2026-10-03. Planning only. No code has been changed.

## 1. Question

Should SPM keep and harden its FastAPI world engine (`evennia_world/`, port 4005), switch to the real Evennia framework (cloned at `../evennia`), or use a hybrid? The answer has to fit SPM's constraints:

- It runs on modest hardware: a 16 GB GPU, with the engine on the CPU.
- It is deterministic. The SRD's Zero-LLM rule covers gating and state.
- Each concurrent chat session gets its own isolated world.
- State persists and reloads.
- It is testable without live services.
- It gives the gating plan what it needs: a room graph, barriers, distances, line of sight and hearing, and ticks.

## 2. History: why the stand-in exists

There is no evidence that anyone ever tried to integrate Evennia and failed. The evidence shows Evennia was skipped from day one and that the stand-in was designed as the engine from the start.

| Date | Evidence |
|---|---|
| 2026-07-25 | The Hermes MVP (`~/Desktop/Experiments/Hermes/SPM_DEMO_SUMMARY.md:52`, `PLAYBOOK_SPM.md:45,302`) lists "Evennia World Engine — SKIPPED (in-memory for demo)" and "needs Evennia (per PRD §3.2) for production". The demo world was a Python dict (`spm-demo-mvp/world_state/spatial_matrix.py`). |
| 2026-07-27 21:39 | Initial stub commit `3a251cb` (Antigravity). `evennia_world/app.py` is already a FastAPI app titled "Evennia World State Engine **Liaison** API". It holds a hard-coded `current_world` and a global `action_tick_counter = 1420`, and its consequences are a mocked list (`cellar_chars = ["rowan","domino","luna"]`). It has no `import evennia`. |
| 2026-07-27 21:40 | `bc435c7` adds the Evennia clone to the playbook's workspace map, as reference only. |
| 2026-07-27 23:01 | `2d92f03` "Phase 2: Evennia World State Liaison Service" fleshes out the same FastAPI stub (REST endpoints, TTL locks, 4 templates, 54 tests). The playbook logs it as complete ("Liaison API"). |
| Sep 2026 | `a6b379a` "implement Evennia spatial integration" adds Postgres persistence (`world_state_sessions`, `objective_world_log`) to the FastAPI app. "Evennia" in commit titles means the stand-in. |
| 2026-10-03 | The state assessment lists "world engine is a FastAPI in-memory stand-in" as a deliberate deviation (§4). The README says "Evennia itself is not used". |

No commit message, known issue or doc records an Evennia install, an `evennia --init` game directory, a migration or a runtime error. No game directory exists under `~/Desktop/Experiments`: there is no `server/conf/settings.py` and no `evennia.db3`. Neither the SPM `.venv` nor the `spm-demo-mvp` venv contains Evennia, Django or Twisted. The PRD frames the API as a "Liaison Protocol" (§3.2.3) between FastAPI and Evennia. The agents built the FastAPI side of that liaison and the engine behind it never arrived.

**What would have made hooking Evennia hard** (inferred from the clone, Evennia 6.1.0, `a89a9b9`, 2026-08-19, BSD-3):

1. **SPM's API is not something Evennia offers.** The REST API (`evennia/web/api/urls.py`, `views.py`) is generic DRF CRUD over `objects/characters/rooms/exits/scripts`. It is off by default (`settings_default.py:1226 REST_API_ENABLED = False`) and requires authentication with "builder" permissions (`:1199-1222`). Nothing like `POST /world/action → gated consequences` exists. That logic would be custom Django views inside a game directory, so the hard part (gating) would be SPM code either way.
2. **Port collision.** Evennia's internal webserver port is 4005 (`settings_default.py:90 WEBSERVER_PORTS = [(4001, 4005)]`), the same port the PRD assigns the liaison. Evennia also uses 4000 (telnet), 4002 (websocket) and 4006 (AMP).
3. **Heavy runtime.** Evennia runs as two Twisted processes (Portal and Server) linked by AMP, started by `evennia start` from a game directory. That requires `evennia --init`, migrations and an interactive superuser prompt (`server/evennia_launcher.py:1467-1488`), and creates a Limbo root room (`server/initial_setup.py:57-85`). Django 6.0 and Twisted < 25 would be pinned next to FastAPI (`pyproject.toml:65-75`).
4. **One persistent world.** Evennia models a single shared world rooted at Limbo. SPM's FR-001 needs one world per chat session.
5. **Wall-clock time.** `utils/gametime.py:142` computes game time from runtime × `TIME_FACTOR`, and `scripts/tickerhandler.py:96-135` fires on Twisted intervals. SPM ticks are counts of user messages (playbook "Temporal Anchor").
6. **Room-scoped perception.** `DefaultObject.msg_contents`/`at_say` (`objects/objects.py:1051`, `:2859`) deliver to the contents of a room. There is no distance, acoustics or barrier model, so SRD 3.3.2 would be custom code anyway.
7. **Sync ORM with an in-process cache.** Django's ORM is synchronous and Evennia caches typeclass instances per process (`utils/idmapper/models.py:28-40`). SPM cannot safely write Evennia's database from its own async process. Every call would have to go through the running Server's thread-pooled WSGI (`server/webserver.py:27-172`).
8. **Different test model.** Evennia's tests use `evennia test --settings` and `EvenniaTest` with a Django test DB (`docs/source/Coding/Unit-Testing.md`). SPM's suite is pytest with a throwaway `spm_test` DB and is required never to touch live services.

## 3. Spec intent

- PRD §3.2.2 names Evennia the "Primary Candidate", chosen for its rooms, containers and exits and for a "headless-first" design. PRD §10 expects Django-ORM schemas in `/evennia_world`.
- The binding requirements are §3.2.1's six criteria: spatial awareness, low latency, scalability "without exponential CPU or memory overhead", agent-readable JSON, persistence in a local DB (e.g. PostgreSQL), and temporal ticks. Any engine meeting them satisfies the intent.
- SRD §2 lists the spatial engine as "Text-based spatial router (**SQLite or Evennia**)". SRD §1 adds "Lightweight Standalone Architecture … Mac, Pi, and mobile".
- SRD §5.3 fixes the contract, not the implementation: REST on 4005, `/world/action`, `/world/state`, and a session and tick lock.

The spec therefore allows a non-Evennia engine. Evennia's "headless-first" premise is not accurate: it is a MUD server with a web front end.

## 4. Options compared

| Criterion | (1) Harden FastAPI engine | (2) Real Evennia as engine | (3) Hybrid |
|---|---|---|---|
| Spec intent | Meets SRD §2/§5.3 and PRD §3.2.1. Deviates from the PRD's "primary candidate" (already accepted as a deviation). | Literal match to PRD §3.2.2. | 3a (borrow Evennia's data model): same as (1). 3b (Evennia only on the Sovereign tier): splits behaviour by tier. |
| Determinism | Full control: pure functions over a session state plus an ordered event log. | Possible only if tickers, `gametime` and `delay` are avoided. Hooks and contribs may use real time or random numbers. | 3a: as (1). 3b: two engines to keep in agreement. |
| Per-session isolation | Native: a dict keyed by session, rows keyed by `session_id`. | Every room, exit and character must carry a session Tag, every search must filter on it, and reset means deleting by tag. Cross-session leaks are possible through the shared cache and global scripts. | 3a: native. 3b: as (2). |
| Persistence and reload | Tables exist. Reload has to be added (≈2–3 days). | Built in (Django ORM, SQLite by default or Postgres). | 3a: as (1). |
| Footprint (low tier) | One uvicorn process at about 61 MB RSS (measured with `ps` today). Could also run in the proxy's process. | Two Twisted processes plus Django. Estimated several hundred MB in total. Four extra ports, migrations, a superuser and a separate DB. | 3a: as (1). 3b: as (2) on the large tier only. |
| SPM↔engine integration | Exists today: 4 call sites (`proxy/api/routes.py:262,753,760`, `evennia_client.py`). | Write a game directory, custom DRF views implementing the liaison contract, auth, and a port remap. The gating maths is rewritten as typeclass code. | 3b: two adapters. |
| Testability | Already pytest-native (54 liaison tests plus the matrix tests). Runs in-process with `TestClient`. | Needs the Django test runner or `django.setup()` in pytest with Evennia settings and a test DB. Heavy and slow. | 3a: as (1). |
| Maintenance | About 1.2 kLOC owned by SPM. | Follows Evennia releases (Django and Twisted pins, Python version caps). The team has no Evennia experience. | 3b: highest. |
| Gating plan needs | Must build: exits with barrier state, path distance, sight and hearing propagation, a per-session tick. All small and deterministic. | Gets rooms, exits, contents and Tags. Still has to build distance, acoustics, line of sight and ticks per message. Evennia's ticker does not help. | 3a: builds them, modelled on Evennia's concepts. |
| Effort | ≈12–17 dev-days to a hardened engine | ≈20–30 dev-days plus ongoing ops | 3a ≈ (1) + 1 day. 3b ≈ (1) + (2). |

### How Option 2 would work concretely

This is for reference, in case the owner wants a spike.

- **Game directory and settings.** Run `evennia --init spm_world`. Set `REST_API_ENABLED=True`, move `WEBSERVER_PORTS` off 4005, set `DATABASES` to a Postgres schema, and set `TELNET_ENABLED=False`.
- **Typeclasses.** `typeclasses/rooms.py` subclasses `DefaultRoom` (`objects/objects.py:3355`). Exits subclass `DefaultExit` (`:3522`) or `contrib/grid/simpledoor` for an open/closed door state. Characters subclass `DefaultCharacter` (`:3018`). Each object gets `tags.add(session_id, category="spm_session")`.
- **Per-session root.** Each session gets a root room and a `WorldScript` (`scripts/scripts.py`) that holds `db.tick` and `db.seed`.
- **Exposing state.** Add a custom view in `web/api/` (pattern in `docs/source/Howtos/Web-Extending-the-REST-API.md`): `POST /api/spm/action`. It looks up the session's WorldScript by tag, increments `db.tick` once per `turn_id`, walks `room.exits` to compute distances and barriers, applies the gating matrix, writes the log, and returns JSON. Do not use `TICKER_HANDLER.add(interval=...)` (`scripts/tickerhandler.py:489`), because it is wall-clock. `OnDemandHandler` (`scripts/ondemandhandler.py:376`) can be driven by a custom clock, but that adds nothing a counter does not already do.
- **Calls from SPM.** SPM calls the view over HTTP with a service account (Basic auth, `settings_default.py:1214`). A websocket on 4002 is unnecessary because SPM does not need pushed events.

All the SPM-specific logic still lives in that custom view. Evennia would provide only object storage and a graph of rooms, exits and contents, which is about 300 lines in the stand-in.

## 5. Recommendation

**Option 3a: keep and harden the FastAPI engine as SPM's own deterministic world engine, and adopt Evennia's data-model ideas without its runtime.**

Behind a stable engine contract (§6), so an Evennia adapter remains possible later.

Reasons:

1. **Evennia would not remove the work.** Gating, distances, acoustics, ticks per message, session isolation and the liaison API are all custom code under either option. Evennia would add a second runtime (Twisted, Django, two processes, AMP) and a second ORM while removing only the trivial part.
2. **Isolation and time.** Per-session worlds and message-count ticks are first-class in (1) and work against Evennia's design in (2).
3. **Hardware, tests and spec.** Low-tier footprint and pytest isolation clearly favour (1), and the SRD already allows "SQLite or Evennia".
4. **What to borrow from Evennia:**
   - Exits as first-class objects with their own state (`DefaultExit`, `simpledoor`).
   - Tags and Attributes on rooms and entities.
   - Containers (`location`/`contents`).
   - Optional XYZ coordinates (`contrib/grid/xyzgrid`) for distances within a room.
   - Lazily computed, tick-delta state (the `OnDemandHandler` idea) instead of background tickers.
5. **3b (Evennia on the Sovereign tier only)** is not recommended. It doubles the surface area, and behaviour would differ by tier, which hurts determinism and the test matrix.

## 6. Engine interface contract

Code SPM against a Python `Protocol` with two adapters:

- `HttpWorldEngine`: today's :4005 service.
- `InProcessWorldEngine`: the same engine imported directly. Use it for tests, and optionally on low tiers to save a process.

Any future Evennia adapter must pass the same contract test suite.

```python
class WorldEngine(Protocol):
    contract_version: str  # "1.0"
    async def ensure_world(self, session_id: str, seed: WorldSeed | None = None) -> WorldSnapshot
    async def advance_tick(self, session_id: str, turn_id: str) -> TickResult        # idempotent per turn_id
    async def submit_action(self, session_id: str, action: Action, turn_id: str) -> ActionResult
    async def perceive(self, session_id: str, observer_id: str, source_id: str,
                       channel: Literal["sound", "sight"]) -> Perception
    async def get_graph(self, session_id: str) -> WorldGraph
    async def get_entity_state(self, session_id: str, entity_id: str) -> EntityState
    async def apply_mutation(self, session_id: str, mutation: Mutation,
                             idempotency_key: str, origin: Literal["gm", "system", "user"]) -> MutationResult
    async def reset(self, session_id: str | None) -> None          # None = all, admin only
    async def health(self) -> EngineHealth
```

### Data model (Pydantic, JSON on the wire)

- `Room{id, name, desc, tags[], attrs{}, size_ft}`
- `Exit{id, from_room, to_room, name, barrier: open_door|closed_door|locked_door|drywall|solid_wall|metal_partition|none, distance_ft, two_way_pair_id?}`
- `Entity{id, kind: character|object, room_id, container_id?, pos_ft?: (x, y)}`
- `Action{actor_id, type: speak|whisper|shout|move|manipulate, target_id?, text, volume?}`
- `ActionResult{tick, consequences: [{recipient_id, gating, sensory_feed, distance_ft, barriers[], path[]}]}`. This is a superset of PRD §3.2.3.
- `Perception{gating: direct|degraded|blackout, distance_ft, barriers[], path[], line_of_sight: bool, audible: bool}`
- `Mutation` is a closed union: `create_room`, `link_rooms`, `set_exit_state`, `move_entity`, `place_entity`, `remove_entity`, `set_attr`. Each is schema-validated, with id format `[a-z0-9_]{1,48}` (reusing `core.identifiers`).

### Semantics

- **Determinism.** State is a pure function of `(seed, ordered mutation and action log)`. No wall clock and no unseeded randomness. Feeds come from string templates. Replaying the log produces byte-identical `get_graph`.
- **Ticks.** Each session has its own counter. `advance_tick` increments only for a new `turn_id`, which is the proxy's user-message index plus a hash. A regenerated message reuses its `turn_id` and does not advance. `submit_action` never advances the tick implicitly.
- **Isolation.** Every call is scoped by session, with no global "current world". Calls on an unknown session lazily load it from the DB.
- **Idempotency.** `apply_mutation` with a repeated key returns the original result. That makes GM_ACTION retries and duplicate parses safe.
- **Errors.** Typed errors: `WorldNotFound`, `InvalidMutation` (422), `Conflict` (409), `EngineUnavailable`. The proxy maps `EngineUnavailable` to a graceful degraded turn (OPEN-010).
- **HTTP mapping.** Keep `/api/v1/world/action` and `/api/v1/world/state` for SRD §5.3 compatibility. Add `/api/v1/world/{session}/tick|graph|perceive|mutations`. Require a shared-secret header and bind to 127.0.0.1.

## 7. Phases, acceptance criteria and tests

All tests use the in-process adapter or `TestClient` and the `spm_test` DB. No live :4005, Lemonade or `config.json`.

**Phase A — Contract and package boundary (≈2 days)**
- Add `world_engine/contract.py` with the models and Protocol above.
- Build a contract test suite parametrized over the adapters.
- Wrap the current app in `InProcessWorldEngine` and `HttpWorldEngine`.
- Make the proxy depend only on the Protocol. Replace the direct calls at `routes.py:262,753,760`.
- *Accept:* the full existing suite passes, and the contract suite passes for both adapters.

**Phase B — Correctness fixes (≈2–3 days)**
- Make the tick per-session, keyed by `turn_id`. Remove the global `action_tick_counter` and the seed value 1420.
- Make distances read the session world. This fixes `app.py:594-619`, which reads `app_state.current_world`, and lets the legacy `current_world` be deleted.
- Remove the hard-coded "upstairs"/"tavern" consequence block (`app.py:184-199`) and the `central_nexus` special case (`:612`).
- Honour `X-Idempotency-Key`. The client sends it today and the engine ignores it.
- *Accept:*
  - Two sessions advance independently.
  - Regenerating a message does not change the tick.
  - Two sessions with identical names and rooms yield independent gating.
  - Replaying a mutation is a no-op.

**Phase C — Persistence and reload (≈2–3 days)**
- Treat `objective_world_log` as the authoritative append-only log. Add `seq` and `origin` columns, plus a `mutation` JSONB column.
- Write a snapshot to `world_state_sessions` (plus `tick`) every N mutations.
- On first access, load the session lazily from snapshot plus log tail.
- Evict idle sessions from memory with an LRU (configurable, default 200).
- Make the reset endpoint per-session.
- *Accept:*
  - Restart test: state and tick are identical after a simulated restart.
  - Replay test: rebuilding from the log alone equals the snapshot.
  - Eviction and reload are transparent.
  - Memory stays bounded with 1,000 synthetic sessions (target < 150 MB RSS).

**Phase D — Graph and perception for the gating plan (≈3–4 days)**
- Make exits first-class with barrier and door state, modelled on `simpledoor`.
- Compute path distance with Dijkstra over `Exit.distance_ft` plus optional in-room `pos_ft`.
- Propagate sound and sight per barrier: blackout for a closed door or metal partition, and the SRD 3.3.2 bands of 5 and 15 ft. Line of sight is blocked by any closed barrier.
- Run hysteresis per session (already the case).
- *Accept:*
  - A table-driven test covers every barrier type × distance band × action type, matching SRD 3.3.2.
  - Results are identical over 1,000 runs.
  - p95 `submit_action` is under 5 ms for 50 rooms and 10 entities in process.

**Phase E — Mutation API for GM_ACTION (≈2 days)**
- Implement the closed mutation union, validation and an allow-list per origin.
- Reject topology-breaking edits, such as dangling exits or moving into a non-existent room.
- Record every mutation in the log with `origin="gm"`.
- *Accept:*
  - Malformed or unknown actions return 422 and leave the world unchanged.
  - Fuzzing GM payloads (hypothesis or a hand-written table) never corrupts the graph.

**Phase F — Ops, security and naming (≈1–2 days)**
- Bind to 127.0.0.1 and add a shared-secret header (covering OPEN-001 for the engine).
- Degrade gracefully when the engine is down (OPEN-010).
- Add a config flag `world_engine.mode = http|inprocess`.
- Rename the package and docs to "SPM World Engine (Evennia-inspired)", keeping a `evennia_world` shim for one release.
- Amend the PRD/SRD deviation note.
- *Accept:*
  - A proxy test with the engine down returns a notice instead of a 500.
  - A test confirms the engine refuses a request without the secret.

**Phase G (optional, owner-gated) — Evennia adapter spike (timebox 5 days)**
- Create a game directory outside the repo and implement the custom DRF view against the contract.
- Run the contract suite as a separate opt-in CI job.
- *Decision gate:* adopt only if it passes the full contract and its footprint and latency are acceptable on the Sovereign tier.

Total for A–F is ≈12–17 dev-days.

## 8. Risks and open questions

- **Spec drift perception.** The PRD still says Evennia. Mitigation: record the decision formally and keep the contract Evennia-compatible.
- **Reinventing too much.** Keep the engine small: a graph, entities, a log and perception. Inventory, NPC AI and combat are out of scope unless specified.
- **Turn identity.** SillyTavern does not send a stable message id. `turn_id` derivation (index plus hash of the user text) must be agreed with the FIFO and gating plans, and edits to an old message need defined semantics.
- **Snapshot cadence vs. write load.** Many sessions × every action could stress Postgres. Batch log writes, as the background task does today.
- **In-room positions.** Free-text roleplay rarely gives coordinates. Default to room-level distance (3 ft in the same room, exit `distance_ft` between rooms) unless a template supplies positions.
- **Open question: shared worlds.** Should two chats with the same characters ever share a world? FR-001 says no today. If that changes, the contract needs a `world_id` separate from `session_id`.

## 9. Dependencies

- **Gating plan.** Consumes `perceive` and `submit_action`, and owns the barrier and distance band table that Phase D implements.
- **GM_ACTION plan.** Consumes `apply_mutation`, its idempotency and origin rules, and its validation errors.
- **FIFO queue plan.** Provides the `turn_id` and the serialization point where `advance_tick` is called once per user message.
- **Token budget plan.** Consumes `ActionResult.sensory_feed` and `get_entity_state` for the room-description slice.
- **Embeddings plan.** Only indirectly (objective-log text may later be embedded).
- **Infra.** `scripts/init_db.sql` migration (Phase C), `tests/conftest.py` fixtures, and the auth work in OPEN-001.

## 10. Owner decisions needed

1. Approve Option 3a, which formally retires "Evennia as runtime" and records the deviation in the PRD/SRD.
2. Approve the Phase G spike, or drop it.
3. Choose the default engine mode for low tiers (`inprocess` saves a process; `http` keeps isolation).
4. Confirm tick semantics: per session, regenerations do not advance, edits to old messages do not rewind.
5. Choose the session eviction limit and snapshot cadence (defaults 200 and every 20 mutations).
6. Decide whether to rename `evennia_world` (recommended, with a shim for one release).
