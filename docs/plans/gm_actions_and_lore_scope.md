# Plan: GM_ACTION as a user option (OPEN-005) and per-chat lore scope (OPEN-007)

Status: plan only, 2026-10-03. Nothing here is implemented yet.
Related plans (written in parallel, referenced only at their interface points): embeddings, gating, FIFO queue / scheduler, token budget, Evennia-vs-FastAPI.

---

## Part A: GM_ACTION as an on/off user option (OPEN-005)

### A.1 Goal

Make LLM-written world changes (`[GM_ACTION: MOVE | CREATE_ROOM]`) an explicit setting:

- **On:** an optional, LLM-assisted mode. Every action is schema-validated and rate-limited before it reaches the world engine.
- **Off:** no GM_ACTION directive in the prompt and nothing dispatched. The world is kept populated by deterministic code.

Record the On mode as a deliberate deviation from the spec.

### A.2 Current state (evidence)

- **The directive is hard-wired.** `proxy/api/routes.py:374-400` appends a "GAME MASTER" directive whenever `prompt_builder.config.inner_monologue_enabled` is true. That flag is a hardware-tier field, not a user setting: `config/hardware_tiers.py:37,47,57` (it is False for the third tier).
  - The directive bundles five concerns: think-tag formatting, lore cross-checking, GM warnings, anti-puppeting, and the two GM_ACTION bullets (`routes.py:380-381`).
  - It is appended after `build_csa_messages`, so the token-budget matrix never counts it.
- **Parsing.** `proxy/core/stream_parser.py:104-107` defines `GM_ACTION_REGEX` and `GM_ACTION_LINE_REGEX`.
  - `_clean_public_line` (`:182-184`) hides public GM_ACTION lines and keeps them for later.
  - `extract_gm_actions` (`:144-162`) deduplicates actions by their sorted JSON.
  - The EOF fallback uses the last GM_ACTION as the point where the monologue ends (`:474-478`).
- **Dispatch has no checks.** `_dispatch_gm_actions` (`routes.py:744-770`) dispatches every parsed action:
  - There is no schema, no allow-list of entities or rooms, and no cap on how many actions run.
  - `entity` defaults to the target character.
  - Each call gets a fresh `uuid.uuid4()` idempotency key, so a regenerate or swipe dispatches the same action again.
  - Parsing runs even when the monologue is off, so stray public tags would still be dispatched.
- **World engine.** `evennia_world/app.py:301-321` (`/world/rooms`) returns 409 when a room already exists, and `:444-481` handles `/world/move`. The client always uses the `dynamic` template (`proxy/backend_client/evennia_client.py:58-87`).
- **The world depends on GM_ACTION.** It starts from the empty `dynamic` template (known_issues BUG resolved by `08d6bd2`). When there are no consequences, the location is `"[Unmapped - Awaiting GM_ACTION: CREATE_ROOM]"` (`routes.py:284-285`) and gating falls back to `direct` (`:271-279`).
- **The spec allows LLM help only in narrow places.**
  - SRD §1/§3.1 (Zero-LLM rule): "LLM inference prohibited in the core unless faster or necessary".
  - PRD 11.1 step 2 itself allows a small model to do the semantic world-extraction pass.
  - PRD 11.1.1 requires templates, no raw room scripts, and "a strict, schema-validated JSON parser" for parameter changes.
- **Configuration.** `config/manager.py:22-42` holds typed defaults. Keys outside `_DEFAULT_VALUES` (for example `periodic_review_cadence`) are written ad hoc. The admin UI loads and saves settings at `proxy/ui/index.html:469-481`, and saves through `POST /admin/config` (`proxy/api/admin_routes.py:158-163`).

### A.3 Design

**Configuration.** Add these keys to `_DEFAULT_VALUES`, with environment overrides, int/bool casting and an enum check in `get_settings()`:

| Key | Type / default | Env | Meaning |
|---|---|---|---|
| `gm_actions_mode` | `"off" \| "move_only" \| "full"`, default `"full"` (current behaviour; see decision D-A1) | `SPM_GM_ACTIONS_MODE` | `move_only` hides CREATE_ROOM from the prompt and rejects it |
| `gm_actions_max_per_turn` | int, 4 | `SPM_GM_ACTIONS_MAX_PER_TURN` | Actions after the cap are dropped and logged |
| `gm_actions_max_rooms_per_session` | int, 40 | `SPM_GM_ACTIONS_MAX_ROOMS` | Further CREATE_ROOM actions are rejected |

- Bool and enum values must be cast. Today `use_alternate_extraction_model` is compared as both `'true'` and `true` (`index.html:470`).
- **Admin UI:** add a "World actions (LLM-assisted)" select with the two numeric fields. The help text says: "Off saves about 90 prompt tokens per turn plus the action output; no LLM requests are saved."

**Split the directive** into named parts in `proxy/rag/prompt_builder.py` (or `res/strings`):

- `monologue_format`
- `lore_check`
- `anti_puppeting`
- `gm_actions_move`
- `gm_actions_create_room`

`routes.py` assembles only the parts the mode allows. This decouples GM_ACTION from `inner_monologue_enabled`. If the monologue is off there is no `<think>` block to put actions in, so the effective mode becomes `off` and that is logged once.

**Token savings when off.** The two bullets are 332 characters, about 85-95 prompt tokens per turn. Output savings are about 30-60 tokens per MOVE and 60-120 per CREATE_ROOM (descriptions), plus whatever spatial reasoning the model writes in `<think>`. Realistically that is about 100-300 tokens per turn in which movement happens.

GM_ACTION adds **no extra LLM requests**: dispatch goes to the world engine over HTTP, not to the LLM. "Save requests" therefore only matters for the request budget of the next turn, and the owner should know that. (Token-budget plan: the directive should be counted in the budget once it is split into parts.)

**Off mode behaviour.**

- The parser still strips stray `[GM_ACTION…]` text, both from the monologue and from public lines. No user-visible change.
- `_dispatch_gm_actions` returns immediately and logs `gm_action.ignored(mode=off)` at debug level with a counter in telemetry.
- The world must not stay empty. The deterministic floor is PRD 11.1.1 template matching: `hybrid_builder.match_template_fuzzy` (`evennia_world/hybrid_builder.py:73`) over the card, scenario and greeting at first contact, then the user and character are placed in the template's start room. Movement after that comes from the gating workstream's deterministic movement detection.
- **Interface:** Part A requires the gating plan to provide `ensure_session_world(session_id, char_id, context_text)` and a movement detector, and to drop the `"Awaiting GM_ACTION"` location label. Part A does not implement them. Until they land, off mode means "everyone is in the same place", which equals today's fallback (gating `direct`).

**On mode: validation layer.** Add a new module `proxy/core/gm_actions.py`. It is pure Python and engine-agnostic, so it holds whatever the Evennia-vs-FastAPI decision is.

- **Pydantic discriminated union.**
  - `MoveAction{type:"MOVE", entity:str, room_id:Slug}`
  - `CreateRoomAction{type:"CREATE_ROOM", room_id:Slug, name:str≤60, desc:str≤400}`
  - `extra="forbid"`
  - Slug: lower-case `^[a-z0-9_]{1,48}$`, after normalising spaces and dashes.
  - Strip control characters, `<...>` tags and nested `[GM_...]` from `name` and `desc`.
- **Semantic checks** against the session world, fetched once per batch through `/world/state`:
  - The MOVE `entity` must be one of: the target character, `user` (the persona name maps to `user`), or a character already present in the session world.
  - The MOVE `room_id` must exist, or be created earlier in the same batch.
  - A CREATE_ROOM for an existing id becomes a no-op ("already exists"), so the world engine no longer logs a 409.
  - Batches are ordered CREATE then MOVE.
- **Limits:** per-turn cap and per-session room cap from the config keys above. Unknown `type` values are rejected.
- **Idempotency key:** `sha256(session_id | turn_anchor | canonical_action_json)` replaces `uuid4()` (`routes.py:756,764`).
  - `turn_anchor` is the hash of the last user message text.
  - A regenerate or swipe of the same turn then cannot double-apply.
  - Later, with Part B's `X-SPM-Gen-Type` header, a swipe can first undo the previous swipe's actions (see A.5).
- **Audit trail:**
  - Every accepted action is written to `objective_world_log` with `action_type='gm_action'` and source `llm`, so state written by the LLM can be told apart.
  - Rejections go to telemetry, with a reason code (`schema`, `unknown_entity`, `unknown_room`, `cap`, `mode`).
  - The admin stats show accepted and rejected counts.

**Spec deviation record.** Add a "Deliberate spec deviations" section (new `docs/SPEC_DEVIATIONS.md`, or a block in `known_issues.md`, per decision D-A3) with this entry:

> **SD-01 — Optional LLM-assisted world actions.** Off by owner choice. When on, the LLM proposes room creation and movement. The proposals reach the world only through the 11.1.1 "strict, schema-validated JSON parser". Routing, gating and tag parsing stay deterministic, so the Zero-LLM rule holds for the core path. The deviation from 11.1.1 is that rooms may be created outside templates.

Then close OPEN-005 as "kept, constrained, optional".

### A.4 Phases, acceptance criteria and tests

| Phase | Work | Acceptance criteria | Tests (spm_test DB, no live services) |
|---|---|---|---|
| A1 (S, ~0.5 d) | Config keys, casting, admin select, `GET/POST /admin/config` round-trip | The setting survives a restart. An invalid mode falls back to the default with a warning. | `tests/test_gm_actions_config.py`: defaults, env override, enum validation, bool/int casting (temp config path as in conftest) |
| A2 (S, ~0.5 d) | Directive split; assembly by mode; decoupled from monologue | With `off`, the backend prompt contains no "GM_ACTION" string. With `move_only`, it contains no "CREATE_ROOM". The other directive parts are unchanged. | Extend `test_gm_actions_flow.py`: capture `csa_messages` via a fake `lemonade_client.generate_stream`; snapshot per mode |
| A3 (S, ~0.5 d) | Gate dispatch on mode | With `off`, a stream containing GM_ACTION tags in the monologue *and* in public lines causes zero world-engine calls and shows no leaked text. | Extend `test_public_gm_action_bleed.py` with a mocked `evennia_client` and assert it is not awaited |
| A4 (M, ~1.5 d) | `proxy/core/gm_actions.py` validator, ordering, caps, deterministic idempotency key, audit rows | Malformed, unknown-entity, unknown-room and over-cap actions are rejected with reason codes. CREATE then MOVE in one batch succeeds. The same turn regenerated gives identical keys. | New `tests/test_gm_action_validation.py`: pure unit tests plus one DB test asserting `objective_world_log` rows in spm_test |
| A5 (S, ~0.5 d) | `SPEC_DEVIATIONS.md` / known_issues update; admin help text | OPEN-005 is closed with a link to SD-01 | none |

Total: about 3.5 days. A1 to A3 are independent of the other plans. Off mode depends on the gating plan for anything beyond "same room".

### A.5 Risks and open questions

- **Swipes.** Two swipes may move a character to different rooms, and only the swipe the user finally keeps should count. Proposal: keep a per-turn action journal and, on `swipe` or `regenerate`, reverse the previous journal entry first. This needs Part B's generation-type header. Until then, the last write wins.
- **Ordering with the next turn.** Dispatch is a background task after the stream (`routes.py:527`). The next turn's `/world/action` can race it. The FIFO/scheduler plan should serialise world mutations per session, or routes should await the previous session's dispatch task before spatial routing.
- **Paid-API compatibility.** The `<think>` prefill (`routes.py:400`) is a local-model technique. OpenAI does not continue assistant prefill, so `full` mode may simply yield no actions there. Off mode is the natural preset for paid backends (token-budget and backend plans).
- **Validation strictness.** Too strict a validator silently starves the world. Watch the rejection counters during the first live week.

### A.6 Owner decisions

- **D-A1: default mode.**
  - Option: `full` (today's behaviour).
  - Option: auto-preset by backend (`off` when `BACKEND_LLM_URL` is a known paid host).
  - Recommendation: `full`, plus a one-click "paid API preset" in the UI.
- **D-A2:** Is `move_only` worth having, or is a plain on/off toggle enough?
- **D-A3:** Where deviations live: a new `docs/SPEC_DEVIATIONS.md` (recommended) or `known_issues.md`.

---

## Part B: Lore rules scoped per chat (OPEN-007), with a chat identity scheme

### B.1 Goal

Give every SillyTavern chat a stable SPM identity, and scope lore rules (and, consistently, memory and world state) to it. This stops lore bleeding between chats (BUG-013 root cause). Returning to an old chat must find its lore again. Optionally, character-wide "canon" lore stays global.

### B.2 Current state (evidence)

- **Lore schema has no session.** `scripts/init_db.sql:100-118` (`create_csa_lore_rules_table`) has the columns `id, rule_text, rule_type, rule_embedding, status, created_at` and no `session_id`.
- **Extractor drops the session.** `proxy/rag/lore_extractor.py:47,124,128` receives `session_id` but never stores it. The insert is at `:112-118`, and the duplicate check by `rule_text` (`res/strings.json:26`) is per character.
- **Retrieval is per character.** `proxy/rag/retriever.py:96-131` (`retrieve_lore_rules`) filters only by character and status. Memory retrieval in the same file *does* filter by `session_id` (`:49`).
- **Admin.** Pending, approve and reject (`admin_routes.py:289-358`) are per character. `DELETE /admin/sessions/{id}` (`:166-213`) deletes memories, imports and world state but **not lore**. Factory reset truncates lore (`:216-260`).
- **Session id today.** `_extract_session_id` (`routes.py:138-157`) uses this precedence: `X-Session-ID` > body `session_id` > `X-Chat-ID` / `X-Conversation-ID` > `st_{user}_{char}`.
  - **Bug:** the endpoint calls it with `request.model_dump()` (`routes.py:251`). `ChatCompletionRequest` (`:67-73`) declares neither `session_id` nor `user`, and pydantic drops extra fields. So the body branch is dead in production, and `user` is always `"user"`, giving `st_user_<char>`.
  - The unit test `tests/test_fr001_session_isolation.py:349-359` passes a raw dict, which hides the bug.
- **What SillyTavern 1.19.0 actually sends** (source at `../SillyTavern`, commit `06bde939f`; the live ST uses `chat_completion_source: "custom"`, `custom_url: http://localhost:5050/v1`):
  - The upstream body for the Custom source is the standard OpenAI fields plus `custom_include_body` (`src/endpoints/backends/chat-completions.js:2394-2410`). There is **no chat id and no user name**: `user_name`, `char_name` and `chat_id` reach the ST server (`public/scripts/openai.js:2817-2819,2902`), but ST forwards `chat_id` only for Fireworks.
  - The **Custom source supports user-defined headers and body fields**: `custom_include_headers` / `custom_include_body`, merged server-side (`chat-completions.js:2409-2410`). They are **macro-substituted per request in the browser** (`openai.js:2924-2926`, `substituteParams`).
  - Every chat file header carries `chat_metadata.integrity`, a UUIDv4 minted when the chat loads (`public/script.js:7665-7666`) and persisted. All four live Arvenia chats have distinct UUIDs.
    - Renaming a chat file (`src/endpoints/chats.js:621`) keeps the metadata, so the identity survives renames.
    - **Branches and checkpoints mint a fresh `integrity`** and record the parent's name in `main_chat` (`public/scripts/bookmarks.js:200-201,283-284`).
  - No built-in macro exposes `integrity` or the chat name. Extensions can register macros (`getContext().registerMacro`, `public/scripts/st-context.js:180`).
  - Built-in macros that help: `{{lastGenerationType}}` (`normal|swipe|regenerate|continue|impersonate|quiet`, `macros/definitions/state-macros.js:35-40`), `{{lastMessageId}}` and `{{lastSwipeId}}`.
  - Chat files are named `<Char> - YYYY-MM-DD@HHhMMmSSsMMMms.jsonl`.
  - ST trims the oldest history when the context is full, so the greeting and first messages **are not reliably present in long chats**.

### B.3 Design: chat identity

Identity is resolved in a new `proxy/core/chat_identity.py`. `_extract_session_id` is replaced by `resolve_chat_identity(req, raw_body, messages) -> ChatIdentity(session_id, source, parent_id, gen_type)`. It reads the raw JSON body, which fixes the model_dump bug. Tiers, strongest first:

1. **Explicit id via the ST extension (recommended).** Add a tiny extension `st-extension/spm-chat-identity/` (manifest plus about 40 lines of JS, installed into `data/default-user/extensions`). It registers these macros:
   - `{{spmChatId}}` → `chat_metadata.integrity`
   - `{{spmParentChat}}` → `chat_metadata.main_chat`, `{{spmChatName}}` → `getCurrentChatId()` (the file name)
   - `{{spmCharId}}` → avatar or group id

   The user pastes once into *Connection → Custom → Additional headers*:

   ```yaml
   X-SPM-Chat-ID: "{{spmChatId}}"
   X-SPM-Parent-Chat: "{{spmParentChat}}"
   X-SPM-Gen-Type: "{{lastGenerationType}}"
   X-SPM-Message-ID: "{{lastMessageId}}"
   ```

   The extension can also offer a "copy header block" button and warn when the headers are missing. With this tier:
   - The session id is `stc_<integrity>`.
   - Renames are stable.
   - A new chat gets a new id.
   - Returning to an old chat recovers its exact id.
   - Branches and checkpoints get their own id plus a parent link.
2. **Explicit id without the extension.** Honour the existing `X-Session-ID` / `X-Chat-ID` headers and a body `session_id` (via `custom_include_body`, now actually read). This is for power users and other frontends.
3. **Fingerprint fallback.** Use this only when no header is present.
   - Store per resolved chat a list of message-content hashes in a new table `spm_chat_identity(session_id, char_id, anchor_hashes TEXT[], first_seen, last_seen, source)`.
   - Hash normalised text (whitespace collapsed, macros already expanded by ST), excluding system messages.
   - **Match rule:** take the request's earliest non-system message whose hash appears in exactly one known chat for this character. Ties are broken by the longest common run of hashes.
     - Matching on the earliest *surviving* message rather than the greeting is what makes this robust to ST's history trimming.
   - **New chat rule:** if only the greeting is present (the first turn), or there is no match, create `stf_<char>_<sha(greeting, first_user_msg, first_seen)>`. The greeting alone is not unique (every new chat with a character starts with it), so a greeting-only first turn is provisional. It is confirmed on the next turn by the first user message.
   - **Edits and swipes:** an edited or swiped *last* message changes only the tail, so matching on earlier anchors still works. Editing the *first* user message before turn 2 can fork the identity. That is acceptable and logged.
   - **Branches:** a branch shares its prefix with the parent and is ambiguous. The rule picks the most recently seen chat and logs `identity.ambiguous`. This limitation is documented, and the reason the extension is recommended.
4. **Legacy fallback:** `st_user_<char>`, used only when there are no messages (health probes). It is never used for writes once tiers 1-3 exist.

The resolved `source` (`header|explicit|fingerprint|legacy`) is shown in the admin session list. The fingerprint tier logs a once-per-session hint to install the extension.

**Generation type.** `X-SPM-Gen-Type` is passed to downstream code as `ChatIdentity.gen_type`. `quiet` and `impersonate` requests should skip memory persistence, lore extraction and GM dispatch (interface with the FIFO/scheduler plan, which can also give `quiet` a lower priority). `swipe` and `regenerate` feed Part A's swipe handling.

### B.4 Design: lore schema and migration

- **DDL**, in `create_csa_lore_rules_table`, idempotent:
  - `ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) NULL`
  - `ADD COLUMN IF NOT EXISTS scope VARCHAR(10) NOT NULL DEFAULT 'chat'` (values `chat` or `global`)
  - Index on `(session_id, status, rule_type)`
  - A NULL `session_id` with `scope='global'` is character canon.
- **Writes:** the extractor stores `session_id`. The duplicate check becomes `rule_text = $1 AND (session_id = $2 OR scope = 'global')`.
- **Reads:** `retrieve_lore_rules(character_id, query_embedding, session_id)` uses `WHERE status='active' AND (session_id = $sid OR scope='global')`. The SQL in `res/strings.json:27-29` is updated too, or removed if it is unused (the extractor and retriever use inline SQL).
- **Admin:**
  - The pending list returns `session_id`, a chat label (character plus first-seen date from `spm_chat_identity`) and `scope`.
  - New `POST /admin/lore/{char}/{rule_id}/promote` sets `scope='global'`, `session_id=NULL`.
  - A session filter is added.
  - `DELETE /admin/sessions/{id}` also deletes that session's lore.
  - UI: a filter dropdown and a "Make canon" button.
- **Migration** (`scripts/migrations/2026_10_lore_session_scope.sql`, idempotent, run by the operator, also applied by `init_db.sql` for fresh databases): add the columns, then for existing rows choose per decision D-B2:
  - (a) mark them `scope='chat', session_id='st_user_<char>'` (the legacy bucket, recommended: stops bleeding immediately, nothing lost), or
  - (b) mark them `scope='global'` (keeps today's behaviour until the owner prunes).
- **Lineage, optional phase B5:** when `X-SPM-Parent-Chat` is set and the child chat has no lore yet, copy the parent's active chat-scoped rules into the child at first contact (copy-on-branch). This avoids fragile read-through chains. The parent name maps to an id through `spm_chat_identity` (the extension sends the parent's file name; we store file name → id by having the extension also send `X-SPM-Chat-Name: "{{spmChatName}}"`).

### B.5 Alignment with session-scoped memory (FR-001)

Memory (`csa_memory_*`), world state (`world_state_sessions`, Evennia `session_worlds`), imports and telemetry are already keyed by `session_id`. They are just fed the per-character `st_user_<char>` value. Switching the resolver to per-chat ids therefore makes FR-001 truly per chat, at no extra schema cost.

Consequences:

- Rows from before the switch stay under `st_user_<char>`. An admin "adopt legacy data into this chat" action can re-key them (D-B3).
- The first request of every existing long chat under the new id triggers the FR-002 bulk import (`routes.py:190`). That is PRD Flow B behaviour. With the stub embedder it imports low-value vectors (embeddings plan), so B3 could gate auto-import behind a setting until the embedder lands.
- The sleep cycle consolidates per (session, day) and needs no change.

### B.6 Phases, acceptance criteria and tests

| Phase | Work | Acceptance criteria | Tests |
|---|---|---|---|
| B1 (S, ~0.5 d) | Fix the model_dump bug: resolver reads the raw body; add `X-SPM-*` headers | A body `session_id` and `X-SPM-Chat-ID` sent through the real endpoint reach the memory rows | `test_chat_identity.py`: TestClient POST to `/v1/chat/completions` with a fake backend; assert the stored `session_id` in spm_test |
| B2 (M, ~1 d) | Lore schema, migration, extractor and retriever scoping, admin delete/promote/filter | Two chats with the same character: a rule approved in chat A is not injected in chat B. A promoted rule appears in both. Migration is re-runnable and keeps every row. | Extend `test_lore_extractor.py`, `test_lore_cadence.py`; new `test_lore_scope.py` (two sessions, approve, retrieve); migration test applying the SQL twice to an spm_test table seeded in the old shape |
| B3 (S, ~1 d) | ST extension, setup doc, "headers missing" hint | With headers configured: new chat → new id; reopening an old chat → same id; renamed chat → same id; branch → new id with parent | JS is verified manually. Python side: header parsing, `{{macro}}` left unsubstituted (extension missing) is detected and treated as absent |
| B4 (M, ~1.5 d) | Fingerprint fallback with `spm_chat_identity` | Built from the captured `tests/fixtures/test_chat_payload.json`: the same chat with its head trimmed resolves to the same id; a new chat with the same greeting gets a new id after turn 2; a last-message swipe or edit keeps the id; a branch is logged as ambiguous | `test_chat_fingerprint.py` (pure unit plus spm_test table) |
| B5 (S, ~0.5 d, optional) | Copy lore on branch; `gen_type` handling (`quiet`/`impersonate` skip writes) | A branch starts with the parent's lore; a quiet request writes no memory or lore | Unit tests with headers |

Total: about 4.5 days (B5 optional). B1 and B2 deliver the OPEN-007 fix with the explicit-header path. B3 makes it usable out of the box. B4 is the safety net.

### B.7 Risks and open questions

- **Users without the extension.** The fingerprint tier is heuristic, and branches are ambiguous. Mitigation: show the identity `source` in the admin UI and warn.
- **Unsubstituted macros.** If the extension is not installed, ST sends the literal `{{spmChatId}}`. The resolver must reject any value that matches `\{\{.*\}\}`.
- **Groups.** Group chats have a chat id but several characters. The identity is per chat, lore stays per character table, and the session column scopes it. This needs a check against the group payload, which is not captured yet.
- **Auth (OPEN-001).** Headers are client-supplied. On an unauthenticated proxy anyone can read another chat's lore by guessing a UUID. That is low risk on localhost and must be revisited with auth.
- **Unverified:** that `lastGenerationType` is already updated when the request headers are built (the macro describes it as "last queued"). Confirm with one logged request when services are next running. Also capture raw headers and body keys once, because the existing fixture is post-`model_dump` and cannot show extra fields.

### B.8 Owner decisions

- **D-B1:** Install the ST extension as the supported path (recommended), or headers-only with manual `X-Session-ID` per chat?
- **D-B2:** Existing lore rows → legacy bucket (recommended) or global canon?
- **D-B3:** Provide "adopt legacy data into this chat" for memories and lore, or let `st_user_<char>` data age out?
- **D-B4:** Should branches inherit lore (copy-on-branch, B5) and also memories, or start clean?
- **D-B5:** Gate automatic bulk import of reopened long chats until the real embedder ships?

---

## Dependencies and interface points (both parts)

- **Gating plan:** provides template bootstrap (`ensure_session_world`) and deterministic movement detection for GM_ACTION off mode. It consumes the per-chat `session_id` from Part B, which makes worlds per chat.
- **FIFO / scheduler plan:** lore extraction is an LLM call and must go through the scheduler. `X-SPM-Gen-Type=quiet` is a priority hint. World-mutation dispatch should be serialised per session before the next turn. GM_ACTION itself makes no LLM call.
- **Token-budget plan:** the split directive parts and the injected `[ACTIVE LORE]` block must be counted. Off mode reduces the directive by about 90 tokens.
- **Embeddings plan:** lore trigger retrieval and the bulk import of reopened chats only become meaningful with a real embedder. Neither part depends on it to ship.
- **Evennia-vs-FastAPI plan:** the validator and identity resolver live in the proxy and are engine-agnostic. World endpoints are used as they are today (`/world/state`, `/world/rooms`, `/world/move`).
