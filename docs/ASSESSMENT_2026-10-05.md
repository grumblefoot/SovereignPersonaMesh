# SPM assessment — 2026-10-05 (post Sprint 4)

Scope: a live and code audit of `V0.4` after all five planned sprints, the human
QA still needed before SPM can be called done, and what it takes to make SPM
installable on another machine.

State at the end of this pass: `V0.4` @ `9590133`, **763 tests passing** (Python 3.13
and 3.14), services live on the fixed code, 0 errors in either service log.

---

## 1. Bugs found in this pass (all fixed, tested, pushed)

Every one of these passed the existing test suite. They were found by probing the
live stack, and each now has a regression test that failed before its fix.

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | **Critical** | **Cross-chat gating leak.** `/world/action` used the empty `dynamic` world for seeded sessions, then fell back to the process-global `current_world` — whichever chat was configured *last*. Chat A's speech was delivered to a character that exists only in chat B. Gating had only appeared to work because tests and single-chat use configure one session at a time. | `a30c469` |
| 2 | High | GM actions wrote to a phantom `dynamic` world: rooms and moves "succeeded" but never reached the world characters perceive, and the move removed the player from the real world. | `a30c469` |
| 3 | High | Moving the player (id `user` in every chat) removed them from *other* chats; `/world/move` and the place/delete endpoints also mutated the shared templates, so placements leaked into later sessions. | `a30c469` |
| 4 | High | The in-proxy sleep cycle (added in Sprint 4) read the wrong DB variables (port 5435): tonight's 04:00 pass would have failed every night. | `70772aa` |
| 5 | Medium | Target-character detection matched the `[scene:KEY]` tag, so minimal prompts ran under a character called `scene` (wrong memory table and recipient). | `cd73e02` |
| 6 | Medium | The LLM names the player by persona ("Tom") but the engine id is `user`: every player move was rejected. | `cd73e02` |
| 7 | Medium | GM move entities kept the LLM's spelling (`Mira`), so the engine added a second occupant instead of moving `mira`. | `6473aab` |
| 8 | Medium | Token-budget calibration never fired live: Lemonade streams `usage` only when asked (`stream_options.include_usage`). Live now: ratio 3.20 → 3.35 chars/token after one turn. | `9590133` |
| 9 | Low | `requirements.txt` declared 7 packages never imported at runtime (incl. ~200 MB `onnxruntime`). Split into runtime (6 packages) and dev. | `d761b82` |

Two existing tests had encoded bug #1 (placing characters in one session and then
asserting another session could see them). They were corrected, not deleted.

**Process note:** five of these were defects in work from the last two days that
unit tests passed. Mocks that always return the happy shape (bug 8) and tests that
configure only one session (bug 1) hid them. The live probes found them in minutes,
which is the main argument for the QA pass in §3.

## 2. Open issues (not fixed in this pass)

| Item | Why it matters | Suggested action |
|---|---|---|
| **Old-model memories invisible to vector recall** | Live rows embedded with the old gemma model (space 1: 4 memories, 13 lore) or with no embedding (13 memories) aren't found by vector search under Qwen3 (space 4). | Run `scripts/reembed.py` once. It's resumable; it uses the embedder, not the chat model. |
| **QA/test residue in the live DB** | One early leak-suite run this morning wrote 88 perception rows, 41 `arvenia` and 2 `watcher` memories into `litellm_postgres`. Today's probes also left a `scene` character (4 memories, 18 lore rules) and test chats. Re-verified: the suite no longer writes to the live DB, with or without `SPM_TEST_DB`. | Delete sessions `st_chat_leak*` and `st_chat_qa*`, plus the `scene` tables. Needs owner OK (live data). |
| Gating latency gate unmeasured | Sprint 2's exit gate had two parts. 0 leaks was proven; "<30 ms p95 at 300 messages" was never measured. | Benchmark it (half a day). |
| `.env.example` is stale | It uses `LLM_BACKEND_URL`, `EVENNIA_API_URL`, `LLM_MAX_CONTEXT_TOKENS`; the code reads `BACKEND_LLM_URL`, `EVENNIA_LIAISON_URL`, `SPM_CONTEXT_WINDOW_OVERRIDE`. Settings are silently ignored. | Rewrite it from `config/manager.py`'s env map (packaging prerequisite). |
| Open on the network, no auth | The proxy (`SPM_HOST`, default `0.0.0.0`) and engine (hardcoded `0.0.0.0:4005`) accept LAN connections. The admin UI shows private monologues and can delete data. OPEN-001 auth is deferred by decision. | Fine on the homelab. Must default to `127.0.0.1` before distribution. |
| Legacy `current_world` alias | It's now written but never read on any gating path. It's dead weight that invites the same bug again. | Remove it in a cleanup pass. |

## 3. Human QA before calling it done

Automated tests can't prove these. They need real SillyTavern, the real model and
your judgement. They are ordered so that a failure early on makes later items moot.

### A. Setup and identity (≈30 min)
1. With the ST extension installed, start a new chat and send a message. In the
   admin perception view (`/admin/api/v1/sessions/{id}/perception`) the session
   should be `st_chat_<uuid>`, not `st_user_<char>`.
2. Reopen that chat: same session id. Start a second chat with the same
   character: a different id, and no shared memories or lore.
3. Branch a chat: new id. Check whether the parent link appears in the logs.
4. Run **Impersonate** and a **Summarize** (quiet) generation. Both should return
   text, and neither may add memories, perception rows or a tick.

### B. Gating — the core promise (≈1 h)
5. **Two chats at once** (this is bug #1's scenario). In chat A say something
   distinctive; in chat B, with a different character, ask what they heard. B must
   not know.
6. Same room versus behind a door: put two characters in different rooms (GM moves,
   or the `[move:room]` tag). Speak to one, then ask the other. Behind a closed door
   they get nothing; through an open door, only muffled sound.
7. Private thoughts: write a `'quoted thought'` containing a secret. No character
   should ever react to it, including several turns later.
8. Whisper with a third character present: `[whisper:Mira] "..."`. The bystander
   notices a murmur and never the words.

### C. World building (≈30 min) — newly working, never seen by a person
9. Narrate walking somewhere new ("we head down to the wine cellar"). The room
   should appear in the snapshot and both characters should be moved into it, once.
   Bugs #2, #6 and #7 meant this never actually worked before today.
10. Set GM actions to **Off** in the admin UI and repeat. Nothing is created and
    the prompt contains no GM directive.
11. Watch the `[GMAction] rejected` lines for a session. A high rejection rate
    means the validator is too strict (the plan's stated risk).

### D. Memory, lore, budget (≈1 h)
12. After `reembed.py`, ask a character about something from 20+ turns ago that's
    outside SillyTavern's context. A relevant memory should come back. Judge whether
    the 0.45 threshold feels right (too many irrelevant memories → lower it).
13. Approve a lore rule in chat A and confirm it doesn't apply in chat B. Promote
    it to canon and confirm it then applies in both.
14. Long chat: set `context_window_override` to 8192 and keep chatting past it.
    The chat should keep working, and the logs should show `[TokenBudget] … trimmed`.
    A huge character card should produce the visible "not sent" notice.
15. Regenerate and swipe several times. There should be no duplicate memories and
    the world tick shouldn't advance (check `/world/snapshot`).

### E. Failure modes (≈20 min)
16. Stop the LLM backend mid-chat: you should get a visible notice, not an
    invented reply. Stop the world engine: chat continues, ungated.
17. Tomorrow morning: check the proxy log for `[SleepCycle] consolidation pass
    complete`. This path has never run for real.

### F. Admin UI (≈15 min)
18. Open `/admin`. Check that the GM-actions, Memory & Embeddings and Token Budget
    sections load, save, and survive a restart. Hermes built these and no person
    has rendered them yet.

**Exit criterion:** A, B and C pass with no leaks or crashes; D and E have no
blocking issues. Anything that fails gets filed and fixed before the tag.

## 4. Packaging SPM for other machines

### What SPM needs at runtime
- **Python 3.13+** with six packages (verified on a clean venv).
- **PostgreSQL 16 with pgvector.** This is the hard part of any installer.
- **An OpenAI-compatible LLM backend**, which the user supplies (Lemonade,
  llama.cpp, KoboldCpp, Ollama, vLLM or OpenAI). An embeddings endpoint is optional;
  without one SPM falls back to full-text search.
- **SillyTavern** plus the `st_extension/SPM-Chat-ID` extension, copied into
  ST's third-party extensions folder.
- Two services, the proxy (:5050) and the world engine (:4005). The engine is pure
  Python and could run inside the proxy process.

### Prerequisites for any packaging route (≈2–3 days)
1. **Safe network defaults:** bind `127.0.0.1` by default; make the engine's host
   and port configurable. This can't wait for OPEN-001, because the admin UI is
   unauthenticated.
2. **One config source:** regenerate `.env.example` from `config/manager.py`. Make
   `config/docker-compose.yml` agree with the code (today it names the container
   `litellm_postgres` and doesn't run SPM itself).
3. **A cross-platform launcher.** `start_spm.sh` is Linux-only (podman, `nc`,
   `notify-send`, `xdg-open`) and lives outside the repo with a hardcoded path.
   Replace it with a Python `spm` CLI: `spm init` (create the DB, apply
   `init_db.sql`, run the migrations), `spm start`, `spm stop`, `spm status`.
4. **First-run setup:** ask for the backend URL, chat model and embedding model
   (or `none`), and print the SillyTavern connection settings and the steps to
   install the extension.
5. **A Windows smoke test.** Nothing has run on Windows. Uvicorn's `uvloop` extra
   is skipped there (it falls back), and any POSIX assumptions need checking.

### Three routes, cheapest first

| Route | What the user does | Effort | Pros | Cons |
|---|---|---|---|---|
| **A. Docker Compose** | Install Docker; `docker compose up` | ≈1–2 days on top of the prerequisites | Same on Linux, macOS and Windows (Docker Desktop); Postgres and pgvector come solved; the Containerfile already exists | Users need Docker; reaching a backend on the host needs `host.docker.internal` setup |
| **B. pip package + embedded Postgres** | `pipx install spm`; `spm init`; `spm start` | ≈3–5 days | No Docker; one command; the six-package runtime installs fast | Needs an embedded Postgres that includes pgvector (the `pgserver` package is a candidate; Windows support needs a spike) |
| **C. Native installer / .exe** | Download and run an installer | ≈2–3 weeks | Most familiar for non-technical users | PyInstaller builds must run on each OS (no cross-compiling), so it needs a CI matrix; Postgres binaries must be bundled; Windows code signing to avoid SmartScreen; route B's work is a prerequisite anyway |

**Recommendation:** build route A first. It's the fastest way to get SPM running on
another device, and it exercises the prerequisites. Then route B, which removes
Docker. Route C wraps B and is only worth it if non-technical users are the
audience.

**Not recommended now:** porting from Postgres to SQLite (with `sqlite-vec`) to
avoid the database dependency. SPM uses pgvector's `<=>` operator, `tsvector`
full-text search, PL/pgSQL table-factory functions and JSONB throughout. That's a
large refactor, and it would put at risk the behaviour that today's QA just verified.
