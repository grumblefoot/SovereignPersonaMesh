# Sovereign Persona Mesh (SPM) — Known Issues & Feature Backlog

**Last Updated:** August 27, 2026  
**Status:** ALL REPORTED DEFECTS REMEDIATED & CERTIFIED  

---

## 🐛 Defect Resolution & Feature Log

### 1. [BUG-001] Session Identifier Hardcoded to Fallback (`st_user_example_assistant`)
- **Status:** 🟢 RESOLVED (`proxy/api/routes.py`)
- **Resolution:** Upgraded `_extract_target_char()` with regex character card parsing (`[Arvenia's Personality=...]`, `[Character: Arvenia]`, etc.). Session ID dynamically resolves to `st_user_arvenia` upon new conversation start.

---

### 2. [BUG-002] Thought Viewer SSE Stream Does Not Auto-Update Live Messages
- **Status:** 🟢 RESOLVED (`proxy/ui/index.html`)
- **Resolution:** Implemented SSE auto-reconnect on error, auto-scroll to bottom of stream box (`scrollTop = scrollHeight`), and real-time DOM rendering without manual page refreshes.

---

### 3. [BUG-003] Thought Viewer Incorrectly Names Monologue Stream (`INNER MONOLOGUE [EXAMPLE_ASSISTANT]`)
- **Status:** 🟢 RESOLVED (`proxy/ui/index.html` & `proxy/core/telemetry.py`)
- **Resolution:** Stream headers dynamically label active character names (e.g. `💭 INNER MONOLOGUE [ARVENIA]`) based on extracted character metadata.

---

### 4. [BUG-004] Vibe Profiler Gauges Are Static During Stream Generation
- **Status:** 🟢 RESOLVED (`proxy/ui/index.html`)
- **Resolution:** Added real-time text analyzer (`updateVibeProfiler()`) computing Dialogue/Action word ratio, verbosity baseline, and perspective classification (First Person vs Third Person).

---

### 5. [BUG-005] Spatial Gating Map Container Fails to Render
- **Status:** 🟢 RESOLVED (`proxy/ui/index.html`)
- **Resolution:** Implemented `fetchSpatialMap()` polling world engine state to render room layout grid (`Dungeon Cellar`, `Tavern Common`), character locations (`[Arvenia]`, `[User]`), and acoustic gating badge (`DIRECT`).

---

### 6. [FEAT-006] Multi-Session Subtab Support in Thought Viewer
- **Status:** 🟢 IMPLEMENTED (`proxy/ui/index.html`)
- **Resolution:** Added interactive Session Selector filter dropdown in Thought Viewer (`All Sessions` vs specific session/character IDs).

---

### 7. [BUG-007] RAG Score Inspector Fails to Refresh Dynamically
- **Status:** 🟢 RESOLVED (`proxy/ui/index.html`)
- **Resolution:** Added `updateRAGInspector()` rendering formula breakdown `(1 - cos) * e^(-λt) * (1 + imp/10)` and live calculated scores.

---

### 8. [BUG-008] Data Management "Execute Factory Reset" Button Ineffective
- **Status:** 🟢 RESOLVED (`proxy/main.py` & `proxy/api/admin_routes.py`)
- **Resolution:** Added asyncpg connection pool initialization on startup event in `proxy/main.py`, injecting `_admin_db_pool` into `admin_routes.py`. `DELETE /admin/api/v1/factory_reset` truncates all tables and returns HTTP 200 OK success.

---

### 9. [BUG-009] Spatial Gating Map Not Resetting to Empty State
- **Status:** 🟢 RESOLVED (`proxy/core/telemetry.py`, `proxy/rag/import_worker.py`, `proxy/ui/index.html`)
- **Resolution:** Updated telemetry and UI renderer to display an empty state placeholder (`🗺️ No spatial locations discovered yet...`) with an `UNBOUND` badge on reset, dynamically populating nodes as chat turns execute or batch imports process.

---

### 10. [BUG-010] Thought Stream UI Persists Events After Factory Reset
- **Status:** 🟢 RESOLVED (`proxy/core/telemetry.py` & `proxy/ui/index.html`)
- **Resolution:** Added `{"event": "reset"}` SSE broadcast on factory reset. Browser JS listens for reset events, clearing in-memory thought arrays and resetting DOM containers.

---

### 11. [BUG-011] Monologue Internal Thoughts & Malformed Tags Bleed into Canon Chat
- **Status:** 🟢 RESOLVED (`proxy/core/stream_parser.py`)
- **Resolution:** Implemented `OPEN_TAG_REGEX` and `CLOSE_TAG_REGEX` regex pattern matchers handling malformed XML tags (`` `<ctrl94   ``), missing closing brackets (`</ctrl94Arvenia`), prompt directive echoes (`"You MUST begin your response...`), and line-buffered markdown scratchpad filtering (`*   *Thought:*`, `Check:`, `*(Self-Correction)*`).

---

## 📋 Remediation Status Matrix

| Priority | Issue ID | Component | Status | Verification |
|---|---|---|---|---|
| P0 | BUG-001 | Proxy Session Resolution | 🟢 FIXED | Verified `st_user_arvenia` |
| P0 | BUG-002 | Admin Dashboard SSE Stream | 🟢 FIXED | Live SSE auto-scroll |
| P0 | BUG-003 | Telemetry Character Tagging | 🟢 FIXED | Dynamic `[ARVENIA]` headers |
| P0 | BUG-011 | Monologue Stream Parser | 🟢 FIXED | 100% clean canon output |
| P1 | BUG-008 | Admin API Factory Reset | 🟢 FIXED | HTTP 200 OK table truncation |
| P1 | BUG-009 | Spatial Gating Map Reset | 🟢 FIXED | Empty placeholder + dynamic nodes |
| P1 | BUG-010 | Thought Stream SSE Reset | 🟢 FIXED | Auto-flush on reset |
| P1 | BUG-004 | Vibe Profiler Gauges | 🟢 FIXED | Live word/ratio analyzer |
| P1 | BUG-005 | Spatial Gating Map | 🟢 FIXED | Active room grid visualizer |
| P1 | BUG-007 | RAG Inspector Telemetry | 🟢 FIXED | Live formula breakdown |
| P2 | FEAT-006| Web Admin Session Filter | 🟢 FIXED | Session selector dropdown |

---

## 🔮 Design Discussions & Future Work

### [DESIGN-001] "Metagaming" & Telepathy / Internal Thought Filtering
**Problem:** Currently, SillyTavern sends user actions, spoken dialogue, and *internal thoughts* (usually formatted with asterisks) in a single plaintext block. Because this entire block is processed as the `sensory_feed`, characters (even non-telepathic ones) can "read" the user's thoughts and react to them, breaking RP immersion.

**Challenges:**
1. **Formatting is Unpredictable:** Regex filters are fragile because users employ vastly different formatting styles (or no formatting at all) for actions vs. thoughts.
2. **Performance Constraints (SLA):** Adding a second LLM in the critical path (User -> Scrubber LLM -> SPM -> Main LLM) to clean the text would double the Time-to-First-Token (TTFT) and violate the <5ms proxy SLA, especially on edge hardware.

**Proposed Architecture (Asynchronous Background Scrubber):**
- To preserve TTFT, the SPM immediately forwards the *raw* user text to the Main LLM for the active turn (allowing slight metagaming in the immediate response, which is often acceptable or unnoticed).
- **Concurrently**, an asynchronous background task (similar to the `ImportWorker`) dispatches the raw text to a tiny, fast model (e.g., `gemma-2-2b` or smaller) via the backend.
- The scrubber model's prompt: *"Extract only observable physical actions and spoken words. Remove all internal thoughts."*
- **Result:** Future memory retrievals for this event will contain *only* the physical actions, preventing the character from remembering or acting upon the user's internal thoughts in future turns.

---

### [DESIGN-002] State of the World Updates & Auto-Populating GM Rules
**Problem:** Currently, GM rules (`invariants` and `game_over` conditions) must be manually defined via SQL or the Admin UI. This creates friction for users wanting zero-config setup for imported chats or new characters. Furthermore, static rules prevent the scenario from organically evolving (e.g., if an invariant states a character is in a castle, but they later leave, the GM will falsely flag violations).

**Proposed Architecture (Human-in-the-Loop Auto-Extraction):**
1. **Initial Extraction (Background Task):** When a new chat or import begins, an asynchronous background task runs the character card and/or chat history through the LLM with an extraction prompt to generate candidate `invariants` and `game_over` rules.
2. **Admin Dashboard Approval:** To prevent "poisoning the well" with hallucinated invariants, extracted rules are set to `pending` and presented in the SPM Admin UI for the user to approve, edit, or reject before they become active.
3. **Periodic State of the World Reviews:** To allow characters and the narrative to grow (e.g., character arcs like Bilbo or Sun Wukong), Hermes or a background LLM task periodically reviews the recent chat history (e.g., every 15 turns). It proposes updates to the active invariants to reflect the evolving world state, ensuring the GM remains a dynamic referee.

### Issue: LLM GM_Action Bleedthrough (Parser Vulnerability)
**Date:** 2026-09-07
**Status:** 🟢 RESOLVED (`proxy/core/stream_parser.py` & `proxy/api/routes.py`)
**Description:**
The LLM frequently ignores closing tags (`</think>`) and transition markers (`[SCENE START]`, `---`) when finishing its scratchpad. Because our `MonologueStreamParser` rigidly waits for a closing tag to switch states, it remained in the monologue state until EOF. This triggered a fail-safe that dumped the entire buffer, causing raw `[GM_ACTION]` JSON tags to bleed into the frontend chat interface. Furthermore, the LLM sometimes wrote planning notes (e.g., "I need to describe...") after the closing tag.

**Resolution (Surgical Pivot Parser & Dual-Layer Filtering):**
1. **Surgical Pivot Parser:** Implemented an EOF fallback in `stream_parser.py`. If the stream finishes without closing `</think>`, it scans the buffer for the *last* `[GM_ACTION]` tag, designating everything before it as monologue, and yielding only text after the pivot.
2. **Prompt Directive Tuning:** Stricter rules in `proxy/api/routes.py` enforcing no planning outside `<think>` tags.
3. **Aggressive Regex Heuristics:** Expanded `_strip_monologue_bleed()` to catch "I need to", "The user is", and "Room Details". 
