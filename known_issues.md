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
