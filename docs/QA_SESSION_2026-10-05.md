# v0.4 QA session 1: 2026-10-05

Live QA with the owner in real SillyTavern, against a third-party card (美 Mei) and then a
group chat (Mei and her mother Lian). It found **23 defects; 22 are fixed, tested and
deployed** (F1–F23 except open O1/O2). It also produced **one design decision (D1)** and
**one v0.5 feature** (group scenes).
Step-by-step evidence: the **QA log** tab of the
[SPM v0.4 QA Test Plan](https://claude.ai/code/artifact/c5632042-b000-4741-9017-2269471ceec1).
Raw logs: `~/Desktop/Experiments/SillyTavern/qa-logs/2026-10-05/`.

State at pause: `V0.4` @ `d40a011`, **801 tests passing**, all work pushed, services stopped.

## Where QA stands

| Section | Status |
|---|---|
| A. Setup and chat identity (A1–A5) | **All pass** (after fixes) |
| B. Gating | B1 effectively pass (A3 isolation + engine leak fix; confirm). **B4 pass.** **B5: retest pending** (re-trigger Lian after the F23 fix). B2, B3, B6 not run. |
| C. World building | C1 effectively pass (seen live several times). C2–C4 not run formally. |
| D. Memory, lore, budget | Not run. Note for D2: approve one of the second Mei chat's pending rules. |
| E. Failure modes | E3 partly seen (first live preemption). E1, E2, E4 not run. E4 needs the proxy running overnight. |
| F. Admin UI | F1/F3 partly seen (the Thought viewer fixes are in use). Not run formally. |
| G. Group chat (v0.4 scope) | **G1–G3 pass; G4 pass with notes** |

**Resume at:** re-trigger Lian for B5 (she should notice Vardus murmuring to Mei, not
the words), then B2/B3/B6, C2–C4, D, E, F.

## Findings

| # | Sev | Finding | Fix |
|---|---|---|---|
| F1 | High | Wrong target character on a third-party card (ASCII-only name pattern; persona description taken as the character) | `bed99b8` |
| F2 | High | Admin Thought viewer showed invented data (`Math.random()` RAG scores, hardcoded "Arvenia" map) | `be53187` |
| F3 | High | Test runs reset the LIVE world engine (hardcoded localhost in factory reset) | `be53187` |
| F4 | Med | Character replies advanced the world tick | `4c49eef` |
| F5 | Med | Salvaged replies never saved (5 of 8 Gemma-4 replies leave `<think>` unclosed) | `7cef99c` |
| F6 | High | Branching a non-ASCII-named character failed in SillyTavern (header encoding) | `2b4d10d` |
| F7 | Med | Branches / pre-SPM chats never seeded | `2f8c5af` |
| F8 | Med | Zombie proxy processes (shutdown hung on the dashboard stream) | `0f59739` |
| F9 | High | Impersonate/Summarize ran through the character pipeline (returned only a header) | `d61a8cb` |
| F10 | High | Group member joining later was never placed, and inherited another's blackout | `47555dd` |
| F11 | **Critical** | Bulk import gave a joining group member the whole transcript (omniscience leak) | `3b4d0e7` |
| F12 | High | Future-tense planning line leaked into a visible reply | `63df103` |
| F13 | Med | SillyTavern's "[Start a new group chat]" marker created junk rooms | `63df103` |
| F14 | High | Salvaged planning shown as the whole reply | `a3ab11a` |
| F15 | Med-High | GM duplicated rooms (never told which exist) | `20d2782` |
| F16 | High | Group nudge heard as the user's speech | `c6a9200` |
| F17 | High | Out-of-character `**directions**` discarded; now delivered as an author's direction | `c6a9200` |
| F18 | Med | Speech and action merged; player labelled "User" | `7cef99c` |
| F19 | Med-High | GM moves and reply recorded in parallel (a character moved in missed the question) | `b797161` |
| F20 | Low | Ticks drifted after engine restarts | `d40a011` |
| F21 | High | Second reply in a group turn erased the first | `ae86757` |
| F22 | Low | Preempted lore task logged an ERROR | `c5a82fb` |
| F23 | Med-High | Whisper bystanders in the same room got nothing (not a murmur) | `d40a011` |
| O1 | Low, open | SillyTavern's connection test is processed as a real turn | — |
| O2 | Low, open | A punctuation-only message becomes in-world speech | — |

**Decision D1 (owner, option A):** an author's direction goes only to the first character
who answers it in a turn; later responders learn by perception.

**Owner convention adopted:** `"speech"`, `*in-character action*`,
`**out-of-character direction**`.

## Data corrections made during QA (test data, owner-approved)

- Deleted 14 leaked bulk-import rows from Lian's memory (F11).
- Realigned positions twice to match the story: Lian and the player to the living room.
- Restored Lian's turn-6 reply, which F21 had erased, from the log; removed Mei's stale
  rows ahead of her regenerate.
- Migration 003 applied to the live DB; turn ticks backfilled from the perception log.

## Carried forward

- **v0.5 headline feature:** group scenes and user-directed scene transitions
  (`docs/plans/group_scenes_v0.5.md`, OPEN-014), now with three live examples.
- **Model behaviour to watch:** Gemma-4-26B leaves `<think>` unclosed on most replies.
  The salvage path is now robust (F5, F12, F14), but native reasoning output from the
  backend would remove the problem at its source. Worth a spike.
- The branch chat's world still holds junk rooms from F13/F15 (harmless; there is no
  room-delete operation).
