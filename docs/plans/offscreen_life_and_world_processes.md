# Design note: off-screen character life and world processes

**Status:** design input, not scheduled. Consider it when designing v0.5 group scenes and any
later entity-tracking work. Owner discussion, 2026-10-06. Nothing here is implemented.

## The question

While a character is out of the scene (Mei gathering snacks in the kitchen while Lian serves tea
in the living room), what happens to them? And how does SPM handle things that change on their
own over time (a spreading fire, a bomb on a timed fuse)? The original idea was a very short
in-character memory for the skipped span. It may add little to roleplay by itself, but the same
machinery matters for richer entity tracking.

## Principle

**The engine owns what happens; the model owns how it is told.** Mechanics (state that changes
over time, rules, what is perceivable where) are simulated deterministically. Narrative (what a
character did and how it felt) is written by the model, and only when the story needs it.

This keeps SPM's core rule: routing and gating never need the model.

## 1. Off-screen character life (narrative)

| Approach | How | Cost |
|---|---|---|
| Templated ambient row (gating plan §7, OPEN-015) | On every skipped turn, store a fixed line ("heard muffled sounds") | Zero inference, but adds little |
| Generate per skipped turn | One model call per absent character per turn | Multiplies calls; on a single-slot backend each queues behind real chat; expensive on paid APIs |
| **Lazy, at re-entry (chosen)** | When the character is next called on, their prompt gets one line of elapsed context; the model covers the off-screen span inside the reply it is already writing; that reply is stored as their memory as usual | **Zero extra calls**; off-screen time is only filled in when the story needs it |

**Decision (owner, 2026-10-06): lazy re-entry.** It makes the best use of a paid-token model and
spends nothing on characters the story never returns to.

The elapsed-context line is deterministic, built from data SPM already has:

> *Since you last acted: 4 turns passed. You were in the kitchen. You heard muffled voices from
> the living room.*

- **Inputs:** turns since the character's last reply (perception log), their room and any room
  changes (world engine), and what they perceived in that span (perception rows, so the line obeys
  gating: never what they couldn't hear).
- **Optional richer source:** a per-character routine from the card or scenario ("Lian cooks
  dinner in the evenings") seeds what they were plausibly doing.
- **OPEN-015 consequence:** don't wire up the templated ambient filter
  (`proxy/core/sensory_filter.py`). Replace it with the elapsed-context line, and remove the dead
  import.

## 2. World processes (mechanics)

A spreading fire, a bomb fuse, weather, a character's schedule: these are **simulated by the world
engine as small state machines**, not narrated by the model.

- A process has state, a rule that advances it per unit of game time, and events it emits. Example:
  a fire spreads to an adjacent room every N minutes, changes that room's state (`burning`), and
  emits "smoke" and "flames" events.
- **Events are world actions**, perceived through normal gating: behind a closed door you smell
  smoke before you see flames; in another wing you perceive nothing.
- Deterministic and testable, costs no inference, and the model can't "forget" a lit fuse.
- Characters and processes interact through the same engine: moving a character out of a burning
  room is a world mutation; a process can end ("fire extinguished").

## Prerequisite: game time

Today a **tick is one user message**, not a unit of story time. Fuses, schedules and "4 turns
passed" all need a game clock:

- **Engine-advanced:** each turn advances the clock by a default amount (configurable per chat).
- **User-nudged:** a fail-open tag such as `[time:+10min]` or `[time:evening]`, in the same family
  as the v0.5 scene tags (`[focus:]`, `[at:]`).
- Elapsed-context lines then speak in story time ("ten minutes passed"), not turns.

## Where this fits

| Release | Scope |
|---|---|
| **v0.5** (group scenes) | The lazy elapsed-context line at re-entry: v0.5 already reworks the skip rule and character placement. OPEN-015 is resolved this way. |
| **Later (v0.6+)** | World processes: a timed-process model in the engine, the game clock and time tags, process events as perceivable actions. |

## Open questions

- Should a re-entering character's prompt also carry a one-line *process* summary ("the fire has
  spread to the hall"), or only what they perceived?
- Default game time per turn: fixed, per chat, or inferred from the scene?
- Should schedules come from the card (author-defined) or be proposed by the GM and approved?
