# Design need (v0.5): one card, many characters (scenario cards)

**Status:** design need, recorded 2026-10-06 from the first open-world test. Not scheduled
in detail; to be designed alongside group scenes (`group_scenes_v0.5.md`) and off-screen life /
world processes (`offscreen_life_and_world_processes.md`).

## The problem

SPM's model is **one SillyTavern card = one character with a body**: a room, a point of view,
its own memories and lore. Many popular cards aren't characters but **scenarios**: an open world
("My Hero Academia RPG World") voiced by one narrator, often with a large lorebook that defines
dozens of characters. In the test: 199 lore entries, about 68k tokens, with entries such as
"Shota Aizawa: Eraser Head" and "World Context: Quirks".

Two consequences, both seen live:

| | What happens today | Why it matters |
|---|---|---|
| **O3: NPCs are invisible** | Aizawa, classmates and villains exist only as text in the narrator's replies. SPM tracks no position, memory or perception for them. | Gating, SPM's core promise, can't separate characters inside a scenario. The narrator knows everything every NPC knows. |
| **F26: the narrator has a body** | The scenario card is placed in a room like a person. When the GM moved the player to a new area, the narrator stayed in the starting room, its reply registered as a blackout for the player, and on the next message the narrator would have been skipped as "out of earshot". | A scenario stops responding after the player's first move. |

## What a scenario card needs

1. **The narrator is a camera, not a body.** It narrates the player's experience, so it perceives
   what the player perceives and follows the player everywhere. It is never gated away from the
   player, and never placed in a room as an occupant.
2. **NPCs are entities.** Characters the scenario introduces get world entities: a position,
   their own perception rows and memories, joining and leaving scenes. This is the same machinery
   as group-chat members, without a SillyTavern card behind each one.
3. **NPC sources, in order of trust:**
   - Lorebook character entries (a name plus a description), proposed as entities when the
     story activates them.
   - GM actions in the narrator's scratchpad ("Aizawa enters"), validated like other GM actions.
   - Possibly the user, through tags (`[at:Aizawa:classroom]` from the group-scenes spec).
4. **Point of view per NPC, written by one model call.** Calling the model once per NPC per turn
   is too expensive (and wrong for paid-token models; same reasoning as lazy re-entry). Instead,
   the narrator's prompt gets a compact per-NPC block: who is present, and what each of them
   perceived since they last appeared. The model writes all of them in one reply, and SPM
   attributes the result back to each NPC's memory.
5. **Lorebooks are knowledge, not budget.** World Info is part of the untrimmable system prompt
   today. For large lorebooks, SPM should know which entries describe which NPC, so an NPC's
   own entry is shown only when that NPC is present.

## Detecting a scenario card

There's no reliable marker in what SillyTavern sends. Options:

- A **per-chat setting** ("this card is a narrator/scenario"), the dependable route.
- A **heuristic default**: name words such as World, RPG, Story, Adventure, Narrator, Game Master,
  Scenario; a lorebook attached; a first message addressed to "you".
- Both: heuristic proposal, setting confirms.

## Interim (v0.4)

Before v0.5, F26 needs a mitigation so scenario cards stay usable: the narrator follows the
player on every move, or is exempt from gating. See QA log F26.

## Open questions

- Should an NPC introduced only in narration become an entity automatically, or only with
  confirmation (owner, GM approval, or a lorebook entry)?
- How should NPC memories be attributed when one reply voices several NPCs?
- Group chat of scenario cards (two narrators): in scope?
