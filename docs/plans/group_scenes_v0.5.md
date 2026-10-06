# Feature (v0.5): group scenes and user-directed scene transitions

**Status:** planned for v0.5 (owner decision, 2026-10-05). Not in v0.4.
**Origin:** v0.4 QA, group-chat test with 美 Mei and her mother Lian (QA log F10, 19:31).

## Intent

SPM's original purpose is spatial gating in **group chats**: several characters share
one world, each perceiving only what reaches them. v0.5 makes that a first-class
storytelling tool by letting the **user direct the scene**:

- The user can move the story's focus to a room they are not in. Example: "Mei hurries
  into the kitchen" while the user's character waits at the entrance.
- The character called on to respond answers **from their own room and point of view**,
  based on what they actually perceived there.
- Characters can talk **to each other** without the user present, each from their own
  POV, with neither hearing what happens elsewhere.

Target output for the QA scenario (the user shifts focus to the kitchen, where Lian is
preparing dinner):

> *Lian sees Mei enter the kitchen, her face alight with an unusually excited look. Lian
> pauses in cutting the vegetables to speak with her.* "Mei, did something happen?"

and, when Mei is called next, *her* prompt describes Lian at the cutting board, not the
user's character at the door.

## How it works today (v0.4)

| Area | v0.4 behaviour |
|---|---|
| Turn model | Every user message is an action **by the user, at the user's location**. The engine computes who perceives it (consequences); then the target character replies, and that reply is itself a world action perceived by whoever shares her room. |
| Per-character POV | **Works.** Each prompt is built from that character's own perception rows (gated history), never the shared transcript. |
| Character-to-character perception | **Works structurally.** Replies are world actions, so two characters in the same room enter each other's history. |
| Initial placement | First turn of a session: the user and the first speaking character are placed together in the seeded world's start room. |
| Characters joining later | Placed **in the user's room** before the user's action (F10 fix, `47555dd`). This avoids silence, but ignores the story: Lian belongs in the kitchen, not at the door. |
| Narration in user messages | `*emotes*` and prose are the **user's** visible actions at the user's location. There is no way to describe an event in another room. |
| Moving others by tag | `[move:Actor->PLACE]` is parsed, but the actor is **discarded** and the **user** is moved. User moves are also applied after the user's message, so the user speaks from the room they are leaving. |
| GM actions | The model may create rooms and move characters from its scratchpad. This is the only way characters get re-positioned today. |
| Skip rule (decision 8) | If the target can't perceive the user's message, SPM skips the model and shows "*X hears only muffled sounds…*". Built for 1:1 chats: in a group, a character in another room is skipped even when she perceived something else worth reacting to. |

## Gaps to close

1. **Story-driven placement.** A joining character should start where the story puts
   them, not wherever the user stands.
2. **Scene focus.** The user needs a way to point the "camera" at a room they are not in.
3. **Narration as a room event.** Scene narration should be perceived by those in the
   focus room as something they *saw*, not as the user speaking from the hallway.
4. **A group-aware skip rule.** Skip a character only when she perceived nothing new
   since her last turn.
5. **The `[move:Actor->PLACE]` actor** must be honoured, and the user's own moves should
   resolve *before* the user's message is perceived.

## Proposed design

All deterministic: the zero-LLM rule for the core path holds. The new tags follow the
decision-11 convention of **fail-open hints**: malformed or absent tags degrade to
today's behaviour, never to a leak.

| Piece | Design |
|---|---|
| `[at:Character:room]` | Place or move a named character. Unknown rooms are created only when GM actions are enabled; otherwise the tag is ignored and logged. |
| `[focus:room]` | Sets the session's **scene focus** for this turn. The rest of the message is narration happening in that room. |
| Narrator events | A focus message is submitted as an action by a `narrator` actor *inside the focus room*. Occupants perceive it per normal gating (direct in-room, muffled through an open door, nothing through walls). The user's character does not perceive it unless present. |
| Group-aware skip | A character is skipped only if she has no new perception rows since her last reply. Otherwise she runs on her own gated history. |
| Placement by name | A joining character is matched against existing room names in the card, scenario and recent messages (e.g. "kitchen" → `kitchen`) before falling back to the user's room. |
| Move ordering | User moves and `[at:]` placements resolve before the turn's speech or narration is perceived. |
| Prompt | The target's prompt gains one line: the room she is in and who she can see there. |

Engine/contract impact: a `narrator` entity kind (perceivable, never a recipient) and a
per-session focus field on the snapshot. Perception rows and gated history are unchanged.

## Acceptance (new QA section and leak-suite scenarios)

1. Focus on a room the user is not in: its occupants respond to the narration, and the
   user's character perceives none of it.
2. Two characters in the kitchen converse over several turns, each prompt in its own POV.
   A character in another room never sees their words.
3. A joining character whose card or scenario names a room starts there.
4. `[at:Lian:kitchen]` moves Lian, not the user.
5. A character in another room who perceived something new is not skipped; one who
   perceived nothing is.
6. All tags malformed or absent: identical to v0.4 behaviour (leak suite stays green).

## Open questions for the owner

- Tag names: `[focus:]` and `[at:]`, or something closer to SillyTavern conventions?
- Should a focus change persist until changed, or apply to one message only?
- When the user's character is not in the focus room, should the reply be written as
  pure third-person narration of that room?

**Estimate:** 2–3 days including tests.
