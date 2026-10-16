# Known issues

Limitations you may run into with the current release. Reports and fixes are welcome.

## Security

- **No authentication.** The proxy, its admin dashboard (including factory reset, shutdown and
  settings) and the world engine accept requests from anyone who can reach them. By default they
  listen on this machine only (127.0.0.1) and refuse requests sent by other web pages in your
  browser; opening them to the network (`SPM_HOST=0.0.0.0`, or `SPM_LAN_ACCESS=1` in Docker)
  exposes the dashboard, including the thought stream with characters' private reasoning, to
  everyone on that network. Do that only on a trusted network.

## Chat behaviour

- **Occasional cut-off turns.** Now and then a turn ends a few seconds in with no reply, and
  regenerating works. SPM logs these as `Client disconnected from turn …`; the cause is still
  being investigated.
- **Withheld replies.** When the model writes only its planning notes, SPM withholds the reply
  and shows `[SPM: the model returned only planning notes…]`. That notice stays in the
  SillyTavern chat as a message; delete it or regenerate.
- **Going back in a chat** (deleting messages, or swiping back to another opening) makes SPM
  forget the abandoned turns, but rooms and positions in the world engine are not rolled back.
  For a different opening, start a new chat. This needs the `X-SPM-User-Turns` header.
- **SillyTavern's Summary and Author's Note** are removed from character prompts, because they
  can't be filtered per character. Turn on `st_passthrough_shared_notes` for chats with no
  secrets if you want them kept.

## Group chats and scenario cards

- **Group chats:** you can't yet direct a scene to a room you are not in; narration is always
  your action at your location; a character joining a group is placed with you rather than where
  the story puts them.
- **Scenario cards** (one card voicing a whole world): the narrator follows you, but characters
  inside the scenario (NPCs) are not tracked by SPM yet, so they all share the narrator's
  knowledge.

## Background work

- **Memory consolidation** (the "sleep cycle") runs inside the proxy at 04:00, so only when
  the proxy is running at that time.
