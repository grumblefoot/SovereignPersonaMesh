# SPM Chat ID: SillyTavern extension setup

SPM needs to know which SillyTavern chat each request belongs to, and a little about the
generation. The `SPM-Chat-ID` extension provides that through four macros, which SillyTavern
sends to SPM as HTTP headers.

## What it does

| Macro | Value |
|---|---|
| `{{spmChatId}}` | The chat's stable identity: `chat_metadata.integrity`, a UUID SillyTavern stores in every chat file. If an old chat has none, the extension creates one and saves it. Empty when no chat is open. |
| `{{spmParentChat}}` | For a branch or checkpoint, the parent chat's file name, percent-encoded (header values must be ASCII); empty otherwise. |
| `{{spmGenType}}` | The type of generation: `normal`, `swipe`, `regenerate`, `continue`, `impersonate` or `quiet`. Empty before the first generation after opening a chat. |
| `{{spmUserTurns}}` | How many user messages the chat holds (hidden ones excluded). SillyTavern trims old messages from long prompts, so SPM can't count turns itself; this lets it tell "the user deleted messages and went back" from "old messages were trimmed". |

SillyTavern fills in these macros in the Custom source's additional headers on every
generation, and its server forwards the headers to SPM.

## Install

Copy the `st_extension/SPM-Chat-ID/` folder from this repository into
`<SillyTavern>/data/<user-handle>/extensions/` (the default handle is `default-user`), then
reload the SillyTavern browser tab. No server restart is needed. It appears under Extensions as
"SPM Chat ID".

## One-time configuration

In SillyTavern, open **API Connections → Chat Completion → Custom (OpenAI-compatible)**:

1. Set the URL to `http://localhost:5050/v1`.
2. In **Additional Headers** ("Include Headers"), paste exactly:

   ```yaml
   X-SPM-Chat-ID: "{{spmChatId}}"
   X-SPM-Parent-Chat: "{{spmParentChat}}"
   X-SPM-Gen-Type: "{{spmGenType}}"
   X-SPM-User-Turns: "{{spmUserTurns}}"
   ```

   The quotes matter: they keep an empty macro an empty string instead of a YAML `null`.

3. Turn off the **Summarize** extension. Its summary is built from the full chat and added to
   every prompt, which would tell characters about scenes they never perceived. SPM keeps its
   own memories, and logs a warning if it sees a summary request.

## Verifying it works

1. **Macros registered:** after reloading the tab, the browser console shows
   `[SPM-Chat-ID] registered macros: {{spmChatId}}, {{spmParentChat}}, {{spmGenType}}, {{spmUserTurns}}`.
   With a chat open, run:

   ```js
   SillyTavern.getContext().substituteParams('{{spmChatId}} | {{spmParentChat}} | {{spmGenType}} | {{spmUserTurns}}')
   ```

   You should get a UUID, the parent name (or nothing), the last generation type and a number.
2. **Headers reach SPM:** send a message. The SPM log shows a session id of the form
   `st_chat_<uuid>` and a `Generation type:` line. In the admin dashboard
   (`http://localhost:5050/admin`) the chat appears under that id.

If a header contains the literal text `{{spmChatId}}`, the extension didn't load: check the
Extensions panel and the browser console. SPM treats such a value as absent.

## Behaviour and limitations

- **Branches and checkpoints** get a new id, with the parent recorded in `{{spmParentChat}}`.
- **Renaming a chat** keeps its id.
- **Exported and re-imported chats** keep their id (two copies of one export share it).
- **Group chats** work the same way.
- **Only the Custom chat-completion source** sends these headers, which is the source SPM uses.
- **Deprecation warnings:** the extension registers macros with `getContext().registerMacro`,
  which SillyTavern 1.19 marks as deprecated but which is the only API that reaches both of its
  macro engines. Three harmless warnings appear in the console at load.
- **Changing SillyTavern's "experimental macro engine" setting** needs a page reload for the
  macros to re-register.
