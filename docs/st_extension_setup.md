# SPM Chat ID — SillyTavern extension setup

Sprint 0 client half of the chat-identity scheme (plan: `docs/plans/gm_actions_and_lore_scope.md` Part B §B.3 tier 1; `docs/plans/SPRINT_PLAN.md` §1.2, decision 3 approved).

## What it does

A tiny, dependency-free SillyTavern UI extension that registers three macros:

| Macro | Value |
|---|---|
| `{{spmChatId}}` | The chat's stable identity: `chat_metadata.integrity`, a UUIDv4 SillyTavern persists in every chat file. If a (pre-1.13-era) chat has none, the macro mints one and saves it. Empty string when no chat is open. |
| `{{spmParentChat}}` | For a branch or checkpoint chat, the parent chat's file name (`chat_metadata.main_chat`), **percent-encoded** (decode with `urllib.parse.unquote`); empty otherwise. Header values must be ASCII, and chat files are named after the character, so a raw `美 Mei - …` made SillyTavern fail every branch with `ERR_INVALID_CHAR`. |
| `{{spmGenType}}` | The type of the generation being produced: `normal`, `swipe`, `regenerate`, `continue`, `impersonate`, `quiet`. Empty before the first generation after opening a chat. |

Pasted into the Custom API's additional-headers field (below), these ride on **every** chat-completion request to the SPM proxy, giving SPM a per-chat session identity that survives chat renames, ST restarts, and returning to an old chat.

**Transport path: HTTP headers (confirmed in ST 1.19.0 source).** SillyTavern macro-substitutes the Custom source's additional headers in the browser on every generation (`public/scripts/openai.js:2926` — `generate_data.custom_include_headers = substituteParams(settings.custom_include_headers)`), and the ST server merges them into the upstream request's headers (`src/endpoints/backends/chat-completions.js:2410` — `mergeObjectWithYaml(headers, request.body.custom_include_headers)`). No body injection is needed.

## Install

- **This machine:** already installed at `SillyTavern/data/default-user/extensions/SPM-Chat-ID/` (manifest.json + index.js). ST loads data-folder extensions at page load, so just **reload the ST browser tab** (no server restart). It appears under Extensions → "SPM Chat ID" as a third-party extension.
- **Other installs:** copy the `st_extension/SPM-Chat-ID/` folder from this repo into `<SillyTavern>/data/<user-handle>/extensions/` (default handle: `default-user`), then reload the ST tab.

## One-time configuration

In SillyTavern: **API Connections → Chat Completion → Custom (OpenAI-compatible) → Additional Headers** (labelled "Include Headers"), paste exactly:

```yaml
X-SPM-Chat-ID: "{{spmChatId}}"
X-SPM-Parent-Chat: "{{spmParentChat}}"
X-SPM-Gen-Type: "{{spmGenType}}"
```

The quotes matter: they keep an empty macro an empty string instead of a YAML `null`.

Optional, until the SPM-side resolver (Part B phase B1) ships: adding the line below makes the **current** proxy key everything per chat immediately, because today's `_extract_session_id` (`proxy/api/routes.py:138-157`) already honours `X-Session-ID` first. Note the resulting session ids are the bare UUID, not the planned `stc_<uuid>` form, so skip this if you want to wait for the canonical resolver.

```yaml
X-Session-ID: "{{spmChatId}}"
```

## Verifying it works

1. **Macros registered** — after reloading the ST tab, open the browser console; you should see `[SPM-Chat-ID] registered macros: {{spmChatId}}, {{spmParentChat}}, {{spmGenType}}`. With a chat open, run:
   ```js
   SillyTavern.getContext().substituteParams('{{spmChatId}} | {{spmParentChat}} | {{spmGenType}}')
   ```
   You should get a UUID, the parent name (or blank), and the last generation type.
2. **Headers leave the browser** — send one message, then in DevTools → Network find the `/api/backends/chat-completions/generate` request: its JSON payload's `custom_include_headers` field must contain the resolved UUID, not the literal `{{spmChatId}}` text.
3. **Headers reach SPM** — once the Part B server half lands, the proxy logs/stores the resolved session id per request (`stc_<uuid>`, source `header`), and memory/lore rows carry it. Until then, the `X-SPM-*` headers arrive at `:5050` but are ignored by `_extract_session_id`; a temporary debug log of `request.headers` in `proxy/api/routes.py` shows them if you want proof today (or use the optional `X-Session-ID` line above and check the `session_id` column in `csa_memory_*`).

If a header ever contains the literal text `{{spmChatId}}`, the extension did not load (check Extensions panel and console errors); SPM's resolver should treat any value starting with `{{` as absent.

## Behaviour and limitations

- **Branches/checkpoints get a fresh id, with the parent recorded.** ST mints a new `integrity` for every branch/checkpoint and stores the parent chat's file name in `main_chat` (`public/scripts/bookmarks.js:200-201, 283-284`); the extension exposes both. Lore/memory lineage (copy-on-branch) is the server's job (plan §B.4 phase B5).
- **Renames are safe.** Renaming a chat file keeps `chat_metadata`, so the id is unchanged.
- **Exports/imports keep metadata.** A `.jsonl` chat export carries the metadata header, so a re-imported chat keeps its id. (Two copies of the same export share one id — by design, they are the same conversation.)
- **Group chats work.** Group chat metadata also carries `integrity` (`public/scripts/group-chats.js:277-278`).
- **Gen type is "last queued".** `{{spmGenType}}` is set when ST's `Generate()` starts (`public/script.js:4299`), which happens *before* headers are substituted for that request (`public/scripts/openai.js:2926`), so it is accurate for the request it rides on. Right after opening a chat it is empty until the first generation.
- **Headers only apply to the Custom chat-completion source.** Other sources (Text Completion, paid presets) do not send `custom_include_headers`; SPM is driven through Custom (`custom_url: http://localhost:5050/v1`), so this is fine.
- **Deprecated-but-correct registration API.** The extension registers via `getContext().registerMacro` (`MacrosParser.registerMacro`), which ST 1.19.0 marks deprecated. It is used deliberately: it is the only surface feeding both the default legacy macro evaluator (`public/scripts/macros.js:676`) and the experimental macro engine (bridge at `macros.js:81-117`); the suggested replacement `macros.register()` reaches only the experimental engine, which is off by default. Three one-line deprecation warnings appear in the console at load; harmless. If ST removes the bridge in a future major version, switch the three calls to `macros.register(name, { handler, description })`.
- **Toggling ST's "experimental macro engine" setting requires a page reload** for the macros to re-register on the active engine (true of all legacy-registered macros, not just these).
