/**
 * SPM Chat ID — stable per-chat identity macros for the Sovereign Persona Mesh.
 *
 * Registers three macros that the user pastes once into the Custom Chat
 * Completion source's "Additional headers" field, so every request to the
 * SPM proxy carries a stable chat identity:
 *
 *   {{spmChatId}}     -> chat_metadata.integrity (UUIDv4, created+saved if missing)
 *   {{spmParentChat}} -> chat_metadata.main_chat (branch/checkpoint parent) or ""
 *   {{spmGenType}}    -> this generation's type: normal|swipe|regenerate|continue|
 *                        impersonate|quiet|... ("" before the first generation)
 *
 * Transport: SillyTavern macro-substitutes the Custom source's additional
 * headers in the browser on every request (public/scripts/openai.js:
 * `generate_data.custom_include_headers = substituteParams(...)`) and the
 * server merges them into the upstream request headers
 * (src/endpoints/backends/chat-completions.js, mergeObjectWithYaml).
 *
 * Dependency-free: uses only the public `SillyTavern.getContext()` API.
 * Verified against SillyTavern 1.19.0.
 */

'use strict';

(function () {
    const MODULE = 'SPM-Chat-ID';

    /** RFC-4122 v4 UUID, same shape SillyTavern's own uuidv4() produces. */
    function uuidv4() {
        try {
            if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
                return crypto.randomUUID();
            }
        } catch {
            // fall through to the polyfill
        }
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === 'x' ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    function getContextSafe() {
        try {
            if (typeof SillyTavern !== 'undefined' && typeof SillyTavern.getContext === 'function') {
                return SillyTavern.getContext();
            }
        } catch (err) {
            console.error(`[${MODULE}] getContext failed`, err);
        }
        return null;
    }

    /**
     * {{spmChatId}}: the chat's stable identity.
     * Reads chat_metadata.integrity; mints and persists one if the chat
     * predates the integrity field. Empty string when no chat is open.
     */
    function spmChatId() {
        const ctx = getContextSafe();
        if (!ctx) return '';
        try {
            // getCurrentChatId() is undefined when no character/group chat is open.
            if (!ctx.getCurrentChatId()) return '';
            const meta = ctx.chatMetadata;
            if (!meta || typeof meta !== 'object') return '';
            if (!meta.integrity) {
                meta.integrity = uuidv4();
                ctx.saveMetadataDebounced();
                console.log(`[${MODULE}] minted missing chat_metadata.integrity:`, meta.integrity);
            }
            return String(meta.integrity);
        } catch (err) {
            console.error(`[${MODULE}] spmChatId failed`, err);
            return '';
        }
    }

    /**
     * {{spmParentChat}}: the parent chat's file name when this chat is a
     * branch or checkpoint (chat_metadata.main_chat), else empty string.
     */
    function spmParentChat() {
        const ctx = getContextSafe();
        if (!ctx) return '';
        try {
            if (!ctx.getCurrentChatId()) return '';
            const parent = ctx.chatMetadata?.main_chat;
            return parent ? String(parent) : '';
        } catch (err) {
            console.error(`[${MODULE}] spmParentChat failed`, err);
            return '';
        }
    }

    // {{spmGenType}}: tracked from GENERATION_STARTED, exactly like ST's own
    // {{lastGenerationType}} (public/scripts/macros/definitions/state-macros.js),
    // but owned by this extension so it works regardless of ST's macro-engine
    // internals. GENERATION_STARTED fires at the top of Generate()
    // (public/script.js:4299), before the request headers are substituted
    // (public/scripts/openai.js:2926), so the value is current for the
    // request being built, not the previous one.
    let genTypeValue = '';

    function spmGenType() {
        return genTypeValue;
    }

    function init() {
        const ctx = getContextSafe();
        if (!ctx) {
            console.error(`[${MODULE}] SillyTavern context unavailable; macros not registered.`);
            return;
        }

        try {
            ctx.eventSource.on(ctx.eventTypes.GENERATION_STARTED, (type, _params, isDryRun) => {
                if (isDryRun) return;
                genTypeValue = type || 'normal';
            });
            ctx.eventSource.on(ctx.eventTypes.CHAT_CHANGED, () => {
                genTypeValue = '';
            });
        } catch (err) {
            console.error(`[${MODULE}] could not subscribe to generation events`, err);
        }

        // ctx.registerMacro (MacrosParser.registerMacro) is marked deprecated,
        // but it is the only registration surface that feeds BOTH macro
        // evaluators in ST 1.19.0: the default legacy evaluator (via
        // MacrosParser.populateEnv, public/scripts/macros.js:676) and the
        // experimental engine (via the bridge at macros.js:81-117). The
        // suggested replacement, macros.register(), only reaches the
        // experimental engine, which is off by default.
        try {
            ctx.registerMacro('spmChatId', spmChatId,
                'SPM: stable id of the current chat (chat_metadata.integrity; created and saved if missing). Empty when no chat is open.');
            ctx.registerMacro('spmParentChat', spmParentChat,
                'SPM: parent chat file name when this chat is a branch/checkpoint (chat_metadata.main_chat), else empty.');
            ctx.registerMacro('spmGenType', spmGenType,
                'SPM: type of the generation being produced (normal, swipe, regenerate, continue, impersonate, quiet). Empty before the first generation in a chat.');
            console.log(`[${MODULE}] registered macros: {{spmChatId}}, {{spmParentChat}}, {{spmGenType}}`);
        } catch (err) {
            console.error(`[${MODULE}] macro registration failed`, err);
        }
    }

    init();
})();
