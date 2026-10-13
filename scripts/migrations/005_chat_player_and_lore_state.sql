-- QA 2026-10-12: per-chat player character and initial-lore state. Idempotent.
-- player_sheet / player_name (N7, N4): the user's "Name: / Age: / Quirk:" block is
-- out-of-character setup, shown to the narrator every turn; its Name is a player alias
-- for GM actions. initial_lore_done (N5): first-turn lore extraction ran, tracked
-- explicitly instead of inferred from perception rows (an aborted first attempt
-- used to skip it for good).

ALTER TABLE spm_chat_settings ADD COLUMN IF NOT EXISTS player_sheet TEXT;
ALTER TABLE spm_chat_settings ADD COLUMN IF NOT EXISTS player_name VARCHAR(255);
ALTER TABLE spm_chat_settings ADD COLUMN IF NOT EXISTS initial_lore_done BOOLEAN NOT NULL DEFAULT FALSE;

-- Chats that already have history ran (or skipped) their first-turn extraction under the
-- old rule; don't re-run it for them.
INSERT INTO spm_chat_settings (session_id, initial_lore_done)
SELECT DISTINCT session_id, TRUE FROM spm_perception
ON CONFLICT (session_id) DO UPDATE SET initial_lore_done = TRUE;
