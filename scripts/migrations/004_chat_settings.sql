-- QA F26 (2026-10-06): per-chat settings. First use: narrator mode for scenario cards
-- ('auto' = detect from the card name, 'on', 'off'). Idempotent.

CREATE TABLE IF NOT EXISTS spm_chat_settings (
    session_id    VARCHAR(255) PRIMARY KEY,
    narrator_mode VARCHAR(8) NOT NULL DEFAULT 'auto',   -- auto | on | off
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
