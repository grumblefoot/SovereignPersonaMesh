-- QA F20 (2026-10-05): persist the world engine's turn -> tick map.
-- It lived only in memory, so after an engine restart a regenerate (or a second
-- character's reply in a group) for an already-seen turn advanced the tick again.
-- Idempotent.

CREATE TABLE IF NOT EXISTS spm_turn_ticks (
    session_id VARCHAR(255) NOT NULL,
    turn_id    VARCHAR(255) NOT NULL,
    tick       BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, turn_id)
);
