-- Sprint 2 chunk 3 (gating phase 4 foundation): per-recipient perception log.
-- One row per (turn, recipient): what each character actually perceived, post-gating.
-- Doubles as the cache the per-character gated history is built from.
-- Idempotent.

CREATE TABLE IF NOT EXISTS spm_perception (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    tick BIGINT NOT NULL,
    turn_id VARCHAR(255),
    actor_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(16) NOT NULL,
    recipient_id VARCHAR(255) NOT NULL,
    gating_level VARCHAR(16) NOT NULL,          -- direct | degraded | blackout
    perceived_text TEXT NOT NULL DEFAULT '',    -- post-gating text ('' for blackout)
    distance_ft REAL,
    barriers JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_spm_perception_recipient
    ON spm_perception (session_id, recipient_id, tick, id);
CREATE INDEX IF NOT EXISTS idx_spm_perception_turn
    ON spm_perception (session_id, tick);
