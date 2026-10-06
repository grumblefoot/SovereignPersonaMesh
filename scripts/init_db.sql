-- Sovereign Persona Mesh (SPM) Database Initialization Script
-- Enables pgvector extension and creates initial database structure.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Global Objective World Log (Ground Truth maintained by Evennia / WSD)
CREATE TABLE IF NOT EXISTS objective_world_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    action_tick BIGINT NOT NULL,
    actor_id VARCHAR(255) NOT NULL,
    location_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    raw_event TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_world_log_session_tick ON objective_world_log(session_id, action_tick);

-- ======================================================================
-- Embedding space registry (embeddings plan phase 2 / migration 001).
-- Which (provider, model, dim) produced a vector; vectors from different
-- spaces are never compared. csa_* tables reference id by value only
-- (dynamic tables: FK enforced by convention, not constraint).
-- ======================================================================
CREATE TABLE IF NOT EXISTS spm_embedding_spaces (
    id SMALLSERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(255) NOT NULL,
    dim INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider, model, dim)
);

-- Dynamic Schema Helper Function for Character Subagent Episodic Memory
-- Usage: SELECT create_csa_memory_table('luna');
-- Keep in sync with scripts/migrations/001_embedding_spaces_and_lore_scope.sql.
CREATE OR REPLACE FUNCTION create_csa_memory_table(char_id TEXT)
RETURNS VOID AS $$
DECLARE
    table_name TEXT := 'csa_memory_' || lower(char_id);
    emb_typmod INT;
BEGIN
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL DEFAULT ''default_session'',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            sensory_input TEXT NOT NULL,
            inner_monologue TEXT,
            public_response TEXT,
            episodic_embedding vector,
            embedding_space_id SMALLINT,
            importance_score INT DEFAULT 5,
            is_core_memory BOOLEAN DEFAULT FALSE,
            is_subjective BOOLEAN DEFAULT TRUE,
            access_count INT DEFAULT 1,
            last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            fts tsvector GENERATED ALWAYS AS (to_tsvector(''simple'',
                coalesce(sensory_input, '''') || '' '' || coalesce(public_response, ''''))) STORED
        );
    ', table_name);

    -- Lazy upgrades for tables created in older shapes (all idempotent).
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS public_response TEXT', table_name);
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS embedding_space_id SMALLINT', table_name);

    -- VECTOR(3584) -> dimension-free vector (the active embedding space
    -- decides the dimension; stored vectors were reset to NULL on 2026-10-03).
    SELECT atttypmod INTO emb_typmod
      FROM pg_attribute
     WHERE attrelid = to_regclass(table_name)
       AND attname = 'episodic_embedding'
       AND NOT attisdropped;
    IF emb_typmod IS NOT NULL AND emb_typmod <> -1 THEN
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN episodic_embedding TYPE vector USING episodic_embedding::vector',
            table_name);
    END IF;

    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS (to_tsvector(''simple'',
            coalesce(sensory_input, '''') || '' '' || coalesce(public_response, ''''))) STORED',
        table_name);

    -- NOTE: no HNSW/IVFFlat index: exact scan is deterministic (SRD) and the
    -- tables hold thousands of rows per character. HNSW above HNSW_MIN_ROWS
    -- is embeddings plan phase 6. The GIN index serves the full-text fallback.
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_core_time ON %I (is_core_memory, timestamp)',
        table_name, table_name
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_fts ON %I USING GIN (fts)',
        table_name, table_name
    );
END;
$$ LANGUAGE plpgsql;

-- Initialize default demo tables
SELECT create_csa_memory_table('rowan');
SELECT create_csa_memory_table('domino');
SELECT create_csa_memory_table('luna');
SELECT create_csa_memory_table('seamus');

-- ======================================================================
-- FR-002: Bulk Chat Import Tracking
-- ======================================================================
CREATE TABLE IF NOT EXISTS spm_chat_imports (
    import_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    character_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    total_messages INT NOT NULL DEFAULT 0,
    processed_messages INT NOT NULL DEFAULT 0,
    error_log TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_imports_session ON spm_chat_imports(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_imports_status ON spm_chat_imports(status);

-- ======================================================================
-- FR-003: Tiered Data Lifecycle & Cold Storage
-- ======================================================================
CREATE TABLE IF NOT EXISTS spm_cold_archives (
    archive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    character_id VARCHAR(255) NOT NULL,
    archive_path TEXT NOT NULL,
    record_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cold_archives_session ON spm_cold_archives(session_id);
CREATE INDEX IF NOT EXISTS idx_cold_archives_character ON spm_cold_archives(character_id);

-- ======================================================================
-- Dynamic Schema Helper Function for Character Lore Rules
-- Usage: SELECT create_csa_lore_rules_table('luna');
-- ======================================================================
-- Keep in sync with scripts/migrations/001_embedding_spaces_and_lore_scope.sql.
-- session_id/scope are schema-only for now; lore retrieval filtering by
-- session lands in Sprint 4 (OPEN-007).
CREATE OR REPLACE FUNCTION create_csa_lore_rules_table(char_id TEXT)
RETURNS VOID AS $$
DECLARE
    table_name TEXT := 'csa_lore_rules_' || lower(char_id);
    emb_typmod INT;
BEGIN
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            rule_text TEXT NOT NULL,
            rule_type VARCHAR(50) NOT NULL, -- invariant, conditional_trigger, game_over
            rule_embedding vector,
            embedding_space_id SMALLINT,
            status VARCHAR(20) DEFAULT ''active'',
            session_id VARCHAR(255) NOT NULL DEFAULT ''legacy_global'',
            scope VARCHAR(10) NOT NULL DEFAULT ''chat'', -- chat | global
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            fts tsvector GENERATED ALWAYS AS (to_tsvector(''simple'', coalesce(rule_text, ''''))) STORED
        );
    ', table_name);

    -- Lazy upgrades for tables created in older shapes (all idempotent).
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT ''active''', table_name);
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS embedding_space_id SMALLINT', table_name);
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) NOT NULL DEFAULT ''legacy_global''', table_name);
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS scope VARCHAR(10) NOT NULL DEFAULT ''chat''', table_name);

    SELECT atttypmod INTO emb_typmod
      FROM pg_attribute
     WHERE attrelid = to_regclass(table_name)
       AND attname = 'rule_embedding'
       AND NOT attisdropped;
    IF emb_typmod IS NOT NULL AND emb_typmod <> -1 THEN
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN rule_embedding TYPE vector USING rule_embedding::vector',
            table_name);
    END IF;

    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS (to_tsvector(''simple'', coalesce(rule_text, ''''))) STORED',
        table_name);

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_session ON %I (session_id, status, rule_type)',
        table_name, table_name
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_fts ON %I USING GIN (fts)',
        table_name, table_name
    );
END;
$$ LANGUAGE plpgsql;

-- Initialize default demo tables
SELECT create_csa_lore_rules_table('rowan');
SELECT create_csa_lore_rules_table('domino');
SELECT create_csa_lore_rules_table('luna');
SELECT create_csa_lore_rules_table('seamus');
SELECT create_csa_lore_rules_table('arvenia');

-- ======================================================================
-- V0.4 World State Sessions
-- ======================================================================
CREATE TABLE IF NOT EXISTS world_state_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    template_key VARCHAR(255) NOT NULL,
    room_id VARCHAR(255) NOT NULL,
    room_data JSONB NOT NULL,
    action_tick BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, template_key, room_id)
);

CREATE INDEX IF NOT EXISTS idx_world_state_session ON world_state_sessions(session_id, template_key);

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

-- QA F26 (2026-10-06): per-chat settings. First use: narrator mode for scenario cards
-- ('auto' = detect from the card name, 'on', 'off'). Idempotent.

CREATE TABLE IF NOT EXISTS spm_chat_settings (
    session_id    VARCHAR(255) PRIMARY KEY,
    narrator_mode VARCHAR(8) NOT NULL DEFAULT 'auto',   -- auto | on | off
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
