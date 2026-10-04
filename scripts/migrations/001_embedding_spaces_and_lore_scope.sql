-- 001: Embedding spaces, dimension-free vectors, full-text fallback columns,
--      and the lore session/scope columns (Sprint 1 Track B).
--
-- Combines the embeddings plan (docs/plans/embeddings.md, phase 2) with the
-- lore-scope schema from docs/plans/gm_actions_and_lore_scope.md Part B §B.4,
-- per SPRINT_PLAN reconciliation §1.5 (one migration touches csa_lore_rules_*).
--
-- Idempotent: every statement is IF NOT EXISTS / CREATE OR REPLACE / guarded
-- by a catalog check, and the final DO block sweeps ALL existing csa_% tables
-- dynamically, so re-running is always safe. Live vectors were all NULL after
-- the 2026-10-03 reset, so the VECTOR(3584) -> vector change converts no data.
--
-- Applied by scripts/apply_migrations.py (tracked in spm_schema_migrations).
-- scripts/init_db.sql carries the same shape for fresh installs.

CREATE EXTENSION IF NOT EXISTS vector;

-- ──────────────────────────────────────────────────────────────────────
-- Embedding space registry: which (provider, model, dim) produced a vector.
-- Vectors from different spaces are never compared, even at equal dims.
-- csa_* tables reference spm_embedding_spaces.id by value only: they are
-- created dynamically per character, so the FK is enforced by convention,
-- not by a constraint.
-- ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS spm_embedding_spaces (
    id SMALLSERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(255) NOT NULL,
    dim INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider, model, dim)
);

-- ──────────────────────────────────────────────────────────────────────
-- Memory tables: new shape + lazy in-place upgrades.
-- The helper runs on every table access, so any table created by an older
-- build upgrades the next time it is touched; the DO block below upgrades
-- all existing ones right now.
-- ──────────────────────────────────────────────────────────────────────
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

    -- VECTOR(3584) -> dimension-free vector (embeddings plan: the active
    -- space decides the dimension; stored vectors were reset to NULL).
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
    -- is embeddings plan phase 6.
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_core_time ON %I (is_core_memory, timestamp)',
        table_name, table_name);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_fts ON %I USING GIN (fts)',
        table_name, table_name);
END;
$$ LANGUAGE plpgsql;

-- ──────────────────────────────────────────────────────────────────────
-- Lore tables: new shape + lazy upgrades. session_id/scope are schema-only
-- here; retrieval filtering by lore session lands in Sprint 4 (OPEN-007).
-- ──────────────────────────────────────────────────────────────────────
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
        table_name, table_name);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_fts ON %I USING GIN (fts)',
        table_name, table_name);
END;
$$ LANGUAGE plpgsql;

-- ──────────────────────────────────────────────────────────────────────
-- Upgrade ALL existing per-character tables now (dynamic: the tables are
-- created on demand, so the list cannot be hard-coded). Re-calling the
-- helpers applies the guarded ALTERs above to each one.
-- ──────────────────────────────────────────────────────────────────────
DO $$
DECLARE
    t RECORD;
BEGIN
    FOR t IN SELECT tablename FROM pg_tables
              WHERE schemaname = 'public' AND tablename LIKE 'csa\_memory\_%' ESCAPE '\'
    LOOP
        PERFORM create_csa_memory_table(substring(t.tablename FROM length('csa_memory_') + 1));
    END LOOP;
    FOR t IN SELECT tablename FROM pg_tables
              WHERE schemaname = 'public' AND tablename LIKE 'csa\_lore\_rules\_%' ESCAPE '\'
    LOOP
        PERFORM create_csa_lore_rules_table(substring(t.tablename FROM length('csa_lore_rules_') + 1));
    END LOOP;
END;
$$;
