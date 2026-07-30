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

-- Dynamic Schema Helper Function for Character Subagent Episodic Memory
-- Usage: SELECT create_csa_memory_table('luna');
CREATE OR REPLACE FUNCTION create_csa_memory_table(char_id TEXT)
RETURNS VOID AS $$
DECLARE
    table_name TEXT := 'csa_memory_' || lower(char_id);
BEGIN
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL DEFAULT ''default_session'',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            sensory_input TEXT NOT NULL,
            inner_monologue TEXT,
            episodic_embedding VECTOR(3584),
            importance_score INT DEFAULT 5,
            is_core_memory BOOLEAN DEFAULT FALSE,
            is_subjective BOOLEAN DEFAULT TRUE,
            access_count INT DEFAULT 1,
            last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    ', table_name);

    -- NOTE: pgvector index operators (HNSW/IVFFlat) are capped at 2000 dimensions.
    -- For 3584-dim embeddings we use brute-force cosine search via the <=> operator.
    -- A partial B-tree index on (is_core_memory, timestamp) speeds up sleep-cycle queries.
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_core_time ON %I (is_core_memory, timestamp)',
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
