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

    EXECUTE format('
        CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (episodic_embedding vector_cosine_ops);
    ', 'idx_' || table_name || '_embedding', table_name);
END;
$$ LANGUAGE plpgsql;

-- Initialize default demo tables
SELECT create_csa_memory_table('rowan');
SELECT create_csa_memory_table('domino');
SELECT create_csa_memory_table('luna');
SELECT create_csa_memory_table('seamus');
