-- Brief A1: one rules table for all chats replaces the per-character csa_lore_rules_*
-- tables for retrieval. Idempotent: creates spm_chat_rules, then copies chat-scoped
-- rows from every table named csa_lore_rules_%:
--   level = 'character_internal', character_id = the table suffix, source = 'migrated'.
-- Canon rows (scope = 'global') and legacy_global rows are skipped (counted in a
-- NOTICE). rule_type 'trigger' becomes 'conditional_trigger'; anything else outside
-- the allowed list becomes 'invariant'. A row whose (session_id, character_id,
-- rule_text) already exists in spm_chat_rules is not copied again, so re-running the
-- migration adds nothing. Old tables that predate the scope/session_id/status columns
-- are skipped with a NOTICE.

CREATE TABLE IF NOT EXISTS spm_chat_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('world','character_internal','character_external')),
    character_id VARCHAR(255),
    rule_text TEXT NOT NULL,
    rule_type VARCHAR(24) NOT NULL CHECK (rule_type IN ('invariant','conditional_trigger','game_over')),
    status VARCHAR(10) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','active')),
    source VARCHAR(12) NOT NULL DEFAULT 'extracted' CHECK (source IN ('extracted','manual','migrated')),
    rule_embedding vector,
    embedding_space_id SMALLINT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (level = 'world' OR character_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_spm_chat_rules_session ON spm_chat_rules (session_id, status);

DO $$
DECLARE
    t           RECORD;
    v_char_id   TEXT;
    v_cols      INT;
    v_copied    INT;
    v_skipped   INT;
    v_total     INT := 0;
BEGIN
    FOR t IN SELECT tablename FROM pg_tables
              WHERE schemaname = 'public' AND tablename LIKE 'csa_lore_rules_%'
              ORDER BY tablename
    LOOP
        -- Old shapes without the OPEN-007 columns cannot be attributed to a chat.
        SELECT COUNT(*) INTO v_cols
          FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = t.tablename
           AND column_name IN ('scope', 'session_id', 'status');
        IF v_cols < 3 THEN
            RAISE NOTICE 'migration 007: skipping % (missing scope/session_id/status columns)',
                t.tablename;
            CONTINUE;
        END IF;

        v_char_id := replace(t.tablename, 'csa_lore_rules_', '');

        EXECUTE format($f$
            WITH src AS (
                SELECT DISTINCT
                       r.session_id,
                       r.rule_text,
                       CASE WHEN r.rule_type = 'trigger' THEN 'conditional_trigger'
                            WHEN r.rule_type IN ('invariant', 'conditional_trigger', 'game_over')
                                 THEN r.rule_type
                            ELSE 'invariant' END AS rule_type,
                       CASE WHEN r.status IN ('pending', 'active') THEN r.status
                            ELSE 'pending' END AS status,   -- NULL/odd values: re-review
                       r.rule_embedding,
                       r.embedding_space_id,
                       r.created_at
                  FROM %I r
                 WHERE r.scope = 'chat' AND r.session_id <> 'legacy_global'
            ),
            ins AS (
                INSERT INTO spm_chat_rules
                    (session_id, level, character_id, rule_text, rule_type, status,
                     source, rule_embedding, embedding_space_id, created_at)
                SELECT s.session_id, 'character_internal', $1, s.rule_text, s.rule_type,
                       s.status, 'migrated', s.rule_embedding, s.embedding_space_id,
                       s.created_at
                  FROM src s
                 WHERE NOT EXISTS (
                     SELECT 1 FROM spm_chat_rules x
                      WHERE x.session_id = s.session_id
                        AND x.character_id = $1
                        AND x.rule_text = s.rule_text)
                RETURNING 1
            )
            SELECT COUNT(*) FROM ins
        $f$, t.tablename) INTO v_copied USING v_char_id;

        EXECUTE format($f$
            SELECT COUNT(*) FROM %I
             WHERE scope = 'global' OR session_id = 'legacy_global'
        $f$, t.tablename) INTO v_skipped;

        v_total := v_total + v_copied;
        IF v_copied > 0 OR v_skipped > 0 THEN
            RAISE NOTICE 'migration 007: % rows copied, % canon/legacy rows skipped from %',
                v_copied, v_skipped, t.tablename;
        END IF;
    END LOOP;

    RAISE NOTICE 'migration 007: % total rows copied into spm_chat_rules', v_total;
END;
$$ LANGUAGE plpgsql;
