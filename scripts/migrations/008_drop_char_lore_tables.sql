-- Applied only when the owner wipes the database for QA (v0.4); not applied automatically.
--
-- Brief C1b-3: per-character csa_lore_rules_* tables are retired; GM rules
-- live in spm_chat_rules (migration 007). Drop every remaining
-- csa_lore_rules_% table in schema public and the helper function that
-- created them. Idempotent: a second run drops nothing and errors nowhere.

DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT tablename FROM pg_tables
         WHERE schemaname = 'public' AND tablename LIKE 'csa\_lore\_rules\_%'
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS public.%I;', r.tablename);
    END LOOP;
END
$$;

DROP FUNCTION IF EXISTS create_csa_lore_rules_table(TEXT);
