-- QA 2026-10-13 (Hermes review B1-B3): fingerprint of the player's exact message per turn.
-- A later responder in a group turn reuses the recorded perception only when the message is
-- unchanged; comparing rendered feeds failed for muffled rows and partial edits. Idempotent.

ALTER TABLE spm_perception ADD COLUMN IF NOT EXISTS action_digest CHAR(16);
