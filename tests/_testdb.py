"""Connection settings for the throwaway PostgreSQL database the test suite runs against.

Tests must never touch a live SPM database ("spm", the default since v0.4, or
"litellm_postgres", the pre-v0.4 default). conftest.py rebuilds this one
from scripts/init_db.sql at the start of every session.
"""
import os

LIVE_DB_NAMES = ("spm", "litellm_postgres")
LIVE_DB_NAME = LIVE_DB_NAMES[1]   # kept for older imports
TEST_DB_NAME = os.environ.get("SPM_TEST_DB", "spm_test")

TEST_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": TEST_DB_NAME,
}
