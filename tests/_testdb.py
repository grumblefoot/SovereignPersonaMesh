"""Connection settings for the throwaway PostgreSQL database the test suite runs against.

Tests must never touch the live SPM database (litellm_postgres). conftest.py rebuilds this one
from scripts/init_db.sql at the start of every session.
"""
import os

LIVE_DB_NAME = "litellm_postgres"
TEST_DB_NAME = os.environ.get("SPM_TEST_DB", "spm_test")

TEST_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "spm_user",
    "password": "spm_secure_password",
    "database": TEST_DB_NAME,
}
