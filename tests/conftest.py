"""Shared fixtures for SPM test suite."""
import asyncio
import pytest_asyncio
import asyncpg


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """Create a real asyncpg connection pool for tests.

    IMPORTANT: Must use loop_scope=session in pytest.ini so the same
    event loop is used for both the fixture and the tests.
    """
    loop = asyncio.get_running_loop()
    assert loop is not None, "No event loop running"
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        user="spm_user",
        password="spm_secure_password",
        database="litellm_postgres",
    )
    yield pool
    await pool.close()
