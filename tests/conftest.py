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

import os
import pytest
import config.manager as config_manager
import proxy.backend_client.lemonade_client as lemonade_client_module
from proxy.api.routes import evennia_client
from proxy.api.routes import lemonade_client

# Closed port: requests fail fast instead of reaching the live Evennia (4005) or Lemonade (13305) services.
_UNREACHABLE = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    """Keep tests off the live world engine and LLM server. RUN_LIVE_LLM_TESTS=1 re-enables Lemonade."""
    monkeypatch.setattr(evennia_client, "base_url", f"{_UNREACHABLE}/api/v1")
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        monkeypatch.setattr(lemonade_client_module, "DEFAULT_BASE_URL", f"{_UNREACHABLE}/v1")
        monkeypatch.setattr(lemonade_client, "base_url", f"{_UNREACHABLE}/v1")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point the settings manager at a temp config.json so tests never write the live config/config.json."""
    monkeypatch.setattr(config_manager, "_CONFIG_PATH", str(tmp_path / "config.json"))
    config_manager.reset_settings_manager()
    yield
    config_manager.reset_settings_manager()

@pytest.fixture(autouse=True)
def reset_clients():
    """Clear cached clients between tests to avoid 'Event loop is closed' errors."""
    evennia_client._client = None
    lemonade_client._client = None
    yield
    evennia_client._client = None
    lemonade_client._client = None
