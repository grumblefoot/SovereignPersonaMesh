"""Tests for config/manager.py — lazy defaults, clean fallback, resilient JSON parse, reset."""

import json
import os
import tempfile
import pytest
from unittest.mock import patch

from config.manager import (
    SettingsManager,
    get_settings_manager,
    get_default_settings,
    reset_settings_manager,
)


# ---------------------------------------------------------------------------
# Bug 1: DEFAULT_CONFIG evaluates os.getenv() at import time
# ---------------------------------------------------------------------------

class TestLazyDefaultGetter:
    """get_default_settings() must read env vars at call time, not at import time."""

    def test_lazy_reads_env_at_call_time(self):
        """get_default_settings should honour the current env, not stale import-time values."""
        with patch.dict(os.environ, {"BACKEND_LLM_URL": "http://new-url:9999/v1"}):
            settings = get_default_settings()
            assert settings["BACKEND_LLM_URL"] == "http://new-url:9999/v1"

    def test_default_values_when_env_unset(self):
        """When no env vars are set, get_default_settings returns sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_default_settings()
            assert settings["BACKEND_LLM_URL"] == "http://localhost:8000/v1"
            assert settings["SPM_PROXY_PORT"] == 5050
            assert isinstance(settings["SPM_PROXY_PORT"], int)
            assert settings["SPM_HARDWARE_TIER"] == "SOVEREIGN"
            assert settings["backend_max_tokens"] == 128000
            assert isinstance(settings["backend_max_tokens"], int)

    def test_env_override_takes_precedence(self):
        """Env vars should override default values."""
        with patch.dict(os.environ, {
            "SPM_PROXY_PORT": "9999",
            "SPM_HARDWARE_TIER": "TITAN",
            "BACKEND_MAX_TOKENS": "64000",
        }):
            settings = get_default_settings()
            assert settings["SPM_PROXY_PORT"] == 9999
            assert isinstance(settings["SPM_PROXY_PORT"], int)
            assert settings["SPM_HARDWARE_TIER"] == "TITAN"
            assert settings["backend_max_tokens"] == 64000
            assert isinstance(settings["backend_max_tokens"], int)


# ---------------------------------------------------------------------------
# Bug 2: get_settings() fallback is redundant
# ---------------------------------------------------------------------------

class TestCleanFallback:
    """get_settings() fallback should not redundantly call os.getenv() again."""

    def test_no_redundant_int_cast(self):
        """Values already read from JSON or defaults should not be re-cast through int()."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "SPM_PROXY_PORT": 5050,
                "backend_max_tokens": 128000,
                "BACKEND_LLM_URL": "http://localhost:8000/v1",
                "BACKEND_API_KEY": "",
                "SPM_HARDWARE_TIER": "SOVEREIGN",
                "EVENNIA_LIAISON_URL": "http://localhost:4005",
            }, f)
            f.flush()
            mgr = SettingsManager(config_path=f.name)

        settings = mgr.get_settings()
        assert isinstance(settings["SPM_PROXY_PORT"], int)
        assert isinstance(settings["backend_max_tokens"], int)
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Bug 3: Resilient JSON parse fallback strategies
# ---------------------------------------------------------------------------

class TestResilientJsonParse:
    """When config.json is corrupt, the system must degrade gracefully."""

    def test_corrupt_json_falls_back_to_defaults(self):
        """A corrupt config.json should not crash; defaults should be used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("NOT VALID JSON {{{")
            f.flush()
            mgr = SettingsManager(config_path=f.name)

        settings = mgr.get_settings()
        assert settings["BACKEND_LLM_URL"] == "http://localhost:8000/v1"
        assert settings["SPM_PROXY_PORT"] == 5050
        os.unlink(f.name)

    def test_partial_json_preserves_valid_keys(self):
        """When config.json has some valid keys and some corruption, valid keys should be used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "SPM_HARDWARE_TIER": "TITAN",
                # intentionally omit others — should fall back
            }, f)
            f.flush()
            mgr = SettingsManager(config_path=f.name)

        settings = mgr.get_settings()
        assert settings["SPM_HARDWARE_TIER"] == "TITAN"
        assert settings["BACKEND_LLM_URL"] == "http://localhost:8000/v1"
        os.unlink(f.name)

    def test_empty_config_uses_defaults(self):
        """An empty (or empty-object) config file should yield all defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            f.flush()
            mgr = SettingsManager(config_path=f.name)

        settings = mgr.get_settings()
        assert settings["BACKEND_LLM_URL"] == "http://localhost:8000/v1"
        assert settings["SPM_PROXY_PORT"] == 5050
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Bug 4: reset_settings_manager() for test safety
# ---------------------------------------------------------------------------

class TestResetSettingsManager:
    """reset_settings_manager() must clear the module-level singleton."""

    def test_reset_clears_singleton(self):
        """After reset, get_settings_manager should return a fresh instance."""
        mgr = get_settings_manager()
        reset_settings_manager()

        fresh_mgr = get_settings_manager()
        assert fresh_mgr is not mgr

    def test_reset_is_idempotent(self):
        """Calling reset twice should not raise."""
        reset_settings_manager()
        reset_settings_manager()  # should not raise
        mgr = get_settings_manager()
        assert isinstance(mgr, SettingsManager)

    def test_reset_allows_repeated_get(self):
        """After reset, get_settings_manager works normally."""
        reset_settings_manager()
        mgr = get_settings_manager()
        assert mgr is not None
        assert isinstance(mgr, SettingsManager)
        settings = mgr.get_settings()
        assert isinstance(settings, dict)
