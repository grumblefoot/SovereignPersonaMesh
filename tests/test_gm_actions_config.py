"""Tests for the GM_ACTION config keys in config/manager.py (task A1).

Uses temp config paths only — never writes the real config/config.json
(same isolation idiom as tests/test_config.py; conftest's isolated_settings
fixture also redirects _CONFIG_PATH to tmp).
"""

import json
import logging

import config.manager as config_manager
from config.manager import SettingsManager, get_default_settings


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestGmActionsDefaults:
    """Unset env + empty config must yield full / 4 / 40."""

    def test_defaults_from_get_default_settings(self, monkeypatch):
        for var in ("SPM_GM_ACTIONS_MODE", "SPM_GM_ACTIONS_MAX_PER_TURN",
                    "SPM_GM_ACTIONS_MAX_ROOMS"):
            monkeypatch.delenv(var, raising=False)
        settings = get_default_settings()
        assert settings["gm_actions_mode"] == "full"
        assert settings["gm_actions_max_per_turn"] == 4
        assert isinstance(settings["gm_actions_max_per_turn"], int)
        assert settings["gm_actions_max_rooms_per_session"] == 40
        assert isinstance(settings["gm_actions_max_rooms_per_session"], int)

    def test_defaults_from_manager_with_empty_config(self, tmp_path, monkeypatch):
        for var in ("SPM_GM_ACTIONS_MODE", "SPM_GM_ACTIONS_MAX_PER_TURN",
                    "SPM_GM_ACTIONS_MAX_ROOMS"):
            monkeypatch.delenv(var, raising=False)
        cfg = tmp_path / "config.json"
        cfg.write_text("{}")
        mgr = SettingsManager(config_path=str(cfg))
        settings = mgr.get_settings()
        assert settings["gm_actions_mode"] == "full"
        assert settings["gm_actions_max_per_turn"] == 4
        assert settings["gm_actions_max_rooms_per_session"] == 40


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

class TestGmActionsEnvOverrides:
    """All three keys honour their env vars."""

    def test_mode_env_override(self, monkeypatch):
        monkeypatch.setenv("SPM_GM_ACTIONS_MODE", "move_only")
        settings = get_default_settings()
        assert settings["gm_actions_mode"] == "move_only"

    def test_off_mode_is_valid(self, monkeypatch):
        monkeypatch.setenv("SPM_GM_ACTIONS_MODE", "off")
        settings = get_default_settings()
        assert settings["gm_actions_mode"] == "off"

    def test_numeric_env_overrides_are_ints(self, monkeypatch):
        monkeypatch.setenv("SPM_GM_ACTIONS_MAX_PER_TURN", "2")
        monkeypatch.setenv("SPM_GM_ACTIONS_MAX_ROOMS", "10")
        settings = get_default_settings()
        assert settings["gm_actions_max_per_turn"] == 2
        assert isinstance(settings["gm_actions_max_per_turn"], int)
        assert settings["gm_actions_max_rooms_per_session"] == 10
        assert isinstance(settings["gm_actions_max_rooms_per_session"], int)

    def test_env_override_via_manager(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPM_GM_ACTIONS_MODE", "off")
        monkeypatch.setenv("SPM_GM_ACTIONS_MAX_PER_TURN", "1")
        monkeypatch.setenv("SPM_GM_ACTIONS_MAX_ROOMS", "5")
        cfg = tmp_path / "config.json"
        cfg.write_text("{}")
        mgr = SettingsManager(config_path=str(cfg))
        settings = mgr.get_settings()
        assert settings["gm_actions_mode"] == "off"
        assert settings["gm_actions_max_per_turn"] == 1
        assert settings["gm_actions_max_rooms_per_session"] == 5


# ---------------------------------------------------------------------------
# Invalid mode → fallback to "full" with exactly ONE warning
# ---------------------------------------------------------------------------

class TestGmActionsModeValidation:
    """An invalid gm_actions_mode must fall back to 'full' with one warning."""

    def test_invalid_env_mode_falls_back_to_full(self, monkeypatch, caplog):
        monkeypatch.setenv("SPM_GM_ACTIONS_MODE", "banana")
        with caplog.at_level(logging.WARNING, logger="config.manager"):
            settings = get_default_settings()
        assert settings["gm_actions_mode"] == "full"
        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "gm_actions_mode" in r.getMessage()]
        assert len(warnings) == 1

    def test_invalid_file_mode_falls_back_with_single_warning(self, tmp_path, monkeypatch, caplog):
        for var in ("SPM_GM_ACTIONS_MODE",):
            monkeypatch.delenv(var, raising=False)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"gm_actions_mode": "banana"}))
        mgr = SettingsManager(config_path=str(cfg))
        with caplog.at_level(logging.WARNING, logger="config.manager"):
            settings = mgr.get_settings()
        assert settings["gm_actions_mode"] == "full"
        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "gm_actions_mode" in r.getMessage()]
        assert len(warnings) == 1

    def test_valid_modes_pass_through(self, monkeypatch):
        for mode in ("off", "move_only", "full"):
            monkeypatch.setenv("SPM_GM_ACTIONS_MODE", mode)
            assert get_default_settings()["gm_actions_mode"] == mode


# ---------------------------------------------------------------------------
# Numeric strings stored in config.json must come back as ints
# ---------------------------------------------------------------------------

class TestGmActionsNumericCasting:
    """config.json may store the caps as strings; get_settings() must int-cast."""

    def test_numeric_strings_in_config_file(self, tmp_path, monkeypatch):
        for var in ("SPM_GM_ACTIONS_MAX_PER_TURN", "SPM_GM_ACTIONS_MAX_ROOMS"):
            monkeypatch.delenv(var, raising=False)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "gm_actions_max_per_turn": "7",
            "gm_actions_max_rooms_per_session": "99",
        }))
        mgr = SettingsManager(config_path=str(cfg))
        settings = mgr.get_settings()
        assert settings["gm_actions_max_per_turn"] == 7
        assert isinstance(settings["gm_actions_max_per_turn"], int)
        assert settings["gm_actions_max_rooms_per_session"] == 99
        assert isinstance(settings["gm_actions_max_rooms_per_session"], int)
