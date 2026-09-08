"""
Dynamic Settings Manager for Sovereign Persona Mesh (SPM).

Reads and writes runtime configuration to config/config.json with environment variable fallbacks.
Supports hot-reloading BACKEND_LLM_URL, BACKEND_API_KEY, and SPM_PROXY_PORT.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.json"
)

# Static defaults (no env var evaluation here — env reads happen lazily).
_DEFAULT_VALUES: Dict[str, Any] = {
    "BACKEND_LLM_URL": "http://localhost:8000/v1",
    "BACKEND_API_KEY": "",
    "SPM_PROXY_PORT": 5050,
    "SPM_HARDWARE_TIER": "SOVEREIGN",
    "EVENNIA_LIAISON_URL": "http://localhost:4005",
    "backend_max_tokens": 128000,
}

# Integer-typed keys that should always produce int values.
_INT_KEYS = frozenset({"SPM_PROXY_PORT", "backend_max_tokens"})

# Mapping from config key → env var name (some differ, e.g. backend_max_tokens → BACKEND_MAX_TOKENS).
_ENV_VAR_MAP: Dict[str, str] = {
    "BACKEND_LLM_URL": "BACKEND_LLM_URL",
    "BACKEND_API_KEY": "BACKEND_API_KEY",
    "SPM_PROXY_PORT": "SPM_PROXY_PORT",
    "SPM_HARDWARE_TIER": "SPM_HARDWARE_TIER",
    "EVENNIA_LIAISON_URL": "EVENNIA_LIAISON_URL",
    "backend_max_tokens": "BACKEND_MAX_TOKENS",
}


def get_default_settings() -> Dict[str, Any]:
    """Return a fresh settings dict with current env-var overrides.

    This is a lazy getter: it reads os.getenv() at call time so that
    environment changes are always reflected.  It does NOT read config.json.
    """
    settings: Dict[str, Any] = {}
    for key, fallback in _DEFAULT_VALUES.items():
        env_key = _ENV_VAR_MAP[key]
        raw = os.getenv(env_key, fallback if fallback != "" else None)
        if key in _INT_KEYS:
            settings[key] = int(raw) if raw is not None else fallback
        else:
            settings[key] = raw if raw is not None else fallback
    return settings


# Deprecated alias for backwards compatibility — prefer get_default_settings().
DEFAULT_CONFIG: Dict[str, Any] = get_default_settings()


class SettingsManager:
    """Manages persistent reading and writing of SPM settings."""

    def __init__(self, config_path: str = _CONFIG_PATH):
        self.config_path = config_path
        self._ensure_config_file()

    def _ensure_config_file(self) -> None:
        """Create default config.json if missing."""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            self.write_settings(get_default_settings())

    def get_settings(self) -> Dict[str, Any]:
        """Read and return current settings from config.json with env fallbacks.

        Merge strategy (most-resilient-first):
        1. Try to load config.json → merge with defaults (file overrides defaults).
        2. If loading fails (missing file, corrupt JSON, I/O error) → use get_default_settings()
           which reads env vars at call time and applies typed defaults.

        This avoids the old behaviour where a corrupt config silently fell through
        to a second os.getenv() call that could diverge from the file's own defaults.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Merge: defaults first, then file values override.
                merged = {**get_default_settings(), **data}
                # Re-cast known integer keys in case the file stored them as strings.
                for k in _INT_KEYS:
                    if k in merged:
                        merged[k] = int(merged[k])
                return merged
            except (json.JSONDecodeError, ValueError, TypeError, OSError) as e:
                logger.warning(
                    "[SettingsManager] Failed to parse %s, falling back to env defaults: %s",
                    self.config_path,
                    e,
                )

        # Fallback: fresh env-var read + typed defaults.
        return get_default_settings()

    def write_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Write new settings to config.json and return updated dict."""
        current = self.get_settings()
        current.update(new_settings)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        logger.info("[SettingsManager] Updated settings in %s", self.config_path)
        return current

    update = write_settings


_manager_instance: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get singleton SettingsManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SettingsManager()
    return _manager_instance


def reset_settings_manager() -> None:
    """Clear the module-level singleton for test safety.

    Call this between tests that mutate the singleton to avoid cross-test
    state leakage.
    """
    global _manager_instance
    _manager_instance = None
