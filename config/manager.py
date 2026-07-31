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

DEFAULT_CONFIG: Dict[str, Any] = {
    "BACKEND_LLM_URL": os.getenv("BACKEND_LLM_URL", "http://localhost:8000/v1"),
    "BACKEND_API_KEY": os.getenv("BACKEND_API_KEY", os.getenv("SPM_SECRET_KEY", "")),
    "SPM_PROXY_PORT": int(os.getenv("SPM_PROXY_PORT", "5050")),
    "SPM_HARDWARE_TIER": os.getenv("SPM_HARDWARE_TIER", "SOVEREIGN"),
    "EVENNIA_LIAISON_URL": os.getenv("EVENNIA_LIAISON_URL", "http://localhost:4005"),
}


class SettingsManager:
    """Manages persistent reading and writing of SPM settings."""

    def __init__(self, config_path: str = _CONFIG_PATH):
        self.config_path = config_path
        self._ensure_config_file()

    def _ensure_config_file(self) -> None:
        """Create default config.json if missing."""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            self.write_settings(DEFAULT_CONFIG)

    def get_settings(self) -> Dict[str, Any]:
        """Read and return current settings from config.json with env fallbacks."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with defaults
                    merged = {**DEFAULT_CONFIG, **data}
                    return merged
            except Exception as e:
                logger.error(f"[SettingsManager] Failed to read {self.config_path}: {e}")

        # Fallback to env vars or defaults
        return {
            "BACKEND_LLM_URL": os.getenv("BACKEND_LLM_URL", DEFAULT_CONFIG["BACKEND_LLM_URL"]),
            "BACKEND_API_KEY": os.getenv("BACKEND_API_KEY", DEFAULT_CONFIG["BACKEND_API_KEY"]),
            "SPM_PROXY_PORT": int(os.getenv("SPM_PROXY_PORT", DEFAULT_CONFIG["SPM_PROXY_PORT"])),
            "SPM_HARDWARE_TIER": os.getenv("SPM_HARDWARE_TIER", DEFAULT_CONFIG["SPM_HARDWARE_TIER"]),
            "EVENNIA_LIAISON_URL": os.getenv("EVENNIA_LIAISON_URL", DEFAULT_CONFIG["EVENNIA_LIAISON_URL"]),
        }

    def write_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Write new settings to config.json and return updated dict."""
        current = self.get_settings()
        current.update(new_settings)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        logger.info(f"[SettingsManager] Updated settings in {self.config_path}")
        return current

    update = write_settings


_manager_instance: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get singleton SettingsManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SettingsManager()
    return _manager_instance
