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
    "gated_history_enabled": True,
    # ── Embeddings (embeddings plan phases 1-3, OPEN-002) ──
    # auto | openai_compat | none | fake  (onnx_local is phase 4 and raises).
    # auto resolves to openai_compat for a loopback/LAN base URL and to
    # 'none' for a cloud URL: memories never go to a cloud embedder
    # implicitly (SPRINT_PLAN decision 5 / PRD SLA-2).
    "EMBEDDING_PROVIDER": "auto",
    # Empty string means "use BACKEND_LLM_URL" (the chat backend serves
    # /v1/embeddings on Lemonade, KoboldCpp --embeddingsmodel, llama-server
    # --embedding). A llama.cpp user can point this at a second instance.
    "EMBEDDING_URL": "",
    # Empty string means "use BACKEND_API_KEY".
    "EMBEDDING_API_KEY": "",
    # Provisional default per decision 4 (bake-off pending): Lemonade serves
    # embed-gemma-300m-FLM at 768 dims in its own 'embedding' slot.
    "EMBEDDING_MODEL": "embed-gemma-300m-FLM",
    # 0 = the model's native dimension (locked on first successful embed).
    "EMBEDDING_DIM": 0,
    "EMBEDDING_TIMEOUT_S": 3,
    # Explicit SLA-2 opt-in: lets `auto` pick a non-local EMBEDDING_URL.
    "EMBEDDING_ALLOW_REMOTE": False,
    # ── GM_ACTION extraction (GM actions plan, task A1) ──
    # off | move_only | full. Invalid values fall back to "full" (see
    # _validate_gm_actions_mode) — the safest mode is the pre-feature behaviour
    # boundary, so a typo never silently disables world actions entirely.
    "gm_actions_mode": "full",
    "gm_actions_max_per_turn": 4,
    "gm_actions_max_rooms_per_session": 40,
}

# Integer-typed keys that should always produce int values.
_INT_KEYS = frozenset({
    "SPM_PROXY_PORT",
    "backend_max_tokens",
    "EMBEDDING_DIM",
    "EMBEDDING_TIMEOUT_S",
    "gm_actions_max_per_turn",
    "gm_actions_max_rooms_per_session",
})

# Allowed values for gm_actions_mode; anything else falls back to "full".
_GM_ACTIONS_MODES = frozenset({"off", "move_only", "full"})

# Mapping from config key → env var name (some differ, e.g. backend_max_tokens → BACKEND_MAX_TOKENS).
_ENV_VAR_MAP: Dict[str, str] = {
    "BACKEND_LLM_URL": "BACKEND_LLM_URL",
    "BACKEND_API_KEY": "BACKEND_API_KEY",
    "SPM_PROXY_PORT": "SPM_PROXY_PORT",
    "SPM_HARDWARE_TIER": "SPM_HARDWARE_TIER",
    "EVENNIA_LIAISON_URL": "EVENNIA_LIAISON_URL",
    "backend_max_tokens": "BACKEND_MAX_TOKENS",
    "gated_history_enabled": "GATED_HISTORY_ENABLED",
    "EMBEDDING_PROVIDER": "EMBEDDING_PROVIDER",
    "EMBEDDING_URL": "EMBEDDING_URL",
    "EMBEDDING_API_KEY": "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL": "EMBEDDING_MODEL",
    "EMBEDDING_DIM": "EMBEDDING_DIM",
    "EMBEDDING_TIMEOUT_S": "EMBEDDING_TIMEOUT_S",
    "EMBEDDING_ALLOW_REMOTE": "EMBEDDING_ALLOW_REMOTE",
    "gm_actions_mode": "SPM_GM_ACTIONS_MODE",
    "gm_actions_max_per_turn": "SPM_GM_ACTIONS_MAX_PER_TURN",
    "gm_actions_max_rooms_per_session": "SPM_GM_ACTIONS_MAX_ROOMS",
}


def _validate_gm_actions_mode(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce an invalid gm_actions_mode back to "full" with ONE warning.

    Called from every place that produces a settings dict so a bad value from
    env or config.json can never reach the request path.
    """
    mode = settings.get("gm_actions_mode")
    if mode not in _GM_ACTIONS_MODES:
        logger.warning(
            "[Settings] Invalid gm_actions_mode %r; falling back to 'full' "
            "(allowed: off, move_only, full)",
            mode,
        )
        settings["gm_actions_mode"] = "full"
    return settings


def _build_default_settings(validate: bool = True) -> Dict[str, Any]:
    """Build the settings dict from defaults + env vars (lazy os.getenv)."""
    settings: Dict[str, Any] = {}
    for key, fallback in _DEFAULT_VALUES.items():
        env_key = _ENV_VAR_MAP[key]
        raw = os.getenv(env_key, fallback if fallback != "" else None)
        if key in _INT_KEYS:
            settings[key] = int(raw) if raw is not None else fallback
        else:
            settings[key] = raw if raw is not None else fallback
    if validate:
        _validate_gm_actions_mode(settings)
    return settings


def get_default_settings() -> Dict[str, Any]:
    """Return a fresh settings dict with current env-var overrides.

    This is a lazy getter: it reads os.getenv() at call time so that
    environment changes are always reflected.  It does NOT read config.json.
    """
    return _build_default_settings()


# Deprecated alias for backwards compatibility — prefer get_default_settings().
DEFAULT_CONFIG: Dict[str, Any] = get_default_settings()


class SettingsManager:
    """Manages persistent reading and writing of SPM settings."""

    def __init__(self, config_path: Optional[str] = None):
        # Resolve the default at call time (not definition time) so tests can redirect _CONFIG_PATH.
        self.config_path = config_path or _CONFIG_PATH
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
                # Merge: defaults first, then file values override.  Defaults
                # are built unvalidated so an invalid gm_actions_mode warns
                # exactly once — on the final merged dict below.
                merged = {**_build_default_settings(validate=False), **data}
                # Re-cast known integer keys in case the file stored them as strings.
                for k in _INT_KEYS:
                    if k in merged:
                        merged[k] = int(merged[k])
                return _validate_gm_actions_mode(merged)
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
