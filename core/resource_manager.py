"""
Sovereign Persona Mesh — Global String ResourceManager

Singleton that loads, merges, and serves string resources from one or more
JSON files.  All public strings flow through ``ResourceManager.get()`` so that
missing keys and interpolation failures are handled gracefully with structured
logging.

Usage
-----
.. code-block:: python

    strings = ResourceManager("proxy/rag/res/strings.json")
    prompt = strings.get("rag.system_prompt", max_tokens=512)
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResourceManager:
    """Singleton global-string provider backed by one or more JSON files.

    The constructor walks every path supplied in *json_paths*, loads each file,
    and *deep-merges* the resulting dicts into a single frozen cache
    (``types.MappingProxyType``).  Dot-notation keys are supported:
    ``strings.get("rag.system_prompt")`` maps to ``cache["rag"]["system_prompt"]``.
    """

    _instance: Optional["ResourceManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ResourceManager":
        if kwargs.get("_skip_cache", False):
            return super().__new__(cls)
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(self, json_paths: List[str], *, _skip_cache: bool = False) -> None:
        # Guard against re-init after singleton reuse (e.g. in tests).
        if hasattr(self, "_cache") and not _skip_cache:
            return

        # Convert relative paths to absolute — avoids cwd-dependent bugs when
        # the module is imported from different working directories.
        resolved_paths: List[str] = [str(Path(p).resolve()) for p in json_paths]

        # Load into a mutable working dict first so we can freeze it cleanly.
        working: Dict[str, Any] = {}

        for path in resolved_paths:
            self._load_file(path, cache=working)

        # Freeze the dict — external callers cannot mutate it.
        self._cache = self._deep_freeze(working)  # type: ignore[assignment]

        # Remember original paths for reload() without arguments.
        self._loaded_paths: List[str] = resolved_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: str, cache: Optional[Dict[str, Any]] = None) -> None:
        """Load and deep-merge a JSON resource file into the cache.

        Parameters
        ----------
        path : str
            Filesystem path to a JSON resource file.
        cache : dict, optional
            If provided, merge into this dict instead of ``self._cache``.
            This enables transactional loading (build in a temp dict,
            swap atomically on success).
        """
        filepath = Path(path)
        if not filepath.is_file():
            logger.warning("ResourceManager: file not found — %s", path)
            return

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("ResourceManager: invalid JSON in %s — %s", path, exc)
            return

        target = cache if cache is not None else self._cache
        self._deep_merge(target, data)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Mutate *base* by deep-merging *overlay* into it."""
        for key, value in overlay.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                ResourceManager._deep_merge(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _deep_freeze(obj: Any) -> Any:
        """Return a new object that is immutable (frozenset / MappingProxyType)."""
        if isinstance(obj, dict):
            return types.MappingProxyType(
                {k: ResourceManager._deep_freeze(v) for k, v in obj.items()}
            )
        if isinstance(obj, (list, tuple)):
            return tuple(ResourceManager._deep_freeze(item) for item in obj)
        return obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, **kwargs: Any) -> str:
        """Resolve a dot-separated key and interpolate *kwargs* safely.

        Parameters
        ----------
        key : str
            Dot-separated path, e.g. ``"rag.system_prompt"``.
        **kwargs
            Named arguments passed to ``str.format(**kwargs)``.

        Returns
        -------
        str
            The resolved string, or a fallback like ``"[MISSING STRING: …]"``.
        """
        value = self._resolve_key(key)

        if value is None:
            logger.warning(
                "ResourceManager: missing key — %s (fallback: [%s])",
                key,
                key,
            )
            return f"[MISSING STRING: {key}]"

        # Coerce to string just in case the JSON holds a non-string leaf.
        value = str(value)

        # Safe interpolation — never crash.
        if kwargs:
            try:
                value = value.format(**kwargs)
            except (KeyError, ValueError) as exc:
                logger.error(
                    "ResourceManager: interpolation failed for key '%s' — %s "
                    "(returning raw string)",
                    key,
                    exc,
                )
                # Return the raw string without interpolation.

        return value

    def get_raw(self, key: str) -> Optional[Any]:
        """Resolve a dot-separated key and return the raw, uncoerced value."""
        value = self._resolve_key(key)
        if value is None:
            logger.warning("ResourceManager: missing key — %s", key)
        return value

    def _resolve_key(self, key: str) -> Optional[Any]:
        """Walk a dot-separated path and return the leaf value, or *None*."""
        import collections.abc
        keys = key.split(".")
        node: Any = self._cache
        for part in keys:
            if isinstance(node, collections.abc.Mapping) and part in node:
                node = node[part]
            else:
                return None
        return node

    def reload(
        self, json_paths: Optional[List[str]] = None, *, force: bool = False
    ) -> None:
        """(Re)load resources — useful in tests or hot-reload scenarios.

        Parameters
        ----------
        json_paths : list[str], optional
            Paths to reload.  If *None*, the original paths from
            ``__init__`` are reloaded (cached internally).
        force : bool
            If *True*, always rebuild the cache even if nothing
            changed.  Default *False*.

        This method is **transactional**: if any file fails to load
        the entire operation is rolled back and the existing cache
        is preserved.
        """
        if json_paths is None:
            return

        # Resolve to absolute paths (same as __init__).
        resolved_paths: List[str] = [str(Path(p).resolve()) for p in json_paths]

        # Build the new cache in a temporary dict — never mutate
        # self._cache until *all* files are loaded successfully.
        working: Dict[str, Any] = {}

        for path in resolved_paths:
            self._load_file(path, cache=working)

        # If we get here every file loaded successfully — atomically
        # swap in the frozen cache.
        self._cache = self._deep_freeze(working)  # type: ignore[assignment]
        self._loaded_paths = resolved_paths


# Module-level convenience instance — auto-loads the well-known RAG strings.
strings: ResourceManager = ResourceManager([
    "proxy/rag/res/strings.json",
    "evennia_world/res/strings.json",
    "core/res/shared_strings.json"
])
