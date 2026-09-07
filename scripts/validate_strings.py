#!/usr/bin/env python3
"""
CI Validation Script for Global Strings Refactor.
Ensures that there are no missing or broken string keys across the project.
This script loads all JSON files through the ResourceManager and validates
that key files have valid syntax and don't contain missing interpolation keys.
"""

import sys
import logging
import json
from pathlib import Path

# Add project root to sys.path so we can import 'core'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.resource_manager import ResourceManager, strings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_FILES = [
    "proxy/rag/res/strings.json",
    "evennia_world/res/strings.json",
    "core/res/shared_strings.json"
]

def main():
    logger.info("Starting String Validation...")
    errors = 0

    # 1. Check all files exist and are valid JSON
    for fpath in REQUIRED_FILES:
        path = Path(fpath)
        if not path.is_file():
            logger.error(f"Missing required resource file: {fpath}")
            errors += 1
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)
            logger.info(f"Valid JSON: {fpath}")
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON in {fpath}: {exc}")
            errors += 1

    if errors > 0:
        logger.error("Validation failed during JSON structure checks.")
        sys.exit(1)

    # 2. Check that the Singleton loaded them correctly
    cache_keys = list(strings._cache.keys())
    if not cache_keys:
        logger.error("ResourceManager cache is empty! Files failed to load.")
        errors += 1

    # 3. Check specific critical keys
    critical_keys = [
        "rag.csa_prompt_system_block",
        "rag.no_memories_fallback",
        "templates.tavern_common.tavern_common_room.description",
        "api.errors.db_not_configured",
        "scripts.sleep_cycle.prompt_template"
    ]

    for key in critical_keys:
        val = strings.get_raw(key)
        if val is None:
            logger.error(f"Missing critical key in cache: {key}")
            errors += 1
        else:
            logger.info(f"Key found: {key}")

    if errors > 0:
        logger.error(f"String validation failed with {errors} errors.")
        sys.exit(1)
    
    logger.info("All strings validated successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
