"""Tests for core.resource_manager.ResourceManager."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from core.resource_manager import ResourceManager


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def strings_json(tmp_path: Path) -> Path:
    """Write a small strings JSON and return the path."""
    data = {
        "rag": {
            "system_prompt": "You are an AI named {name}.",
            "greeting": "Hello, friend.",
        },
        "gm": {
            "scene_start": "SCENE START in {location}.",
        },
        "fallback_greeting": "Hi there.",
    }
    path = tmp_path / "strings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def rm(strings_json: Path) -> ResourceManager:
    """Return a fresh ResourceManager (bypasses singleton cache)."""
    return ResourceManager([str(strings_json)], _skip_cache=True)


# ------------------------------------------------------------------ #
#  Tests — loading
# ------------------------------------------------------------------ #

def test_loads_and_merges_json(rm: ResourceManager) -> None:
    """ResourceManager loads a JSON file and makes all keys accessible."""
    assert rm.get("rag.system_prompt") == "You are an AI named {name}."
    assert rm.get("rag.greeting") == "Hello, friend."
    assert rm.get("gm.scene_start") == "SCENE START in {location}."
    assert rm.get("fallback_greeting") == "Hi there."


def test_loads_multiple_files(tmp_path: Path) -> None:
    """Multiple JSON files are deep-merged."""
    file_a = tmp_path / "a.json"
    file_a.write_text(json.dumps({"rag": {"greeting": "Aloha"}}), encoding="utf-8")

    file_b = tmp_path / "b.json"
    file_b.write_text(json.dumps({"rag": {"system_prompt": "Be nice."}}), encoding="utf-8")

    rm = ResourceManager([str(file_a), str(file_b)], _skip_cache=True)

    # file_b's key is new; file_a's key still exists.
    assert rm.get("rag.system_prompt") == "Be nice."
    assert rm.get("rag.greeting") == "Aloha"


def test_missing_file_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Loading a non-existent file does NOT raise — it logs a WARNING."""
    with caplog.at_level(logging.WARNING):
        rm = ResourceManager(["/nonexistent/path.json"], _skip_cache=True)

    assert any("file not found" in record.message for record in caplog.records)


def test_invalid_json_logs_error(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Malformed JSON does NOT raise — it logs an ERROR."""
    bad = tmp_path / "bad.json"
    bad.write_text("{this is not valid json", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        rm = ResourceManager([str(bad)], _skip_cache=True)

    assert any("invalid JSON" in record.message for record in caplog.records)


# ------------------------------------------------------------------ #
#  Tests — safe lookup
# ------------------------------------------------------------------ #

def test_get_returns_correct_string(rm: ResourceManager) -> None:
    """Dot-notation lookup returns the exact stored string."""
    assert rm.get("rag.greeting") == "Hello, friend."
    assert rm.get("fallback_greeting") == "Hi there."


def test_nested_dot_notation(rm: ResourceManager) -> None:
    """Deeply nested keys are resolved correctly."""
    assert rm.get("rag.system_prompt") == "You are an AI named {name}."
    assert rm.get("gm.scene_start") == "SCENE START in {location}."


# ------------------------------------------------------------------ #
#  Tests — missing key behaviour
# ------------------------------------------------------------------ #

def test_missing_key_logs_warning_and_returns_fallback(
    rm: ResourceManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-existent key logs WARNING and returns [MISSING STRING: …]."""
    with caplog.at_level(logging.WARNING):
        result = rm.get("does.not.exist")

    assert result == "[MISSING STRING: does.not.exist]"
    assert any("missing key" in record.message for record in caplog.records)


def test_missing_key_partial_path(rm: ResourceManager) -> None:
    """Missing intermediate segment returns fallback."""
    result = rm.get("rag.nonexistent.greeting")
    assert result == "[MISSING STRING: rag.nonexistent.greeting]"


# ------------------------------------------------------------------ #
#  Tests — interpolation safety
# ------------------------------------------------------------------ #

def test_interpolation_succeeds(rm: ResourceManager) -> None:
    """Valid kwargs are interpolated."""
    result = rm.get("rag.system_prompt", name="Lemonaid")
    assert result == "You are an AI named Lemonaid."

    result = rm.get("gm.scene_start", location="The Cellar")
    assert result == "SCENE START in The Cellar."


def test_interpolation_missing_kwargs_returns_raw_string(
    rm: ResourceManager, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing interpolation arguments do NOT crash — raw string returned."""
    with caplog.at_level(logging.ERROR):
        result = rm.get("rag.system_prompt", dummy=True)  # {name} not supplied

    assert "{name}" in result  # raw, un-interpolated
    assert any("interpolation failed" in record.message for record in caplog.records)


def test_interpolation_extra_kwargs_ignored(rm: ResourceManager) -> None:
    """Extra kwargs that don't match format slots are ignored."""
    result = rm.get("rag.greeting", extra="ignored")
    assert result == "Hello, friend."


# ------------------------------------------------------------------ #
#  Tests — reload() transactional behavior (Bug #1)
# ------------------------------------------------------------------ #


@pytest.fixture()
def two_files(tmp_path: Path):
    """Two JSON files; second one will be swapped to invalid JSON."""
    good_a = tmp_path / "a.json"
    good_a.write_text(json.dumps({"section": {"key": "value_a"}}), encoding="utf-8")
    good_b = tmp_path / "b.json"
    good_b.write_text(json.dumps({"section": {"key": "value_b"}}), encoding="utf-8")
    bad_b = tmp_path / "bad_b.json"
    bad_b.write_text("NOT VALID JSON {{{", encoding="utf-8")
    return str(good_a), str(good_b), str(bad_b)


def test_reload_partial_failure_preserves_cache(
    two_files: tuple[str, str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """If reload() fails halfway, the original cache must be preserved (transactional).

    1. Load files a.json and b.json successfully.
    2. Call reload() with a.json and a *bad* JSON file.
    3. The original data from step 1 should still be present.
    """
    a_path, b_path, bad_path = two_files
    rm = ResourceManager([a_path, b_path], _skip_cache=True)
    assert rm.get("section.key") == "value_b"  # b merges on top of a

    with caplog.at_level(logging.ERROR):
        rm.reload([a_path, bad_path])

    # Bug #1: without transactional reload, _cache.clear() already happened,
    # so the data from a.json is gone and section.key is missing.
    assert rm.get("section.key") == "value_a"  # should still have a's data


# ------------------------------------------------------------------ #
#  Tests — reload() force parameter (Bug #2)
# ------------------------------------------------------------------ #


def test_reload_accepts_force_argument(rm: ResourceManager, tmp_path: Path) -> None:
    """reload(json_paths, force=True) should accept the force keyword."""
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps({"new_key": "new_value"}), encoding="utf-8")

    # Should not raise TypeError — force must be a valid kwarg.
    rm.reload([str(new_path)], force=True)
    assert rm.get("new_key") == "new_value"


# ------------------------------------------------------------------ #
#  Tests — singleton uses absolute paths (Bug #3)
# ------------------------------------------------------------------ #


def test_reload_with_absolute_paths(tmp_path: Path) -> None:
    """Paths inside ResourceManager should be resolved to absolute paths
    so the singleton works regardless of cwd.

    1. Create a JSON file in tmp_path.
    2. Instantiate a ResourceManager with a *relative* path.
    3. Verify it resolves correctly (uses absolute paths internally).
    """
    strings_file = tmp_path / "absolute_test.json"
    strings_file.write_text(
        json.dumps({"singleton_test": {"flag": "ok"}}), encoding="utf-8"
    )

    # Save cwd, change to tmp_path, use relative path from there.
    import os

    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        relative_path = "absolute_test.json"
        rm = ResourceManager([relative_path], _skip_cache=True)
        assert rm.get("singleton_test.flag") == "ok"
    finally:
        os.chdir(cwd)


def test_skip_cache_bypasses_singleton(rm: ResourceManager) -> None:
    """Two instances with _skip_cache=True must be different objects."""
    second = ResourceManager([], _skip_cache=True)
    assert second is not rm
    assert isinstance(second, ResourceManager)
