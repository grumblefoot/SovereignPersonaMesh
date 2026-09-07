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
