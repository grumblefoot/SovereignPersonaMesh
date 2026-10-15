"""Tests for the Docker bundle: compose file (name, grace periods) and the
one-click scripts in scripts/docker/ (both Windows and Linux/macOS forms).

These are pure file-content checks: nothing here starts, stops or removes any
container, image or volume.
"""

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "config" / "docker-compose.yml"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "docker"

SCRIPT_NAMES = ["start", "stop", "status", "logs", "update"]


# ---------------------------------------------------------------- compose file

def _compose_text() -> str:
    assert COMPOSE_FILE.is_file(), f"missing {COMPOSE_FILE}"
    return COMPOSE_FILE.read_text(encoding="utf-8")


def test_compose_has_project_name_spm():
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    text = _compose_text()
    if yaml is not None:
        data = yaml.safe_load(text)
        assert data.get("name") == "spm", "compose file must set top-level 'name: spm'"
    else:
        assert re.search(r"(?m)^name:\s*spm\s*$", text), "compose file must set top-level 'name: spm'"


def test_compose_stop_grace_periods():
    text = _compose_text()
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(text)
        services = data.get("services", {})
        assert services["postgres"].get("stop_grace_period") == "30s"
        assert services["spm"].get("stop_grace_period") == "20s"
    else:
        # Simple text checks: each service block declares its grace period.
        assert re.search(r"(?m)^\s+stop_grace_period:\s*30s\s*$", text)
        assert re.search(r"(?m)^\s+stop_grace_period:\s*20s\s*$", text)
    # Both services must survive a reboot unless the user stopped them.
    assert text.count("restart: unless-stopped") >= 2


def test_compose_spm_service_has_no_backend_llm_url():
    """Brief FX-C: the compose `environment:` block must not duplicate
    BACKEND_LLM_URL — env_file (the repo's .env) provides it, and SPM's own
    default plus the container's localhost rewrite cover the no-.env case."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    text = _compose_text()
    if yaml is not None:
        data = yaml.safe_load(text)
        env = data["services"]["spm"].get("environment") or {}
        assert "BACKEND_LLM_URL" not in env
    else:
        spm_block = text[text.index("  spm:"):]
        env_block = spm_block[spm_block.index("environment:"):spm_block.index("extra_hosts:")]
        # Strip comment lines so the explanatory note cannot trip the check.
        code = "\n".join(l for l in env_block.splitlines() if not l.lstrip().startswith("#"))
        assert "BACKEND_LLM_URL" not in code


# ---------------------------------------------------------------- scripts

def test_scripts_dir_exists():
    assert SCRIPTS_DIR.is_dir(), f"missing {SCRIPTS_DIR}"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_exists_in_both_forms(name):
    assert (SCRIPTS_DIR / f"{name}.cmd").is_file(), f"missing scripts/docker/{name}.cmd"
    assert (SCRIPTS_DIR / f"{name}.sh").is_file(), f"missing scripts/docker/{name}.sh"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_references_compose_file_relative_to_itself(name):
    sh = (SCRIPTS_DIR / f"{name}.sh").read_text(encoding="utf-8")
    cmd = (SCRIPTS_DIR / f"{name}.cmd").read_text(encoding="utf-8")
    # .sh: resolves ../.. from the script's own directory.
    assert 'dirname "$0"' in sh, f"{name}.sh must resolve paths from its own location"
    assert "/config/docker-compose.yml" in sh, f"{name}.sh must point at config/docker-compose.yml"
    # .cmd: resolves ..\..\ from %~dp0 (the script's own directory).
    assert "%~dp0" in cmd, f"{name}.cmd must resolve paths from its own location"
    assert re.search(r"%~dp0\.\.\\\.\.\\config\\docker-compose\.yml", cmd), (
        f"{name}.cmd must point at ..\\..\\config\\docker-compose.yml"
    )


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_scripts_pass_repo_env_file_to_compose(name):
    """Brief FX-C: every script passes `--env-file <repo>/.env` when that file
    exists, so ${POSTGRES_*} interpolation and the compose env_file agree."""
    sh = (SCRIPTS_DIR / f"{name}.sh").read_text(encoding="utf-8")
    cmd = (SCRIPTS_DIR / f"{name}.cmd").read_text(encoding="utf-8")
    assert "--env-file" in sh and ".env" in sh, f"{name}.sh must pass --env-file <repo>/.env"
    assert "--env-file" in cmd and ".env" in cmd, f"{name}.cmd must pass --env-file <repo>\\.env"


@pytest.mark.parametrize("ext", [".cmd", ".sh"])
def test_stop_never_destroys(ext):
    stop = (SCRIPTS_DIR / f"stop{ext}").read_text(encoding="utf-8")
    # Check every actual compose invocation (not incidental text like `command -v`):
    # the stop script may only run `stop` — never `down`, never `-v`. An optional
    # --env-file expansion (brief FX-C) may sit between the command and -f.
    invocations = []
    for line in stop.splitlines():
        line = line.strip()
        if re.match(r'^(\$COMPOSE|docker compose)(\s+(\$\{ENV_FILE_ARGS.*?\}|%ENV_FILE_ARGS%))?\s+-f\b', line):
            line = re.sub(r'-f\s+"?\$?\{?COMPOSE_FILE\}?"?', "", line)
            invocations.append(line)
    assert invocations, f"stop{ext} should contain a compose invocation"
    for inv in invocations:
        assert "down" not in inv, f"stop{ext} must not run 'down' (it would invite data loss): {inv}"
        assert "-v" not in inv, f"stop{ext} must not pass -v (it would remove volumes): {inv}"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_sh_scripts_are_executable_with_shebang(name):
    path = SCRIPTS_DIR / f"{name}.sh"
    raw = path.read_bytes()
    assert raw.startswith(b"#!"), f"{name}.sh must start with a shebang"
    if os.name != "nt":
        assert os.access(path, os.X_OK), f"{name}.sh must be executable (chmod +x)"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_cmd_scripts_use_crlf(name):
    raw = (SCRIPTS_DIR / f"{name}.cmd").read_bytes()
    assert b"\r\n" in raw, f"{name}.cmd must use CRLF line endings"
    # Every LF must be preceded by CR (no stray bare-LF lines).
    without_crlf = raw.replace(b"\r\n", b"")
    assert b"\n" not in without_crlf, f"{name}.cmd must not contain bare LF line endings"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_sh_scripts_fall_back_to_podman(name):
    sh = (SCRIPTS_DIR / f"{name}.sh").read_text(encoding="utf-8")
    assert "docker compose" in sh and "podman compose" in sh, (
        f"{name}.sh must try 'docker compose' then 'podman compose'"
    )
