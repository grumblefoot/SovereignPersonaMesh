"""Brief P2: one config source — .env.example generated from the code, one DB namespace."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "gen_env_example.py"
ENV_EXAMPLE = ROOT / ".env.example"
SEARCH_DIRS = ["proxy", "evennia_world", "core", "config", "scripts", "spm"]

_VAR_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=")


def _generated_vars() -> list[str]:
    return [m.group(1) for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
            if (m := _VAR_LINE.match(line))]


def test_check_mode_passes():
    """The checked-in .env.example matches what the generator produces."""
    r = subprocess.run([sys.executable, str(GEN), "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"--check failed:\n{r.stdout}\n{r.stderr}"


def test_every_generated_var_is_read_in_production_code():
    """No variable in .env.example that nothing reads (the old file's dead vars)."""
    corpus = ""
    dirs = [ROOT / d for d in SEARCH_DIRS if (ROOT / d).is_dir()]
    for d in dirs:
        for py in d.rglob("*.py"):
            corpus += py.read_text(encoding="utf-8")
    # The generator itself lists the names; exclude its own source from the corpus.
    corpus = corpus.replace((GEN).read_text(encoding="utf-8"), "")
    missing = [v for v in _generated_vars() if v not in corpus]
    assert not missing, f"generated vars never read in {SEARCH_DIRS}: {missing}"


def test_file_is_marked_generated():
    head = ENV_EXAMPLE.read_text(encoding="utf-8")[:400]
    assert "GENERATED" in head and "gen_env_example.py" in head


def test_db_defaults_match_code():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "SPM_HOST=127.0.0.1" in text
    assert "SPM_ENGINE_HOST=127.0.0.1" in text
    assert "POSTGRES_PASSWORD=spm_secure_password" in text
    assert "POSTGRES_DB=litellm_postgres" in text
    # Dead vars from the old hand-written file are gone.
    for dead in ("EVENNIA_SECRET_KEY", "EVENNIA_API_URL", "LLM_MAX_CONTEXT_TOKENS",
                 "POSTGRES_MIN_POOL", "POSTGRES_MAX_POOL", "your_secure_password_here"):
        assert dead not in text, f"dead variable {dead} still in .env.example"


def test_engine_db_config_prefers_postgres_env(monkeypatch):
    import evennia_world.app as world_app

    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_HOST",
                 "POSTGRES_PORT", "SPM_DB_USER", "SPM_DB_PASSWORD", "SPM_DB_NAME",
                 "SPM_DB_HOST", "SPM_DB_PORT"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("POSTGRES_USER", "pg_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg_pass")
    monkeypatch.setenv("POSTGRES_DB", "pg_db")
    monkeypatch.setenv("POSTGRES_HOST", "pg.host")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("SPM_DB_USER", "legacy_user")
    monkeypatch.setenv("SPM_DB_PASSWORD", "legacy_pass")
    monkeypatch.setenv("SPM_DB_NAME", "legacy_db")
    monkeypatch.setenv("SPM_DB_HOST", "legacy.host")
    monkeypatch.setenv("SPM_DB_PORT", "1234")

    cfg = world_app._db_config()
    assert cfg == {"user": "pg_user", "password": "pg_pass", "database": "pg_db",
                   "host": "pg.host", "port": 6543}


def test_engine_db_config_falls_back_to_spm_db(monkeypatch):
    import evennia_world.app as world_app

    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_HOST",
                 "POSTGRES_PORT", "SPM_DB_USER", "SPM_DB_PASSWORD", "SPM_DB_NAME",
                 "SPM_DB_HOST", "SPM_DB_PORT"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("SPM_DB_USER", "legacy_user")
    monkeypatch.setenv("SPM_DB_NAME", "legacy_db")
    cfg = world_app._db_config()
    assert cfg["user"] == "legacy_user"
    assert cfg["database"] == "legacy_db"
    # Unset in both namespaces → the historical defaults.
    assert cfg["password"] == "spm_secure_password"
    assert cfg["host"] == "localhost"
    assert cfg["port"] == 5432
