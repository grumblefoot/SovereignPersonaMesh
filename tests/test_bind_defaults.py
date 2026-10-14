"""P1: safe network defaults.

The admin dashboard has no authentication, so both entry points must bind to
127.0.0.1 unless the user explicitly asks for a wider interface. These tests
execute each module's `__main__` block with `uvicorn.run` patched and assert
which host the server would be started on.
"""

import os
import runpy

import pytest

# runpy re-executes an already-imported module as __main__ (harmless here).
pytestmark = pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")


def _run_main(module_name, monkeypatch, env):
    """Run `module_name` as __main__ with uvicorn.run and the file-handler
    side effect patched; return the kwargs uvicorn.run was called with."""
    calls = []

    import uvicorn

    def fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    # proxy.main's __main__ block opens a rotating log file; neutralise it.
    from proxy.core import logger as proxy_logger
    monkeypatch.setattr(proxy_logger, "attach_rotating_file_handler", lambda *a, **k: None)

    for var in ("SPM_HOST", "SPM_PORT", "SPM_ENGINE_HOST", "SPM_ENGINE_PORT", "SPM_RELOAD"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    runpy.run_module(module_name, run_name="__main__", alter_sys=False)

    assert len(calls) == 1, f"{module_name} should call uvicorn.run exactly once"
    return calls[0]["kwargs"]


def test_proxy_main_default_host_is_loopback(monkeypatch):
    kwargs = _run_main("proxy.main", monkeypatch, {})
    assert kwargs["host"] == "127.0.0.1"


def test_engine_main_default_host_is_loopback(monkeypatch):
    kwargs = _run_main("evennia_world.app", monkeypatch, {})
    assert kwargs["host"] == "127.0.0.1"


def test_proxy_main_honours_spm_host(monkeypatch):
    kwargs = _run_main("proxy.main", monkeypatch, {"SPM_HOST": "0.0.0.0"})
    assert kwargs["host"] == "0.0.0.0"


def test_engine_main_honours_spm_engine_host(monkeypatch):
    kwargs = _run_main("evennia_world.app", monkeypatch, {"SPM_ENGINE_HOST": "0.0.0.0"})
    assert kwargs["host"] == "0.0.0.0"
