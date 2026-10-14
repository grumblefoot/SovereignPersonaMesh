"""Brief W1b: the detected context window drives every prompt (OPEN-024).

Rig pattern from tests/test_gating_leaks.py: the real proxy runs ASGI-mounted with a
faked LLM that captures every call; backend_profile.window_for is patched so no probe
ever leaves the process (8,192 'reported', like the LM Studio incident that started this).

  1 a long history is trimmed to fit the detected window (max_tokens + prompt estimate)
  2 a virtual model id reaches window_for and the turn still works
  3 native mode with a tight window runs the turn as 'none' (thinking off)
  4 GET /admin/api/v1/backend/window reports tokens/source/override/effective
"""
import itertools
import time
import uuid as _uuid

import httpx
import pytest
from starlette.testclient import TestClient

import evennia_world.app as world_app
from evennia_world.hybrid_builder import HybridWorldBuilder
from proxy.main import app as proxy_app

_counter = itertools.count(1)
_RUN = _uuid.uuid4().hex[:8]


class Rig:
    def __init__(self, client, engine, calls):
        self.client = client            # proxy TestClient
        self.engine = engine            # ENGINE TestClient (same module state)
        self.calls = calls              # captured LLM calls: one dict of kwargs each
        self.chat_id = f"win{_RUN}{next(_counter)}"
        self.history = []               # accumulated user/assistant log (ST resends it)

    @property
    def session_id(self):
        return f"st_chat_{self.chat_id}"

    def post(self, messages, model="Gemma-4-26B-A4B-it"):
        """One chat turn; returns the response. (The gating rig polls the perception
        rows afterwards; these tests only read the captured LLM call.)"""
        payload = [m for m in messages if m.get("role") == "system"] \
            + self.history + [m for m in messages if m.get("role") != "system"]
        self.history = self.history + [m for m in messages if m.get("role") != "system"] \
            + [{"role": "assistant", "content": "A measured reply."}]
        r = self.client.post("/v1/chat/completions",
                             headers={"X-SPM-Chat-ID": self.chat_id},
                             json={"model": model, "stream": True, "messages": payload})
        assert r.status_code == 200, r.text
        return r, payload

    def seed(self, template="dungeon_cellar", placements=()):
        r = self.engine.post("/api/v1/world/configure", json={
            "template_key": template, "session_id": self.session_id,
            "placements": [{"character_id": c, "room_id": room} for c, room in placements]})
        assert r.status_code == 200, r.text


@pytest.fixture
def rig(monkeypatch):
    # Same isolation as tests/test_gating_leaks.py: memory-only engine, faked LLM.
    monkeypatch.setenv("SPM_WORLD_DB", "0")
    st = world_app.app_state
    st._db_pool = None
    st.current_world = {}
    st.session_worlds = {}
    st.session_ticks = {}
    st.session_turn_ticks = {}
    st.idempotency_seen = {}
    st.session_edges = {}
    st.mutation_log = {}
    world_app.world_builder = HybridWorldBuilder()
    if not hasattr(world_app.app.state, "start_time"):
        world_app.app.state.start_time = 0.0

    import proxy.api.routes as routes
    calls = []

    async def fake_stream(*args, **kwargs):
        calls.append(dict(kwargs))
        yield "A measured reply."

    monkeypatch.setattr(routes.lemonade_client, "generate_stream", fake_stream)
    monkeypatch.setattr(routes, "_dispatch_lore_extraction", lambda *a, **k: None)
    routes.evennia_client.base_url = "http://world.test/api/v1"
    routes.evennia_client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=world_app.app), base_url="http://world.test")

    with TestClient(proxy_app) as client, TestClient(world_app.app) as engine:
        yield Rig(client, engine, calls)


@pytest.fixture
def window_8192(monkeypatch):
    """Patch window_for itself (no probes): reported 8,192, recording the model ids it
    was asked about — the way LM Studio reported Gemma's JIT default (OPEN-024)."""
    from proxy.backend_client import backend_profile
    seen = []

    async def fake_window_for(client, model):
        seen.append(model)
        return backend_profile.WindowInfo(8192, "models-list", f"{model} context_length 8192")

    monkeypatch.setattr(backend_profile, "window_for", fake_window_for)
    return seen


def sysmsgs(char="Mira"):
    return [
        {"role": "system",
         "content": f"Write {char}'s next reply in a fictional chat between {char} and Tom."},
        {"role": "system", "content": f"[Character: {char}] A test character."},
    ]


def long_messages(pairs=60):
    """A chat long enough that only the detected window can make it fit: ~35k token
    estimate against an 8,192-token window."""
    msgs = sysmsgs()
    for i in range(pairs):
        filler = f"Scene {i}: " + ("the lamplight fell across the wet stones and " * 30)
        msgs.append({"role": "user", "content": f'"{filler} she said."'})
        msgs.append({"role": "assistant", "content": filler + " She answered at length."})
    msgs.append({"role": "user", "content": '"And now?"'})
    return msgs


# ── 1. long history trims to the detected window ────────────────────────────

def test_long_history_trimmed_to_detected_window(rig, window_8192):
    from proxy.rag import budget
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar")])
    _, payload = rig.post(long_messages())

    assert len(rig.calls) >= 1, "no LLM call captured"
    call = rig.calls[-1]
    sent = call["messages"]
    est = budget.estimate_messages(sent, "Gemma-4-26B-A4B-it")
    # Trimmed: the prompt holds a fraction of the chat that was sent in.
    assert len(sent) < len(payload)
    # Fits the 8,192 window, and max_tokens is the clamp against THAT window
    # (8192 - estimate - 5% margin), not the 128,000 settings ceiling.
    assert est < 8192
    margin = max(64, int(8192 * 0.05))
    assert call["max_tokens"] == max(256, min(128000, 8192 - est - margin))


# ── 2. virtual model id ──────────────────────────────────────────────────────

def test_virtual_model_id_reaches_window_for_and_turn_works(rig, window_8192):
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar")])
    r, _ = rig.post(sysmsgs() + [{"role": "user", "content": '"Hello?"'}],
                    model="spm-sovereign-mesh")
    assert "spm-sovereign-mesh" in window_8192
    assert "measured reply" in r.text


# ── 3. native mode downgrades when the window is tight ──────────────────────

def test_native_mode_runs_as_none_when_room_below_reasoning_max(rig, window_8192, caplog):
    from config.manager import get_settings_manager
    get_settings_manager().update({"reasoning_mode": "native",
                                   "reasoning_max_tokens": 3000,
                                   "gm_actions_channel": "inline"})
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar")])
    rig.post(long_messages())

    call = rig.calls[-1]
    body = call["extra_body"]
    assert body is not None
    assert body["reasoning_effort"] == "none"
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert any("8192" in rec.getMessage() for rec in caplog.records
               if rec.levelname == "WARNING")


# ── 4. admin endpoint ────────────────────────────────────────────────────────

def test_admin_backend_window_endpoint(rig, monkeypatch):
    from proxy.backend_client import backend_profile
    from proxy.backend_client.lemonade_client import DEFAULT_CHAT_MODEL

    async def fake_window_for(client, model):
        return backend_profile.WindowInfo(128000, "lmstudio-loaded", f"{model} loaded at 128000")

    monkeypatch.setattr(backend_profile, "window_for", fake_window_for)
    r = rig.client.get("/admin/api/v1/backend/window")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tokens"] == 128000
    assert body["source"] == "lmstudio-loaded"
    assert body["detail"]
    assert body["override"] is None
    assert body["effective"] == 128000
    assert body["tier"] == "SOVEREIGN"
    assert body["model"] == DEFAULT_CHAT_MODEL
    assert body["backend_url"]
