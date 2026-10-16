"""BACKEND_API_KEY reaches a hosted model server, and only that server.

All HTTP goes through httpx.MockTransport; no real server, key or network.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy.backend_client.auth import BackendKeyAuth, same_origin
from proxy.backend_client.lemonade_client import LemonadeLLMClient

BASE = "https://api.example.test/v1"
KEY = "sk-test-0000000000000000"


def _recorder():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"data": [{"id": "remote-model"}]})

    return handler, seen


@pytest.mark.parametrize("a,b,expected", [
    ("https://api.example.test/v1/models", BASE, True),
    ("https://api.example.test:443/v1", BASE, True),        # default port filled in
    ("http://api.example.test/v1", BASE, False),            # scheme differs
    ("https://api.example.test:8443/v1", BASE, False),      # port differs
    ("https://evil.example.test/v1", BASE, False),          # host differs
    ("http://127.0.0.1:4005/api/v1", BASE, False),          # the world engine
    ("not a url", BASE, False),
])
def test_same_origin(a, b, expected):
    assert same_origin(a, b) is expected


async def test_key_goes_to_the_backend_origin_only():
    handler, seen = _recorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 auth=BackendKeyAuth(BASE, KEY)) as http:
        await http.get(f"{BASE}/models")
        await http.get("https://other.example.test/v1/models")
        await http.get("http://127.0.0.1:13305/v1/embeddings")
        await http.get(f"{BASE}/models", headers={"Authorization": "Bearer caller-set"})
    assert seen == [
        (f"{BASE}/models", f"Bearer {KEY}"),
        ("https://other.example.test/v1/models", None),
        ("http://127.0.0.1:13305/v1/embeddings", None),
        (f"{BASE}/models", "Bearer caller-set"),
    ]


async def test_no_key_no_header():
    handler, seen = _recorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 auth=BackendKeyAuth(BASE, "")) as http:
        await http.get(f"{BASE}/models")
    assert seen == [(f"{BASE}/models", None)]


async def test_live_client_reads_the_key_from_the_settings_at_call_time():
    """The proxy's client follows BACKEND_API_KEY from the settings (hot reload)."""
    from config.manager import get_settings_manager

    handler, seen = _recorder()
    client = LemonadeLLMClient(base_url=BASE)
    http = client.client
    http._transport = httpx.MockTransport(handler)
    mgr = get_settings_manager()

    await http.get(f"{BASE}/models")
    mgr.write_settings({"BACKEND_API_KEY": KEY})
    await http.get(f"{BASE}/models")
    await http.aclose()
    assert [h for _, h in seen] == [None, f"Bearer {KEY}"]


async def test_explicit_key_wins_and_model_lookup_is_authorised():
    handler, seen = _recorder()
    client = LemonadeLLMClient(base_url=BASE, api_key=KEY)
    client.client._transport = httpx.MockTransport(handler)
    ids = await client._chat_model_ids()
    await client.close()
    assert ids is not None and "remote-model" in ids
    assert seen and all(h == f"Bearer {KEY}" for _, h in seen)


def test_embeddings_get_the_backend_key_only_from_the_same_server():
    from proxy.embeddings.service import create_embedding_service

    def key_for(settings):
        svc = create_embedding_service(settings)
        return getattr(getattr(svc, "_provider", None), "api_key", None)

    remote = {"EMBEDDING_PROVIDER": "openai_compat", "BACKEND_LLM_URL": BASE,
              "BACKEND_API_KEY": KEY, "EMBEDDING_MODEL": "e"}
    # Embeddings from the chat backend's own server: its key applies.
    assert key_for({**remote, "EMBEDDING_URL": ""}) == KEY
    # Embeddings on a local server: the hosted chat key never goes there.
    assert key_for({**remote, "EMBEDDING_URL": "http://127.0.0.1:13305/v1"}) == ""
    # A dedicated embedding key is used as given.
    assert key_for({**remote, "EMBEDDING_URL": "http://127.0.0.1:13305/v1",
                    "EMBEDDING_API_KEY": "emb-key"}) == "emb-key"


def test_dashboard_never_receives_the_key():
    from proxy.main import app

    client = TestClient(app)
    r = client.post("/admin/api/v1/config", json={"BACKEND_API_KEY": KEY})
    assert r.status_code == 200
    assert KEY not in r.text
    assert r.json()["settings"]["BACKEND_API_KEY_SET"] is True

    r = client.get("/admin/api/v1/config")
    assert KEY not in r.text
    body = r.json()
    assert body["BACKEND_API_KEY"] == "" and body["BACKEND_API_KEY_SET"] is True
    assert body["EMBEDDING_API_KEY_SET"] is False

    # The flag sent back with a form is not saved as a setting.
    client.post("/admin/api/v1/config", json={"BACKEND_API_KEY_SET": False})
    from config.manager import get_settings_manager
    assert "BACKEND_API_KEY_SET" not in get_settings_manager().saved_keys()
    assert get_settings_manager().get_settings()["BACKEND_API_KEY"] == KEY


async def test_report_check_sends_the_key_and_explains_a_refusal(monkeypatch, tmp_path):
    from spm import diagnostics as dx

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(401, json={"error": "invalid api key"})

    real_factory = httpx.AsyncClient

    async def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_factory(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(dx, "_make_http", factory)
    env = {"BACKEND_LLM_URL": BASE, "BACKEND_API_KEY": KEY, "SPM_DEFAULT_MODEL": "m"}
    ctx = dx._Ctx(root=tmp_path, redactor=dx.Redactor())   # not the real checkout
    c = await dx.check_model_server(env, ctx)
    assert seen and seen[0] == f"Bearer {KEY}"
    assert c.status == "warn" and "check BACKEND_API_KEY" in c.summary
    assert c.details["backend_key_set"] is True
    assert KEY not in repr(c.details) and KEY not in c.summary
