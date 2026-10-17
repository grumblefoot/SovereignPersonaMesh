"""tests/test_backend_profile.py — context-window detection (brief W1a).

One test per server shape from docs/plans/window_detection.md, the malformed-answer
guards (review item 7), the cache/refresh/backoff rules (review item 3), and the
budget precedence of review item 4. All servers are httpx MockTransports.
"""
import asyncio
import time

import httpx
import pytest

from proxy.backend_client import backend_profile
from proxy.backend_client.backend_profile import (
    REPORTED_SOURCES,
    WindowInfo,
    detect_window,
    reset_cache,
    window_for,
)
from proxy.backend_client.lemonade_client import model_matches
from proxy.rag import budget
from config.hardware_tiers import HardwareConfig


@pytest.fixture(autouse=True)
def _fresh_profile_cache():
    reset_cache()
    yield
    reset_cache()


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(backend_profile, "_now", c)
    return c


class Transport:
    """MockTransport wrapper that counts requests per path."""

    def __init__(self, routes):
        self.counts = {}
        self._routes = routes
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle),
                                        timeout=5.0)

    async def _handle(self, request):
        self.counts[request.url.path] = self.counts.get(request.url.path, 0) + 1
        handler = self._routes.get(request.url.path)
        if handler is None:
            return httpx.Response(404, json={"error": "no route"})
        if callable(handler):
            result = handler(request)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        return handler

    async def aclose(self):
        await self.client.aclose()


def json_response(payload, status=200):
    return httpx.Response(status, json=payload)


class FakeClient:
    """Duck-typed stand-in for LemonadeLLMClient (base_url + probe http)."""

    def __init__(self, base_url, transport, resolve=None):
        self.base_url = base_url
        self._window_http = transport.client
        self._resolve = resolve

    async def _resolve_model(self, model):
        if self._resolve is None:
            return model
        return await self._resolve(model)


LM = "Gemma-4-26B-A4B-it-GGUF"


async def probe(routes, base_url="http://localhost:13305/v1", model=LM):
    t = Transport(routes)
    try:
        return await detect_window(base_url, model, http=t.client)
    finally:
        await t.aclose()


# ── model_matches (extracted matcher, review item 2) ────────────────────────

@pytest.mark.parametrize("a, b, expected", [
    (LM, "google/gemma-4-26b-a4b-qat", True),                       # cross-server naming
    (LM, "unsloth/gemma-4-26b-a4b-it-gguf:UD-Q4_K_M", True),        # Lemonade checkpoint
    ("Gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M", LM, True),                # ':tag' stripped
    ("google/gemma-4-26b-a4b-it", LM, True),                        # retired HF-style id
    ("gemma-4-26b-a4b-it-gguf", LM, True),                          # case-insensitive
    ("Qwen3.8-27B-GGUF", "qwen/qwen3.8-27b", True),
    (LM, "qwen/qwen3.8-27b", False),                                # different model
    (LM, "", False),
    ("", "", False),
])
def test_model_matches(a, b, expected):
    assert model_matches(a, b) is expected


# ── one test per server shape ───────────────────────────────────────────────

async def test_lmstudio_loaded():
    info = await probe({
        "/api/v0/models": json_response({"data": [
            {"id": "google/gemma-4-26b-a4b-qat", "state": "loaded", "loaded_context_length": 32768},
        ]}),
    })
    assert (info.tokens, info.source) == (32768, "lmstudio-loaded")
    assert "32768" in info.detail


async def test_lmstudio_listed_but_not_loaded_jit_default():
    info = await probe({
        "/api/v0/models": json_response({"data": [
            {"id": "google/gemma-4-26b-a4b-qat", "state": "not loaded"},
        ]}),
    })
    assert (info.tokens, info.source) == (8192, "lmstudio-jit-default")


async def test_models_list_context_length():
    info = await probe({"/v1/models": json_response({"data": [
        {"id": "google/gemma-4-26b-a4b-it", "context_length": 98304},
    ]})})
    assert (info.tokens, info.source) == (98304, "models-list")


async def test_models_list_max_model_len_vllm():
    info = await probe({"/v1/models": json_response({"data": [
        {"id": "Gemma-4-26B-A4B-it-GGUF", "max_model_len": 16384},
    ]})})
    assert (info.tokens, info.source) == (16384, "models-list")


async def test_lemonade_health_ctx_size_checkpoint():
    info = await probe({
        "/api/v1/health": json_response({"all_models_loaded": [
            {"model_name": "Gemma-4-26B-A4B-it-GGUF",
             "checkpoint": "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M",
             "launch_command": "lemonade-server run X --ctx-size 262144"},
            {"model_name": "embedding-model",
             "launch_command": "lemonade-server run Y --ctx-size 32768"},
        ]}),
    })
    assert (info.tokens, info.source) == (262144, "lemonade-launch")


async def test_lemonade_health_launch_command_as_argv_list():
    # Lemonade 2026.42+ reports launch_command as a list (QA 2026-10-17); a smaller
    # --ctx-size than the model's maximum must win over the /models context_length.
    info = await probe({
        "/api/v1/health": json_response({"all_models_loaded": [
            {"model_name": "Gemma-4-26B-A4B-it-GGUF",
             "launch_command": ["llama-server", "-m", "gemma.gguf", "--ctx-size", "32768",
                                "--port", "8001"]},
        ]}),
        "/v1/models": json_response({"data": [
            {"id": "Gemma-4-26B-A4B-it-GGUF", "context_length": 262144},
        ]}),
    })
    assert (info.tokens, info.source) == (32768, "lemonade-launch")


async def test_llamacpp_props():
    for payload, expected in [
        ({"default_generation_settings": {"n_ctx": 32768}}, 32768),
        ({"n_ctx": 4096}, 4096),
    ]:
        info = await probe({"/props": json_response(payload)},
                           base_url="http://localhost:8080/v1")
        assert (info.tokens, info.source) == (expected, "llamacpp-props")


# ── malformed answers (review item 7) ───────────────────────────────────────

async def test_html_200_is_ignored():
    info = await probe({"/api/v0/models": httpx.Response(
        200, text="<html><body>proxy login</body></html>",
        headers={"content-type": "text/html"})})
    assert info.source == "unknown" and info.tokens is None


async def test_non_dict_json_is_ignored():
    info = await probe({"/v1/models": httpx.Response(200, json=[1, 2, 3])})
    assert info.source == "unknown"


async def test_context_length_auto_is_ignored():
    info = await probe({"/v1/models": json_response({"data": [
        {"id": LM, "context_length": "auto"},
    ]})})
    assert info.source == "unknown"


async def test_negative_context_is_ignored():
    info = await probe({"/v1/models": json_response({"data": [
        {"id": LM, "context_length": -1},
    ]})})
    assert info.source == "unknown"


async def test_lmstudio_state_other_than_loaded_is_not_loaded():
    # 'error' etc. must not be read as loaded even when loaded_context_length is present.
    info = await probe({"/api/v0/models": json_response({"data": [
        {"id": LM, "state": "error", "loaded_context_length": 262144},
    ]})})
    assert info.source == "lmstudio-jit-default"
    assert info.tokens == 8192


async def test_all_probes_failing_is_unknown():
    info = await probe({})     # every route 404
    assert (info.tokens, info.source) == (None, "unknown")


async def test_unreachable_server_is_unknown():
    t = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
    info = await detect_window("http://localhost:13305/v1", LM, http=t)
    await t.aclose()
    assert info.source == "unknown"


# ── window_for: cloud skip, latency cap, cache, refresh, backoff ────────────

async def test_cloud_url_makes_no_request():
    t = Transport({"/v1/models": json_response(
        {"data": [{"id": LM, "context_length": 123}]})})
    client = FakeClient("https://api.openai.com/v1", t)
    try:
        info = await window_for(client, LM)
        assert info.source == "unknown" and info.tokens is None
        assert t.counts == {}
    finally:
        await t.aclose()


async def test_overall_two_second_cap():
    async def slow(_request):
        await asyncio.sleep(5)
        return json_response({"data": []})

    start = time.monotonic()
    info = await probe({"/api/v0/models": slow,
                        "/api/v1/health": slow,
                        "/v1/models": slow})
    elapsed = time.monotonic() - start
    assert info.source == "unknown"
    assert elapsed < 3.0, f"probes exceeded the overall cap: {elapsed:.2f}s"


async def test_cache_hit_makes_no_request(clock):
    t = Transport({"/v1/models": json_response(
        {"data": [{"id": LM, "context_length": 98304}]})})
    client = FakeClient("http://localhost:8080/v1", t)
    try:
        first = await window_for(client, LM)
        second = await window_for(client, LM)
        assert (first.tokens, first.source) == (98304, "models-list")
        assert second.tokens == 98304
        assert t.counts["/v1/models"] == 1
        clock.t += 299                     # still inside the 300 s TTL
        await window_for(client, LM)
        assert t.counts["/v1/models"] == 1
    finally:
        await t.aclose()


async def test_stale_value_served_while_background_refresh_runs(clock):
    window = {"n": 8192}
    t = Transport({"/v1/models": lambda r: _current(window)})
    client = FakeClient("http://localhost:8080/v1", t)
    try:
        first = await window_for(client, LM)
        assert first.tokens == 8192
        window["n"] = 32768                # the server changed (model reloaded)
        clock.t += 301                     # cache is now stale
        served = await window_for(client, LM)
        assert served.tokens == 8192       # stale value served immediately...
        pending = list(backend_profile._inflight.values())
        assert pending                       # ...while a refresh runs in the background
        await asyncio.gather(*pending)
        refreshed = await window_for(client, LM)
        assert refreshed.tokens == 32768
    finally:
        await t.aclose()


async def _current(window):
    return json_response({"data": [{"id": LM, "context_length": window["n"]}]})


async def test_backoff_after_three_unknowns(clock):
    t = Transport({})                      # every probe 404 -> unknown
    client = FakeClient("http://localhost:13305/v1", t)
    key_hits = lambda: t.counts.get("/api/v0/models", 0)
    try:
        for i in range(3):
            info = await window_for(client, LM)
            assert info.source == "unknown"
            assert key_hits() == i + 1
            clock.t += 61                  # unknown TTL is 60 s
        await window_for(client, LM)       # the 3rd unknown is now on the books
        assert key_hits() == 3
        # 3 unknowns in a row: back off for 600 s — cached unknown, no new probes.
        clock.t += 61
        info = await window_for(client, LM)
        assert info.source == "unknown"
        assert key_hits() == 3
        clock.t += 601                     # backoff over: probe once more
        await window_for(client, LM)
        assert key_hits() == 4
    finally:
        await t.aclose()


async def test_found_value_resets_the_unknown_streak(clock):
    t = Transport({})                      # unknown
    client = FakeClient("http://localhost:13305/v1", t)
    hits = lambda: t.counts.get("/api/v0/models", 0)
    try:
        for _ in range(2):                 # two unknowns (streak 2)
            await window_for(client, LM)
            clock.t += 61
        t._routes["/v1/models"] = json_response(
            {"data": [{"id": LM, "context_length": 4096}]})
        clock.t += 61
        info = await window_for(client, LM)
        assert (info.tokens, info.source) == (4096, "models-list")
        assert backend_profile._unknown_streak[(client.base_url, LM)] == 0
        del t._routes["/v1/models"]        # backend goes dark again
        # Stale found value: served while the background refresh probes (and fails).
        clock.t += 301
        before = hits()
        served = await window_for(client, LM)
        assert served.tokens == 4096       # stale value served immediately
        await asyncio.gather(*backend_profile._inflight.values())
        assert hits() == before + 1
        assert backend_profile._unknown_streak[(client.base_url, LM)] == 1
        # Expired unknown re-probes synchronously; streak reset means this second
        # unknown does NOT trigger the backoff (without the reset it would).
        clock.t += 61
        before = hits()
        await window_for(client, LM)
        assert hits() == before + 1, "probe skipped: premature backoff after reset"
        assert backend_profile._unknown_streak[(client.base_url, LM)] == 2
    finally:
        await t.aclose()


async def test_virtual_id_resolved_via_client(clock):
    resolved_calls = []

    async def fake_resolve(model):
        resolved_calls.append(model)
        return "google/gemma-4-26b-a4b-qat"

    t = Transport({"/api/v0/models": json_response({"data": [
        {"id": "google/gemma-4-26b-a4b-qat", "state": "loaded", "loaded_context_length": 32768},
    ]})})
    client = FakeClient("http://10.0.0.5:1234/v1", t, resolve=fake_resolve)
    try:
        info = await window_for(client, "spm-sovereign-mesh")
        assert (info.tokens, info.source) == (32768, "lmstudio-loaded")
        assert resolved_calls == ["spm-sovereign-mesh"]
        await window_for(client, "spm-sovereign-mesh")
        assert resolved_calls == ["spm-sovereign-mesh"]   # resolution is cached too
    finally:
        await t.aclose()


async def test_resolve_failure_falls_back_to_given_id(clock):
    async def boom(model):
        raise RuntimeError("backend down")

    t = Transport({"/v1/models": json_response(
        {"data": [{"id": LM, "context_length": 16384}]})})
    client = FakeClient("http://localhost:8080/v1", t, resolve=boom)
    try:
        info = await window_for(client, LM)
        assert (info.tokens, info.source) == (16384, "models-list")
    finally:
        await t.aclose()


async def test_client_without_resolve_or_probe_http_never_raises(clock):
    class Bare:
        base_url = "http://localhost:9999/v1"     # nothing listening: probes fail
    info = await window_for(Bare(), LM)
    assert info.source == "unknown"


# ── budget precedence (review item 4) ───────────────────────────────────────

TIER = HardwareConfig(gtt_vram_budget_gb=32.0, max_context_tokens=32768,
                      system_prompt_budget=4096, rag_context_budget=8192,
                      spatial_context_budget=4096, active_chat_budget=16384,
                      top_k_memories=8, inner_monologue_enabled=True)


@pytest.mark.parametrize("settings, detected, source, expected", [
    # override caps a REPORTED window: min of the two, either direction
    ({"context_window_override": 16384}, 32768, "lmstudio-loaded", 16384),
    ({"context_window_override": 65536}, 32768, "models-list", 32768),
    ({"context_window_override": 16384}, 262144, "lemonade-launch", 16384),
    ({"context_window_override": 8192}, 32768, "llamacpp-props", 8192),
    # override wins over the guesses
    ({"context_window_override": 16384}, 8192, "lmstudio-jit-default", 16384),
    ({"context_window_override": 16384}, None, "unknown", 16384),
    # detected only
    ({}, 8192, "lmstudio-loaded", 8192),
    ({}, 8192, "lmstudio-jit-default", 8192),
    ({}, None, "unknown", 32768),           # neither -> tier
    ({}, None, "", 32768),
    # malformed override behaves as unset
    ({"context_window_override": "auto"}, 4096, "models-list", 4096),
    ({"context_window_override": "auto"}, None, "", 32768),
])
def test_context_window_precedence(settings, detected, source, expected):
    assert budget.context_window(settings, TIER, detected=detected,
                                 detected_source=source) == expected
    # no detection at all keeps the old behaviour exactly
    assert budget.context_window(settings, TIER) == (
        int(settings["context_window_override"])
        if str(settings.get("context_window_override") or "").isdigit()
        else 32768)


def test_reported_sources_membership():
    assert REPORTED_SOURCES == {"lmstudio-loaded", "models-list",
                                "lemonade-launch", "llamacpp-props"}


def _msgs(chars):
    return [{"role": "user", "content": "x" * chars}]


def test_clamp_and_allocate_pass_detection_through():
    # window 8192 (detected, no override): a 12k-char prompt (~3.7k tokens) leaves less
    # room than the 32k tier would; compare against the no-detection clamp.
    msgs = _msgs(12000)
    ceiling = {"backend_max_tokens": 8000}
    with_det = budget.clamp_max_tokens(msgs, ceiling, TIER, detected=8192,
                                       detected_source="lmstudio-loaded")
    without = budget.clamp_max_tokens(msgs, ceiling, TIER)
    assert with_det < without
    assert with_det >= budget.OUTPUT_FLOOR

    _, _, _, report = budget.allocate(settings={}, hw_config=TIER, model="",
                                      system_text="card", memories=[], lore={},
                                      history=[], detected=8192,
                                      detected_source="lmstudio-loaded")
    assert report.window == 8192
    # override caps the detection inside allocate too
    _, _, _, report2 = budget.allocate(
        settings={"context_window_override": 6000}, hw_config=TIER, model="",
        system_text="card", memories=[], lore={}, history=[],
        detected=8192, detected_source="lmstudio-loaded")
    assert report2.window == 6000
    # and against a guess the override wins
    _, _, _, report3 = budget.allocate(
        settings={"context_window_override": 16384}, hw_config=TIER, model="",
        system_text="card", memories=[], lore={}, history=[],
        detected=8192, detected_source="lmstudio-jit-default")
    assert report3.window == 16384


def test_best_match_prefers_the_exact_name_over_a_longer_sibling():
    """Review of W1a: a single any-match pass found two candidates for an exact name and
    resolved nothing."""
    from proxy.backend_client.lemonade_client import best_match
    cands = ["qwen/qwen3.8-27b", "qwen/qwen3.8-27b-instruct"]
    assert best_match("qwen/qwen3.8-27b", cands) == "qwen/qwen3.8-27b"
    assert best_match("Gemma-4-26B-A4B-it-GGUF", ["google/gemma-4-26b-a4b-qat", "google/gemma-4-e4b"]) \
        == "google/gemma-4-26b-a4b-qat"
    assert best_match("gemma-4", ["google/gemma-4-26b-a4b-qat", "google/gemma-4-e4b"]) is None   # ambiguous
    assert best_match("", cands) is None
