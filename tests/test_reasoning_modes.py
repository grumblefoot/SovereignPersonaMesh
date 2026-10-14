"""Native-reasoning spike (2026-10-13): reasoning modes and backend probing."""
import asyncio

import pytest

from proxy.core import reasoning
from proxy.rag.prompt_builder import CognitivePromptBuilder


class FakeClient:
    def __init__(self, chunks, base_url="http://192.168.50.207:1234/v1"):
        self.chunks, self.base_url, self.calls = chunks, base_url, 0

    async def generate_stream(self, **kw):
        self.calls += 1
        self.kw = kw
        for c in self.chunks:
            yield c


def test_extra_body_switches_thinking_for_local_backends_only():
    s = {"reasoning_effort": "medium"}
    assert reasoning.extra_body("native", s, "http://localhost:13305/v1") == {
        "reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True}}
    assert reasoning.extra_body("none", s, "http://10.0.0.5:1234/v1") == {
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    assert reasoning.extra_body("native", s, "https://api.openai.com/v1") is None
    assert reasoning.extra_body("prefill", s, "http://localhost:13305/v1") is None


@pytest.mark.asyncio
async def test_auto_probe_detects_native_and_caches():
    reasoning.reset_cache()
    c = FakeClient(["<thinking>", "The user wants OK.", "</thinking>", "OK"])
    assert await reasoning.resolve_mode({"reasoning_mode": "auto"}, c, "gemma") == "native"
    assert await reasoning.resolve_mode({"reasoning_mode": "auto"}, c, "gemma") == "native"
    assert c.calls == 1                                         # cached
    assert c.kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_auto_probe_without_reasoning_field_is_none():
    reasoning.reset_cache()
    assert await reasoning.resolve_mode({"reasoning_mode": "auto"}, FakeClient(["OK"]), "mixtral") == "none"


@pytest.mark.asyncio
async def test_auto_never_probes_a_cloud_backend():
    reasoning.reset_cache()
    c = FakeClient(["<thinking>"], base_url="https://api.openai.com/v1")
    assert await reasoning.resolve_mode({"reasoning_mode": "auto"}, c, "gpt") == "none"
    assert c.calls == 0


@pytest.mark.asyncio
async def test_explicit_mode_wins_and_failed_probe_is_none():
    reasoning.reset_cache()
    assert await reasoning.resolve_mode({"reasoning_mode": "prefill"}, FakeClient([]), "m") == "prefill"

    class Broken(FakeClient):
        async def generate_stream(self, **kw):
            raise ConnectionError("down")
            yield ""
    assert await reasoning.resolve_mode({"reasoning_mode": "auto"}, Broken([]), "m") == "none"


def _build(mode):
    b = CognitivePromptBuilder()
    assert b.config.inner_monologue_enabled
    return b.build_csa_messages(
        system_prompt="You are Mei.", sensory_feed="Vardus says hi.", retrieved_memories=[],
        chat_history=[{"role": "user", "content": "hello"},
                      {"role": "assistant", "content": "<think> *Processing context...* </think>\nMei bows."},
                      {"role": "user", "content": "how are you"}],
        spatial_context="Location: hall", gm_mode="full", max_history=None, reasoning_mode=mode)


def test_native_prompt_has_no_text_scratchpad_anywhere():
    for mode in ("native", "none"):
        msgs = _build(mode)
        joined = "\n".join(m["content"] for m in msgs)
        assert "<think>" not in joined and "GM_ACTION" not in joined, mode
        assert msgs[2] == {"role": "assistant", "content": "Mei bows."}


def test_prefill_prompt_is_unchanged_legacy():
    msgs = _build("prefill")
    joined = "\n".join(m["content"] for m in msgs)
    assert "<think>" in joined and "GM_ACTION" in joined


@pytest.mark.asyncio
async def test_reasoning_status_reports_remote_cost_notice():
    import json as _json
    from config.manager import get_settings_manager
    from proxy.api.admin_routes import reasoning_status
    get_settings_manager().update({"BACKEND_LLM_URL": "https://api.openai.com/v1", "reasoning_mode": "auto"})
    d = _json.loads((await reasoning_status()).body)
    assert d["local"] is False and "charge per token" in d["notice"] and d["configured"] == "auto"
    get_settings_manager().update({"BACKEND_LLM_URL": "http://192.168.50.207:1234/v1"})
    d = _json.loads((await reasoning_status()).body)
    assert d["local"] is True and d["notice"] == ""


# --- separate GM channel (proxy/core/gm_planner.py) ---

from proxy.core import gm_planner


class GMClient:
    def __init__(self, replies, reject_schema=False):
        self.replies, self.reject_schema, self.calls = list(replies), reject_schema, []

    async def generate_stream(self, **kw):
        self.calls.append(kw)
        if self.reject_schema and "response_format" in (kw.get("extra_body") or {}):
            raise RuntimeError("HTTP 400: response_format not supported")
        yield "<thinking>"
        yield "hidden reasoning {not json}"
        yield "</thinking>"
        yield self.replies.pop(0)


@pytest.mark.asyncio
async def test_gm_plan_uses_structured_output_and_ignores_thinking():
    c = GMClient(['{"actions": [{"type": "MOVE", "entity": "user", "room_id": "garden"}]}'])
    acts = await gm_planner.plan(c, "m", [{"role": "user", "content": "x"}], local=True)
    assert acts == [{"type": "MOVE", "entity": "user", "room_id": "garden"}]
    eb = c.calls[0]["extra_body"]
    assert eb["response_format"]["type"] == "json_schema" and eb["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_gm_plan_falls_back_to_plain_when_schema_is_rejected():
    c = GMClient(['Sure: {"actions": []}'], reject_schema=True)
    assert await gm_planner.plan(c, "m", [{"role": "user", "content": "x"}], local=True) == []
    assert len(c.calls) == 2 and "response_format" not in (c.calls[1]["extra_body"] or {})


@pytest.mark.asyncio
async def test_gm_plan_sends_no_extra_fields_to_a_cloud_backend_except_schema():
    c = GMClient(['{"actions": []}'])
    await gm_planner.plan(c, "m", [{"role": "user", "content": "x"}], local=False)
    eb = c.calls[0]["extra_body"]
    assert "chat_template_kwargs" not in eb and "reasoning_effort" not in eb


def test_gm_parse_rejects_junk():
    assert gm_planner.parse("no json here") == []
    assert gm_planner.parse('{"actions": "move"}') == []
    assert gm_planner.parse('{"actions": [1, {"type": "MOVE", "room_id": "x"}]}') == [{"type": "MOVE", "room_id": "x"}]


def test_gm_grounded_keeps_only_quoted_actions():
    acts = [{"type": "MOVE", "entity": "lian", "room_id": "kitchen", "evidence": "Lian went to the kitchen"},
            {"type": "CREATE_ROOM", "room_id": "garden", "name": "Garden", "evidence": "garden"},
            {"type": "MOVE", "entity": "user", "room_id": "garden", "evidence": "They step out through the back door"}]
    kept = gm_planner.grounded(acts, '*They step out through the back door.*', "Lian smiled.")
    assert kept == [{"type": "CREATE_ROOM", "room_id": "garden", "name": "Garden"},
                    {"type": "MOVE", "entity": "user", "room_id": "garden"}]
    assert gm_planner.grounded([{"type": "MOVE", "entity": "user", "room_id": "x"}], "anything") == []
