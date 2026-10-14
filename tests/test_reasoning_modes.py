"""Native-reasoning spike (2026-10-13): reasoning modes and backend probing."""
import asyncio

import pytest

from proxy.core import reasoning
from proxy.rag.prompt_builder import CognitivePromptBuilder


class FakeClient:
    def __init__(self, chunks, base_url="http://10.0.0.5:1234/v1"):
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
    get_settings_manager().update({"BACKEND_LLM_URL": "http://10.0.0.5:1234/v1"})
    d = _json.loads((await reasoning_status()).body)
    assert d["local"] is True and d["notice"] == ""


# --- separate GM channel (proxy/core/gm_planner.py) ---

from proxy.core import gm_planner


@pytest.fixture(autouse=True)
def _fresh_schema_memory():
    gm_planner._NO_SCHEMA.clear()
    yield
    gm_planner._NO_SCHEMA.clear()


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


def test_gm_reuse_rooms_maps_same_place_only():
    rooms = [{"room_id": "principals_office", "name": "Principal's Office"},
             {"room_id": "ua_high_school", "name": "U.A. High School"}]
    acts = [{"type": "CREATE_ROOM", "room_id": "principal_office", "name": "The Principal's Office"},
            {"type": "MOVE", "entity": "user", "room_id": "principal_office"},
            {"type": "CREATE_ROOM", "room_id": "u_a_high_school", "name": "UA High School"},
            {"type": "MOVE", "entity": "mei", "room_id": "u_a_high_school"}]
    assert gm_planner.reuse_rooms(acts, rooms) == [{"type": "MOVE", "entity": "user", "room_id": "principals_office"},
                                                   {"type": "MOVE", "entity": "mei", "room_id": "ua_high_school"}]
    # An extra word is a different place (review C4).
    for rid, name in (("vice_principals_office", "Vice Principal's Office"),
                      ("ua_high_school_main_gates", "U.A. High School Main Gates")):
        new = [{"type": "CREATE_ROOM", "room_id": rid, "name": name}, {"type": "MOVE", "entity": "user", "room_id": rid}]
        assert gm_planner.reuse_rooms(new, rooms) == new


def test_gm_grounded_trims_fields_the_validator_forbids():
    acts = [{"type": "CREATE_ROOM", "entity": "", "room_id": "garden", "name": "Garden", "desc": "",
             "evidence": "we step out into the garden"},
            {"type": "MOVE", "entity": "user", "room_id": "garden", "name": "", "evidence": "we step out into the garden"}]
    out = gm_planner.grounded(acts, "Hand in hand, we step out into the garden.")
    assert out == [{"type": "CREATE_ROOM", "room_id": "garden", "name": "Garden"},
                   {"type": "MOVE", "entity": "user", "room_id": "garden"}]


def test_gm_reuse_rooms_keeps_the_last_move_per_entity():
    acts = [{"type": "CREATE_ROOM", "room_id": "kitchen"}, {"type": "MOVE", "entity": "lian", "room_id": "kitchen"},
            {"type": "CREATE_ROOM", "room_id": "vardus_room"}, {"type": "MOVE", "entity": "lian", "room_id": "vardus_room"}]
    assert gm_planner.reuse_rooms(acts, []) == [{"type": "CREATE_ROOM", "room_id": "vardus_room"},
                                                {"type": "MOVE", "entity": "lian", "room_id": "vardus_room"}]


def test_gm_grounded_ignores_evidence_inside_dialogue():
    reply = ('"Mei!" Lian called out. "Go to the guest room and ensure the linens are crisp." '
             'Lian walks back into the kitchen and sets the kettle down.')
    acts = [{"type": "MOVE", "entity": "mei", "room_id": "guest_room", "evidence": "Go to the guest room"},
            {"type": "MOVE", "entity": "lian", "room_id": "kitchen", "evidence": "Lian walks back into the kitchen"}]
    assert gm_planner.grounded(acts, "", reply) == [{"type": "MOVE", "entity": "lian", "room_id": "kitchen"}]


def test_gm_grounded_move_needs_a_quote_that_shows_movement():
    text = "The figure standing at the mouth of the alley watches. Then she walks into the alley."
    acts = [{"type": "MOVE", "entity": "user", "room_id": "street", "evidence": "The figure standing at the mouth of the alley"},
            {"type": "MOVE", "entity": "user", "room_id": "alley", "evidence": "she walks into the alley"}]
    assert gm_planner.grounded(acts, "", text) == [{"type": "MOVE", "entity": "user", "room_id": "alley"}]


def test_gm_grounded_player_move_stands_on_the_players_narration():
    acts = [{"type": "MOVE", "entity": "user", "room_id": "training_grounds",
             "evidence": "The P.E. Grounds stretch out ahead"}]
    reply = "The P.E. Grounds stretch out ahead, a massive expanse of dirt."
    assert gm_planner.grounded(acts, "I go to the training grounds", reply) == [
        {"type": "MOVE", "entity": "user", "room_id": "training_grounds"}]
    # Movement only in the player's dialogue is not a declared move.
    assert gm_planner.grounded(acts, '"Let\'s go to the training grounds later," I say.', reply) == []


@pytest.mark.asyncio
async def test_gm_plan_remembers_a_backend_that_rejects_structured_output():
    c = GMClient(['{"actions": []}', '{"actions": []}'], reject_schema=True)
    await gm_planner.plan(c, "m", [{"role": "user", "content": "x"}], local=True)
    await gm_planner.plan(c, "m", [{"role": "user", "content": "x"}], local=True)
    assert len(c.calls) == 3                       # schema+plain, then plain only
    assert "response_format" not in (c.calls[2]["extra_body"] or {})


def test_gm_grounded_binds_the_move_to_the_named_character():
    reply = "Lian walks into the kitchen and sets the kettle down."
    acts = [{"type": "MOVE", "entity": "vardus", "room_id": "kitchen", "evidence": "Lian walks into the kitchen"},
            {"type": "MOVE", "entity": "lian", "room_id": "kitchen", "evidence": "Lian walks into the kitchen"}]
    assert gm_planner.grounded(acts, "", reply) == [{"type": "MOVE", "entity": "lian", "room_id": "kitchen"}]


def test_gm_grounded_player_shortcut_needs_the_place_named():
    junk = [{"type": "MOVE", "entity": "user", "room_id": "moon_base", "evidence": "xyz"}]
    assert gm_planner.grounded(junk, "The key is back in the box.", "Nothing happens.") == []


def test_gm_grounded_ignores_dialogue_that_spans_lines():
    reply = 'She said\n"Go to the guest\nroom now."\nMei stayed seated.'
    acts = [{"type": "MOVE", "entity": "mei", "room_id": "guest_room", "evidence": "Go to the guest room now"}]
    assert gm_planner.grounded(acts, "", reply) == []


@pytest.mark.asyncio
async def test_gm_plan_gives_up_after_the_timeout(monkeypatch):
    class Hung:
        async def generate_stream(self, **kw):
            await asyncio.sleep(10)
            yield "{}"
    monkeypatch.setattr(gm_planner, "PLAN_TIMEOUT_S", 0.05)
    assert await gm_planner.plan(Hung(), "m", [{"role": "user", "content": "x"}], local=True) == []


@pytest.mark.asyncio
async def test_failed_probe_is_retried_soon_and_shared(monkeypatch):
    reasoning.reset_cache()
    calls = {"n": 0}

    async def slow_fail(client, model):
        calls["n"] += 1
        await asyncio.sleep(0.05)
        raise RuntimeError("queue full")
    monkeypatch.setattr(reasoning, "_probe", slow_fail)
    c = FakeClient([])
    s = {"reasoning_mode": "auto"}
    assert await asyncio.gather(reasoning.resolve_mode(s, c, "m"), reasoning.resolve_mode(s, c, "m")) == ["none", "none"]
    assert calls["n"] == 1                                   # one probe for both turns
    when, mode, ttl = reasoning._probe_cache[(c.base_url, "m")]
    assert mode == "none" and ttl == reasoning.PROBE_RETRY_S < reasoning.PROBE_TTL_S
    reasoning.reset_cache()


def test_gm_grounded_leaving_a_place_moves_nobody_into_it():
    """Tester session 2026-10-13 (T5): 'open the door to leave his office' moved the
    player INTO the office."""
    acts = [{"type": "MOVE", "entity": "user", "room_id": "principals_office",
             "evidence": "I bow and open the door to leave his office"}]
    assert gm_planner.grounded(acts, "I bow and open the door to leave his office", "He nods.") == []
    went = [{"type": "MOVE", "entity": "user", "room_id": "training_grounds",
             "evidence": "He left for the training grounds at dawn"}]
    assert gm_planner.grounded(went, "", "He left for the training grounds at dawn.") == [
        {"type": "MOVE", "entity": "user", "room_id": "training_grounds"}]


def test_gm_drop_no_ops_skips_moves_into_the_current_room():
    acts = [{"type": "CREATE_ROOM", "room_id": "class_1a"},
            {"type": "MOVE", "entity": "user", "room_id": "class_1a"},
            {"type": "MOVE", "entity": "mei", "room_id": "garden"}]
    assert gm_planner.drop_no_ops(acts, {"user": "class_1a", "mei": "kitchen"}) == [
        {"type": "MOVE", "entity": "mei", "room_id": "garden"}]
