"""Token budget P0 (OPEN-008): the clamp, the heuristic counter, calibration."""
import pytest

from config.hardware_tiers import HARDWARE_TIERS, HardwareTierEnum
from proxy.rag import budget

TIER = HARDWARE_TIERS[HardwareTierEnum.SOVEREIGN]   # 32768 window


def msgs(n_chars, n=1):
    return [{"role": "user", "content": "x" * n_chars} for _ in range(n)]


def test_small_prompt_gets_the_full_ceiling():
    out = budget.clamp_max_tokens(msgs(400), {"backend_max_tokens": 2048}, TIER)
    assert out == 2048


def test_huge_ceiling_is_clamped_to_window_minus_prompt():
    # the old behaviour sent 128000 regardless; now the window governs
    out = budget.clamp_max_tokens(msgs(4000), {"backend_max_tokens": 128000}, TIER)
    prompt = budget.estimate_messages(msgs(4000))
    assert out <= 32768 - prompt
    assert out > budget.OUTPUT_FLOOR


def test_overfull_prompt_still_sends_the_floor():
    out = budget.clamp_max_tokens(msgs(200000), {"backend_max_tokens": 128000}, TIER)
    assert out == budget.OUTPUT_FLOOR


def test_window_override_wins_over_tier():
    settings = {"backend_max_tokens": 128000, "context_window_override": 8192}
    out = budget.clamp_max_tokens(msgs(400), settings, TIER)
    assert out <= 8192


def test_estimate_counts_per_message_overhead():
    assert budget.estimate_messages([]) == 0
    one = budget.estimate_messages(msgs(32))
    assert one == budget.estimate_tokens("x" * 32) + budget.PER_MESSAGE_OVERHEAD


def test_calibration_moves_the_ratio_toward_observed():
    model = "cal-test-model"
    base = budget.estimate_tokens("y" * 320, model)      # 3.2 -> ~100
    budget.calibrate(model, prompt_chars=320, reported_prompt_tokens=200)  # 1.6 observed
    after = budget.estimate_tokens("y" * 320, model)
    assert after > base                                   # ratio dropped -> more tokens
    budget._calibrated_ratio.pop(model, None)


def test_bad_inputs_are_ignored():
    budget.calibrate("m", 0, 10)
    budget.calibrate("m", 10, 0)
    assert "m" not in budget._calibrated_ratio


# ── P1: allocator ───────────────────────────────────────────────────────────

def _alloc(history=None, memories=None, lore=None, system="You are Mira.", window=None):
    settings = {"context_window_override": window} if window else {}
    return budget.allocate(
        settings=settings, hw_config=TIER, model="m", system_text=system,
        memories=memories or [], lore=lore or {"invariants": [], "triggers": []},
        history=history if history is not None else [])


def test_200_turn_history_fits_the_window():
    history = []
    for i in range(200):
        history.append({"role": "user", "content": f"user turn {i} " + "words " * 60})
        history.append({"role": "assistant", "content": f"reply {i} " + "words " * 60})
    history.append({"role": "user", "content": "the latest question"})
    mem, lore, kept, report = _alloc(history=history)
    assert not report.refused
    assert budget.estimate_messages(kept) + report.used["system_card"] <= report.window
    assert kept[-1]["content"] == "the latest question"      # never trimmed
    assert report.trimmed["history"] > 0


def test_latest_user_turn_survives_even_a_tiny_window():
    history = [{"role": "user", "content": "old " * 500},
               {"role": "assistant", "content": "older reply " * 500},
               {"role": "user", "content": "the latest question"}]
    mem, lore, kept, report = _alloc(history=history, window=4096)
    assert not report.refused
    assert any(m.get("content") == "the latest question" for m in kept)


def test_oversized_card_refuses_instead_of_truncating():
    mem, lore, kept, report = _alloc(system="card " * 40000, window=8192)
    assert report.refused
    assert "card" in report.refusal_reason or "system prompt" in report.refusal_reason


def test_monologue_stripped_from_old_turns_before_dropping():
    long_think = "<think>" + "secret plan " * 400 + "</think>visible reply"
    history = ([{"role": "user", "content": "q0"},
                {"role": "assistant", "content": long_think}]
               + [{"role": "user", "content": f"q{i} " + "w " * 40} for i in range(1, 40)]
               + [{"role": "user", "content": "latest"}])
    mem, lore, kept, report = _alloc(history=history, window=4096)
    assert budget.estimate_messages(kept) > 0
    old_assistant = [m for m in kept if m.get("role") == "assistant"]
    assert old_assistant, "the assistant turn should be stripped, not dropped"
    for m in old_assistant:
        assert "secret plan" not in m["content"]              # stripped, not kept verbatim
        assert "visible reply" in m["content"]                # the public part survives
    assert report.trimmed["history"] == 0                     # stripping sufficed


def test_memories_and_lore_trim_worst_first():
    memories = [{"sensory_input": f"memory {i} " + "detail " * 200} for i in range(20)]
    lore = {"invariants": [{"rule_text": "inv " + "r " * 100}],
            "triggers": [{"rule_text": f"trig {i} " + "r " * 200} for i in range(10)]}
    mem, kept_lore, hist, report = _alloc(memories=memories, lore=lore, window=8192)
    assert 0 < len(mem) < 20                                  # best-first kept
    assert mem[0]["sensory_input"].startswith("memory 0")
    assert kept_lore["invariants"]                            # invariants survive longest
    assert report.trimmed["memories"] > 0
    assert report.trimmed["lore"] > 0


# ── P2: streamed usage feeds the calibration EMA ────────────────────────────

@pytest.mark.asyncio
async def test_streamed_usage_calibrates_the_model_ratio():
    import httpx
    from proxy.backend_client.lemonade_client import LemonadeLLMClient

    model = "usage-test-model"
    body = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5}}\n\n'
        "data: [DONE]\n\n")

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model}]})
        return httpx.Response(200, text=body,
                              headers={"content-type": "text/event-stream"})

    client = LemonadeLLMClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                       base_url="http://fake")
    client.base_url = "http://fake/v1"
    budget._calibrated_ratio.pop(model, None)
    out = []
    async for c in client.generate_stream(messages=[{"role": "user", "content": "x" * 320}],
                                          model=model):
        out.append(c)
    assert "hi" in "".join(out)
    assert model in budget._calibrated_ratio            # EMA updated from usage
    assert client.last_usage["prompt_tokens"] == 100
    budget._calibrated_ratio.pop(model, None)
    await client.close()
