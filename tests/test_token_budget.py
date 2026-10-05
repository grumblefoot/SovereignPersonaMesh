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
