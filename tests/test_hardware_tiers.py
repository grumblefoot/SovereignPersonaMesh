"""
TDD tests for config/hardware_tiers.py bugs.

Bugs being tested:
  1. HARDWARE_TIERS dict is a mutable module-level global — should be frozen.
  2. get_hardware_config() silently returns SOVEREIGN on invalid tier — should raise.
  3. _detect_hardware_tier() triggers SOVEREIGN at 64 GB instead of 128 GB.
  4. HardwareConfig.tier is redundant dead code.
  5. get_hardware_config() accepts any str via .upper() — needs input validation.
"""

import pytest
from unittest.mock import patch, mock_open

# ── 1. HardwareConfig has no redundant tier field ─────────────────────────
def test_hardware_config_no_tier_field():
    """HardwareConfig.tier is redundant (enum is already the dict key)."""
    from config.hardware_tiers import HardwareConfig

    cfg = HardwareConfig(
        gtt_vram_budget_gb=96.0,
        max_context_tokens=32768,
        system_prompt_budget=4096,
        rag_context_budget=8192,
        spatial_context_budget=4096,
        active_chat_budget=16384,
        top_k_memories=5,
        inner_monologue_enabled=True,
    )
    # Must not have a 'tier' attribute — it's dead code.
    assert not hasattr(cfg, "tier"), (
        "HardwareConfig.tier is redundant — the enum is already the dict key."
    )


# ── 2. get_hardware_config raises on invalid tier (not silent fallback) ───
def test_get_hardware_config_invalid_tier_raises():
    """Invalid tier should raise ValueError, not silently return SOVEREIGN."""
    from config.hardware_tiers import get_hardware_config

    with pytest.raises(ValueError, match="Unknown tier"):
        get_hardware_config("FAKE_TIER")


def test_get_hardware_config_valid_tier_returns_config():
    """Valid tier names should work."""
    from config.hardware_tiers import get_hardware_config, HardwareTierEnum

    cfg = get_hardware_config("SOVEREIGN")
    assert cfg.gtt_vram_budget_gb == 96.0

    cfg = get_hardware_config("performance")
    assert cfg.gtt_vram_budget_gb == 16.0

    cfg = get_hardware_config("experimental")
    assert cfg.inner_monologue_enabled is False


# ── 3. get_hardware_config input validation: rejects bad types ────────────
def test_get_hardware_config_rejects_non_string_input():
    """Passing non-string should raise, not silently coerce via .upper()."""
    from config.hardware_tiers import get_hardware_config

    with pytest.raises(TypeError):
        get_hardware_config(123)

    with pytest.raises(TypeError):
        get_hardware_config(None)


# ── 4. HARDWARE_TIERS is immutable ────────────────────────────────────────
def test_hardware_tiers_is_immutable():
    """HARDWARE_TIERS should not be a plain mutable dict."""
    from config.hardware_tiers import HARDWARE_TIERS

    # Must not allow in-place mutation.
    with pytest.raises((TypeError, AttributeError)):
        HARDWARE_TIERS["new_key"] = "value"


def test_hardware_tiers_is_still_accessible():
    """Frozen dict should still support key lookup (via __getitem__)."""
    from config.hardware_tiers import HARDWARE_TIERS, HardwareTierEnum

    # FrozenProxy should still implement __getitem__.
    assert HARDWARE_TIERS[HardwareTierEnum.SOVEREIGN].gtt_vram_budget_gb == 96.0


# ── 5. _detect_hardware_tier threshold is 128 GB, not 64 ─────────────────
def test_detect_hardware_tier_128gb_returns_sovereign():
    """A machine with exactly 128 GB RAM should be SOVEREIGN tier."""
    from proxy.rag.import_worker import _detect_hardware_tier, HardwareTierEnum

    # 128 * 1024 * 1024 = 134217728 kB  → 134217728 / 1048576 = 128.0 GB
    mock_mem = "MemTotal:     134217728 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_mem)):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.SOVEREIGN, (
            f"128 GB should return SOVEREIGN, got {tier}"
        )


def test_detect_hardware_tier_64gb_returns_performance():
    """A machine with exactly 64 GB RAM should NOT be SOVEREIGN — PERFORMANCE."""
    from proxy.rag.import_worker import _detect_hardware_tier, HardwareTierEnum

    # 64 * 1024 * 1024 = 67108864 kB  → 67108864 / 1048576 = 64.0 GB
    mock_mem = "MemTotal:      67108864 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_mem)):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.PERFORMANCE, (
            f"64 GB should return PERFORMANCE, got {tier}"
        )


def test_detect_hardware_tier_16gb_returns_performance():
    """16 GB should map to PERFORMANCE."""
    from proxy.rag.import_worker import _detect_hardware_tier, HardwareTierEnum

    # 16 * 1024 * 1024 = 16777216 kB
    mock_mem = "MemTotal:      16777216 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_mem)):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.PERFORMANCE


def test_detect_hardware_tier_below_16gb_returns_experimental():
    """Below 16 GB should map to EXPERIMENTAL."""
    from proxy.rag.import_worker import _detect_hardware_tier, HardwareTierEnum

    # 8 * 1024 * 1024 = 8388608 kB
    mock_mem = "MemTotal:       8388608 kB\n"
    with patch("builtins.open", mock_open(read_data=mock_mem)):
        tier = _detect_hardware_tier()
        assert tier == HardwareTierEnum.EXPERIMENTAL
