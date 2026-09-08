"""
Hardware Tier Configurations for Sovereign Persona Mesh (SPM).
Determines context sizes, RAG limits, and fallback strategies based on host memory profile.
"""

from enum import Enum
from pydantic import BaseModel


class HardwareTierEnum(str, Enum):
    SOVEREIGN = "SOVEREIGN"       # AMD Strix Halo 128GB Unified GTT (Balanced 96GB GTT Profile)
    PERFORMANCE = "PERFORMANCE"   # Discrete GPU VRAM >= 16GB / Host RAM >= 32GB
    EXPERIMENTAL = "EXPERIMENTAL" # Low Resource Shared Memory (16GB)


class HardwareConfig(BaseModel):
    gtt_vram_budget_gb: float
    max_context_tokens: int
    system_prompt_budget: int
    rag_context_budget: int
    spatial_context_budget: int
    active_chat_budget: int
    top_k_memories: int
    inner_monologue_enabled: bool


# Pre-configured Tiers per PRD & SRD Specs
_HARDWARE_TIERS_RAW = {
    HardwareTierEnum.SOVEREIGN: HardwareConfig(
        gtt_vram_budget_gb=96.0,
        max_context_tokens=32768,
        system_prompt_budget=4096,
        rag_context_budget=8192,
        spatial_context_budget=4096,
        active_chat_budget=16384,
        top_k_memories=5,
        inner_monologue_enabled=True,
    ),
    HardwareTierEnum.PERFORMANCE: HardwareConfig(
        gtt_vram_budget_gb=16.0,
        max_context_tokens=8192,
        system_prompt_budget=2048,
        rag_context_budget=2048,
        spatial_context_budget=1024,
        active_chat_budget=3072,
        top_k_memories=3,
        inner_monologue_enabled=True,
    ),
    HardwareTierEnum.EXPERIMENTAL: HardwareConfig(
        gtt_vram_budget_gb=8.0,
        max_context_tokens=4096,
        system_prompt_budget=1024,
        rag_context_budget=1024,
        spatial_context_budget=512,
        active_chat_budget=1536,
        top_k_memories=2,
        inner_monologue_enabled=False,
    ),
}


class _FrozenProxy:
    """Read-only proxy around a dict — raises on any mutation attempt."""

    def __init__(self, backing_dict: dict):
        self._data = backing_dict

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        raise TypeError("HARDWARE_TIERS is immutable — use _replace_tier() instead")

    def __delitem__(self, key):
        raise TypeError("HARDWARE_TIERS is immutable — use _replace_tier() instead")

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


HARDWARE_TIERS: _FrozenProxy = _FrozenProxy(_HARDWARE_TIERS_RAW)


def get_hardware_config(tier_name: str) -> HardwareConfig:
    """Retrieve hardware configuration by tier name.

    Parameters
    ----------
    tier_name : str
        One of 'SOVEREIGN', 'PERFORMANCE', 'EXPERIMENTAL' (case-insensitive).

    Raises
    ------
    TypeError
        If ``tier_name`` is not a string.
    ValueError
        If ``tier_name`` does not match any registered tier.
    """
    if not isinstance(tier_name, str):
        raise TypeError(
            f"tier_name must be a string, got {type(tier_name).__name__}"
        )

    try:
        tier_enum = HardwareTierEnum(tier_name.upper())
    except ValueError:
        raise ValueError(f"Unknown tier: {tier_name!r}")

    return HARDWARE_TIERS[tier_enum]
