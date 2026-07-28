"""
Hardware Tier Configurations for Sovereign Persona Mesh (SPM).
Determines context sizes, RAG limits, and fallback strategies based on host memory profile.
"""

from enum import Enum
from pydantic import BaseModel


class HardwareTierEnum(str, Enum):
    SOVEREIGN = "SOVEREIGN"       # AMD Strix Halo 128GB Unified GTT (Balanced 96GB GTT Profile)
    PERFORMANCE = "PERFORMANCE"   # Discrete GPU VRAM >= 16GB / Host RAM >= 32GB
    EXPERIMENTAL = "EXPERIMENTAL"# Low Resource Shared Memory (16GB)


class HardwareConfig(BaseModel):
    tier: HardwareTierEnum
    gtt_vram_budget_gb: float
    max_context_tokens: int
    system_prompt_budget: int
    rag_context_budget: int
    spatial_context_budget: int
    active_chat_budget: int
    top_k_memories: int
    inner_monologue_enabled: bool


# Pre-configured Tiers per PRD & SRD Specs
HARDWARE_TIERS = {
    HardwareTierEnum.SOVEREIGN: HardwareConfig(
        tier=HardwareTierEnum.SOVEREIGN,
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
        tier=HardwareTierEnum.PERFORMANCE,
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
        tier=HardwareTierEnum.EXPERIMENTAL,
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


def get_hardware_config(tier_name: str) -> HardwareConfig:
    """Retrieve hardware configuration by tier name."""
    try:
        tier_enum = HardwareTierEnum(tier_name.upper())
        return HARDWARE_TIERS[tier_enum]
    except ValueError:
        return HARDWARE_TIERS[HardwareTierEnum.SOVEREIGN]
