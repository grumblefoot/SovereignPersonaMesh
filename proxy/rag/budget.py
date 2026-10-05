"""
Token budget (OPEN-008, docs/plans/token_budget.md).

Phase P0: the safety layer. Every generation request gets
``max_tokens = min(configured ceiling, window - prompt estimate)`` with a hard
output floor, so no request ever asks the backend for more room than the context
window has left (the old code always sent ``backend_max_tokens`` — 128000 — flat).

The counter is the calibrated heuristic from the plan §3: chars / ratio + a fixed
per-message template overhead. The ratio starts conservative (3.2 chars/token) and
phase P2 updates it per model from streamed ``usage.prompt_tokens`` via an EMA.
The P1 allocator (partitions, trim order) builds on these same primitives.
"""
import logging
from typing import Any, Dict, List, Optional

from config.hardware_tiers import HardwareConfig

logger = logging.getLogger(__name__)

HEURISTIC_CHARS_PER_TOKEN = 3.2     # conservative: over-counts slightly, never starves output
PER_MESSAGE_OVERHEAD = 6            # chat-template tokens per message (plan §3)
OUTPUT_FLOOR = 256                  # floors.output_reserve default: never ask for less

# P2 calibration state: model id -> EMA chars/token (updated from streamed usage).
_calibrated_ratio: Dict[str, float] = {}
_EMA_ALPHA = 0.3


def ratio_for(model: str) -> float:
    return _calibrated_ratio.get(model, HEURISTIC_CHARS_PER_TOKEN)


def calibrate(model: str, prompt_chars: int, reported_prompt_tokens: int) -> None:
    """P2: fold the backend's reported prompt_tokens into the per-model ratio."""
    if reported_prompt_tokens <= 0 or prompt_chars <= 0:
        return
    observed = prompt_chars / reported_prompt_tokens
    prev = _calibrated_ratio.get(model, HEURISTIC_CHARS_PER_TOKEN)
    _calibrated_ratio[model] = prev + _EMA_ALPHA * (observed - prev)


def estimate_tokens(text: str, model: str = "") -> int:
    if not text:
        return 0
    return int(len(text) / ratio_for(model)) + 1


def estimate_messages(messages: List[Dict[str, Any]], model: str = "") -> int:
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        total += estimate_tokens(content or "", model) + PER_MESSAGE_OVERHEAD
    return total


def context_window(settings: Dict[str, Any], hw_config: HardwareConfig) -> int:
    """Window precedence (plan §1): explicit override > hardware tier. Backend
    detection (BackendProfile) joins in P1; the tier fallback is the PRD value."""
    override = settings.get("context_window_override")
    try:
        if override:
            return int(override)
    except (TypeError, ValueError):
        pass
    return int(hw_config.max_context_tokens)


def clamp_max_tokens(
    messages: List[Dict[str, Any]],
    settings: Dict[str, Any],
    hw_config: HardwareConfig,
    model: str = "",
    configured_ceiling: Optional[int] = None,
) -> int:
    """P0: the max_tokens actually sent to the backend.

    min(configured ceiling, window - prompt estimate - safety margin), floored at
    OUTPUT_FLOOR. When even the floor doesn't fit, the floor is sent anyway and a
    warning logged — refusing the turn outright is the P1 allocator's job, once it
    can tell trimmable history from the untrimmable card.
    """
    window = context_window(settings, hw_config)
    margin = max(64, int(window * 0.05))
    prompt = estimate_messages(messages, model)
    room = window - prompt - margin
    try:
        ceiling = int(configured_ceiling if configured_ceiling is not None
                      else settings.get("backend_max_tokens", 2048))
    except (TypeError, ValueError):
        ceiling = 2048
    result = max(OUTPUT_FLOOR, min(ceiling, room))
    if room < OUTPUT_FLOOR:
        logger.warning(
            f"[TokenBudget] prompt estimate {prompt} leaves {room} tokens of a "
            f"{window} window; sending the {OUTPUT_FLOOR} floor (P1 trimming pending).")
    return result
