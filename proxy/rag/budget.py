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


# ── P1: allocator (plan §4) ─────────────────────────────────────────────────
# PRD 5.3 split (decision 16), percent mode. 'history' is elastic: it gets
# whatever the other partitions leave behind.
PARTITIONS_PCT = {"system_card": 12, "lore": 5, "memories": 15, "sensory_spatial": 6,
                  "monologue_reserve": 8, "output_reserve": 12}
HISTORY_FLOOR = 1024
DIRECTIVE_OVERHEAD = 700            # SPM directive + prefill + GM block, estimated


from dataclasses import dataclass, field as _field


@dataclass
class BudgetReport:
    window: int = 0
    refused: bool = False
    refusal_reason: str = ""
    used: Dict[str, int] = _field(default_factory=dict)
    trimmed: Dict[str, int] = _field(default_factory=dict)   # items dropped per partition
    notes: List[str] = _field(default_factory=list)


def _strip_think(text: str) -> str:
    import re
    return re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", "", text, flags=re.DOTALL)


def allocate(
    *,
    settings: Dict[str, Any],
    hw_config: HardwareConfig,
    model: str,
    system_text: str,
    memories: List[Dict[str, Any]],
    lore: Dict[str, List[Dict[str, Any]]],
    history: List[Dict[str, str]],
):
    """Fit memories, lore and history into the window (plan §4). Returns
    (memories', lore', history', report).

    Never trimmed: the system card, the SPM directive (reserved via
    DIRECTIVE_OVERHEAD) and the latest user turn. If those alone do not fit,
    report.refused is set and the caller must send a visible notice instead of a
    silently truncated card (decision 16: refuse, never truncate the card).

    Trim order: memories (lowest-ranked first — the lists arrive best-first),
    lore triggers (invariants survive longest), history oldest-first in whole
    user/assistant pairs, with monologue stripped from older assistant turns
    before any pair is dropped.
    """
    report = BudgetReport(window=context_window(settings, hw_config))
    window_eff = report.window - max(64, int(report.window * 0.05))
    pct = lambda name: int(report.window * PARTITIONS_PCT[name] / 100)

    out_res = max(OUTPUT_FLOOR, pct("output_reserve"))
    mono_res = pct("monologue_reserve")
    card = estimate_tokens(system_text, model) + DIRECTIVE_OVERHEAD
    latest_user = next((m for m in reversed(history) if m.get("role") == "user"), None)
    latest_tokens = estimate_tokens((latest_user or {}).get("content", ""), model) + PER_MESSAGE_OVERHEAD

    report.used["system_card"] = card
    if card + latest_tokens + out_res > window_eff:
        report.refused = True
        report.refusal_reason = (
            f"The character card/system prompt alone needs ~{card} tokens of a "
            f"{report.window}-token window; nothing can be trimmed to make room.")
        return [], {"invariants": [], "triggers": []}, history, report

    # memories: keep best-first while they fit their partition
    kept_mem, mem_used = [], 0
    for m in memories:
        t = estimate_tokens(str(m.get("sensory_input", "")) + str(m.get("inner_monologue", "")), model)
        if mem_used + t <= pct("memories"):
            kept_mem.append(m)
            mem_used += t
    report.used["memories"] = mem_used
    report.trimmed["memories"] = len(memories) - len(kept_mem)

    # lore: invariants first (trimmed last), then triggers best-first
    kept_inv, kept_trig, lore_used = [], [], 0
    for r in lore.get("invariants", []):
        t = estimate_tokens(str(r.get("rule_text", "")), model)
        if lore_used + t <= pct("lore"):
            kept_inv.append(r)
            lore_used += t
    for r in lore.get("triggers", []):
        t = estimate_tokens(str(r.get("rule_text", "")), model)
        if lore_used + t <= pct("lore"):
            kept_trig.append(r)
            lore_used += t
    report.used["lore"] = lore_used
    report.trimmed["lore"] = (len(lore.get("invariants", [])) - len(kept_inv)
                              + len(lore.get("triggers", [])) - len(kept_trig))

    # history: elastic remainder, never below the floor if the room exists
    hist_budget = window_eff - card - latest_tokens - out_res - mono_res - mem_used - lore_used - pct("sensory_spatial")
    hist_budget = max(hist_budget, min(HISTORY_FLOOR, window_eff - card - latest_tokens - out_res))

    kept_hist = list(history)
    stripped = 0

    def hist_tokens():
        return estimate_messages(kept_hist, model)

    # 1) strip monologue from assistant turns older than the last two
    if hist_tokens() > hist_budget:
        for i, m in enumerate(kept_hist[:-4]):
            if m.get("role") == "assistant" and "<think" in (m.get("content") or ""):
                kept_hist[i] = {**m, "content": _strip_think(m["content"])}
                stripped += 1
                if hist_tokens() <= hist_budget:
                    break
    # 2) drop whole turns from the old end (keeping the latest user turn)
    dropped = 0
    while kept_hist and hist_tokens() > hist_budget:
        if len(kept_hist) == 1 and kept_hist[0] is latest_user:
            break
        kept_hist.pop(0)
        dropped += 1
    if latest_user is not None and latest_user not in kept_hist:
        kept_hist.append(latest_user)           # the latest user turn is never trimmed

    report.used["history"] = hist_tokens()
    report.trimmed["history"] = dropped
    if stripped:
        report.notes.append(f"monologue stripped from {stripped} older assistant turn(s)")
    if dropped:
        report.notes.append(f"{dropped} oldest history message(s) dropped")
    return kept_mem, {"invariants": kept_inv, "triggers": kept_trig}, kept_hist, report
