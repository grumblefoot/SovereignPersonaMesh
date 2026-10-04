"""Sprint 0 regression tests: request plumbing, prompt assembly, GM idempotency keys,
hardware-tier resolution, and embeddings phase 0 (no fake vectors, ever)."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from starlette.datastructures import Headers

from config.hardware_tiers import HARDWARE_TIERS, HardwareTierEnum
from proxy.api.routes import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    _assemble_system_prompt,
    _extract_session_id,
    _gm_action_key,
)
from proxy.rag.prompt_builder import CognitivePromptBuilder
from proxy.rag.retriever import EpisodicRAGRetriever
from scripts.onnx_embedder import CPUEmbeddingEngine

FIXTURE = Path(__file__).parent / "fixtures" / "test_chat_payload.json"


def _req(headers=None):
    return SimpleNamespace(headers=Headers(headers or {}))


# ── B2: body session_id reachable again; B9: max_tokens default gone ────────

def test_request_model_declares_identity_fields():
    r = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                              session_id="chat-42", user="zosazin")
    dumped = r.model_dump()
    assert dumped["session_id"] == "chat-42"
    assert dumped["user"] == "zosazin"
    assert r.max_tokens is None  # ST sends its own cap; 128000 default is gone


def test_body_session_id_is_used():
    body = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                 session_id="chat-42").model_dump()
    assert _extract_session_id(_req(), body) == "chat-42"


def test_spm_chat_id_header_wins_and_is_sanitized():
    body = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                 session_id="body-id").model_dump()
    sid = _extract_session_id(_req({"X-SPM-Chat-ID": "4FD2-aB; DROP TABLE x"}), body)
    assert sid == "st_chat_4fd2_ab_drop_table_x"
    assert all(c.isalnum() or c == "_" for c in sid)


def test_unresolved_macro_header_is_ignored():
    body = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                 session_id="body-id").model_dump()
    assert _extract_session_id(_req({"X-SPM-Chat-ID": "{{spmChatId}}"}), body) == "body-id"


def test_legacy_fallback_unchanged():
    body = {"messages": [{"role": "system", "content": "[Character: Mira]"},
                         {"role": "user", "content": "hi"}], "user": "tom"}
    assert _extract_session_id(_req(), body) == "st_tom_mira"


# ── B1 + decision 12: system prompt assembly ────────────────────────────────

SETTINGS_DEFAULT = {"st_passthrough_shared_notes": False}


def _sys(*contents):
    return [ChatCompletionMessage(role="system", content=c) for c in contents]


def test_all_system_messages_survive_not_just_the_first():
    msgs = _sys("Write Mira's next reply.", "[World Info: the port town floods in autumn]",
                "[Character: Mira] A tavern keeper.")
    out = _assemble_system_prompt(msgs, SETTINGS_DEFAULT)
    assert "World Info" in out and "tavern keeper" in out and out.startswith("Write Mira's")


def test_summary_and_authors_note_stripped_by_default():
    msgs = _sys("Main prompt.",
                "[Summary: the antagonist secretly plans to betray the party]",
                "[Author's Note: keep the villain's motives hidden]",
                "Author's note: style guidance here",
                "[Character: Mira]")
    out = _assemble_system_prompt(msgs, SETTINGS_DEFAULT)
    assert "betray" not in out and "hidden" not in out and "style guidance" not in out
    assert "Main prompt." in out and "[Character: Mira]" in out


def test_passthrough_setting_keeps_shared_notes():
    msgs = _sys("Main.", "[Summary: no secrets in this chat]")
    out = _assemble_system_prompt(msgs, {"st_passthrough_shared_notes": True})
    assert "no secrets" in out


def test_real_fixture_keeps_card_and_persona_drops_summary():
    payload = json.loads(FIXTURE.read_text())
    msgs = [ChatCompletionMessage(**m) for m in payload["messages"]]
    out = _assemble_system_prompt(msgs, SETTINGS_DEFAULT)
    assert '[character("Arvenia")' in out            # card kept
    assert "Vardus is a tall and fit human male" in out  # persona visible to characters
    assert "[Summary:" not in out                    # shared summary stripped


# ── B5: deterministic GM idempotency keys ───────────────────────────────────

def test_gm_action_key_is_deterministic_and_distinct():
    a = {"type": "MOVE", "entity": "Tom", "room_id": "cellar"}
    k1 = _gm_action_key("s1", 7, 0, a)
    assert k1 == _gm_action_key("s1", 7, 0, dict(a))          # regenerate → same key
    assert k1 != _gm_action_key("s1", 8, 0, a)                # next turn → new key
    assert k1 != _gm_action_key("s1", 7, 1, a)                # second action → new key
    assert k1 != _gm_action_key("s2", 7, 0, a)                # other session → new key
    assert k1.startswith("gm-") and len(k1) == 35


# ── B8: hardware tier resolved from the environment ─────────────────────────

def test_prompt_builder_reads_hardware_tier_env(monkeypatch):
    monkeypatch.setenv("SPM_HARDWARE_TIER", "EXPERIMENTAL")
    assert CognitivePromptBuilder().config is HARDWARE_TIERS[HardwareTierEnum.EXPERIMENTAL]
    monkeypatch.setenv("SPM_HARDWARE_TIER", "nonsense")
    assert CognitivePromptBuilder().config is HARDWARE_TIERS[HardwareTierEnum.SOVEREIGN]


# ── Embeddings phase 0: the stub never fabricates vectors ───────────────────

@pytest.mark.asyncio
async def test_stub_embedder_returns_none_and_reports_unavailable():
    eng = CPUEmbeddingEngine()
    assert eng.available is False
    assert await eng.generate_embedding("some text") is None
    assert await eng.batch_generate_embeddings(["a", "b"]) == [None, None]


@pytest.mark.asyncio
async def test_retriever_skips_vector_search_without_query_embedding():
    retriever = EpisodicRAGRetriever(MagicMock())  # pool must never be touched
    out = await retriever.retrieve_memories(character_id="mira", query_embedding=None,
                                            session_id="s1")
    assert out == []
