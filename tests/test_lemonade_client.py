import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from proxy.backend_client.lemonade_client import LemonadeLLMClient, LLMBackendError

@pytest.fixture
async def client():
    c = LemonadeLLMClient(base_url="http://test-server/v1")
    yield c
    await c.close()

class MockResponse:
    def __init__(self, status_code, json_data=None, lines=None):
        self.status_code = status_code
        self._json_data = json_data
        self._lines = lines or []
        
    def json(self):
        return self._json_data
        
    async def aiter_lines(self):
        for line in self._lines:
            yield line

class MockStreamContext:
    def __init__(self, response):
        self.response = response
        
    async def __aenter__(self):
        return self.response
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_resolve_model_found(client):
    mock_resp = MockResponse(200, {"data": [{"id": "google/gemma-4-26B-A4B-it"}]})
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        result = await client._resolve_model("google/gemma-4-26B-A4B-it")
        assert result == "google/gemma-4-26B-A4B-it"

@pytest.mark.asyncio
async def test_resolve_model_fallback(client):
    mock_resp = MockResponse(200, {"data": [{"id": "some-other-model"}]})
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        result = await client._resolve_model("unknown-model")
        # Never silently switch to whatever model is listed first
        assert result == "unknown-model"

@pytest.mark.asyncio
async def test_resolve_model_substring(client):
    mock_resp = MockResponse(200, {"data": [{"id": "google/gemma-4"}]})
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        result = await client._resolve_model("gemma-4")
        assert result == "google/gemma-4"

@pytest.mark.asyncio
async def test_resolve_model_error(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Network")
        result = await client._resolve_model("requested")
        assert result == "requested"

@pytest.mark.asyncio
async def test_generate_stream_success(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, 'stream') as mock_stream:
         
        mock_get.return_value = MockResponse(200, {"data": [{"id": "test-model"}]})
        
        lines = [
            'data: {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}',
            'data: {"choices": [{"delta": {"content": "hello"}}]}',
            'data: [DONE]'
        ]
        mock_stream.return_value = MockStreamContext(MockResponse(200, lines=lines))
        
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi", model="test-model"):
            tokens.append(chunk)

        assert tokens == ["<thinking>", "thinking...", "</thinking>", "hello"]

@pytest.mark.asyncio
async def test_generate_stream_404_fallback(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, 'stream') as mock_stream:
         
        mock_get.return_value = MockResponse(200, {"data": [{"id": "test-model"}]})
        
        lines = [
            'data: {"choices": [{"text": "Fallback"}]}',
            'data: [DONE]'
        ]
        mock_stream.side_effect = [
            MockStreamContext(MockResponse(404)),
            MockStreamContext(MockResponse(200, lines=lines))
        ]
        
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi", model="test-model"):
            tokens.append(chunk)

        assert tokens == ["Fallback"]

@pytest.mark.asyncio
async def test_generate_stream_decode_error(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, 'stream') as mock_stream:
         
        mock_get.return_value = MockResponse(200, {"data": [{"id": "test-model"}]})
        
        lines = [
            'data: invalid json',
            'data: {"choices": [{"delta": {"content": "Valid"}}]}',
            'data: [DONE]'
        ]
        mock_stream.return_value = MockStreamContext(MockResponse(200, lines=lines))
        
        tokens = []
        async for chunk in client.generate_stream(prompt="Hi"):
            tokens.append(chunk)

        assert tokens == ["Valid"]

@pytest.mark.asyncio
async def test_generate_stream_error_status(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, 'stream') as mock_stream:
         
        mock_get.return_value = MockResponse(200, {"data": []})
        mock_stream.return_value = MockStreamContext(MockResponse(500))
        
        tokens = []
        with pytest.raises(LLMBackendError, match="HTTP 500"):
            async for chunk in client.generate_stream():
                tokens.append(chunk)
        # The error must not be emitted as if it were story text
        assert tokens == []

@pytest.mark.asyncio
async def test_generate_stream_connection_error_raises_instead_of_inventing_reply(client):
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, 'stream') as mock_stream:
         
        mock_get.return_value = MockResponse(200, {"data": []})
        mock_stream.side_effect = Exception("Connection Refused")
        
        tokens = []
        with pytest.raises(LLMBackendError, match="unreachable"):
            async for chunk in client.generate_stream():
                tokens.append(chunk)
        assert tokens == []


def _models(*entries):
    return MockResponse(200, {"data": [{"id": i, "labels": l} for i, l in entries]})


LEMONADE_LIST = _models(
    ("Flux-2-Klein-9B-GGUF", ["image"]),
    ("Gemma-4-26B-A4B-it-GGUF", ["chat"]),
    ("Gemma-4-E4B-it-GGUF", ["chat"]),
    ("Qwen3.8-27B-GGUF", ["chat", "reasoning"]),
    ("gemma3-4b-FLM", ["chat"]),
    ("hermes-coder", ["chat", "coding"]),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested, expected", [
    ("google/gemma-4-26B-A4B-it", "Gemma-4-26B-A4B-it-GGUF"),   # retired id -> GGUF build
    ("google/gemma-4-E4B-it", "Gemma-4-E4B-it-GGUF"),
    ("spm-sovereign-mesh", "Gemma-4-26B-A4B-it-GGUF"),          # SPM virtual id -> default
    ("", "Gemma-4-26B-A4B-it-GGUF"),
    ("Qwen3.8-27B-GGUF", "Qwen3.8-27B-GGUF"),                   # exact
    ("qwen3.8-27b-gguf", "Qwen3.8-27B-GGUF"),                   # case-insensitive
    ("e4b-it", "Gemma-4-E4B-it-GGUF"),                          # unique substring
    ("gemma", "gemma"),                                         # ambiguous: passed through, not guessed
    ("hermes-coder", "hermes-coder"),                           # coding alias is never chosen for chat...
    ("Flux-2-Klein-9B-GGUF", "Flux-2-Klein-9B-GGUF"),           # ...nor an image model; passed through as asked
])
async def test_resolve_model_against_lemonade_catalogue(client, requested, expected):
    with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = LEMONADE_LIST
        assert await client._resolve_model(requested) == expected


@pytest.mark.asyncio
async def test_unknown_model_never_falls_back_to_image_or_coding_model(client):
    with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _models(("Flux-2-Klein-9B-GGUF", ["image"]), ("hermes-coder", ["chat", "coding"]))
        assert await client._resolve_model("some-roleplay-model") == "some-roleplay-model"


@pytest.mark.asyncio
async def test_model_list_is_cached(client):
    with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = LEMONADE_LIST
        await client._resolve_model("Qwen3.8-27B-GGUF")
        await client._resolve_model("Gemma-4-E4B-it-GGUF")
        assert mock_get.await_count == 1


@pytest.mark.asyncio
async def test_extra_body_is_merged_into_payload(client):
    with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get, \
         patch.object(client.client, "stream") as mock_stream:
        mock_get.return_value = MockResponse(200, {"data": []})
        mock_stream.return_value = MockStreamContext(MockResponse(200, lines=["data: [DONE]"]))
        async for _ in client.generate_stream(prompt="x", extra_body={"chat_template_kwargs": {"enable_thinking": False}}):
            pass
        assert mock_stream.call_args.kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}


# --- External backend (LM Studio on another machine, 2026-10-13) --------------------

LMSTUDIO_LIST = MockResponse(200, {"data": [
    {"id": "google/gemma-4-26b-a4b-qat", "object": "model"},
    {"id": "qwen/qwen3.8-27b", "object": "model"},
    {"id": "google/gemma-4-e4b", "object": "model"},
    {"id": "text-embedding-nomic-embed-text-v1.5", "object": "model"},
]})


@pytest.mark.asyncio
@pytest.mark.parametrize("requested, expected", [
    ("Gemma-4-26B-A4B-it-GGUF", "google/gemma-4-26b-a4b-qat"),   # Lemonade name -> LM Studio id
    ("google/gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b-qat"),  # retired HF-style id
    ("Qwen3.8-27B-GGUF", "qwen/qwen3.8-27b"),
    ("Gemma-4-E4B-it-GGUF", "google/gemma-4-e4b"),
    ("gemma-4", "gemma-4"),                                        # ambiguous: passed through
    ("some-roleplay-model", "some-roleplay-model"),
])
async def test_resolve_model_across_naming_schemes(client, requested, expected):
    with patch.object(client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = LMSTUDIO_LIST
        assert await client._resolve_model(requested) == expected


@pytest.mark.asyncio
async def test_base_url_follows_backend_setting(monkeypatch):
    """The client used to be hard-wired to localhost:13305 whatever BACKEND_LLM_URL said.
    Settings are re-read on every call, so a config change needs no restart."""
    import config.manager as cm
    current = {"BACKEND_LLM_URL": "http://10.0.0.5:1234/v1/"}
    stub = MagicMock(); stub.get_settings = lambda: dict(current)
    monkeypatch.setattr(cm, "get_settings_manager", lambda: stub)
    c = LemonadeLLMClient()
    assert c.base_url == "http://10.0.0.5:1234/v1"
    current["BACKEND_LLM_URL"] = "http://localhost:13305/v1"
    assert c.base_url == "http://localhost:13305/v1"
    current["BACKEND_LLM_URL"] = ""
    from proxy.backend_client import lemonade_client as lc
    assert c.base_url == lc.DEFAULT_BASE_URL.rstrip("/")   # empty setting -> built-in default
    explicit = LemonadeLLMClient(base_url="http://test-server/v1/")
    assert explicit.base_url == "http://test-server/v1"    # explicit URL wins
    await c.close(); await explicit.close()
