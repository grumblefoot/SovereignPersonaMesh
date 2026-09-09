"""TDD tests for _dispatch_lore_extraction — the real production function.

These tests call `_dispatch_lore_extraction` directly and verify its real behavior
(mocking only the external dependencies: LoreExtractionWorker constructor and its
async methods, settings manager, and the module-level _background_tasks set).

The key principle: we do NOT mirror the production code in a helper and test that.
We test the actual function.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from config.manager import reset_settings_manager
from proxy.api.routes import (
    set_db_pool,
    _dispatch_lore_extraction,
    _background_tasks,
    ChatCompletionMessage,
    LoreExtractionWorker,
)


@pytest.fixture(autouse=True)
def clean_background_tasks():
    """Clear _background_tasks between tests to prevent cross-test contamination."""
    _background_tasks.clear()
    yield
    _background_tasks.clear()


@pytest.fixture
def mock_db_pool():
    """Create a mock db_pool for testing."""
    pool = MagicMock()
    conn_mock = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__.return_value = conn_mock
    acquire_cm.__aexit__.return_value = False
    pool.acquire = MagicMock(return_value=acquire_cm)
    conn_mock.fetchval = AsyncMock(return_value=None)
    return pool


def _make_request(messages, model="test-model"):
    """Create a ChatCompletionRequest mock with given messages."""
    req = MagicMock()
    req.messages = messages
    req.model = model
    return req


def _make_messages(user_count=1, include_system=True):
    """Create a list of ChatCompletionMessage objects."""
    msgs = []
    if include_system:
        msgs.append(ChatCompletionMessage(role="system", content="System prompt"))
    for i in range(user_count):
        msgs.append(ChatCompletionMessage(role="user", content=f"User message {i+1}"))
    return msgs


# ============================================================
# RED-GREEN tests for _dispatch_lore_extraction
# ============================================================


class TestDispatchNoDbPool:
    """Test: when _db_pool is None, _dispatch_lore_extraction returns early."""

    def test_no_db_pool_early_return(self):
        """FAIL: Production code must check _db_pool and return immediately."""
        set_db_pool(None)
        req = _make_request(_make_messages(user_count=1))

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            # This should NOT call get_settings_manager at all
            _dispatch_lore_extraction(req, "s1", "char", "thoughts", "response")
            mock_sm.assert_not_called()


class TestDispatchInitialExtraction:
    """Test: first user message triggers extract_initial_rules."""

    @pytest.mark.asyncio
    async def test_first_user_triggers_initial(self, mock_db_pool):
        """FAIL: Production code must detect user_msg_count <= 1 and call extract_initial_rules."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request([
            ChatCompletionMessage(role="system", content="System prompt"),
            ChatCompletionMessage(role="user", content="First message"),
        ])

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "thoughts", "response")

                # LoreExtractionWorker must be instantiated
                assert mock_worker_cls.call_count == 1
                # extract_initial_rules must be called (task created)
                assert mock_extractor.extract_initial_rules.called

    @pytest.mark.asyncio
    async def test_initial_passes_full_turn_history(self, mock_db_pool):
        """FAIL: Production code must build full_turn_history including assistant turn."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request([
            ChatCompletionMessage(role="system", content="System prompt"),
            ChatCompletionMessage(role="user", content="Hello world"),
        ])

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(
                    req, "s1", "char",
                    "I should greet the user.",
                    "Hello there, traveler!"
                )

                # Get the context string passed to extract_initial_rules
                call_args = mock_extractor.extract_initial_rules.call_args
                context_text = call_args[0][2]  # positional arg index 2

                # The context must include the assistant turn with both monologue and public response
                assert "I should greet the user." in context_text
                assert "Hello there, traveler!" in context_text

    @pytest.mark.asyncio
    async def test_initial_task_added_to_background_tasks(self, mock_db_pool):
        """FAIL: Production code must add task to _background_tasks set."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=1))

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_worker_cls.return_value = AsyncMock()

                before = len(_background_tasks)
                _dispatch_lore_extraction(req, "s1", "char", "t", "r")
                after = len(_background_tasks)

                assert after == before + 1
                assert len(_background_tasks) == 1


class TestDispatchPeriodicExtraction:
    """Test: on-cadence messages trigger periodic_review_rules."""

    @pytest.mark.asyncio
    async def test_cadence_3_fires_at_4th_user_msg(self, mock_db_pool):
        """FAIL: (user_msg_count - 1) % cadence == 0 -> 4th message (count=4, 3%3=0)."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request([
            ChatCompletionMessage(role="system", content="System prompt"),
            ChatCompletionMessage(role="user", content="Msg 1"),
            ChatCompletionMessage(role="user", content="Msg 2"),
            ChatCompletionMessage(role="user", content="Msg 3"),
            ChatCompletionMessage(role="user", content="Msg 4"),
        ])

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                assert mock_worker_cls.call_count == 1
                assert mock_extractor.periodic_review_rules.called
                assert not mock_extractor.extract_initial_rules.called

    @pytest.mark.asyncio
    async def test_off_cadence_does_not_extract(self, mock_db_pool):
        """FAIL: (user_msg_count - 1) % cadence != 0 -> no extraction."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=3))
        # 3 user msgs: (3-1) % 3 = 2 != 0 -> no extraction

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                assert mock_worker_cls.call_count == 1
                assert not mock_extractor.periodic_review_rules.called
                assert not mock_extractor.extract_initial_rules.called

    @pytest.mark.asyncio
    async def test_periodic_passes_recent_messages(self, mock_db_pool):
        """FAIL: periodic_review_rules must receive full_turn_history[-(cadence*2):]."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=4))

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                # recent = last cadence*2 = 6 messages
                # But we only have 4 messages total + 1 assistant = 5
                # So recent = all 5
                call_args = mock_extractor.periodic_review_rules.call_args
                recent = call_args[0][2]
                assert isinstance(recent, list)
                assert len(recent) >= 4  # At least the 4 user messages


class TestDispatchModelSelection:
    """Test: alternate model selection from settings."""

    @pytest.mark.asyncio
    async def test_uses_alternate_model_when_configured(self, mock_db_pool):
        """FAIL: Production code must read use_alternate_extraction_model from settings."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=1), model="main-model")

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
                "use_alternate_extraction_model": True,
                "alternate_extraction_model_name": "alt-model-v2",
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                # ext_model should be "alt-model-v2"
                call_args = mock_extractor.extract_initial_rules.call_args
                assert call_args.kwargs['model'] == "alt-model-v2"

    @pytest.mark.asyncio
    async def test_uses_request_model_when_no_alternate(self, mock_db_pool):
        """FAIL: Production code must fall back to request.model."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=1), model="main-model")

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
                "use_alternate_extraction_model": False,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                call_args = mock_extractor.extract_initial_rules.call_args
                assert call_args.kwargs['model'] == "main-model"


class TestDispatchAssistantTurnFormat:
    """Test: assistant turn includes both monologue and public response."""

    @pytest.mark.asyncio
    async def test_assistant_turn_includes_monologue_and_public(self, mock_db_pool):
        """FAIL: assistant_turn content must be f'</think>\n{inner_monologue}\n</think>\n{public_resp}'."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request(_make_messages(user_count=1))

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(
                    req, "s1", "char",
                    "I need to think about this carefully.",
                    "Welcome to the tavern!"
                )

                call_args = mock_extractor.extract_initial_rules.call_args
                context_text = call_args[0][2]

                assert "I need to think about this carefully." in context_text
                assert "Welcome to the tavern!" in context_text


class TestDispatchPrefiltering:
    """Test: prefill stubs (</think>) are filtered from user count."""

    @pytest.mark.asyncio
    async def test_prefill_stubs_filtered_from_count(self, mock_db_pool):
        """FAIL: assistant messages containing only '</think>' or '<think>' should not count."""
        set_db_pool(mock_db_pool)
        reset_settings_manager()

        req = _make_request([
            ChatCompletionMessage(role="system", content="System"),
            ChatCompletionMessage(role="user", content="Hello"),
            # These are prefill stubs from the streaming init
            ChatCompletionMessage(role="assistant", content="<think>"),
            ChatCompletionMessage(role="assistant", content="</think>"),
        ])

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            mock_sm.return_value.get_settings.return_value = {
                "periodic_review_cadence": 3,
            }
            with patch('proxy.api.routes.LoreExtractionWorker') as mock_worker_cls:
                mock_extractor = AsyncMock()
                mock_worker_cls.return_value = mock_extractor

                _dispatch_lore_extraction(req, "s1", "char", "t", "r")

                # Should trigger initial (1 user msg, stubs filtered)
                assert mock_extractor.extract_initial_rules.called


class TestDispatchNoExtractionWhenNoPool:
    """Test: no extraction happens when _db_pool is None."""

    @pytest.mark.asyncio
    async def test_no_extraction_without_db_pool(self):
        """FAIL: Production code must return early when _db_pool is None."""
        set_db_pool(None)
        req = _make_request(_make_messages(user_count=5))

        with patch('proxy.api.routes.get_settings_manager') as mock_sm:
            _dispatch_lore_extraction(req, "s1", "char", "t", "r")
            mock_sm.assert_not_called()


class TestLogTaskDone:
    """Test: _log_task_done logs exceptions from failed background tasks."""

    @pytest.mark.asyncio
    async def test_log_task_done_catches_exception(self):
        """FAIL: _log_task_done must catch and log exceptions from task.result()."""
        from proxy.api.routes import _log_task_done

        # Create a task that will raise
        async def failing_task():
            raise ValueError("extraction failed")

        task = asyncio.create_task(failing_task())
        await asyncio.sleep(0)  # Let task fail

        with patch('proxy.api.routes.logger') as mock_logger:
            _log_task_done(task)
            # Should NOT raise; instead logs the error
            mock_logger.error.assert_called_once()
            assert "extraction failed" in str(mock_logger.error.call_args)
