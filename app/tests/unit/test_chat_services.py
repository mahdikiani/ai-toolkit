"""Unit tests for chat services."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.core.exceptions import BaseHTTPException

from apps.language.chat.services import (
    SessionTitleSuggestion,
    complete_assistant_message,
    evaluate_session_title,
    maybe_apply_session_title_if_ready,
    messages_as_openrouter,
    openrouter_headers,
    proxy_chat_completions,
    thread_model,
)
from apps.language.completion import services as completion_services


@pytest.mark.unit
class TestThreadModel:
    """Tests for thread_model function."""

    def test_returns_thread_model_when_set(self) -> None:
        """thread_model should return thread's chat_model when set."""
        thread = MagicMock()
        thread.chat_model = "anthropic/claude-3"

        result = thread_model(thread)

        assert result == "anthropic/claude-3"

    def test_falls_back_to_default_when_none(self) -> None:
        """thread_model should fall back to Settings.default_model when None."""
        thread = MagicMock()
        thread.chat_model = None

        with patch("apps.language.chat.services.Settings") as mock_settings:
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = thread_model(thread)

        assert result == "openai/gpt-4o-mini"


@pytest.mark.unit
class TestOpenrouterHeaders:
    """Tests for openrouter_headers function."""

    def test_returns_headers_when_key_configured(self) -> None:
        """openrouter_headers should return headers when API key is set."""
        with patch(
            "apps.language.chat.services.openrouter_client.build_headers",
            return_value={"Authorization": "Bearer test_key"},
        ):
            headers = openrouter_headers()

        assert "Authorization" in headers

    def test_raises_503_when_key_missing(self) -> None:
        """openrouter_headers should raise 503 when API key is not configured."""
        with (
            patch(
                "apps.language.chat.services.openrouter_client.build_headers",
                side_effect=ValueError("No API key"),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            openrouter_headers()

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 503


@pytest.mark.unit
class TestCompletionProxyServices:
    """Tests for OpenAI-compatible completion proxy services."""

    async def test_proxy_adds_default_model_and_meters_success(self) -> None:
        """Verify successful proxying uses the default model and meters usage."""
        mock_response = MagicMock()
        mock_response.content = b'{"choices": [], "model": "openai/gpt-4o-mini"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.status_code = 200

        with (
            patch.object(
                completion_services.Settings, "default_model", "default/model"
            ),
            patch(
                "apps.language.completion.services.openrouter_client.post_chat_completion_unchecked",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as post_mock,
            patch(
                "apps.language.completion.services.meter_completion_response",
                new_callable=AsyncMock,
            ) as meter_mock,
        ):
            content, ctype, status = await completion_services.proxy_chat_completions(
                {"messages": []},
                user_id="user-1",
            )

        assert content == mock_response.content
        assert ctype == "application/json"
        assert status == 200
        assert post_mock.await_args.args[0]["model"] == "default/model"
        meter_mock.assert_awaited_once_with(mock_response.content, user_id="user-1")

    async def test_meter_completion_response_records_usage(self) -> None:
        """Verify metering records usage from a completion response."""
        with (
            patch(
                "apps.language.completion.services.openrouter_client.extract_provider_meta",
                return_value={
                    "model": "m",
                    "usage": {"total_tokens": 10},
                    "raw_cost": None,
                },
            ) as meta_mock,
            patch(
                "apps.language.completion.services.finance.estimate_text_cost",
                return_value=12.5,
            ) as cost_mock,
            patch(
                "apps.language.completion.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as meter_mock,
        ):
            await completion_services.meter_completion_response(
                b'{"id": "r1"}', user_id="user-1"
            )

        meta_mock.assert_called_once()
        cost_mock.assert_called_once()
        meter_mock.assert_awaited_once()

    async def test_stream_adds_default_model(self) -> None:
        """Verify streamed requests use the configured default model."""

        async def mock_stream(payload: dict[str, object]) -> AsyncIterator[bytes]:
            assert payload["model"] == "default/model"
            await asyncio.sleep(0)
            yield b"data: chunk\n\n"

        with (
            patch.object(
                completion_services.Settings, "default_model", "default/model"
            ),
            patch(
                "apps.language.completion.services.openrouter_client.stream_chat_completion_bytes",
                side_effect=mock_stream,
            ),
        ):
            stream = completion_services.proxy_chat_completions_raw_stream({
                "messages": []
            })
            chunks = [chunk async for chunk in stream]

        assert chunks == [b"data: chunk\n\n"]

    async def test_stream_maps_openrouter_error(self) -> None:
        """Verify streamed OpenRouter errors map to API errors."""

        async def failing_stream(payload: dict[str, object]) -> AsyncIterator[bytes]:
            await asyncio.sleep(0)
            raise completion_services.openrouter_client.OpenRouterError(429, "limited")
            yield  # pragma: no cover

        with (
            patch(
                "apps.language.completion.services.openrouter_client.stream_chat_completion_bytes",
                return_value=failing_stream({}),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            async for _ in completion_services.proxy_chat_completions_raw_stream({
                "model": "m"
            }):
                pass

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 429


@pytest.mark.unit
class TestMessagesAsOpenrouter:
    """Tests for messages_as_openrouter function."""

    async def test_returns_formatted_messages(self) -> None:
        """messages_as_openrouter should return messages in OpenRouter format."""
        thread = MagicMock()
        thread.user_id = "user_123"
        thread.uid = "thread_uid_123"

        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_msg1.content = "Hello"

        mock_msg2 = MagicMock()
        mock_msg2.role = "assistant"
        mock_msg2.content = "Hi there!"

        with patch(
            "apps.language.chat.services.ChatMessage.list_items",
            new_callable=AsyncMock,
            return_value=[mock_msg1, mock_msg2],
        ):
            result = await messages_as_openrouter(thread)

        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    async def test_returns_empty_list_when_no_messages(self) -> None:
        """messages_as_openrouter should return empty list when no messages."""
        thread = MagicMock()
        thread.user_id = "user_123"
        thread.uid = "thread_uid_123"

        with patch(
            "apps.language.chat.services.ChatMessage.list_items",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await messages_as_openrouter(thread)

        assert result == []


@pytest.mark.unit
class TestCompleteAssistantMessage:
    """Tests for complete_assistant_message function."""

    async def test_creates_assistant_message_on_success(self) -> None:
        """complete_assistant_message should create and return assistant message."""
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"
        thread.user_id = "user_123"

        mock_messages = [{"role": "user", "content": "Hello"}]
        mock_openrouter_response = {
            "choices": [{"message": {"content": "Hello back!"}}],
            "model": "openai/gpt-4o-mini",
            "usage": {"total_tokens": 20},
        }
        mock_created_msg = MagicMock()
        mock_created_msg.content = "Hello back!"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=mock_messages,
            ),
            patch(
                "apps.language.chat.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                return_value=mock_openrouter_response,
            ),
            patch(
                "apps.language.chat.services.ChatMessage.create_item",
                new_callable=AsyncMock,
                return_value=mock_created_msg,
            ),
        ):
            result = await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        assert result == mock_created_msg

    async def test_raises_400_when_no_messages(self) -> None:
        """complete_assistant_message should raise 400 when thread has no messages."""
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[],
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 400

    async def test_raises_503_when_api_key_missing(self) -> None:
        """complete_assistant_message should raise 503 when API key is missing."""
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[{"role": "user", "content": "Hello"}],
            ),
            patch(
                "apps.language.chat.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                side_effect=ValueError("No API key"),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 503

    async def test_raises_502_on_runtime_error(self) -> None:
        """complete_assistant_message should raise 502 on OpenRouter RuntimeError."""
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[{"role": "user", "content": "Hello"}],
            ),
            patch(
                "apps.language.chat.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 502


@pytest.mark.unit
class TestProxyChatCompletions:
    """Tests for proxy_chat_completions function."""

    async def test_returns_response_content_and_status(self) -> None:
        """proxy_chat_completions should return content, content-type, and status."""
        mock_response = MagicMock()
        mock_response.content = b'{"choices": []}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.status_code = 200

        with patch(
            "apps.language.chat.services.openrouter_client.post_chat_completion_unchecked",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            content, ctype, status = await proxy_chat_completions({"model": "gpt-4"})

        assert content == b'{"choices": []}'
        assert ctype == "application/json"
        assert status == 200

    async def test_raises_503_when_api_key_missing(self) -> None:
        """proxy_chat_completions should raise 503 when API key is not configured."""
        with (
            patch(
                "apps.language.chat.services.openrouter_client.post_chat_completion_unchecked",
                new_callable=AsyncMock,
                side_effect=ValueError("No API key"),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            await proxy_chat_completions({"model": "gpt-4"})

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 503


@pytest.mark.unit
class TestProxyChatCompletionsRawStream:
    """Tests for proxy_chat_completions_raw_stream function."""

    async def test_yields_streaming_chunks(self) -> None:
        """proxy_chat_completions_raw_stream should yield SSE bytes from OpenRouter."""

        async def mock_stream() -> AsyncIterator[bytes]:
            await asyncio.sleep(0)
            for chunk in [
                b"data: chunk1\n\n",
                b"data: chunk2\n\n",
                b"data: [DONE]\n\n",
            ]:
                yield chunk

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_completion_bytes",
            return_value=mock_stream(),
        ):
            from apps.language.chat.services import proxy_chat_completions_raw_stream

            chunks = [
                chunk
                async for chunk in proxy_chat_completions_raw_stream({"model": "gpt-4"})
            ]

        assert len(chunks) == 3
        assert chunks[0] == b"data: chunk1\n\n"
        assert chunks[1] == b"data: chunk2\n\n"
        assert chunks[2] == b"data: [DONE]\n\n"

    async def test_raises_503_when_api_key_missing(self) -> None:
        """Raise 503 when the API key is missing."""

        async def failing_stream() -> AsyncIterator[bytes]:
            await asyncio.sleep(0)
            error = ValueError("No API key")
            raise error
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_completion_bytes",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import proxy_chat_completions_raw_stream

            with pytest.raises(BaseHTTPException) as exc_info:
                async for _ in proxy_chat_completions_raw_stream({"model": "gpt-4"}):
                    pass

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY is not configured" in exc_info.value.detail

    async def test_raises_http_exception_on_openrouter_error(self) -> None:
        """Raise an HTTPException for an OpenRouter error."""

        async def failing_stream() -> AsyncIterator[bytes]:
            from utils.integrations.openrouter import OpenRouterError

            await asyncio.sleep(0)
            raise OpenRouterError(status_code=429, detail="Rate limit exceeded")
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_completion_bytes",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import proxy_chat_completions_raw_stream

            with pytest.raises(BaseHTTPException) as exc_info:
                async for _ in proxy_chat_completions_raw_stream({"model": "gpt-4"}):
                    pass

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail


@pytest.mark.unit
class TestIterOpenrouterSseDeltas:
    """Tests for iter_openrouter_sse_deltas function."""

    async def test_yields_text_deltas(self) -> None:
        """Yield text deltas from a streaming response."""

        async def mock_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0)
            for delta in ["Hello", " ", "world", "!"]:
                yield delta

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=mock_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            deltas = [
                delta async for delta in iter_openrouter_sse_deltas({"model": "gpt-4"})
            ]

        assert deltas == ["Hello", " ", "world", "!"]

    async def test_raises_503_when_api_key_missing(self) -> None:
        """iter_openrouter_sse_deltas should raise 503 when API key is missing."""

        async def failing_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0)
            error = ValueError("No API key")
            raise error
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            with pytest.raises(BaseHTTPException) as exc_info:
                async for _ in iter_openrouter_sse_deltas({"model": "gpt-4"}):
                    pass

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY is not configured" in exc_info.value.detail

    async def test_raises_502_on_runtime_error(self) -> None:
        """iter_openrouter_sse_deltas should raise 502 on RuntimeError."""

        async def failing_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0)
            error = RuntimeError("Stream processing error")
            raise error
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            with pytest.raises(BaseHTTPException) as exc_info:
                async for _ in iter_openrouter_sse_deltas({"model": "gpt-4"}):
                    pass

        assert isinstance(exc_info.value, BaseHTTPException)
        assert exc_info.value.status_code == 502
        assert "Stream processing error" in str(exc_info.value.detail)


@pytest.mark.unit
class TestSessionTitleHelpers:
    """Tests for promptic-backed session title evaluation."""

    async def test_evaluate_session_title_when_model_says_not_ready(self) -> None:
        """evaluate_session_title should respect has_title=false."""
        thread = MagicMock()
        thread.user_id = "user_1"
        thread.uid = "thread_1"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[{"role": "user", "content": "سلام"}],
            ),
            patch(
                "apps.language.chat.services.load_data",
                return_value={"model": "free/model", "temperature": 0.3},
            ),
            patch(
                "apps.language.chat.services.PromptEngine.generate",
                return_value=("system", "user", {"type": "json_schema"}),
            ),
            patch(
                "apps.language.chat.services.call_openrouter",
                new_callable=AsyncMock,
                return_value=(
                    '{"session_title": {"has_title": false, "title": ""}}',
                    {"model": "free/model"},
                ),
            ),
            patch(
                "apps.language.chat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            result = await evaluate_session_title(user_id="user_1", thread=thread)

        assert result.has_title is False
        assert result.title is None

    async def test_evaluate_session_title_when_model_says_ready(self) -> None:
        """evaluate_session_title should return title when has_title=true."""
        thread = MagicMock()
        thread.user_id = "user_1"
        thread.uid = "thread_1"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[
                    {"role": "user", "content": "Explain quantum computing"},
                    {"role": "assistant", "content": "Sure, here is an overview."},
                ],
            ),
            patch(
                "apps.language.chat.services.load_data",
                return_value={"model": "free/model", "temperature": 0.3},
            ),
            patch(
                "apps.language.chat.services.PromptEngine.generate",
                return_value=("system", "user", {"type": "json_schema"}),
            ),
            patch(
                "apps.language.chat.services.call_openrouter",
                new_callable=AsyncMock,
                return_value=(
                    (
                        '{"session_title": {"has_title": true, '
                        '"title": "Quantum Computing"}}'
                    ),
                    {"model": "free/model"},
                ),
            ),
            patch(
                "apps.language.chat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            result = await evaluate_session_title(user_id="user_1", thread=thread)

        assert result.has_title is True
        assert result.title == "Quantum Computing"

    async def test_maybe_apply_skips_when_has_title_false(self) -> None:
        """maybe_apply_session_title_if_ready should not set title when not ready."""
        session = MagicMock()
        session.title = None
        session.suggest_title = True
        thread = MagicMock()

        with patch(
            "apps.language.chat.services.evaluate_session_title",
            new_callable=AsyncMock,
            return_value=SessionTitleSuggestion(has_title=False),
        ):
            result = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id="user_1",
            )

        assert result.title is None
        session.save.assert_not_called()
