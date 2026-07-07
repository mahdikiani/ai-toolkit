"""Unit tests for chat services."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from apps.language.chat.services import (
    complete_assistant_message,
    messages_as_openrouter,
    openrouter_headers,
    proxy_chat_completions,
    thread_model,
)


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
            pytest.raises(HTTPException) as exc_info,
        ):
            openrouter_headers()

        assert exc_info.value.status_code == 503


@pytest.mark.unit
class TestMessagesAsOpenrouter:
    """Tests for messages_as_openrouter function."""

    async def test_returns_formatted_messages(self) -> None:
        """messages_as_openrouter should return messages in OpenRouter format."""
        thread = MagicMock()
        thread.tenant_id = "tenant_123"
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
        thread.tenant_id = "tenant_123"
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
        thread.tenant_id = "tenant_123"
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
                tenant_id="tenant_123",
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
            pytest.raises(HTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
                tenant_id="tenant_123",
            )

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
            pytest.raises(HTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
                tenant_id="tenant_123",
            )

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
            pytest.raises(HTTPException) as exc_info,
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
                tenant_id="tenant_123",
            )

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
            pytest.raises(HTTPException) as exc_info,
        ):
            await proxy_chat_completions({"model": "gpt-4"})

        assert exc_info.value.status_code == 503


@pytest.mark.unit
class TestProxyChatCompletionsRawStream:
    """Tests for proxy_chat_completions_raw_stream function."""

    async def test_yields_streaming_chunks(self) -> None:
        """proxy_chat_completions_raw_stream should yield SSE bytes from OpenRouter."""

        async def mock_stream():
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

            chunks = [chunk async for chunk in proxy_chat_completions_raw_stream({"model": "gpt-4"})]

        assert len(chunks) == 3
        assert chunks[0] == b"data: chunk1\n\n"
        assert chunks[1] == b"data: chunk2\n\n"
        assert chunks[2] == b"data: [DONE]\n\n"

    async def test_raises_503_when_api_key_missing(self) -> None:
        """proxy_chat_completions_raw_stream should raise 503 when API key is missing."""

        async def failing_stream():
            raise ValueError("No API key")
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_completion_bytes",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import proxy_chat_completions_raw_stream

            with pytest.raises(HTTPException) as exc_info:
                async for _ in proxy_chat_completions_raw_stream({"model": "gpt-4"}):
                    pass

        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY is not configured" in exc_info.value.detail

    async def test_raises_http_exception_on_openrouter_error(self) -> None:
        """proxy_chat_completions_raw_stream should raise HTTPException on OpenRouterError."""

        async def failing_stream():
            from utils.openrouter import OpenRouterError

            raise OpenRouterError(status_code=429, detail="Rate limit exceeded")
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_completion_bytes",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import proxy_chat_completions_raw_stream

            with pytest.raises(HTTPException) as exc_info:
                async for _ in proxy_chat_completions_raw_stream({"model": "gpt-4"}):
                    pass

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail


@pytest.mark.unit
class TestIterOpenrouterSseDeltas:
    """Tests for iter_openrouter_sse_deltas function."""

    async def test_yields_text_deltas(self) -> None:
        """iter_openrouter_sse_deltas should yield text deltas from streaming response."""

        async def mock_stream():
            for delta in ["Hello", " ", "world", "!"]:
                yield delta

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=mock_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            deltas = [delta async for delta in iter_openrouter_sse_deltas({"model": "gpt-4"})]

        assert deltas == ["Hello", " ", "world", "!"]

    async def test_raises_503_when_api_key_missing(self) -> None:
        """iter_openrouter_sse_deltas should raise 503 when API key is missing."""

        async def failing_stream():
            raise ValueError("No API key")
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            with pytest.raises(HTTPException) as exc_info:
                async for _ in iter_openrouter_sse_deltas({"model": "gpt-4"}):
                    pass

        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY is not configured" in exc_info.value.detail

    async def test_raises_502_on_runtime_error(self) -> None:
        """iter_openrouter_sse_deltas should raise 502 on RuntimeError."""

        async def failing_stream():
            raise RuntimeError("Stream processing error")
            yield  # pragma: no cover

        with patch(
            "apps.language.chat.services.openrouter_client.stream_chat_deltas",
            return_value=failing_stream(),
        ):
            from apps.language.chat.services import iter_openrouter_sse_deltas

            with pytest.raises(HTTPException) as exc_info:
                async for _ in iter_openrouter_sse_deltas({"model": "gpt-4"}):
                    pass

        assert exc_info.value.status_code == 502
        assert "Stream processing error" in str(exc_info.value.detail)
