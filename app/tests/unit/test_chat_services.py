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
    iter_billed_reply_stream,
    maybe_apply_session_title_if_ready,
    messages_as_openrouter,
    openrouter_headers,
    proxy_chat_completions,
    thread_model,
)
from apps.language.completion import services as completion_services
from utils.billing import finance


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
        mock_json_response = MagicMock()
        mock_json_response.status_code = 200
        mock_json_response.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "model": "openai/gpt-4o-mini",
            "usage": {"total_tokens": 10},
        }

        with (
            patch.object(
                completion_services.Settings, "default_model", "default/model"
            ),
            patch(
                "apps.openai_compat.services.post_chat_completion_unchecked",
                new_callable=AsyncMock,
                return_value=mock_json_response,
            ) as post_mock,
            patch(
                "apps.openai_compat.services.finance.check_quota",
                new_callable=AsyncMock,
            ),
            patch(
                "apps.openai_compat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            content, ctype, status = await completion_services.proxy_chat_completions(
                {"messages": []},
                user_id="user-1",
            )

        assert ctype == "application/json"
        assert status == 200
        assert post_mock.await_args.args[0]["model"] == "default/model"
        assert b"chatcmpl-" in content

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
            yield (
                b'data: {"choices": [{"delta": {"content": "chunk"}}], '
                b'"model": "default/model"}\n\n'
            )

        with (
            patch.object(
                completion_services.Settings, "default_model", "default/model"
            ),
            patch(
                "apps.openai_compat.services.stream_chat_completion_bytes",
                side_effect=mock_stream,
            ),
            patch(
                "apps.openai_compat.services.finance.check_quota",
                new_callable=AsyncMock,
            ),
            patch(
                "apps.openai_compat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            stream = completion_services.proxy_chat_completions_raw_stream({
                "messages": []
            })
            chunks = [chunk async for chunk in stream]

        joined = b"".join(chunks)
        assert b"default/model" in joined
        assert b"[DONE]" in joined
        assert b"chunk" in joined

    async def test_stream_maps_openrouter_error(self) -> None:
        """Verify streamed OpenRouter errors are emitted as SSE error chunks."""

        from utils.integrations.openrouter import OpenRouterError

        async def failing_stream(payload: dict[str, object]) -> AsyncIterator[bytes]:
            await asyncio.sleep(0)
            raise OpenRouterError(429, "limited")
            yield  # pragma: no cover

        with (
            patch(
                "apps.openai_compat.services.stream_chat_completion_bytes",
                side_effect=failing_stream,
            ),
            patch(
                "apps.openai_compat.services.finance.check_quota",
                new_callable=AsyncMock,
            ),
            patch(
                "apps.openai_compat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            chunks = [
                chunk
                async for chunk in completion_services.proxy_chat_completions_raw_stream({
                    "model": "m",
                    "messages": [],
                })
            ]

        joined = b"".join(chunks)
        assert b"upstream_error" in joined
        assert b"limited" in joined
        assert b"[DONE]" in joined


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

    async def test_insufficient_quota_stops_before_calling_openrouter(self) -> None:
        """
        Regression check.

        complete_assistant_message used to have no pre-flight quota
        check at all -- a broke user could keep triggering real
        OpenRouter spend indefinitely, with the service only
        discovering (and not even enforcing) insufficient funds after
        the fact.
        """
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
                "apps.language.chat.services.finance.check_quota",
                new_callable=AsyncMock,
                side_effect=finance._insufficient_funds_error("not enough"),
            ),
            patch(
                "apps.language.chat.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
            ) as mock_complete, pytest.raises(Exception, match="not enough")
        ):
            await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        mock_complete.assert_not_awaited()

    async def test_metering_failure_still_delivers_the_message(self) -> None:
        """
        Regression check.

        meter_cost failing used to propagate uncaught and discard an
        already-generated (already-paid-for-in-OpenRouter-cost)
        assistant reply.
        """
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"

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
                return_value=[{"role": "user", "content": "Hello"}],
            ),
            patch(
                "apps.language.chat.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                return_value=mock_openrouter_response,
            ),
            patch(
                "apps.language.chat.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
            patch(
                "apps.language.chat.services.ChatMessage.create_item",
                new_callable=AsyncMock,
                return_value=mock_created_msg,
            ) as mock_create,
        ):
            result = await complete_assistant_message(
                thread=thread,
                user_id="user_123",
            )

        assert result == mock_created_msg
        assert mock_create.call_args.args[0]["completion_extra"]["usage_id"] is None


@pytest.mark.unit
class TestIterBilledReplyStream:
    """
    Tests for iter_billed_reply_stream.

    Regression: streaming replies used to be reachable for free -- no
    quota check, no metering -- since the streaming code path never
    called finance at all, unlike complete_assistant_message.
    """

    async def test_insufficient_quota_blocks_streaming_before_any_call(self) -> None:
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
                "apps.language.chat.services.finance.check_quota_or_error",
                new_callable=AsyncMock,
                side_effect=BaseHTTPException(
                    status_code=402,
                    error_code="insufficient_quota",
                    detail="not enough coins",
                    message={"en": "not enough coins"},
                ),
            ),
            patch(
                "apps.language.chat.services.openrouter_client."
                "stream_chat_deltas_with_usage"
            ) as mock_stream,
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            async for _ in iter_billed_reply_stream(thread=thread, user_id="user_123"):
                pass

        assert exc_info.value.status_code == 402
        mock_stream.assert_not_called()

    async def test_meters_real_usage_reported_by_the_stream(self) -> None:
        thread = MagicMock()
        thread.uid = "thread_uid_123"
        thread.chat_model = "openai/gpt-4o-mini"

        async def mock_stream(*args: object, **kwargs: object):
            yield "hello", None
            yield "", {"total_tokens": 42}

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[{"role": "user", "content": "Hello"}],
            ),
            patch(
                "apps.language.chat.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=1000,
            ),
            patch(
                "apps.language.chat.services.finance.estimate_text_cost",
                return_value=0.02,
            ) as mock_estimate,
            patch(
                "apps.language.chat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
            patch(
                "apps.language.chat.services.openrouter_client."
                "stream_chat_deltas_with_usage",
                side_effect=mock_stream,
            ),
        ):
            chunks = [
                chunk
                async for chunk in iter_billed_reply_stream(
                    thread=thread, user_id="user_123"
                )
            ]

        assert chunks == ["hello"]
        final_call = mock_estimate.call_args_list[-1]
        assert final_call.kwargs["usage"] == {"total_tokens": 42}
        mock_meter.assert_awaited_once()
        assert mock_meter.call_args.args[1] == pytest.approx(0.02)

    async def test_raises_400_when_no_messages(self) -> None:
        thread = MagicMock()
        thread.uid = "thread_uid_123"

        with (
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[],
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            async for _ in iter_billed_reply_stream(thread=thread, user_id="user_123"):
                pass

        assert exc_info.value.status_code == 400


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
        """
        maybe_apply_session_title_if_ready should not set a title when
        not ready AND the conversation is still short of the fallback
        threshold.
        """
        session = MagicMock()
        session.title = None
        session.suggest_title = True
        thread = MagicMock()

        with (
            patch(
                "apps.language.chat.services.evaluate_session_title",
                new_callable=AsyncMock,
                return_value=SessionTitleSuggestion(has_title=False),
            ),
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[{"role": "user", "content": "hi"}],
            ),
        ):
            result = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id="user_1",
            )

        assert result.title is None
        session.save.assert_not_called()

    async def test_maybe_apply_falls_back_after_enough_messages(self) -> None:
        """
        Regression check.

        The LLM can keep saying "not specific enough" indefinitely for
        a short/ambiguous conversation -- after enough messages without
        a natural title, a session must still end up with *some* real
        title instead of staying "بدون عنوان" forever.
        """
        session = MagicMock()
        session.title = None
        session.suggest_title = True
        session.save = AsyncMock()
        thread = MagicMock()

        with (
            patch(
                "apps.language.chat.services.evaluate_session_title",
                new_callable=AsyncMock,
                return_value=SessionTitleSuggestion(has_title=False),
            ),
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[
                    {"role": "user", "content": "how do I center a div in css"},
                    {"role": "assistant", "content": "..."},
                    {"role": "user", "content": "still not working"},
                    {"role": "assistant", "content": "..."},
                ],
            ),
        ):
            result = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id="user_1",
            )

        assert result.title == "how do I center a div in css"
        session.save.assert_called_once()

    async def test_maybe_apply_fallback_skips_when_no_user_content(self) -> None:
        """No user message content to fall back to -- title stays unset."""
        session = MagicMock()
        session.title = None
        session.suggest_title = True
        thread = MagicMock()

        with (
            patch(
                "apps.language.chat.services.evaluate_session_title",
                new_callable=AsyncMock,
                return_value=SessionTitleSuggestion(has_title=False),
            ),
            patch(
                "apps.language.chat.services.messages_as_openrouter",
                new_callable=AsyncMock,
                return_value=[
                    {"role": "assistant", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "assistant", "content": "..."},
                ],
            ),
        ):
            result = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id="user_1",
            )

        assert result.title is None
        session.save.assert_not_called()
