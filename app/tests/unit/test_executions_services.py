"""Unit tests for execution services."""

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.language.promptic.services import (
    call_openrouter,
    call_openrouter_stream,
    check_schemas,
    invoke_stream,
    process_execution_task,
)


def _task_mock(**attrs: object) -> MagicMock:
    """Build a task double that supports awaitable update_and_emit."""
    task = MagicMock()
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save = AsyncMock()

    async def _update_and_emit(**kwargs: object) -> MagicMock:
        for key, value in kwargs.items():
            setattr(task, key, value)
        return task

    task.update_and_emit = AsyncMock(side_effect=_update_and_emit)
    return task


@pytest.mark.unit
class TestCheckSchemas:
    """Tests for check_schemas function."""

    def test_valid_prompt_exists(self, tmp_path: Path) -> None:
        """check_schemas should not raise when prompt file exists."""
        prompt_file = tmp_path / "my_prompt.yaml"
        prompt_file.write_text("task:\n  system: {}\n  user: hello\n")

        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            data = MagicMock()
            check_schemas("my_prompt", data)  # Should not raise

    def test_missing_prompt_raises_404(self, tmp_path: Path) -> None:
        """check_schemas should raise HTTPException 404 for missing prompts."""
        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            data = MagicMock()

            with pytest.raises(BaseHTTPException) as exc_info:
                check_schemas("nonexistent_prompt", data)

            assert isinstance(exc_info.value, BaseHTTPException)
            assert exc_info.value.status_code == 404
            assert "nonexistent_prompt" in exc_info.value.detail


@pytest.mark.unit
class TestCallOpenrouter:
    """Tests for call_openrouter function."""

    async def test_returns_content_on_success(self) -> None:
        """call_openrouter should return content from API response."""
        mock_response = {"choices": [{"message": {"content": "Hello from AI"}}]}
        with patch(
            "apps.language.promptic.services.openrouter_client.complete_chat_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await call_openrouter(system="You are helpful", user="Hello")

        assert result == "Hello from AI"

    async def test_strips_whitespace_from_content(self) -> None:
        """call_openrouter should strip leading/trailing whitespace."""
        mock_response = {"choices": [{"message": {"content": "  Trimmed response  "}}]}
        with patch(
            "apps.language.promptic.services.openrouter_client.complete_chat_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await call_openrouter(system="sys", user="usr")

        assert result == "Trimmed response"

    async def test_raises_on_empty_choices(self) -> None:
        """call_openrouter should raise RuntimeError when choices is empty."""
        mock_response = {"choices": []}
        with (
            patch(
                "apps.language.promptic.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(RuntimeError, match="No response from model"),
        ):
            await call_openrouter(system="sys", user="usr")

    async def test_raises_on_missing_choices(self) -> None:
        """call_openrouter should raise RuntimeError when choices key is missing."""
        mock_response = {}
        with (
            patch(
                "apps.language.promptic.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(RuntimeError, match="No response from model"),
        ):
            await call_openrouter(system="sys", user="usr")

    async def test_uses_default_model(self) -> None:
        """call_openrouter should use Settings.default_model when model is None."""
        mock_response = {"choices": [{"message": {"content": "ok"}}]}
        with (
            patch(
                "apps.language.promptic.services.openrouter_client.complete_chat_json",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_complete,
            patch("apps.language.promptic.services.Settings") as mock_settings,
        ):
            mock_settings.default_model = "openai/gpt-4o-mini"
            await call_openrouter(system="sys", user="usr")

        call_body = mock_complete.call_args[0][0]
        assert call_body["model"] == "openai/gpt-4o-mini"

    async def test_uses_custom_model(self) -> None:
        """call_openrouter should use provided model when specified."""
        mock_response = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "apps.language.promptic.services.openrouter_client.complete_chat_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_complete:
            await call_openrouter(system="sys", user="usr", model="anthropic/claude-3")

        call_body = mock_complete.call_args[0][0]
        assert call_body["model"] == "anthropic/claude-3"

    async def test_includes_response_format_when_provided(self) -> None:
        """call_openrouter should include response_format in body when provided."""
        mock_response = {"choices": [{"message": {"content": "{}"}}]}
        response_format = {"type": "json_schema", "json_schema": {"name": "resp"}}
        with patch(
            "apps.language.promptic.services.openrouter_client.complete_chat_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_complete:
            await call_openrouter(
                system="sys", user="usr", response_format=response_format
            )

        call_body = mock_complete.call_args[0][0]
        assert call_body["response_format"] == response_format


@pytest.mark.unit
class TestCallOpenrouterStream:
    """Tests for call_openrouter_stream function."""

    async def test_yields_chunks_from_stream(self) -> None:
        """call_openrouter_stream should yield chunks from the streaming API."""

        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
            await asyncio.sleep(0)
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        with patch(
            "apps.language.promptic.services.openrouter_client.stream_chat_deltas",
            side_effect=mock_stream,
        ):
            chunks = [
                chunk
                async for chunk in call_openrouter_stream(system="sys", user="usr")
            ]

        assert chunks == ["Hello", " ", "World"]

    async def test_empty_stream_yields_nothing(self) -> None:
        """call_openrouter_stream should handle empty streams gracefully."""

        async def mock_empty_stream(
            *args: object, **kwargs: object
        ) -> AsyncIterator[str]:
            await asyncio.sleep(0)
            return
            yield  # make it a generator

        with patch(
            "apps.language.promptic.services.openrouter_client.stream_chat_deltas",
            side_effect=mock_empty_stream,
        ):
            chunks = [
                chunk
                async for chunk in call_openrouter_stream(system="sys", user="usr")
            ]

        assert chunks == []


@pytest.mark.unit
class TestInvokeStream:
    """Tests for invoke_stream forwarding a prompt's declared model config."""

    async def test_forwards_prompt_declared_model_config(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "model: openai/gpt-5.6-terra\n"
            "temperature: 0.4\n"
            "max_tokens: 8000\n"
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )
        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
        )

        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
            await asyncio.sleep(0)
            yield "chunk"

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter_stream",
                side_effect=mock_stream,
            ) as mock_call,
        ):
            mock_settings.prompts_dir = tmp_path
            [chunk async for chunk in invoke_stream(task)]

        assert mock_call.call_args.kwargs["model"] == "openai/gpt-5.6-terra"
        assert mock_call.call_args.kwargs["temperature"] == pytest.approx(0.4)
        assert mock_call.call_args.kwargs["max_tokens"] == 8000


@pytest.mark.unit
class TestChunkedProcessing:
    """
    Tests for process_execution_task's long-document chunking path.

    Regression: translating (or turning into study notes) a whole book
    used to always run as a single completion call -- no matter how
    high max_tokens was set, a single call's output ceiling is far
    below what a whole book needs, so output silently cut off partway
    through. Content longer than chunk_max_chars must be split, each
    part processed separately, and the parts stitched back together.
    """

    @staticmethod
    def _write_prompt(tmp_path: Path, **extra_yaml: str) -> None:
        extra = "".join(f"{k}: {v}\n" for k, v in extra_yaml.items())
        (tmp_path / "translate.yaml").write_text(
            "model: openai/gpt-5.6-terra\n"
            f"{extra}"
            "task:\n  system:\n    persona: translate\n  user: '{{ content }}'\n"
        )

    async def test_short_content_skips_chunking(self, tmp_path: Path) -> None:
        self._write_prompt(tmp_path, chunk_max_chars="1000", use_glossary="true")
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": "short text"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value=("translated", {"model": "x", "usage": {}}),
            ) as mock_call,
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "apps.language.promptic.services.finance.estimate_text_cost",
                return_value=1.0,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        # a single call: no glossary pass, no chunk splitting
        assert mock_call.await_count == 1
        assert result.result == "translated"
        assert result.provider_meta == {"model": "x", "usage": {}}

    async def test_long_content_splits_into_chunks_and_stitches_result(
        self, tmp_path: Path
    ) -> None:
        self._write_prompt(tmp_path, chunk_max_chars="20", use_glossary="false")
        long_content = "\n\n".join(f"paragraph {i} text here" for i in range(6))
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": long_content},
            user_id="user_123",
        )

        call_count = 0

        async def fake_call(
            *args: object, **kwargs: object
        ) -> tuple[str, dict[str, object]]:
            nonlocal call_count
            call_count += 1
            return f"chunk-{call_count}", {
                "model": "x",
                "usage": {"total_tokens": 10},
            }

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                side_effect=fake_call,
            ),
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "apps.language.promptic.services.finance.estimate_text_cost",
                return_value=1.0,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.completed
        assert call_count > 1
        assert result.result.count("chunk-") == call_count
        assert result.provider_meta["chunked"] is True
        assert result.provider_meta["chunk_count"] == call_count

    async def test_glossary_is_built_once_from_full_content_before_chunking(
        self, tmp_path: Path
    ) -> None:
        self._write_prompt(tmp_path, chunk_max_chars="20", use_glossary="true")
        long_content = "\n\n".join(f"paragraph {i} text here" for i in range(4))
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": long_content, "language": "Persian"},
            user_id="user_123",
        )

        seen_users: list[str] = []

        async def fake_call(
            system: str, user: str, **kwargs: object
        ) -> tuple[str, dict[str, object]]:
            seen_users.append(user)
            return "ok", {"model": "x", "usage": {}}

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                side_effect=fake_call,
            ),
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "apps.language.promptic.services.finance.estimate_text_cost",
                return_value=1.0,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        # the first call awaited is the glossary pass over the FULL content
        assert long_content in seen_users[0]
        assert result.provider_meta["chunk_count"] == len(seen_users) - 1

    async def test_chunk_failure_fails_the_whole_task(self, tmp_path: Path) -> None:
        self._write_prompt(tmp_path, chunk_max_chars="20", use_glossary="false")
        long_content = "\n\n".join(f"paragraph {i} text here" for i in range(6))
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": long_content},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=RuntimeError("chunk failed"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert "chunk failed" in result.error


@pytest.mark.unit
class TestPrompticBillingGate:
    """
    Tests for the pre-flight quota check and non-destructive metering.

    Regression: process_promptic used to have no pre-flight quota check
    at all -- a chunked job (a whole book) could run dozens of LLM
    calls to completion before any billing decision was made. And
    meter_cost was called unguarded right before delivering the
    result, so a transient billing-service failure discarded an
    already-computed, already-paid-for-in-LLM-cost result.
    """

    async def test_insufficient_quota_stops_before_any_llm_call(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "translate.yaml"
        prompt_file.write_text(
            "model: openai/gpt-5.6-terra\n"
            "task:\n  system:\n    persona: translate\n  user: '{{ content }}'\n"
        )
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": "x" * 10_000},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
            ) as mock_call,
        ):
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        mock_call.assert_not_awaited()
        assert result.task_status == TaskStatusEnum.error
        assert result.error == "insufficient_quota"

    async def test_sufficient_quota_proceeds_with_the_call(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "translate.yaml"
        prompt_file.write_text(
            "model: openai/gpt-5.6-terra\n"
            "task:\n  system:\n    persona: translate\n  user: '{{ content }}'\n"
        )
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": "hello"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value=("translated", {"model": "x", "usage": {}}),
            ) as mock_call,
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        mock_call.assert_awaited()
        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "translated"

    async def test_metering_failure_still_delivers_the_completed_result(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "translate.yaml"
        prompt_file.write_text(
            "model: openai/gpt-5.6-terra\n"
            "task:\n  system:\n    persona: translate\n  user: '{{ content }}'\n"
        )
        task = _task_mock(
            prompt_name="translate",
            input_variables={"content": "hello"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value=("translated", {"model": "x", "usage": {}}),
            ),
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        # the translation is real, paid-for LLM output -- a billing outage
        # must not throw it away
        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "translated"
        assert result.usage_id is None


@pytest.mark.unit
class TestProcessExecutionTask:
    """Tests for process_execution_task function."""

    async def test_sets_completed_on_success(self, tmp_path: Path) -> None:
        """process_execution_task should set task status to completed on success."""
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="AI result",
            ),
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "apps.language.promptic.services.finance.estimate_text_cost",
                return_value=1.0,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "AI result"

    async def test_forwards_prompt_declared_model_config_to_openrouter(
        self, tmp_path: Path
    ) -> None:
        """
        Regression check.

        A prompt file's own model/temperature/max_tokens used to be
        parsed and then silently discarded -- every prompt ran on the
        global default model at a flat temperature with no max_tokens,
        regardless of what the prompt declared (e.g. a long-document
        task like summarize needs a much higher max_tokens than the
        provider default, or output gets cut short).
        """
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "model: openai/gpt-5.6-terra\n"
            "temperature: 0.4\n"
            "max_tokens: 8000\n"
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="AI result",
            ) as mock_call,
            patch(
                "apps.language.promptic.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "apps.language.promptic.services.finance.estimate_text_cost",
                return_value=1.0,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            await process_execution_task(task)

        assert mock_call.call_args.kwargs["model"] == "openai/gpt-5.6-terra"
        assert mock_call.call_args.kwargs["temperature"] == pytest.approx(0.4)
        assert mock_call.call_args.kwargs["max_tokens"] == 8000

    async def test_sets_error_when_prompt_missing(self, tmp_path: Path) -> None:
        """process_execution_task should set error status when prompt is missing."""
        task = _task_mock(prompt_name="missing_prompt", input_variables={})

        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "missing_prompt" in result.error

    async def test_sets_error_on_openrouter_exception(self, tmp_path: Path) -> None:
        """process_execution_task should set error status when OpenRouter raises."""
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
            user_id="user_123",
        )

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "API error" in result.error


@pytest.mark.unit
class TestExecutionErrorHandling:
    """Tests for execution error handling scenarios."""

    def test_missing_prompt_returns_404(self, tmp_path: Path) -> None:
        """
        check_schemas should raise HTTPException 404 for missing prompts.

        **Validates: Requirements 12.1, 12.6**
        """
        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            data = MagicMock()

            with pytest.raises(BaseHTTPException) as exc_info:
                check_schemas("nonexistent_prompt", data)

            assert isinstance(exc_info.value, BaseHTTPException)
            assert exc_info.value.status_code == 404
            assert "nonexistent_prompt" in exc_info.value.detail
            assert "not found" in exc_info.value.detail.lower()

    def test_invalid_input_variables_validation(self, tmp_path: Path) -> None:
        """
        check_schemas should validate that prompt exists before processing.

        **Validates: Requirements 12.7, 12.9**

        Note: Input variable validation happens at the Pydantic schema level,
        not in check_schemas. This test verifies the prompt existence check
        which is the first validation step.
        """
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            data = MagicMock()

            # Should not raise for valid prompt
            check_schemas("test_prompt", data)

    async def test_openrouter_api_error_sets_task_to_failed(
        self, tmp_path: Path
    ) -> None:
        """
        OpenRouter API errors should set task status to failed.

        **Validates: Requirements 12.1, 12.9**
        """
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
            user_id="user_123",
        )

        # Test with RuntimeError (API error)
        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=RuntimeError(
                    "OpenRouter API error: 500 Internal Server Error"
                ),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "OpenRouter API error" in result.error
        task.update_and_emit.assert_called()

    async def test_openrouter_network_error_sets_task_to_failed(
        self, tmp_path: Path
    ) -> None:
        """
        Network errors during OpenRouter calls should set task to failed.

        **Validates: Requirements 12.1, 12.9**
        """
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = _task_mock(
            prompt_name="test_prompt",
            input_variables={"text": "hello"},
            user_id="user_123",
        )

        # Test with generic Exception (network error)
        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=Exception("Connection timeout"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "Connection timeout" in result.error
        task.update_and_emit.assert_called()

    async def test_missing_prompt_in_process_task_sets_error(
        self, tmp_path: Path
    ) -> None:
        """
        process_execution_task should set error status for missing prompts.

        **Validates: Requirements 12.6, 12.9**
        """
        task = _task_mock(
            prompt_name="nonexistent_prompt",
            input_variables={"text": "hello"},
        )

        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "nonexistent_prompt" in result.error
        assert "not found" in result.error.lower()
        task.update_and_emit.assert_called()
