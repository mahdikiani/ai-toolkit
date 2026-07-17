"""Unit tests for execution services."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.language.promptic.services import (
    call_openrouter,
    call_openrouter_stream,
    check_schemas,
    process_execution_task,
)


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

            with pytest.raises(HTTPException) as exc_info:
                check_schemas("nonexistent_prompt", data)

            assert isinstance(exc_info.value, HTTPException)
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
class TestProcessExecutionTask:
    """Tests for process_execution_task function."""

    async def test_sets_completed_on_success(self, tmp_path: Path) -> None:
        """process_execution_task should set task status to completed on success."""
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
        )

        task = MagicMock()
        task.prompt_name = "test_prompt"
        task.input_variables = {"text": "hello"}
        task.save = AsyncMock()

        with (
            patch("apps.language.promptic.services.Settings") as mock_settings,
            patch(
                "apps.language.promptic.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="AI result",
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "AI result"

    async def test_sets_error_when_prompt_missing(self, tmp_path: Path) -> None:
        """process_execution_task should set error status when prompt is missing."""
        task = MagicMock()
        task.prompt_name = "missing_prompt"
        task.input_variables = {}
        task.save = AsyncMock()

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

        task = MagicMock()
        task.prompt_name = "test_prompt"
        task.input_variables = {"text": "hello"}
        task.save = AsyncMock()

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

            with pytest.raises(HTTPException) as exc_info:
                check_schemas("nonexistent_prompt", data)

            assert isinstance(exc_info.value, HTTPException)
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

        task = MagicMock()
        task.prompt_name = "test_prompt"
        task.input_variables = {"text": "hello"}
        task.save = AsyncMock()

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
        task.save.assert_called()

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

        task = MagicMock()
        task.prompt_name = "test_prompt"
        task.input_variables = {"text": "hello"}
        task.save = AsyncMock()

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
        task.save.assert_called()

    async def test_missing_prompt_in_process_task_sets_error(
        self, tmp_path: Path
    ) -> None:
        """
        process_execution_task should set error status for missing prompts.

        **Validates: Requirements 12.6, 12.9**
        """
        task = MagicMock()
        task.prompt_name = "nonexistent_prompt"
        task.input_variables = {"text": "hello"}
        task.save = AsyncMock()

        with patch("apps.language.promptic.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            result = await process_execution_task(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.error is not None
        assert "nonexistent_prompt" in result.error
        assert "not found" in result.error.lower()
        task.save.assert_called()
