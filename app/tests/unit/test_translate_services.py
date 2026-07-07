"""Unit tests for translation services."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apps.translate.services import save_result
from fastapi_mongo_base.tasks import TaskStatusEnum


@pytest.mark.unit
class TestSaveResult:
    """Tests for translation save_result function."""

    async def test_sets_completed_status(self) -> None:
        """save_result should set task status to completed."""
        task = MagicMock()
        task.save = AsyncMock(return_value=task)

        await save_result(task, "Translated text")

        assert task.task_status == TaskStatusEnum.completed

    async def test_normalizes_text(self) -> None:
        """save_result should normalize the result text."""
        task = MagicMock()
        task.save = AsyncMock(return_value=task)

        await save_result(task, "  Translated text  ")

        assert task.result == "Translated text"

    async def test_saves_usage_info(self) -> None:
        """save_result should save usage amount and ID."""
        task = MagicMock()
        task.save = AsyncMock(return_value=task)

        await save_result(task, "text", usage_amount=1.0, usage_id="usage_789")

        assert task.usage_amount == pytest.approx(1.0)
        assert task.usage_id == "usage_789"


@pytest.mark.unit
class TestProcessTranslate:
    """Tests for process_translate function."""

    async def test_sets_error_when_prompt_missing(self, tmp_path: Path) -> None:
        """process_translate should set error when translate.yaml is missing."""
        from apps.translate.services import process_translate

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "Persian"
        task.save = AsyncMock(return_value=task)

        with patch("apps.translate.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path  # No translate.yaml here

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.error
        assert "translate.yaml" in result.error

    async def test_translates_text_successfully(self, tmp_path: Path) -> None:
        """process_translate should translate text and set completed status."""
        from apps.translate.services import process_translate

        # Create a minimal translate.yaml
        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "Persian"
        task.save = AsyncMock(return_value=task)

        mock_usage = MagicMock()
        mock_usage.amount = 1.0
        mock_usage.uid = "usage_123"

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="سلام دنیا",
            ),
            patch(
                "apps.translate.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=mock_usage,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "سلام دنیا"

    async def test_sets_error_on_openrouter_exception(self, tmp_path: Path) -> None:
        """process_translate should set error status when OpenRouter raises."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello"
        task.language = "Persian"
        task.save = AsyncMock(return_value=task)

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.error
        assert "API error" in result.error


@pytest.mark.unit
class TestTranslationErrorHandling:
    """Tests for translation error handling scenarios."""

    async def test_handles_unsupported_language_gracefully(
        self, tmp_path: Path
    ) -> None:
        """process_translate should handle unsupported languages gracefully."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "Klingon"  # Unsupported/fictional language
        task.save = AsyncMock(return_value=task)

        mock_usage = MagicMock()
        mock_usage.amount = 1.0
        mock_usage.uid = "usage_123"

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="Unable to translate to Klingon",
            ),
            patch(
                "apps.translate.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=mock_usage,
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        # Should complete successfully - AI handles unsupported languages
        assert result.task_status == TaskStatusEnum.completed
        assert result.result is not None

    async def test_handles_empty_language_with_default(self, tmp_path: Path) -> None:
        """process_translate should use default language when language is None."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = None  # No language specified
        task.save = AsyncMock(return_value=task)

        mock_usage = MagicMock()
        mock_usage.amount = 1.0
        mock_usage.uid = "usage_123"

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="سلام دنیا",
            ),
            patch(
                "apps.translate.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=mock_usage,
            ),
            patch("apps.translate.services.PromptEngine") as mock_engine_class,
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            mock_engine = MagicMock()
            mock_engine.generate.return_value = (
                "You are a translator",
                "Translate: Hello world to Persian",
                None,
            )
            mock_engine_class.return_value = mock_engine

            result = await process_translate(task)

        # Verify default language "Persian" was used
        mock_engine.generate.assert_called_once()
        call_args = mock_engine.generate.call_args[0]
        input_vars = call_args[1]
        assert input_vars["language"] == "Persian"
        assert result.task_status == TaskStatusEnum.completed

    async def test_handles_api_timeout_error(self, tmp_path: Path) -> None:
        """process_translate should handle API timeout errors."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "Spanish"
        task.save = AsyncMock(return_value=task)

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=TimeoutError("Request timeout"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.error
        assert "Request timeout" in result.error

    async def test_handles_api_connection_error(self, tmp_path: Path) -> None:
        """process_translate should handle API connection errors."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "French"
        task.save = AsyncMock(return_value=task)

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Failed to connect to API"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.error
        assert "Failed to connect to API" in result.error

    async def test_handles_invalid_yaml_prompt(self, tmp_path: Path) -> None:
        """process_translate should handle invalid YAML prompt structure."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        # Write invalid YAML (not a mapping)
        translate_yaml.write_text("- invalid\n- yaml\n- list\n")

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "German"
        task.save = AsyncMock(return_value=task)

        with patch("apps.translate.services.Settings") as mock_settings:
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        assert result.task_status == TaskStatusEnum.error
        assert "must be a YAML mapping" in result.error

    async def test_handles_finance_metering_failure(self, tmp_path: Path) -> None:
        """process_translate should handle finance metering failures gracefully."""
        from apps.translate.services import process_translate

        translate_yaml = tmp_path / "translate.yaml"
        translate_yaml.write_text(
            "task:\n"
            "  system:\n"
            "    persona: You are a translator\n"
            "  user: 'Translate: {{ content }} to {{ language }}'\n"
        )

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.text = "Hello world"
        task.language = "Italian"
        task.save = AsyncMock(return_value=task)

        with (
            patch("apps.translate.services.Settings") as mock_settings,
            patch(
                "apps.translate.services.call_openrouter",
                new_callable=AsyncMock,
                return_value="Ciao mondo",
            ),
            patch(
                "apps.translate.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Metering service unavailable"),
            ),
        ):
            mock_settings.prompts_dir = tmp_path
            mock_settings.default_model = "openai/gpt-4o-mini"

            result = await process_translate(task)

        # Should fail because metering is critical
        assert result.task_status == TaskStatusEnum.error
        assert "Metering service unavailable" in result.error
