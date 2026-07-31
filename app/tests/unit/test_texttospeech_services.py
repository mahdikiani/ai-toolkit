"""Unit tests for text-to-speech task processing services."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.texttospeech.services import process_tts


def _task(**attrs: object) -> MagicMock:
    task = MagicMock()
    task.uid = "task_1"
    task.user_id = "user_1"
    task.text = "hello world"
    task.model = "openai/gpt-4o-mini-tts"
    task.voice = "alloy"
    task.response_format = "mp3"
    task.speed = 1.0
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save_report = AsyncMock()
    return task


@pytest.mark.unit
class TestProcessTts:
    """Tests for process_tts."""

    async def test_missing_api_key_sets_error(self) -> None:
        task = _task()
        with patch(
            "apps.texttospeech.services.resolve_api_key",
            side_effect=ValueError("no key"),
        ):
            result = await process_tts(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("no key")

    async def test_insufficient_quota_stops_before_generation(self) -> None:
        """
        Regression check.

        process_tts used to have no pre-flight quota check -- speech was
        generated and billed with no gate preventing a broke user from
        triggering paid generation.
        """
        task = _task()
        with (
            patch("apps.texttospeech.services.resolve_api_key"),
            patch(
                "apps.texttospeech.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch("apps.texttospeech.services.httpx.AsyncClient") as mock_client_cls,
        ):
            result = await process_tts(task)

        mock_client_cls.assert_not_called()
        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("insufficient_quota")

    async def test_successful_generation_meters_and_completes(self) -> None:
        task = _task()
        mock_resp = MagicMock()
        mock_resp.content = b"audio-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("apps.texttospeech.services.resolve_api_key"),
            patch(
                "apps.texttospeech.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.texttospeech.services.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "apps.texttospeech.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
        ):
            result = await process_tts(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_data == b"audio-bytes"
        mock_meter.assert_awaited_once()

    async def test_metering_failure_still_completes_with_result(self) -> None:
        """meter_cost failing must not discard already-generated audio."""
        task = _task()
        mock_resp = MagicMock()
        mock_resp.content = b"audio-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("apps.texttospeech.services.resolve_api_key"),
            patch(
                "apps.texttospeech.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.texttospeech.services.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "apps.texttospeech.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            result = await process_tts(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_data == b"audio-bytes"

    async def test_generation_http_error_sets_error(self) -> None:
        task = _task()
        request = httpx.Request("POST", "https://x")
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=httpx.Response(429, text="rate limited", request=request),
            )
        )

        with (
            patch("apps.texttospeech.services.resolve_api_key"),
            patch(
                "apps.texttospeech.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.texttospeech.services.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            result = await process_tts(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "TTS failed" in task.save_report.await_args.args[0]
