"""Unit tests for video generation task processing services."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.videogen.services import process_video


def _task(**attrs: object) -> MagicMock:
    task = MagicMock()
    task.uid = "task_1"
    task.user_id = "user_1"
    task.prompt = "a cat surfing"
    task.model = "luma/ray-2-720p"
    task.image_url = None
    task.provider = "openrouter"
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save_report = AsyncMock()
    return task


@pytest.mark.unit
class TestProcessVideo:
    """Tests for process_video."""

    async def test_insufficient_quota_stops_before_generation(self) -> None:
        """
        Regression check.

        process_video used to have no pre-flight quota check -- video
        generation (the most expensive task type) was billed only after
        the fact, with no gate preventing a broke user from triggering it.
        """
        task = _task()
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch("apps.videogen.services.resolve_api_key") as mock_resolve,
        ):
            result = await process_video(task)

        mock_resolve.assert_not_called()
        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("insufficient_quota")

    async def test_missing_api_key_sets_error_for_openrouter(self) -> None:
        task = _task(provider="openrouter")
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.videogen.services.resolve_api_key",
                side_effect=ValueError("no key"),
            ),
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "Video generation failed" in task.save_report.await_args.args[0]

    async def test_openrouter_success_meters_and_completes(self) -> None:
        task = _task(provider="openrouter")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"video_url": "https://x/video.mp4"})
                    }
                }
            ]
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch("apps.videogen.services.resolve_api_key"),
            patch(
                "apps.videogen.services.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "apps.videogen.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://x/video.mp4"
        mock_meter.assert_awaited_once()

    async def test_replicate_provider_uses_create_prediction(self) -> None:
        task = _task(provider="replicate")
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.videogen.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": ["https://x/video.mp4"]},
            ) as mock_predict,
            patch(
                "apps.videogen.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://x/video.mp4"
        # Regression: this call used to pass timeout=, which isn't a
        # parameter create_prediction accepts (it's timeout_secs) --
        # every replicate-provider video generation crashed with a
        # TypeError before reaching the network call.
        assert mock_predict.call_args.kwargs == {"timeout_secs": 600.0}

    async def test_metering_failure_still_completes_with_result(self) -> None:
        """meter_cost failing must not discard an already-produced video."""
        task = _task(provider="replicate")
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.videogen.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": "https://x/video.mp4"},
            ),
            patch(
                "apps.videogen.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://x/video.mp4"

    async def test_no_video_url_sets_error(self) -> None:
        task = _task(provider="replicate")
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.videogen.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": None},
            ),
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("No video URL returned")

    async def test_generation_failure_sets_error(self) -> None:
        task = _task(provider="replicate")
        with (
            patch(
                "apps.videogen.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.videogen.services.create_prediction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("replicate down"),
            ),
        ):
            result = await process_video(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "Video generation failed" in task.save_report.await_args.args[0]
