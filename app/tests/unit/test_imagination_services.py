"""Unit tests for imagination task processing services."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.imagination.services import process_imagination


def _task(**attrs: object) -> MagicMock:
    task = MagicMock()
    task.uid = "task_1"
    task.user_id = "user_1"
    task.prompt = "a cat"
    task.model = "openai/dall-e-3"
    task.size = "1024x1024"
    task.enhance_prompt = False
    task.enhanced_prompt = None
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save_report = AsyncMock()
    return task


@pytest.mark.unit
class TestProcessImagination:
    """Tests for process_imagination."""

    async def test_missing_api_key_sets_error(self) -> None:
        task = _task()
        with patch(
            "apps.imagination.services.resolve_api_key",
            side_effect=ValueError("no key"),
        ):
            result = await process_imagination(task)

        assert result.task_status == TaskStatusEnum.error
        assert result.result_url is None
        task.save_report.assert_awaited_once_with("no key")

    async def test_insufficient_quota_stops_before_generation(self) -> None:
        """
        Regression check.

        process_imagination used to have no pre-flight quota check --
        image generation cost was only ever metered after the fact,
        with no gate preventing a broke user from generating images.
        """
        task = _task()
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
            ) as mock_generate,
        ):
            result = await process_imagination(task)

        mock_generate.assert_not_awaited()
        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("insufficient_quota")

    async def test_successful_generation_meters_and_completes(self) -> None:
        task = _task()
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
                return_value={"data": [{"url": "https://example.com/img.png"}]},
            ),
            patch(
                "apps.imagination.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_meter,
        ):
            result = await process_imagination(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://example.com/img.png"
        mock_meter.assert_awaited_once()

    async def test_metering_failure_still_completes_with_result(self) -> None:
        """
        Regression check.

        meter_cost failing must not discard an already-generated
        (already-paid-for) image.
        """
        task = _task()
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
                return_value={"data": [{"url": "https://example.com/img.png"}]},
            ),
            patch(
                "apps.imagination.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            result = await process_imagination(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://example.com/img.png"

    async def test_enhance_prompt_failure_falls_back_to_original(self) -> None:
        task = _task(enhance_prompt=True)
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.imagination.services._enhance_prompt",
                new_callable=AsyncMock,
                side_effect=RuntimeError("enhance failed"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
                return_value={"data": [{"url": "https://example.com/img.png"}]},
            ),
            patch(
                "apps.imagination.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await process_imagination(task)

        assert result.enhanced_prompt == task.prompt
        assert result.task_status == TaskStatusEnum.completed

    async def test_generation_http_error_sets_error(self) -> None:
        task = _task()
        response = httpx.Response(
            status_code=429, text="rate limited", request=httpx.Request("POST", "x")
        )
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
                side_effect=httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                ),
            ),
        ):
            result = await process_imagination(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "rate limited" in task.save_report.await_args.args[0]

    async def test_no_image_data_returned_sets_error(self) -> None:
        task = _task()
        with (
            patch("apps.imagination.services.resolve_api_key"),
            patch(
                "apps.imagination.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.imagination.services._generate_image",
                new_callable=AsyncMock,
                return_value={"data": []},
            ),
        ):
            result = await process_imagination(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("No image data returned")
