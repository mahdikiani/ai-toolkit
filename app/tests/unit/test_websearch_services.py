"""Unit tests for web search task processing services."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.websearch.services import process_search


def _task(**attrs: object) -> MagicMock:
    task = MagicMock()
    task.uid = "task_1"
    task.user_id = "user_1"
    task.query = "python asyncio"
    task.num_results = 10
    task.include_domains = None
    task.exclude_domains = None
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save_report = AsyncMock()
    return task


@pytest.mark.unit
class TestProcessSearch:
    """Tests for process_search."""

    async def test_insufficient_quota_stops_before_searching(self) -> None:
        """
        Regression check.

        process_search used to have no pre-flight quota check -- a
        broke user could keep triggering real (paid) Exa searches
        indefinitely.
        """
        task = _task()
        with (
            patch(
                "apps.websearch.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch(
                "apps.websearch.services.exa_search",
                new_callable=AsyncMock,
            ) as mock_search,
        ):
            result = await process_search(task)

        mock_search.assert_not_awaited()
        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("insufficient_quota")

    async def test_successful_search_meters_and_completes(self) -> None:
        task = _task()
        with (
            patch(
                "apps.websearch.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.websearch.services.exa_search",
                new_callable=AsyncMock,
                return_value={"results": []},
            ),
            patch(
                "apps.websearch.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_meter,
        ):
            result = await process_search(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == {"results": []}
        mock_meter.assert_awaited_once()

    async def test_metering_failure_still_delivers_the_result(self) -> None:
        """
        Regression check.

        meter_cost failing must not discard already-produced (already
        paid for via Exa) search results.
        """
        task = _task()
        with (
            patch(
                "apps.websearch.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.websearch.services.exa_search",
                new_callable=AsyncMock,
                return_value={"results": []},
            ),
            patch(
                "apps.websearch.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            result = await process_search(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == {"results": []}

    async def test_search_failure_sets_error(self) -> None:
        task = _task()
        with (
            patch(
                "apps.websearch.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.websearch.services.exa_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("exa down"),
            ),
        ):
            result = await process_search(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "exa down" in task.save_report.await_args.args[0]
