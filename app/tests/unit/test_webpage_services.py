"""Unit tests for webpage extraction services."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.webpage.models import WebpageTask
from apps.webpage.services import process_webpage


def _task() -> MagicMock:
    """Create a webpage task double with the fields used by the service."""
    task = MagicMock(spec=WebpageTask)
    task.url = "https://example.com/article"
    task.save_report = AsyncMock()
    return task


def _client(response: MagicMock) -> MagicMock:
    """Create an async HTTP client double returning the supplied response."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.unit
class TestProcessWebpage:
    """Verify Jina Reader webpage extraction outcomes."""

    async def test_saves_extracted_content(self) -> None:
        """Mark the task complete when Jina Reader returns page content."""
        task = _task()
        response = MagicMock(text="# Extracted content")
        response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient", return_value=_client(response)):
            result = await process_webpage(task)

        assert result is task
        assert task.task_status == TaskStatusEnum.completed
        assert task.result == "# Extracted content"
        assert task.provider_meta == {
            "provider": "jina-reader",
            "url": task.url,
        }
        task.save_report.assert_awaited_once_with("Task processed successfully")

    async def test_rejects_empty_content(self) -> None:
        """Record an error when Jina Reader returns only whitespace."""
        task = _task()
        response = MagicMock(text="  \n")
        response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient", return_value=_client(response)):
            result = await process_webpage(task)

        assert result is task
        assert task.task_status == TaskStatusEnum.error
        assert task.result == "No content extracted from webpage"
        task.save_report.assert_awaited_once_with(task.result)

    async def test_records_http_status_error(self) -> None:
        """Record the upstream status when Jina Reader rejects a request."""
        task = _task()
        response = httpx.Response(404, request=httpx.Request("GET", task.url))
        error = httpx.HTTPStatusError(
            "not found", request=response.request, response=response
        )
        client = _client(MagicMock())
        client.get = AsyncMock(side_effect=error)

        with patch("httpx.AsyncClient", return_value=client):
            result = await process_webpage(task)

        assert result is task
        assert task.task_status == TaskStatusEnum.error
        assert task.result == "Jina Reader error: 404"
        task.save_report.assert_awaited_once_with(task.result)

    async def test_records_request_error(self) -> None:
        """Record transport failures from Jina Reader."""
        task = _task()
        error = httpx.ConnectError("offline", request=httpx.Request("GET", task.url))
        client = _client(MagicMock())
        client.get = AsyncMock(side_effect=error)

        with patch("httpx.AsyncClient", return_value=client):
            result = await process_webpage(task)

        assert result is task
        assert task.task_status == TaskStatusEnum.error
        assert task.result == "Request failed: offline"
        task.save_report.assert_awaited_once_with(task.result)
