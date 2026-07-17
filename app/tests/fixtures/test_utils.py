"""Utility functions for common test assertions."""

from typing import Protocol

from fastapi_mongo_base.tasks import TaskStatusEnum


class TaskLike(Protocol):
    """Represent the task attributes checked by these assertions."""

    task_status: TaskStatusEnum
    result: object | None
    error: str | None


class ApiResponseLike(Protocol):
    """Represent the response attributes checked by these assertions."""

    status_code: int
    text: str

    def json(self) -> dict[str, object]:
        """Return the response body as JSON."""


def assert_task_status(task: TaskLike, expected_status: TaskStatusEnum) -> None:
    """Assert that a task has the expected status."""
    assert task.task_status == expected_status, (
        f"Expected task status {expected_status}, got {task.task_status}"
    )


def assert_task_completed(task: TaskLike) -> None:
    """Assert that a task completed successfully."""
    assert task.task_status == TaskStatusEnum.completed, (
        f"Expected task to be completed, got {task.task_status}"
    )
    assert task.result is not None, "Expected task to have a result"
    assert task.error is None, f"Expected no error, got: {task.error}"


def assert_task_failed(task: TaskLike) -> None:
    """Assert that a task failed with an error."""
    assert task.task_status == TaskStatusEnum.error, (
        f"Expected task to be in error state, got {task.task_status}"
    )
    assert task.error is not None, "Expected task to have an error message"


def assert_paginated_response(data: dict, max_limit: int | None = None) -> None:
    """Assert that a response has valid pagination structure."""
    assert "items" in data, "Response missing 'items' field"
    assert "total" in data, "Response missing 'total' field"
    assert "offset" in data, "Response missing 'offset' field"
    assert "limit" in data, "Response missing 'limit' field"

    assert isinstance(data["items"], list), "'items' should be a list"
    assert isinstance(data["total"], int), "'total' should be an integer"
    assert data["total"] >= 0, "'total' should be non-negative"
    assert data["offset"] >= 0, "'offset' should be non-negative"
    assert data["limit"] >= 1, "'limit' should be at least 1"

    if max_limit is not None:
        assert len(data["items"]) <= max_limit, (
            f"Returned {len(data['items'])} items but limit is {max_limit}"
        )

    # Pagination invariant: offset + count <= total
    if data["total"] > 0 and data["offset"] < data["total"]:
        assert data["offset"] + len(data["items"]) <= data["total"], (
            f"offset ({data['offset']}) + count ({len(data['items'])}) "
            f"> total ({data['total']})"
        )


def assert_valid_transition(
    from_status: TaskStatusEnum,
    to_status: TaskStatusEnum,
    valid_transitions: dict[TaskStatusEnum, set[TaskStatusEnum]],
) -> None:
    """Assert that a task status transition is valid."""
    allowed = valid_transitions.get(from_status, set())
    assert to_status in allowed, (
        f"Invalid transition: {from_status} -> {to_status}. "
        f"Allowed transitions from {from_status}: {allowed}"
    )


def assert_api_error_response(response: ApiResponseLike, expected_status: int) -> None:
    """Assert that an API response is an error with the expected status code."""
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.text}"
    )
    data = response.json()
    assert "detail" in data, "Error response should have 'detail' field"
