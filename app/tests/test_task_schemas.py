"""Tests for promptic schemas."""

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import ValidationError

from apps.language.promptic.schemas import PrompticCreate, PrompticSchema


class TestPrompticCreate:
    """Test PrompticCreate schema."""

    def test_minimal_valid_creation(self) -> None:
        """Test creating task with minimal required fields."""
        task = PrompticCreate(
            input_variables={"key": "value"},
        )
        assert task.input_variables == {"key": "value"}
        assert task.webhook_url is None
        assert task.idempotency_key is None
        assert task.meta_data == {}

    def test_full_creation(self) -> None:
        """Test creating task with all fields."""
        task = PrompticCreate(
            input_variables={"key": "value"},
            webhook_url="https://example.com/webhook",
            idempotency_key="custom_key",
            meta_data={"chat_id": 456},
        )
        assert task.input_variables == {"key": "value"}
        assert task.webhook_url == "https://example.com/webhook"
        assert task.idempotency_key == "custom_key"
        assert task.meta_data == {"chat_id": 456}

    def test_default_values(self) -> None:
        """Test default values are applied correctly."""
        task = PrompticCreate()
        assert task.input_variables == {}
        assert task.meta_data == {}


class TestPrompticSchema:
    """Test PrompticSchema."""

    def test_schema_inheritance(self) -> None:
        """Test that ExecutionTaskSchema inherits all fields."""
        task = PrompticSchema(
            prompt_name="test_prompt",
            input_variables={"key": "value"},
            idempotency_key="generated_key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.init,
        )

        assert task.prompt_name == "test_prompt"
        assert task.input_variables == {"key": "value"}
        assert task.idempotency_key == "generated_key_123"
        assert task.user_id == "user_123"
        assert task.result is None
        assert task.error is None
        assert task.completed_at is None
        assert task.webhook_failed is False

    def test_idempotency_key_required_in_schema(self) -> None:
        """Test that idempotency_key is required in ExecutionTaskSchema."""
        with pytest.raises(ValidationError) as exc_info:
            PrompticSchema(
                prompt_name="test",
                user_id="user_123",
                uid="550e8400-e29b-41d4-a716-446655440000",
                task_status=TaskStatusEnum.init,
            )

        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("idempotency_key",) for e in errors)

    def test_result_and_error_fields(self) -> None:
        """Test result and error fields."""
        task = PrompticSchema(
            prompt_name="test",
            idempotency_key="key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.completed,
            result="LLM response text",
            error=None,
        )

        assert task.result == "LLM response text"
        assert task.error is None

    def test_webhook_failed_default(self) -> None:
        """Test webhook_failed defaults to False."""
        task = PrompticSchema(
            prompt_name="test",
            idempotency_key="key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.init,
        )

        assert task.webhook_failed is False
