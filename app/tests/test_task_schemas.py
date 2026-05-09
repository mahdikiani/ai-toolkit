"""Tests for execution task schemas."""

import pytest
from apps.executions.schemas import ExecutionTaskCreate, ExecutionTaskSchema
from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import ValidationError


class TestExecutionTaskCreate:
    """Test ExecutionTaskCreate schema."""

    def test_minimal_valid_creation(self) -> None:
        """Test creating task with minimal required fields."""
        task = ExecutionTaskCreate(
            template_name="test_template",
            input_variables={"key": "value"},
        )
        assert task.template_name == "test_template"
        assert task.input_variables == {"key": "value"}
        assert task.user_id is None
        assert task.webhook_url is None
        assert task.idempotency_key is None
        assert task.meta_data == {}
        assert task.blocking_mode is True
        assert task.force_refresh is False

    def test_full_creation(self) -> None:
        """Test creating task with all fields."""
        task = ExecutionTaskCreate(
            template_name="test_template",
            input_variables={"key": "value"},
            user_id="user_123",
            webhook_url="https://example.com/webhook",
            idempotency_key="custom_key",
            meta_data={"chat_id": 456},
            blocking_mode=False,
            force_refresh=True,
        )
        assert task.template_name == "test_template"
        assert task.input_variables == {"key": "value"}
        assert task.user_id == "user_123"
        assert task.webhook_url == "https://example.com/webhook"
        assert task.idempotency_key == "custom_key"
        assert task.meta_data == {"chat_id": 456}
        assert task.blocking_mode is False
        assert task.force_refresh is True

    def test_missing_template_name(self) -> None:
        """Test that template_name is required."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionTaskCreate(input_variables={"key": "value"})

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("template_name",) for e in errors)

    def test_default_values(self) -> None:
        """Test default values are applied correctly."""
        task = ExecutionTaskCreate(
            template_name="test",
        )
        assert task.input_variables == {}
        assert task.meta_data == {}
        assert task.blocking_mode is True
        assert task.force_refresh is False


class TestExecutionTaskSchema:
    """Test ExecutionTaskSchema."""

    def test_schema_inheritance(self) -> None:
        """Test that ExecutionTaskSchema inherits all fields."""
        task_data = {
            "template_name": "test_template",
            "input_variables": {"key": "value"},
            "idempotency_key": "generated_key_123",
            "user_id": "user_123",
            "uid": "550e8400-e29b-41d4-a716-446655440000",
            "task_status": TaskStatusEnum.init,
        }

        task = ExecutionTaskSchema(**task_data)

        assert task.template_name == "test_template"
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
            ExecutionTaskSchema(
                template_name="test",
                user_id="user_123",
                uid="550e8400-e29b-41d4-a716-446655440000",
                task_status=TaskStatusEnum.init,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("idempotency_key",) for e in errors)

    def test_result_and_error_fields(self) -> None:
        """Test result and error fields."""
        task = ExecutionTaskSchema(
            template_name="test",
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
        task = ExecutionTaskSchema(
            template_name="test",
            idempotency_key="key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.init,
        )

        assert task.webhook_failed is False
