"""Schemas for promptic task management."""

from datetime import datetime

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, Field


class PrompticCreate(BaseModel):
    """Schema for creating a promptic run."""

    input_variables: dict[str, object] = Field(
        default_factory=dict,
        description="Jinja2 variables for template rendering",
    )
    webhook_url: str | None = Field(
        default=None,
        description="Callback URL for async notifications",
    )
    webhook_custom_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers to send with webhook notifications",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Custom idempotency key for deduplication",
    )
    meta_data: dict[str, object] = Field(
        default_factory=dict,
        description="Custom metadata from client",
    )


class PrompticSchema(UserOwnedEntitySchema, TaskMixin, PrompticCreate):
    """
    Complete promptic schema with lifecycle tracking.

    Inherits from UserOwnedEntitySchema for user ownership and permissions.
    Inherits from TaskMixin for task lifecycle management and webhooks.
    Inherits from PrompticCreate for core promptic fields.
    """

    prompt_name: str = Field(..., description="Name of the prompt template")

    # Override idempotency_key to make it required in stored tasks
    # (it's auto-generated if not provided in the request)
    idempotency_key: str = Field(
        ...,
        description="SHA256 hash for deduplication (auto-generated if not provided)",
    )

    result: str | None = Field(
        default=None,
        description="LLM response text",
    )
    provider_meta: dict[str, object] | None = Field(
        default=None,
        description="Raw provider/model usage metadata for later pay-as-you-go billing",
    )
    usage_amount: float | None = Field(
        default=None,
        description="Metered usage amount charged through finance",
    )
    usage_id: str | None = Field(
        default=None,
        description="Finance usage record uid",
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when task completed",
    )
    webhook_failed: bool = Field(
        default=False,
        description="True if webhook delivery failed after all retries",
    )


ExecutionTaskCreate = PrompticCreate
ExecutionTaskSchema = PrompticSchema
