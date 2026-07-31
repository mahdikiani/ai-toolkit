"""Schemas for promptic task management."""

from datetime import datetime

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class PrompticCreate(TaskCreateFieldsMixin):
    """Schema for creating a promptic run."""

    input_variables: dict[str, object] = Field(
        default_factory=dict,
        description="Jinja2 variables for template rendering",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Custom idempotency key for deduplication",
    )


class PrompticSchema(
    TenantUserEntitySchema, TaskMixin, WorkspaceScopedSchema, PrompticCreate
):
    """Complete promptic schema with lifecycle tracking."""

    prompt_name: str = Field(..., description="Name of the prompt template")

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
