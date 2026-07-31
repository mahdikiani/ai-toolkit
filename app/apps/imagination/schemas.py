"""Imagination task schemas."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class ImaginationTaskSchemaCreate(TaskCreateFieldsMixin, WorkspaceScopedSchema):
    """Fields accepted when creating an imagination task."""

    prompt: str = Field(..., min_length=1, description="Imagination prompt")
    model: str | None = None
    size: str = "1024x1024"
    enhance_prompt: bool = True


class ImaginationTaskSchema(
    TenantUserEntitySchema,
    TaskMixin,
    ImaginationTaskSchemaCreate,
):
    """Complete imagination task schema with lifecycle tracking."""

    result_url: str | None = None
    result_b64: str | None = None
    enhanced_prompt: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
