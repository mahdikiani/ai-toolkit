"""Video generation task schemas."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class VideoGenTaskSchemaCreate(TaskCreateFieldsMixin, WorkspaceScopedSchema):
    """Schema for creating a video generation task."""

    prompt: str = Field(
        ..., min_length=1, description="Text prompt for video generation"
    )
    model: str = "luma/ray-2-720p"
    image_url: str | None = None
    duration: int | None = None
    negative_prompt: str | None = None


class VideoGenTaskSchema(
    TenantUserEntitySchema, TaskMixin, VideoGenTaskSchemaCreate
):
    """Full video generation task schema, including provider result fields."""

    result_url: str | None = None
    provider: str = "openrouter"
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
