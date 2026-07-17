"""Webpage extraction task schemas."""

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field


class WebpageTaskSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a webpage extraction task."""

    url: str = Field(..., min_length=1, description="Public webpage URL to extract")
    webhook_custom_headers: dict | None = None


class WebpageTaskSchema(UserOwnedEntitySchema, TaskMixin, WebpageTaskSchemaCreate):
    """Complete webpage extraction task schema with result."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
