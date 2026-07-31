"""Translation task schemas and data models."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class TranslateSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a new translation task."""

    text: str
    language: str | None = Field(
        default="Persian",
        description="Target language for the translation",
    )


class TranslateSchema(
    TenantUserEntitySchema, TaskMixin, WorkspaceScopedSchema, TranslateSchemaCreate
):
    """Complete translation task schema including result and usage fields."""

    result: str | None = None
    provider_meta: dict | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
