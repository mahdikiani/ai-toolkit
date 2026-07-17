"""Translation task schemas and data models."""

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, Field


class TranslateSchemaCreate(BaseModel):
    """Schema for creating a new translation task."""

    text: str
    language: str | None = Field(
        default="Persian",
        description="Target language for the translation",
    )
    user_id: str | None = None


class TranslateSchema(UserOwnedEntitySchema, TaskMixin, TranslateSchemaCreate):
    """Complete translation task schema including result and usage fields."""

    result: str | None = None
    provider_meta: dict | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
