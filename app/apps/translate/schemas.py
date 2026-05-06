"""Provide module functionality."""
from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel


class TranslateSchemaCreate(BaseModel):
    """Represent TranslateSchemaCreate."""

    text: str
    user_id: str | None = None


class TranslateSchema(UserOwnedEntitySchema, TaskMixin, TranslateSchemaCreate):  # type: ignore[misc]
    """Represent TranslateSchema."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
