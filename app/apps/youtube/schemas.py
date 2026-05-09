"""YouTube transcription task schemas."""

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel


class YoutubeTaskSchemaCreate(BaseModel):
    """Schema for creating a YouTube transcription task."""

    video_id: str
    user_id: str | None = None


class YoutubeTaskSchema(UserOwnedEntitySchema, TaskMixin, YoutubeTaskSchemaCreate):  # type: ignore[misc]
    """Complete YouTube transcription task schema with result."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
