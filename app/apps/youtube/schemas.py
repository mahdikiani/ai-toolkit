"""YouTube transcription task schemas."""

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field, field_validator

from .video_id import YouTubeVideoIdTypeError, parse_youtube_video_id


class YoutubeTranscriptTaskSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a YouTube transcription task."""

    video_id: str = Field(
        ...,
        description="YouTube video id or full YouTube URL",
    )

    @field_validator("video_id", mode="before")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        """Normalize a YouTube video id or supported URL to a bare video id."""
        if not isinstance(v, str):
            raise YouTubeVideoIdTypeError
        return parse_youtube_video_id(v)


class YoutubeTranscriptTaskSchema(
    UserOwnedEntitySchema, TaskMixin, YoutubeTranscriptTaskSchemaCreate
):
    """Complete YouTube transcription task schema with result."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
