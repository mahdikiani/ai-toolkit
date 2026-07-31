"""Text-to-speech task schemas."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class TextToSpeechTaskSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a text-to-speech task."""

    text: str = Field(..., min_length=1, description="Text to convert to speech")
    model: str = "openai/gpt-4o-mini-tts"
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


class TextToSpeechTaskSchema(
    TenantUserEntitySchema,
    TaskMixin,
    WorkspaceScopedSchema,
    TextToSpeechTaskSchemaCreate,
):
    """Full text-to-speech task schema, including provider result fields."""

    result_url: str | None = None
    result_data: bytes | None = None
    provider: str = "openrouter"
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
