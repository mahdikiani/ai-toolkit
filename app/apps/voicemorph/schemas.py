"""Voice morphing task schemas."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class VoiceMorphTaskSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a voice morph task."""

    audio_url: str = Field(..., description="Source audio URL to morph")
    voice_reference_url: str | None = Field(None, description="Reference voice URL")
    model: str = "openai/whisper-1"
    pitch_shift: float | None = None
    speed_factor: float | None = None


class VoiceMorphTaskSchema(
    TenantUserEntitySchema,
    TaskMixin,
    WorkspaceScopedSchema,
    VoiceMorphTaskSchemaCreate,
):
    """Full voice morph task schema, including provider result fields."""

    result_url: str | None = None
    result_data: bytes | None = None
    provider: str = "replicate"
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
