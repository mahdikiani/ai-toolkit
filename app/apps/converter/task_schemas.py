"""Conversion task schemas (async orchestration over Artifact convert)."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from apps.artifacts.enums import ArtifactFormat
from utils.workspace import WorkspaceScopedSchema


class ConversionTaskSchemaCreate(TaskCreateFieldsMixin, WorkspaceScopedSchema):
    """
    Create a conversion task from a Media source.

    Entry points planned:
    - POST /conversions/from-media (this schema) — implemented
    - POST /conversions/from-upload — deferred (security review)
    - POST /conversions/from-base64 — deferred (security review)
    """

    source_uri: str = Field(
        ...,
        min_length=1,
        description="Media durable URI (media:{uid}) or Media /f/{uid} URL",
    )
    source_format: ArtifactFormat = Field(
        ...,
        description="Format of the Media object (required; no guessing in v1)",
    )
    target_format: ArtifactFormat
    title: str | None = None
    original_name: str | None = None


class ConversionTaskSchema(
    TenantUserEntitySchema, TaskMixin, ConversionTaskSchemaCreate
):
    """Conversion task with Artifact IDs after completion."""

    source_artifact_id: str | None = None
    result_artifact_id: str | None = None
    result_storage_uri: str | None = None
    result: dict | None = None
    provider_meta: dict | None = None
