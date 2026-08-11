"""Pydantic schemas for Artifact SoR."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from pydantic import BaseModel, Field

from utils.workspace import WorkspaceScopedSchema

from .enums import ArtifactFormat


class ArtifactCreate(WorkspaceScopedSchema):
    """Create an Artifact from inline text content (Phase 1 JSON API)."""

    type: str = Field(default="document", description="Logical artifact kind")
    format: ArtifactFormat
    content: str | None = Field(
        default=None,
        description="Inline text content (required for text formats like markdown)",
    )
    title: str | None = None
    original_name: str | None = None
    language: str | None = None
    source: str = Field(
        default="upload",
        description="Origin: upload | ocr | youtube | webpage | converter | …",
    )
    parent_artifact_id: str | None = None
    meta_data: dict | None = None


class ArtifactSchema(TenantUserEntitySchema, WorkspaceScopedSchema):
    """Stored Artifact metadata; bytes live in Media via storage_uri."""

    type: str = "document"
    format: ArtifactFormat
    storage_uri: str = Field(
        ...,
        description="Durable Media reference, e.g. media:{uid}",
    )
    mime_type: str
    original_name: str | None = None
    title: str | None = None
    language: str | None = None
    source: str = "upload"
    parent_artifact_id: str | None = None


class ArtifactConvertHint(BaseModel):
    """Lightweight response fragment when nesting convert results."""

    uid: str
    format: ArtifactFormat
    title: str | None = None
