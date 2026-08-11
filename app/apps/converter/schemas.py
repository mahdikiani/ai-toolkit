"""Converter request/response schemas."""

from pydantic import BaseModel, Field

from apps.artifacts.enums import ArtifactFormat
from utils.workspace import WorkspaceScopedSchema


class ConvertRequest(WorkspaceScopedSchema):
    """Convert an existing Artifact into a derived Artifact."""

    artifact_id: str = Field(..., min_length=1)
    target_format: ArtifactFormat


class ConversionFormatEdge(BaseModel):
    """One registered conversion edge for discovery APIs."""

    source_format: ArtifactFormat
    target_format: ArtifactFormat
    name: str
