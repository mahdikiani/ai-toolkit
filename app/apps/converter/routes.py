"""USSO-authenticated Artifact conversion API."""

from fastapi import APIRouter, Depends
from usso import UserData

from apps.artifacts.models import Artifact
from apps.artifacts.schemas import ArtifactSchema
from utils.usso import get_usso

from . import registry
from .schemas import ConversionFormatEdge, ConvertRequest
from .services import convert_artifact

router = APIRouter(prefix="/convert", tags=["Converter"])
auth = get_usso(raise_exception=True)


@router.post("", response_model=ArtifactSchema, status_code=201)
async def convert(
    data: ConvertRequest,
    user: UserData = Depends(auth),
) -> Artifact:
    """Convert an Artifact to ``target_format`` and return the derived Artifact."""
    return await convert_artifact(
        artifact_id=data.artifact_id,
        target_format=data.target_format,
        user_id=user.uid,
        tenant_id=user.tenant_id,
        workspace_id=data.workspace_id or getattr(user, "workspace_id", None),
    )


@router.get("/formats", response_model=list[ConversionFormatEdge])
async def list_conversion_formats(
    user: UserData = Depends(auth),
) -> list[ConversionFormatEdge]:
    """List registered conversion edges in the Artifact format graph."""
    del user  # auth-gated discovery; no user-specific filtering yet
    registry.ensure_builtin_strategies()
    return [
        ConversionFormatEdge(
            source_format=edge.source_format,
            target_format=edge.target_format,
            name=edge.name,
        )
        for edge in registry.list_edges()
    ]
