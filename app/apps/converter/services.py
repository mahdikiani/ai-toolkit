"""Artifact→Artifact conversion orchestration."""

from __future__ import annotations

from fastapi_mongo_base.errors import BadRequestError

from apps.artifacts.enums import EXTENSION_BY_FORMAT, MIME_BY_FORMAT, ArtifactFormat
from apps.artifacts.models import Artifact
from apps.artifacts.services import (
    create_artifact_from_bytes,
    get_artifact_for_user,
    read_artifact_bytes,
)

from . import registry


def _output_filename(source: Artifact, target_format: ArtifactFormat) -> str:
    stem = (source.title or source.original_name or "document").strip() or "document"
    # strip a known extension from the stem
    lower = stem.lower()
    for ext in EXTENSION_BY_FORMAT.values():
        if lower.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return f"{stem}{EXTENSION_BY_FORMAT[target_format]}"


async def convert_artifact(
    *,
    artifact_id: str,
    target_format: ArtifactFormat,
    user_id: str,
    tenant_id: str,
    workspace_id: str | None = None,
) -> Artifact:
    """
    Convert a source Artifact into a derived Artifact of ``target_format``.

    Looks up a direct registry edge; unsupported pairs raise 400.
    """
    registry.ensure_builtin_strategies()
    source = await get_artifact_for_user(
        artifact_id=artifact_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )
    if source.format == target_format:
        raise BadRequestError(
            error_code="same_format",
            detail="source and target formats are identical",
            message={
                "en": "Already in the requested format",
                "fa": "آرتیفکت از قبل در همین فرمت است",
            },
        )

    edge = registry.get_edge(source.format, target_format)
    if edge is None:
        raise BadRequestError(
            error_code="unsupported_conversion",
            detail=(
                f"No conversion edge registered for "
                f"{source.format.value} → {target_format.value}"
            ),
            message={
                "en": "Unsupported conversion",
                "fa": "این تبدیل پشتیبانی نمی‌شود",
            },
        )

    source_bytes = await read_artifact_bytes(source)
    title = source.title or ""
    out_bytes = edge.strategy(source_bytes, title=title)
    filename = _output_filename(source, target_format)

    return await create_artifact_from_bytes(
        data=out_bytes,
        filename=filename,
        content_type=MIME_BY_FORMAT[target_format],
        artifact_format=target_format,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id or source.workspace_id,
        artifact_type=source.type,
        title=source.title,
        original_name=filename,
        language=source.language,
        source="converter",
        parent_artifact_id=str(source.uid),
        meta_data={
            "conversion": {
                "source_format": source.format.value,
                "target_format": target_format.value,
                "strategy": edge.name,
            }
        },
    )


def render_markdown_to_format(
    markdown: str,
    *,
    target_format: ArtifactFormat,
    title: str = "",
) -> bytes:
    """
    Sync helper for legacy streaming endpoints (document-convert).

    Uses the same registry strategies as Artifact conversion without
    persisting intermediate Artifacts.
    """
    registry.ensure_builtin_strategies()
    edge = registry.get_edge(ArtifactFormat.markdown, target_format)
    if edge is None:
        raise BadRequestError(
            error_code="unsupported_conversion",
            detail=(
                f"No conversion edge registered for "
                f"markdown → {target_format.value}"
            ),
        )
    return edge.strategy(markdown.encode("utf-8"), title=title)
