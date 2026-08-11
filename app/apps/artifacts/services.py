"""Artifact create/read services (Media-backed content)."""

from __future__ import annotations

from io import BytesIO

from fastapi_mongo_base.errors import BadRequestError, NotFoundError

from utils.integrations.media import download_by_storage_uri, upload_file_durable

from .enums import EXTENSION_BY_FORMAT, MIME_BY_FORMAT, ArtifactFormat
from .models import Artifact


def _default_filename(
    *,
    artifact_format: ArtifactFormat,
    title: str | None,
    original_name: str | None,
) -> str:
    if original_name:
        return original_name
    stem = (title or "document").strip() or "document"
    ext = EXTENSION_BY_FORMAT[artifact_format]
    if stem.lower().endswith(ext):
        return stem
    return f"{stem}{ext}"


async def create_artifact_from_bytes(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    artifact_format: ArtifactFormat,
    user_id: str,
    tenant_id: str,
    workspace_id: str | None = None,
    artifact_type: str = "document",
    title: str | None = None,
    original_name: str | None = None,
    language: str | None = None,
    source: str = "upload",
    parent_artifact_id: str | None = None,
    meta_data: dict | None = None,
) -> Artifact:
    """Upload bytes to Media and persist Artifact metadata with durable URI."""
    buf = BytesIO(data)
    buf.seek(0)
    uploaded = await upload_file_durable(
        buf,
        user_id=user_id,
        workspace_id=workspace_id,
        filename=filename,
        content_type=content_type,
    )
    return await Artifact.create_item(
        {
            "type": artifact_type,
            "format": artifact_format,
            "storage_uri": uploaded.storage_uri,
            "mime_type": content_type or MIME_BY_FORMAT[artifact_format],
            "original_name": original_name or filename,
            "title": title,
            "language": language,
            "source": source,
            "parent_artifact_id": parent_artifact_id,
            "meta_data": meta_data,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }
    )


async def create_artifact_from_text(
    *,
    text: str,
    user_id: str,
    tenant_id: str,
    artifact_format: ArtifactFormat = ArtifactFormat.markdown,
    workspace_id: str | None = None,
    artifact_type: str = "document",
    title: str | None = None,
    original_name: str | None = None,
    language: str | None = None,
    source: str = "upload",
    parent_artifact_id: str | None = None,
    meta_data: dict | None = None,
) -> Artifact:
    """Create an Artifact from UTF-8 text (typically markdown)."""
    if artifact_format not in {ArtifactFormat.markdown, ArtifactFormat.html}:
        raise BadRequestError(
            error_code="invalid_text_format",
            detail=f"format {artifact_format.value} cannot be created from text",
        )
    filename = _default_filename(
        artifact_format=artifact_format, title=title, original_name=original_name
    )
    return await create_artifact_from_bytes(
        data=text.encode("utf-8"),
        filename=filename,
        content_type=MIME_BY_FORMAT[artifact_format],
        artifact_format=artifact_format,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        artifact_type=artifact_type,
        title=title,
        original_name=original_name or filename,
        language=language,
        source=source,
        parent_artifact_id=parent_artifact_id,
        meta_data=meta_data,
    )


async def get_artifact_for_user(
    *,
    artifact_id: str,
    user_id: str,
    tenant_id: str,
) -> Artifact:
    """Load an Artifact scoped to tenant/user or raise 404."""
    artifact = await Artifact.get_item(
        uid=artifact_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    if artifact is None:
        raise NotFoundError(
            error_code="artifact_not_found",
            detail=f"Artifact {artifact_id} not found",
            message={
                "en": "Artifact not found",
                "fa": "آرتیفکت یافت نشد",
            },
        )
    return artifact


async def read_artifact_bytes(artifact: Artifact) -> bytes:
    """Resolve Artifact content bytes via durable Media storage_uri."""
    return await download_by_storage_uri(artifact.storage_uri)
