"""Media service utilities for file upload via the media API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO

import httpx

from server.config import Settings

MEDIA_URI_PREFIX = "media:"


class MediaUploadError(RuntimeError):
    """Raised when Media service cannot return an owned temporary file URL."""


class MediaDownloadError(RuntimeError):
    """Raised when Media content cannot be resolved from a durable URI."""


@dataclass(frozen=True, slots=True)
class DurableUploadResult:
    """Durable Media identity plus optional ephemeral signed URL."""

    file_uid: str
    storage_uri: str
    signed_url: str | None = None


@asynccontextmanager
async def get_media_client() -> AsyncGenerator[httpx.AsyncClient]:
    """
    Create an async HTTP client configured for the media API.

    Yields:
        Configured httpx.AsyncClient for media API calls.
    """
    async with httpx.AsyncClient(
        base_url=Settings.media_base_url or "https://media.uln.me/api/media/v1/",
        headers={"x-api-key": Settings.media_api_key or ""},
    ) as client:
        yield client


def media_storage_uri(file_uid: str) -> str:
    """Build a durable ``media:{uid}`` storage URI."""
    return f"{MEDIA_URI_PREFIX}{file_uid}"


def parse_media_uid(storage_uri: str) -> str:
    """Extract the Media file uid from a durable ``media:{uid}`` URI."""
    if not storage_uri.startswith(MEDIA_URI_PREFIX):
        raise MediaDownloadError("invalid_storage_uri")
    file_uid = storage_uri.removeprefix(MEDIA_URI_PREFIX).strip()
    if not file_uid:
        raise MediaDownloadError("invalid_storage_uri")
    return file_uid


async def _upload_raw(
    file: BytesIO,
    *,
    user_id: str,
    workspace_id: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """POST bytes to Media and return the owned file uid."""
    async with get_media_client() as media_client:
        data = {"user_id": user_id}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if filename:
            file_tuple: tuple = (
                filename,
                file,
                content_type or "application/octet-stream",
            )
            files = {"file": file_tuple}
        else:
            files = {"file": file}
        upload_response = await media_client.post(
            "/f/upload",
            files=files,
            data=data,
        )
        upload_response.raise_for_status()
        file_id = upload_response.json().get("uid")
        if not file_id:
            raise MediaUploadError("missing_file_uid")
        return str(file_id)


async def _fetch_signed_url(file_uid: str) -> str:
    """Resolve an ephemeral signed URL for a Media file uid."""
    async with get_media_client() as media_client:
        response = await media_client.get(
            f"/f/{file_uid}", params={"signed_url": True}
        )
        # Media returns the signed URL in the Location header of a 307
        # response.  Do not follow it: the caller needs that URL itself.
        if not response.is_redirect:
            response.raise_for_status()
        signed_url = response.headers.get("location")
        if not signed_url:
            raise MediaUploadError("missing_signed_url")
        return signed_url


async def upload_file(
    file: BytesIO,
    *,
    user_id: str,
    workspace_id: str | None = None,
) -> str:
    """
    Upload a private, owned file and ask Media service for temporary access.

    Args:
        file: BytesIO object containing file data.
        user_id: USSO owner identifier attributed by Media service.
        workspace_id: Optional owning Workspace identifier.

    Returns:
        A temporary URL signed by the Media service/storage backend.
    """
    file_id = await _upload_raw(file, user_id=user_id, workspace_id=workspace_id)
    return await _fetch_signed_url(file_id)


async def upload_file_durable(
    file: BytesIO,
    *,
    user_id: str,
    workspace_id: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    with_signed_url: bool = False,
) -> DurableUploadResult:
    """
    Upload a file and return a durable ``media:{uid}`` storage URI.

    Unlike ``upload_file``, this does not require a signed URL for SoR
    persistence — Artifact.storage_uri must survive URL expiry.
    """
    file_uid = await _upload_raw(
        file,
        user_id=user_id,
        workspace_id=workspace_id,
        filename=filename,
        content_type=content_type,
    )
    signed_url = await _fetch_signed_url(file_uid) if with_signed_url else None
    return DurableUploadResult(
        file_uid=file_uid,
        storage_uri=media_storage_uri(file_uid),
        signed_url=signed_url,
    )


async def signed_url_for_storage_uri(storage_uri: str) -> str:
    """Resolve an ephemeral signed URL for a durable ``media:{uid}`` URI."""
    return await _fetch_signed_url(parse_media_uid(storage_uri))


async def download_by_storage_uri(storage_uri: str) -> bytes:
    """Download file bytes for a durable ``media:{uid}`` URI."""
    signed_url = await signed_url_for_storage_uri(storage_uri)
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(signed_url)
        response.raise_for_status()
        return response.content
