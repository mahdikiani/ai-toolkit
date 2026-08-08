"""Media service utilities for file upload via the media API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO

import httpx

from server.config import Settings


class MediaUploadError(RuntimeError):
    """Raised when Media service cannot return an owned temporary file URL."""


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
        follow_redirects=True,
    ) as client:
        yield client


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
    async with get_media_client() as media_client:
        data = {"user_id": user_id}
        if workspace_id:
            data["workspace_id"] = workspace_id
        upload_response = await media_client.post(
            "/f/upload",
            files={"file": file},
            data=data,
        )
        upload_response.raise_for_status()
        file_id = upload_response.json().get("uid")
        if not file_id:
            raise MediaUploadError("missing_file_uid")

        response = await media_client.get(
            f"/f/{file_id}", params={"signed_url": True}
        )
        response.raise_for_status()
        signed_url = response.headers.get("location")
        if not signed_url:
            raise MediaUploadError("missing_signed_url")
        return signed_url
