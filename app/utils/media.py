"""Media service utilities for file upload via the media API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO

import httpx

from server.config import Settings


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


async def upload_file(file: BytesIO) -> str:
    """
    Upload a file to the media service and make it publicly accessible.

    Args:
        file: BytesIO object containing file data.

    Returns:
        Public URL of the uploaded file.
    """
    async with get_media_client() as media_client:
        upload_response = await media_client.post("/f/upload", files={"file": file})
        upload_response.raise_for_status()
        file_id = upload_response.json().get("uid")

        response = await media_client.patch(
            f"/f/{file_id}", json={"public_permission": {"permission": 10}}
        )
        response.raise_for_status()
        return upload_response.json().get("url")
