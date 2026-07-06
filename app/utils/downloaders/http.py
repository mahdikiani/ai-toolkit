"""HTTP downloader utilities."""

from io import BytesIO

import httpx


async def download_bytes(url: str, *, http_timeout: float | None = None) -> BytesIO:
    """Download a URL into an in-memory bytes buffer."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=http_timeout,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    buffer = BytesIO(response.content)
    buffer.seek(0)
    return buffer
