"""Download file bytes from web URLs."""

from io import BytesIO

import httpx

from .gdrive import (
    gdrive_direct_download_url,
    is_gdrive_url,
    parse_large_file_confirm_token,
    resolve_gdrive_download_url,
)


async def download_bytes(url: str, *, http_timeout: float | None = None) -> BytesIO:
    """Download a URL into an in-memory bytes buffer."""
    download_url = resolve_gdrive_download_url(url) if is_gdrive_url(url) else url

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=http_timeout,
    ) as client:
        response = await client.get(download_url)
        response.raise_for_status()

        if is_gdrive_url(url) and "text/html" in response.headers.get(
            "content-type", ""
        ):
            token = parse_large_file_confirm_token(response.text)
            if token:
                file_id = download_url.split("id=", 1)[-1]
                confirm_url = gdrive_direct_download_url(file_id) + f"&confirm={token}"
                response = await client.get(confirm_url)
                response.raise_for_status()

    buffer = BytesIO(response.content)
    buffer.seek(0)
    return buffer
