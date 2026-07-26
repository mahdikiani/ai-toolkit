"""Download file bytes from web URLs."""

from io import BytesIO

import httpx

from .gdrive import (
    gdrive_direct_download_url,
    is_gdrive_url,
    parse_large_file_confirm_token,
    resolve_gdrive_download_url,
)
from .url_safety import assert_safe_url


async def download_bytes(url: str, *, http_timeout: float | None = 120.0) -> BytesIO:
    """Download a URL into an in-memory bytes buffer with SSRF protection."""
    assert_safe_url(url)
    download_url = resolve_gdrive_download_url(url) if is_gdrive_url(url) else url
    assert_safe_url(download_url)

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=http_timeout,
    ) as client:
        response = await client.get(download_url)
        # Manually follow redirects so each hop is re-validated.
        redirects = 0
        while response.is_redirect and redirects < 5:
            location = response.headers.get("location")
            if not location:
                break
            assert_safe_url(location)
            response = await client.get(location)
            redirects += 1

        response.raise_for_status()

        if is_gdrive_url(url) and "text/html" in response.headers.get(
            "content-type", ""
        ):
            token = parse_large_file_confirm_token(response.text)
            if token:
                file_id = download_url.split("id=", 1)[-1]
                confirm_url = gdrive_direct_download_url(file_id) + f"&confirm={token}"
                assert_safe_url(confirm_url)
                response = await client.get(confirm_url)
                response.raise_for_status()

    buffer = BytesIO(response.content)
    buffer.seek(0)
    return buffer
