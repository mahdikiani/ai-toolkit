"""Google Drive public share-link resolution."""

from __future__ import annotations

import re

GDRIVE_FILE_PATTERN = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
GDRIVE_OPEN_PATTERN = re.compile(
    r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
GDRIVE_UC_PATTERN = re.compile(
    r"https?://drive\.google\.com/uc\?",
    re.IGNORECASE,
)
DOCS_PATTERN = re.compile(
    r"https?://docs\.google\.com/(\w+)/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
CONFIRM_TOKEN_PATTERN = re.compile(r"confirm=([0-9A-Za-z_]+)")


def extract_gdrive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from a public share URL."""
    for pattern in (GDRIVE_FILE_PATTERN, GDRIVE_OPEN_PATTERN):
        match = pattern.search(url)
        if match:
            return match.group(1)
    docs_match = DOCS_PATTERN.search(url)
    if docs_match:
        return docs_match.group(2)
    return None


def is_gdrive_url(url: str) -> bool:
    """Return True when the URL appears to be a Google Drive/Docs share link."""
    return extract_gdrive_file_id(url) is not None or bool(
        GDRIVE_UC_PATTERN.search(url)
    )


def gdrive_direct_download_url(file_id: str) -> str:
    """Build a direct-download URL for a public Drive file id."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def resolve_gdrive_download_url(url: str) -> str:
    """Resolve a Google Drive share URL to a direct download URL."""
    file_id = extract_gdrive_file_id(url)
    if not file_id:
        return url
    return gdrive_direct_download_url(file_id)


def parse_large_file_confirm_token(html: str) -> str | None:
    """Parse Drive's virus-scan confirmation token from an HTML interstitial."""
    match = CONFIRM_TOKEN_PATTERN.search(html)
    return match.group(1) if match else None
