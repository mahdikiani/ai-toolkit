"""Media source_uri normalization for conversion-from-media."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from fastapi_mongo_base.errors import BadRequestError

from server.config import Settings
from utils.integrations.media import MEDIA_URI_PREFIX, media_storage_uri

# /f/{uid} with optional trailing path/query
_MEDIA_PATH_UID_RE = re.compile(r"/f/([A-Za-z0-9_-]+)")

# Known production Media host; always allowed alongside Settings.media_base_url.
_DEFAULT_MEDIA_HOST = "media.uln.me"


def _allowed_media_hosts() -> frozenset[str]:
    """Exact hostnames accepted for HTTPS Media /f/<uid> URLs."""
    hosts: set[str] = {_DEFAULT_MEDIA_HOST}
    base = Settings.media_base_url or f"https://{_DEFAULT_MEDIA_HOST}/api/media/v1/"
    configured = (urlparse(base).hostname or "").lower().rstrip(".")
    if configured:
        hosts.add(configured)
    return frozenset(hosts)


def normalize_media_source_uri(source_uri: str) -> str:
    """
    Normalize a Media reference to durable ``media:{uid}``.

    Accepted:
    - ``media:{uid}``
    - Media HTTPS URL containing ``/f/<uid>`` on an allowlisted Media host

    Rejected: arbitrary public URLs (use a future non-Media ingest path).
    """
    raw = (source_uri or "").strip()
    if not raw:
        raise BadRequestError(
            error_code="missing_source_uri",
            detail="source_uri is required",
        )

    if raw.startswith(MEDIA_URI_PREFIX):
        uid = raw.removeprefix(MEDIA_URI_PREFIX).strip()
        if not uid:
            raise BadRequestError(
                error_code="invalid_media_uri",
                detail="empty media uid",
            )
        return media_storage_uri(uid)

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.path:
        match = _MEDIA_PATH_UID_RE.search(parsed.path)
        if match:
            host = (parsed.hostname or "").lower().rstrip(".")
            if host not in _allowed_media_hosts():
                raise BadRequestError(
                    error_code="non_media_url",
                    detail="from-media only accepts Media service URLs",
                    message={
                        "en": "Only Media URLs are accepted on this endpoint",
                        "fa": "این endpoint فقط لینک سرویس Media را می‌پذیرد",
                    },
                )
            return media_storage_uri(match.group(1))

    raise BadRequestError(
        error_code="invalid_media_source",
        detail="source_uri must be media:<uid> or a Media /f/<uid> URL",
        message={
            "en": "Invalid Media source URI",
            "fa": "آدرس منبع Media نامعتبر است",
        },
    )
