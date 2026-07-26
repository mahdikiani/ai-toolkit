"""Signed token helpers for inbound Soniox transcription webhooks."""

from __future__ import annotations

import hashlib
import hmac
from typing import Self
from urllib.parse import urlencode

from fastapi_mongo_base.core.exceptions import BaseHTTPException

from server.config import Settings


class WebhookAuthError(BaseHTTPException):
    """Raised when a Soniox webhook request fails authentication."""

    def __init__(self, detail: str, *, status_code: int = 401) -> None:
        """Initialize with detail and HTTP status."""
        super().__init__(
            status_code=status_code,
            error_code="webhook_unauthorized",
            detail=detail,
            message={"en": detail},
        )

    @classmethod
    def secret_not_configured(cls) -> Self:
        """Webhook HMAC secret is missing from settings."""
        return cls(
            "SONIOX_WEBHOOK_SECRET is not configured",
            status_code=503,
        )

    @classmethod
    def missing_token(cls) -> Self:
        """Request has no token query/header value."""
        return cls("Missing webhook token")

    @classmethod
    def invalid_token(cls) -> Self:
        """Token does not match the expected HMAC."""
        return cls("Invalid webhook token")


def _secret() -> str | None:
    """Return configured Soniox webhook HMAC secret, if any."""
    return getattr(Settings, "soniox_webhook_secret", None) or None


def build_webhook_token(uid: str) -> str:
    """HMAC token bound to a transcription task uid."""
    secret = _secret()
    if not secret:
        raise WebhookAuthError.secret_not_configured()
    return hmac.new(
        secret.encode("utf-8"),
        uid.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def append_webhook_auth(url: str, uid: str) -> str:
    """Append signed token query param when webhook secret is configured."""
    secret = _secret()
    if not secret:
        return url
    token = build_webhook_token(uid)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({'token': token})}"


def verify_webhook_request(*, uid: str, token: str | None) -> None:
    """Fail-closed verification for inbound Soniox webhooks."""
    secret = _secret()
    if not secret:
        raise WebhookAuthError.secret_not_configured()
    if not token:
        raise WebhookAuthError.missing_token()
    expected = build_webhook_token(uid)
    if not hmac.compare_digest(expected, token):
        raise WebhookAuthError.invalid_token()
