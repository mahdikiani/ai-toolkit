"""WebSocket authentication helpers using the USSO FastAPI integration."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import WebSocket
from starlette.datastructures import MutableHeaders
from usso import UserData
from usso.exceptions import USSOException

from utils.usso import get_usso


@dataclass(frozen=True, slots=True)
class _WebSocketAuthView:
    """Duck-typed view so USSO can read headers/cookies (with query fallback)."""

    headers: MutableHeaders
    cookies: dict[str, str]


def websocket_auth_view(websocket: WebSocket) -> WebSocket | _WebSocketAuthView:
    """
    Return a request-like object for USSO WebSocket auth.

    Browser clients often cannot set ``Authorization`` / ``x-api-key`` on the
    handshake; fall back to query ``access_token`` / ``api_key``.
    """
    access_token = websocket.query_params.get("access_token")
    api_key = websocket.query_params.get("api_key")
    if not access_token and not api_key:
        return websocket

    headers = MutableHeaders(websocket.headers)
    if access_token and not headers.get("Authorization"):
        headers["Authorization"] = f"Bearer {access_token}"
    if api_key and not headers.get("x-api-key"):
        headers["x-api-key"] = api_key
    return _WebSocketAuthView(headers, dict(websocket.cookies))


def authenticate_websocket(websocket: WebSocket) -> UserData:
    """
    Authenticate a WebSocket via USSO JWT Bearer or ``x-api-key``.

    Raises:
        USSOException: When credentials are missing or invalid.
    """
    usso = get_usso(raise_exception=True)
    user = usso.jwt_access_security_ws(websocket_auth_view(websocket))
    if user is None:
        raise USSOException(
            status_code=401,
            error_code="unauthorized",
            detail="No token provided",
        )
    return user
