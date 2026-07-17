"""Unit tests for shared authorization helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from utils.auth import authorize_create_on_behalf, is_service_request


class DummyRequest:
    """Provide a minimal request object with headers."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        """Initialize headers for a test request."""
        self.headers = headers or {}


class DummyData:
    """Provide minimal serializable task creation data."""

    def __init__(self, user_id: str | None = None) -> None:
        """Initialize optional target user data."""
        self.user_id = user_id

    def model_dump(self) -> dict[str, str | None]:
        """Return the task data in model-dump form."""
        return {"user_id": self.user_id}


def test_is_service_request_requires_api_key_header() -> None:
    """Verify only API-key requests are recognized as service requests."""
    assert is_service_request(DummyRequest({"x-api-key": "uak-test"})) is True
    assert is_service_request(DummyRequest()) is False


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_allows_service_request() -> None:
    """Verify service requests can create tasks for another user."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="service-user")
    data = DummyData(user_id="end-user")

    await authorize_create_on_behalf(
        router,
        DummyRequest({"x-api-key": "uak-test"}),
        user,
        data,
    )

    router.authorize.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_keeps_jwt_authorization() -> None:
    """Verify JWT requests retain create-on-behalf authorization checks."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user")
    data = DummyData(user_id="other-user")

    await authorize_create_on_behalf(router, DummyRequest(), user, data)

    router.authorize.assert_awaited_once_with(
        action="create",
        user=user,
        filter_data={"user_id": "other-user"},
    )


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_defaults_to_authenticated_user() -> None:
    """Verify requests without a target user use the authenticated user."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user")
    data = DummyData()

    await authorize_create_on_behalf(router, DummyRequest(), user, data)

    assert data.user_id == "auth-user"
    router.authorize.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_requires_self_auth_when_flagged() -> None:
    """Verify privileged create flows authorize JWT users even for themselves."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user")
    data = DummyData()

    await authorize_create_on_behalf(
        router,
        DummyRequest(),
        user,
        data,
        require_create_authorization=True,
    )

    assert data.user_id == "auth-user"
    router.authorize.assert_awaited_once_with(
        action="create",
        user=user,
        filter_data={"user_id": "auth-user"},
    )


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_keeps_service_bypass_when_flagged() -> None:
    """Verify service requests bypass create authorization even when restricted."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="service-user")
    data = DummyData(user_id="end-user")

    await authorize_create_on_behalf(
        router,
        DummyRequest({"x-api-key": "uak-test"}),
        user,
        data,
        require_create_authorization=True,
    )

    router.authorize.assert_not_awaited()
