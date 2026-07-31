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


class DummyWorkspaceData(DummyData):
    """
    DummyData variant whose schema also declares workspace_id.

    Mirrors the ~11 task apps' CreateSchema classes, which mix in
    WorkspaceScopedSchema -- unlike plain DummyData, which represents a
    payload with no workspace_id field at all.
    """

    def __init__(
        self, user_id: str | None = None, workspace_id: str | None = None
    ) -> None:
        """Initialize optional target user and workspace data."""
        super().__init__(user_id=user_id)
        self.workspace_id = workspace_id

    def model_dump(self) -> dict[str, str | None]:
        """Return the task data in model-dump form."""
        return {**super().model_dump(), "workspace_id": self.workspace_id}


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


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_ignores_data_without_workspace_field() -> (
    None
):
    """Verify payloads with no workspace_id field are left untouched."""
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user", workspace_id="user-ws")
    data = DummyData(user_id="auth-user")

    await authorize_create_on_behalf(router, DummyRequest(), user, data)

    assert not hasattr(data, "workspace_id")


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_service_request_defaults_workspace() -> None:
    """
    Verify a service request with no explicit workspace_id inherits the
    calling principal's own workspace_id.
    """
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="service-user", workspace_id="svc-ws")
    data = DummyWorkspaceData(user_id="end-user", workspace_id=None)

    await authorize_create_on_behalf(
        router, DummyRequest({"x-api-key": "uak-test"}), user, data
    )

    assert data.workspace_id == "svc-ws"


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_service_request_keeps_explicit_workspace() -> (
    None
):
    """
    Verify a service request's explicit workspace_id is honored, not
    overridden by the calling principal's own workspace -- this is the
    mechanism mirza-bot uses to bill a Telegram user's own workspace
    instead of the shared service account's.
    """
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="service-user", workspace_id="svc-ws")
    data = DummyWorkspaceData(user_id="end-user", workspace_id="telegram-user-ws")

    await authorize_create_on_behalf(
        router, DummyRequest({"x-api-key": "uak-test"}), user, data
    )

    assert data.workspace_id == "telegram-user-ws"


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_jwt_user_cannot_spoof_workspace() -> None:
    """
    Verify a JWT end-user can never claim a workspace other than their
    own, even if their payload tries to set a different workspace_id --
    prevents a browser client from spoofing membership in someone else's
    workspace.
    """
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user", workspace_id="real-ws")
    data = DummyWorkspaceData(user_id="auth-user", workspace_id="attacker-ws")

    await authorize_create_on_behalf(router, DummyRequest(), user, data)

    assert data.workspace_id == "real-ws"


@pytest.mark.asyncio
async def test_authorize_create_on_behalf_jwt_user_gets_own_workspace_when_unset() -> (
    None
):
    """
    Verify a JWT end-user with no workspace_id in their payload gets
    their own current workspace_id stamped on (or None, if they aren't
    in one).
    """
    router = SimpleNamespace(authorize=AsyncMock())
    user = SimpleNamespace(user_id="auth-user", workspace_id="real-ws")
    data = DummyWorkspaceData(user_id="auth-user", workspace_id=None)

    await authorize_create_on_behalf(router, DummyRequest(), user, data)

    assert data.workspace_id == "real-ws"
