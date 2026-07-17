"""Shared authorization helpers for user-owned resource creation."""

from collections.abc import Awaitable, Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class RequestWithHeaders(Protocol):
    """Minimal request interface needed to identify API-key requests."""

    headers: Mapping[str, str]


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Minimal authenticated-user interface needed for object ownership."""

    user_id: str


@runtime_checkable
class UserOwnedCreateData(Protocol):
    """Minimal object payload interface used for delegated creation."""

    user_id: str | None

    def model_dump(self) -> Mapping[str, object]:
        """Serialize the object payload for authorization."""


@runtime_checkable
class AuthorizingRouter(Protocol):
    """Minimal router authorization interface used for delegated creation."""

    def authorize(self, **kwargs: object) -> Awaitable[object]:
        """Authorize an action using the supplied request context."""


def is_service_request(request: object) -> bool:
    """Return True when the already-authenticated request used an API key."""
    if not isinstance(request, RequestWithHeaders):
        raise TypeError
    return bool(request.headers.get("x-api-key"))


async def authorize_create_on_behalf(
    router: object,
    request: object,
    user: object,
    data: object,
    *,
    require_create_authorization: bool = False,
) -> None:
    """
    Allow API-key service calls to create objects for an end user.

    JWT-authenticated users still need normal owner/scope authorization when
    they submit an object for a different user_id.

    When `require_create_authorization` is True, JWT users must pass create
    authorization even for their own user_id (e.g. billing or usage metering).
    Service requests authenticated via API key are still allowed through.
    """
    if not isinstance(router, AuthorizingRouter):
        raise TypeError
    if not isinstance(user, AuthenticatedUser):
        raise TypeError
    if not isinstance(data, UserOwnedCreateData):
        raise TypeError
    data.user_id = data.user_id or user.user_id
    if is_service_request(request):
        return
    if not require_create_authorization and data.user_id == user.user_id:
        return
    await router.authorize(action="create", user=user, filter_data=data.model_dump())
