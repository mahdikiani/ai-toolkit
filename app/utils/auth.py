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


def resolve_on_behalf_ids(
    request: object,
    user_id: str,
    workspace_id: str | None,
    *,
    override_user_id: str | None,
    override_workspace_id: str | None,
) -> tuple[str, str | None]:
    """
    Let a service (API-key) caller act on behalf of a specific end user.

    Without this, every request through a shared API key (e.g. mirza-bot,
    acting for many Telegram users through one key) bills and checks quota
    against the key's own identity instead of the actual end user, no
    matter who's asking. JWT end-users can never override this: always
    billed/limited as themselves, never a client-claimed identity.
    """
    if not is_service_request(request):
        return user_id, workspace_id
    return override_user_id or user_id, override_workspace_id or workspace_id


def _resolve_workspace_id_on_behalf(
    request: object, user: object, data: object
) -> None:
    """
    Resolve workspace_id on a create payload that declares the field.

    Mirrors the user_id on-behalf pattern: service (API-key) requests may
    set an explicit workspace_id in the payload to attribute the created
    resource to a specific workspace (e.g. mirza-bot billing a Telegram
    user's own workspace instead of the shared service account's). JWT
    end-users can never claim a workspace other than their own -- always
    pinned to the requester's workspace_id, so a browser client can't
    spoof membership in someone else's workspace.

    No-op for payloads that don't declare a workspace_id field.
    """
    if not hasattr(data, "workspace_id"):
        return
    user_workspace_id = getattr(user, "workspace_id", None)
    if is_service_request(request):
        data.workspace_id = getattr(data, "workspace_id", None) or user_workspace_id
    else:
        data.workspace_id = user_workspace_id


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
    _resolve_workspace_id_on_behalf(request, user, data)
    if is_service_request(request):
        return
    if not require_create_authorization and data.user_id == user.user_id:
        return
    await router.authorize(action="create", user=user, filter_data=data.model_dump())
