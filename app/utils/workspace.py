"""Shared workspace-scoping field for task/resource schemas."""

from pydantic import BaseModel


class WorkspaceScopedSchema(BaseModel):
    """
    Optional workspace_id, stamped server-side from the requester's JWT.

    When set, this record's cost is billed against the workspace's shared
    quota (see utils.billing.finance) instead of the creating user's
    personal balance.
    """

    workspace_id: str | None = None
