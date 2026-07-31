"""Web search task schemas."""

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.workspace import WorkspaceScopedSchema


class WebSearchTaskSchemaCreate(TaskCreateFieldsMixin):
    """Fields accepted when creating a web search task."""

    query: str = Field(..., min_length=1, description="Search query")
    num_results: int = Field(default=10, ge=1, le=50)
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None


class WebSearchTaskSchema(
    TenantUserEntitySchema,
    TaskMixin,
    WorkspaceScopedSchema,
    WebSearchTaskSchemaCreate,
):
    """Complete web search task schema with lifecycle tracking."""

    result: dict | None = None
    search_provider: str = "exa"
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
