"""Web search API routes for search task management."""

from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import WebSearchTask
from .schemas import WebSearchTaskSchema, WebSearchTaskSchemaCreate


class WebSearchRouter(AbstractTaskUSSORouter):
    """Router for web search task endpoints."""

    model = WebSearchTask
    schema = WebSearchTaskSchema

    def __init__(self) -> None:
        """Configure the /web-search router prefix and tags."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/web-search",
            tags=["Web Search"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Register routes, plus a custom /{uid}/result endpoint."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)
        self.router.add_api_route(
            "/{uid}/result",
            self.get_result,
            methods=["GET"],
        )

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[WebSearchTaskSchema]:
        """List web search tasks, paginated."""
        return cast(
            PaginatedResponse[WebSearchTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: WebSearchTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> WebSearchTask:
        """Create a web search task and start processing in the background."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
            "workspace_id": data.workspace_id,
        })
        background_tasks.add_task(item.start_processing)
        return item

    async def get_result(self, request: Request, uid: str) -> dict:
        """Return the search result once the task has completed."""
        task: WebSearchTask = await self.retrieve_item(request, uid)
        if task.task_status != "completed":
            return {"status": task.task_status, "message": "Search not yet complete"}
        return {"status": "completed", "query": task.query, "result": task.result}


router = WebSearchRouter().router
