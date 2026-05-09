"""Invocation API routes."""

from collections.abc import AsyncIterator

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from fastapi_mongo_base.routes import AbstractTaskRouter, PaginatedResponse
from fastapi_mongo_base.tasks import TaskStatusEnum
from fastapi_mongo_base.utils import usso_routes
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings

from . import services
from .models import ExecutionTask
from .schemas import ExecutionTaskCreate, ExecutionTaskSchema


class ExecutionRouter(AbstractTaskRouter, usso_routes.AbstractTenantUSSORouter):
    """Router for invocation API endpoints."""

    model = ExecutionTask
    schema = ExecutionTaskSchema

    def __init__(self) -> None:
        """Initialize the router."""
        super().__init__(
            user_dependency=USSOAuthentication(),
            draftable=False,
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure routes for the router."""
        super().config_routes(
            prefix="executions", update_route=False, webhook_route=False, **kwargs
        )

    async def list_items(
        self,
        request: Request,
        prompt_name: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[ExecutionTask]:
        """List invocations for a prompt."""
        return await self._list_items(
            request, offset, limit, prompt_name=prompt_name, user_id=user_id
        )

    async def create_item(
        self,
        request: Request,
        prompt_name: str,
        data: ExecutionTaskCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
        stream: bool = False,
    ) -> ExecutionTask | StreamingResponse:
        """Create a new invocation for a prompt."""
        user = await self.get_user(request)

        services.check_schemas(prompt_name, data)

        data.input_variables.setdefault("language", "Persian")

        item: ExecutionTask = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "prompt_name": prompt_name,
            "user_id": user.uid,
            "tenant_id": user.tenant_id,
        })

        # For streaming, always process synchronously and return stream
        item.task_status = TaskStatusEnum.processing
        await item.save()

        if stream:

            async def generate_stream() -> AsyncIterator[str]:
                import json

                yield item.model_dump_json(ensure_ascii=False) + "\n"
                async for chunk in services.invoke_stream(item):
                    yield json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "finish"}, ensure_ascii=False) + "\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        if not blocking:
            background_tasks.add_task(item.start_processing)
            return item

        return await item.start_processing()


router = ExecutionRouter().router
