"""Promptic API routes."""

import hashlib
from collections.abc import AsyncIterator

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from fastapi_mongo_base.routes import PaginatedResponse
from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import BaseModel

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from . import services
from .models import PrompticTask
from .schemas import PrompticCreate, PrompticSchema


class PrompticRouter(AbstractTaskUSSORouter):
    """Router for promptic API endpoints."""

    model = PrompticTask
    schema = PrompticSchema

    def __init__(self) -> None:
        """Initialize the router."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/promptic",
            tags=["Promptic"],
        )

    def config_schemas(self, schema: type, **kwargs: object) -> None:
        """Configure schemas using PrompticCreate for create requests."""
        super().config_schemas(schema, create_request_schema=PrompticCreate, **kwargs)

    def config_routes(self, **kwargs: object) -> None:
        """Configure routes for the router."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)

    async def list_items(
        self,
        request: Request,
        prompt_name: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[BaseModel]:
        """List promptic runs for a prompt."""
        return await self._list_items(
            request, offset, limit, prompt_name=prompt_name, user_id=user_id
        )

    async def create_item(
        self,
        request: Request,
        prompt_name: str,
        data: PrompticCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
        stream: bool = False,
    ) -> PrompticTask | StreamingResponse:
        """Create a new promptic run for a prompt."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        services.check_schemas(prompt_name, data)

        data.input_variables.setdefault("language", "Persian")

        item_data = data.model_dump(exclude_none=True)
        item_data.setdefault(
            "idempotency_key",
            hashlib.sha256(
                f"{prompt_name}:{data.model_dump_json()}".encode()
            ).hexdigest(),
        )

        item: PrompticTask = await self.model.create_item({
            **item_data,
            "prompt_name": prompt_name,
            "user_id": data.user_id or user.uid,
            "tenant_id": user.tenant_id,
        })

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


router = PrompticRouter().router
