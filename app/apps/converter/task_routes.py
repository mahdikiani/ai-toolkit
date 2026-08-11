"""
Conversion task HTTP API.

Implemented:
  POST /conversions/from-media

Deferred (security review before enabling):
  POST /conversions/from-upload
  POST /conversions/from-base64
"""

from __future__ import annotations

from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.errors import BadRequestError
from fastapi_mongo_base.routes import PaginatedResponse
from fastapi_mongo_base.schemas import BaseEntitySchema

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from . import registry
from .media_source import normalize_media_source_uri
from .task_models import ConversionTask
from .task_schemas import ConversionTaskSchema, ConversionTaskSchemaCreate


class ConversionTaskRouter(AbstractTaskUSSORouter):
    """Async conversion tasks (from-media first)."""

    model = ConversionTask
    schema = ConversionTaskSchema
    resource = "conversions"

    def __init__(self) -> None:
        """Mount under /conversions."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/conversions",
            tags=["Conversions"],
        )

    def config_schemas(self, schema: type[BaseEntitySchema], **kwargs: object) -> None:
        """Use from-media create schema by default."""
        super().config_schemas(schema, **kwargs)
        self.create_request_schema = ConversionTaskSchemaCreate

    def config_routes(self, **kwargs: object) -> None:
        """Register task routes and the from-media create path."""
        super().config_routes(
            update_route=False,
            create_route=False,
            webhook_route=True,
            **kwargs,
        )
        self.router.add_api_route(
            "/from-media",
            self.create_from_media,
            methods=["POST"],
            response_model=ConversionTaskSchema,
            status_code=201,
        )

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[ConversionTaskSchema]:
        """List conversion tasks."""
        return cast(
            PaginatedResponse[ConversionTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_from_media(
        self,
        request: Request,
        data: ConversionTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> ConversionTask:
        """Create a conversion task from a Media URI and start processing."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        # Fail fast: normalize URI + check registry edge before enqueue.
        normalize_media_source_uri(data.source_uri)
        registry.ensure_builtin_strategies()
        if registry.get_edge(data.source_format, data.target_format) is None:
            raise BadRequestError(
                error_code="unsupported_conversion",
                detail=(
                    f"No conversion edge for "
                    f"{data.source_format.value} → {data.target_format.value}"
                ),
                message={
                    "en": "Unsupported conversion",
                    "fa": "این تبدیل پشتیبانی نمی‌شود",
                },
            )

        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
            "workspace_id": data.workspace_id,
        })
        background_tasks.add_task(item.start_processing)
        return item


router = ConversionTaskRouter().router
