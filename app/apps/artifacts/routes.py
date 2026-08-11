"""USSO-authenticated Artifact CRUD (create + retrieve)."""

from fastapi import Request
from fastapi_mongo_base.errors import BadRequestError
from fastapi_mongo_base.schemas import BaseEntitySchema
from fastapi_mongo_base.utils import usso_routes
from usso.integrations.fastapi import USSOAuthentication

from .enums import ArtifactFormat
from .models import Artifact
from .schemas import ArtifactCreate, ArtifactSchema
from .services import create_artifact_from_text


class ArtifactRouter(usso_routes.AbstractTenantUSSORouter):
    """Create and retrieve durable Artifacts."""

    model = Artifact
    schema = ArtifactSchema
    resource = "artifacts"

    def __init__(self) -> None:
        """Initialize Artifact router under /artifacts."""
        super().__init__(
            user_dependency=USSOAuthentication(),
            prefix="/artifacts",
            tags=["Artifacts"],
        )

    def config_schemas(self, schema: type[BaseEntitySchema], **kwargs: object) -> None:
        """Use ArtifactCreate for POST bodies."""
        super().config_schemas(schema, **kwargs)
        self.create_request_schema = ArtifactCreate

    def config_routes(self, **kwargs: object) -> None:
        """Expose create + retrieve only in Phase 1."""
        super().config_routes(
            list_route=False,
            retrieve_route=True,
            create_route=True,
            update_route=False,
            delete_route=False,
            statistics_route=False,
            mine_route=False,
            **kwargs,
        )

    async def create_item(self, request: Request, data: ArtifactCreate) -> Artifact:
        """Upload content to Media and persist Artifact metadata."""
        user = await self.get_user(request)
        payload = data.model_dump(exclude_none=True)
        await self.authorize(action="create", user=user, filter_data=payload)

        if data.content is None or data.content == "":
            raise BadRequestError(
                error_code="missing_content",
                detail="content is required to create an Artifact",
                message={
                    "en": "content is required",
                    "fa": "محتوا الزامی است",
                },
            )
        if data.format not in {ArtifactFormat.markdown, ArtifactFormat.html}:
            raise BadRequestError(
                error_code="unsupported_create_format",
                detail=(
                    "JSON create currently supports text formats "
                    "(markdown, html); use bytes ingest for binary formats"
                ),
            )

        return await create_artifact_from_text(
            text=data.content,
            artifact_format=data.format,
            user_id=user.uid,
            tenant_id=user.tenant_id,
            workspace_id=data.workspace_id or getattr(user, "workspace_id", None),
            artifact_type=data.type,
            title=data.title,
            original_name=data.original_name,
            language=data.language,
            source=data.source,
            parent_artifact_id=data.parent_artifact_id,
            meta_data=data.meta_data,
        )


router = ArtifactRouter().router
