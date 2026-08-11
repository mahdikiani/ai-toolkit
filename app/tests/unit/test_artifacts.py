"""Unit tests for Artifact SoR services (Media mocked)."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.errors import BadRequestError, NotFoundError

from apps.artifacts.enums import ArtifactFormat
from apps.artifacts.routes import ArtifactRouter
from apps.artifacts.schemas import ArtifactCreate
from apps.artifacts.services import (
    create_artifact_from_bytes,
    create_artifact_from_text,
    get_artifact_for_user,
    read_artifact_bytes,
)
from utils.integrations.media import DurableUploadResult


def _uploaded(uid: str = "file-abc") -> DurableUploadResult:
    return DurableUploadResult(
        file_uid=uid,
        storage_uri=f"media:{uid}",
        signed_url=None,
    )


@pytest.mark.unit
class TestCreateArtifactFromText:
    async def test_uploads_utf8_and_persists_durable_uri(self) -> None:
        created = MagicMock(uid="art-1")
        with (
            patch(
                "apps.artifacts.services.upload_file_durable",
                AsyncMock(return_value=_uploaded()),
            ) as upload,
            patch(
                "apps.artifacts.services.Artifact.create_item",
                AsyncMock(return_value=created),
            ) as create_item,
        ):
            result = await create_artifact_from_text(
                text="# Hello\n\nbody",
                user_id="u1",
                tenant_id="t1",
                title="Greeting",
                source="upload",
            )

        assert result is created
        upload.assert_awaited_once()
        call = upload.await_args
        assert isinstance(call.args[0], BytesIO)
        assert call.kwargs["filename"] == "Greeting.md"
        assert call.kwargs["content_type"] == "text/markdown"
        payload = create_item.await_args.args[0]
        assert payload["storage_uri"] == "media:file-abc"
        assert payload["format"] == ArtifactFormat.markdown
        assert payload["user_id"] == "u1"
        assert payload["tenant_id"] == "t1"
        assert payload["source"] == "upload"

    async def test_rejects_binary_format_from_text(self) -> None:
        with pytest.raises(BadRequestError) as exc_info:
            await create_artifact_from_text(
                text="nope",
                user_id="u1",
                tenant_id="t1",
                artifact_format=ArtifactFormat.pdf,
            )
        assert exc_info.value.error_code == "invalid_text_format"

    async def test_original_name_wins_over_title_for_filename(self) -> None:
        created = MagicMock(uid="art-1")
        with (
            patch(
                "apps.artifacts.services.upload_file_durable",
                AsyncMock(return_value=_uploaded()),
            ) as upload,
            patch(
                "apps.artifacts.services.Artifact.create_item",
                AsyncMock(return_value=created),
            ),
        ):
            await create_artifact_from_text(
                text="x",
                user_id="u1",
                tenant_id="t1",
                title="Ignored",
                original_name="custom.md",
            )
        assert upload.await_args.kwargs["filename"] == "custom.md"


@pytest.mark.unit
class TestCreateArtifactFromBytes:
    async def test_sets_parent_and_converter_source(self) -> None:
        created = MagicMock(uid="art-2")
        with (
            patch(
                "apps.artifacts.services.upload_file_durable",
                AsyncMock(return_value=_uploaded("out-1")),
            ),
            patch(
                "apps.artifacts.services.Artifact.create_item",
                AsyncMock(return_value=created),
            ) as create_item,
        ):
            await create_artifact_from_bytes(
                data=b"%PDF-1.4",
                filename="doc.pdf",
                content_type="application/pdf",
                artifact_format=ArtifactFormat.pdf,
                user_id="u1",
                tenant_id="t1",
                source="converter",
                parent_artifact_id="parent-1",
            )

        payload = create_item.await_args.args[0]
        assert payload["parent_artifact_id"] == "parent-1"
        assert payload["source"] == "converter"
        assert payload["storage_uri"] == "media:out-1"


@pytest.mark.unit
class TestReadAndGetArtifact:
    async def test_get_artifact_raises_when_missing(self) -> None:
        with patch(
            "apps.artifacts.services.Artifact.get_item",
            AsyncMock(return_value=None),
        ), pytest.raises(NotFoundError) as exc_info:
            await get_artifact_for_user(
                artifact_id="missing",
                user_id="u1",
                tenant_id="t1",
            )
        assert exc_info.value.error_code == "artifact_not_found"

    async def test_read_artifact_bytes_uses_storage_uri(self) -> None:
        artifact = SimpleNamespace(storage_uri="media:file-xyz")
        with patch(
            "apps.artifacts.services.download_by_storage_uri",
            AsyncMock(return_value=b"markdown body"),
        ) as download:
            data = await read_artifact_bytes(artifact)

        assert data == b"markdown body"
        download.assert_awaited_once_with("media:file-xyz")


@pytest.mark.unit
class TestArtifactRouterCreate:
    async def test_create_item_requires_content(self) -> None:
        router = ArtifactRouter.__new__(ArtifactRouter)
        user = SimpleNamespace(uid="u1", tenant_id="t1", workspace_id=None)
        router.get_user = AsyncMock(return_value=user)
        router.authorize = AsyncMock()
        data = ArtifactCreate(format=ArtifactFormat.markdown, content=None)

        with pytest.raises(BadRequestError) as exc_info:
            await router.create_item(MagicMock(), data)
        assert exc_info.value.error_code == "missing_content"

    async def test_create_item_delegates_to_service(self) -> None:
        router = ArtifactRouter.__new__(ArtifactRouter)
        user = SimpleNamespace(uid="u1", tenant_id="t1", workspace_id="ws1")
        router.get_user = AsyncMock(return_value=user)
        router.authorize = AsyncMock()
        created = MagicMock(uid="art-9")
        data = ArtifactCreate(
            format=ArtifactFormat.markdown,
            content="# hi",
            title="T",
        )

        with patch(
            "apps.artifacts.routes.create_artifact_from_text",
            AsyncMock(return_value=created),
        ) as create:
            result = await router.create_item(MagicMock(), data)

        assert result is created
        assert create.await_args.kwargs["artifact_format"] == ArtifactFormat.markdown
        assert create.await_args.kwargs["workspace_id"] == "ws1"
