"""Unit tests for ConversionTask from-media orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.errors import BadRequestError
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.artifacts.enums import ArtifactFormat
from apps.converter.media_source import normalize_media_source_uri
from apps.converter.task_services import process_conversion_from_media


class TestNormalizeMediaSourceUri:
    def test_media_prefix(self) -> None:
        assert normalize_media_source_uri("media:abc123") == "media:abc123"

    def test_media_https_path(self) -> None:
        uri = normalize_media_source_uri(
            "https://media.uln.me/api/media/v1/f/fileUid99?signed_url=1"
        )
        assert uri == "media:fileUid99"

    def test_rejects_arbitrary_url(self) -> None:
        with pytest.raises(BadRequestError) as exc:
            normalize_media_source_uri("https://evil.example/f/x")
        assert exc.value.error_code in {"non_media_url", "invalid_media_source"}

    def test_rejects_substring_lookalike_hosts(self) -> None:
        for url in (
            "https://evilmedia.com/f/x",
            "https://media.attacker.test/f/x",
            "https://not-uln.me/f/x",
        ):
            with pytest.raises(BadRequestError) as exc:
                normalize_media_source_uri(url)
            assert exc.value.error_code == "non_media_url"


@pytest.mark.asyncio
async def test_process_conversion_from_media_happy_path() -> None:
    task = MagicMock()
    task.uid = "task-1"
    task.user_id = "u1"
    task.tenant_id = "t1"
    task.workspace_id = None
    task.source_uri = "media:src1"
    task.source_format = ArtifactFormat.markdown
    task.target_format = ArtifactFormat.pdf
    task.title = "Demo"
    task.original_name = "demo.md"
    task.save_status = AsyncMock()
    task.update_and_emit = AsyncMock()

    source_art = SimpleNamespace(uid="art-src", storage_uri="media:src1")
    derived = SimpleNamespace(uid="art-out", storage_uri="media:out1")

    with (
        patch(
            "apps.converter.task_services.download_by_storage_uri",
            AsyncMock(return_value=b"# hi\n"),
        ),
        patch(
            "apps.converter.task_services.create_artifact_from_bytes",
            AsyncMock(return_value=source_art),
        ) as create_art,
        patch(
            "apps.converter.task_services.convert_artifact",
            AsyncMock(return_value=derived),
        ) as convert,
        patch("apps.converter.task_services.registry.ensure_builtin_strategies"),
        patch(
            "apps.converter.task_services.registry.get_edge",
            return_value=SimpleNamespace(name="markdown_pdf"),
        ),
    ):
        result = await process_conversion_from_media(task)

    assert result.result_artifact_id == "art-out"
    assert result.source_artifact_id == "art-src"
    assert result.result_storage_uri == "media:out1"
    create_art.assert_awaited_once()
    convert.assert_awaited_once()
    task.update_and_emit.assert_awaited()
    kwargs = task.update_and_emit.await_args.kwargs
    assert kwargs["task_status"] == TaskStatusEnum.completed


@pytest.mark.asyncio
async def test_process_conversion_unsupported_edge_errors() -> None:
    task = MagicMock()
    task.uid = "task-2"
    task.source_uri = "media:x"
    task.source_format = ArtifactFormat.html
    task.target_format = ArtifactFormat.pdf
    task.save_status = AsyncMock()
    task.update_and_emit = AsyncMock()

    with (
        patch("apps.converter.task_services.registry.ensure_builtin_strategies"),
        patch("apps.converter.task_services.registry.get_edge", return_value=None),
    ):
        await process_conversion_from_media(task)

    kwargs = task.update_and_emit.await_args.kwargs
    assert kwargs["task_status"] == TaskStatusEnum.error
    assert "unsupported_conversion" in kwargs["task_report"]
