"""Unit tests for converter registry and Artifact conversion services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.errors import BadRequestError
from usso import UserData

from apps.artifacts.enums import ArtifactFormat
from apps.converter import registry
from apps.converter.routes import convert, list_conversion_formats
from apps.converter.schemas import ConvertRequest
from apps.converter.services import convert_artifact, render_markdown_to_format
from apps.converter.strategies.markdown_docx import markdown_text_to_docx
from apps.converter.strategies.markdown_pdf import markdown_text_to_pdf


@pytest.mark.unit
class TestConversionRegistry:
    def test_builtin_edges_include_markdown_docx_and_pdf(self) -> None:
        registry.ensure_builtin_strategies()
        edges = {(e.source_format, e.target_format) for e in registry.list_edges()}
        assert (ArtifactFormat.markdown, ArtifactFormat.docx) in edges
        assert (ArtifactFormat.markdown, ArtifactFormat.pdf) in edges

    def test_unsupported_edge_returns_none(self) -> None:
        registry.ensure_builtin_strategies()
        assert registry.get_edge(ArtifactFormat.pdf, ArtifactFormat.docx) is None

    def test_clear_and_ensure_restores_builtins(self) -> None:
        registry.clear_registry()
        assert registry.list_edges() == []
        registry.ensure_builtin_strategies()
        assert len(registry.list_edges()) >= 2


@pytest.mark.unit
class TestMarkdownStrategies:
    def test_markdown_to_docx_produces_ooxml_zip(self) -> None:
        body = markdown_text_to_docx("# عنوان\n\nمتن **ضخیم**", title="گزارش")
        assert body[:2] == b"PK"  # DOCX is a zip

    def test_markdown_to_pdf_produces_pdf_header(self) -> None:
        body = markdown_text_to_pdf("# Report\n\nBody", title="Report")
        assert body.startswith(b"%PDF-")

    def test_render_helper_routes_through_registry(self) -> None:
        docx = render_markdown_to_format(
            "hello", target_format=ArtifactFormat.docx, title="t"
        )
        pdf = render_markdown_to_format(
            "hello", target_format=ArtifactFormat.pdf, title="t"
        )
        assert docx[:2] == b"PK"
        assert pdf.startswith(b"%PDF-")

    def test_render_helper_rejects_unsupported_target(self) -> None:
        with pytest.raises(BadRequestError) as exc_info:
            render_markdown_to_format("x", target_format=ArtifactFormat.html)
        assert exc_info.value.error_code == "unsupported_conversion"


@pytest.mark.unit
class TestConvertArtifactService:
    async def test_converts_and_creates_child_artifact(self) -> None:
        source = SimpleNamespace(
            uid="src-1",
            format=ArtifactFormat.markdown,
            title="Notes",
            original_name="notes.md",
            language="fa",
            type="document",
            workspace_id="ws-1",
            storage_uri="media:src",
        )
        child = MagicMock(uid="child-1", format=ArtifactFormat.docx)

        with (
            patch(
                "apps.converter.services.get_artifact_for_user",
                AsyncMock(return_value=source),
            ),
            patch(
                "apps.converter.services.read_artifact_bytes",
                AsyncMock(return_value=b"# hi"),
            ),
            patch(
                "apps.converter.services.create_artifact_from_bytes",
                AsyncMock(return_value=child),
            ) as create_child,
            patch(
                "apps.converter.registry.get_edge",
                return_value=SimpleNamespace(
                    name="markdown_docx",
                    strategy=lambda data, title="": b"PK-fake-docx",
                ),
            ),
        ):
            result = await convert_artifact(
                artifact_id="src-1",
                target_format=ArtifactFormat.docx,
                user_id="u1",
                tenant_id="t1",
            )

        assert result is child
        kwargs = create_child.await_args.kwargs
        assert kwargs["artifact_format"] == ArtifactFormat.docx
        assert kwargs["source"] == "converter"
        assert kwargs["parent_artifact_id"] == "src-1"
        assert kwargs["data"] == b"PK-fake-docx"
        assert kwargs["workspace_id"] == "ws-1"

    async def test_unsupported_edge_raises_400(self) -> None:
        source = SimpleNamespace(
            uid="src-1",
            format=ArtifactFormat.pdf,
            title=None,
            original_name=None,
            language=None,
            type="document",
            workspace_id=None,
        )
        with (
            patch(
                "apps.converter.services.get_artifact_for_user",
                AsyncMock(return_value=source),
            ),
            patch(
                "apps.converter.registry.get_edge",
                return_value=None,
            ), pytest.raises(BadRequestError) as exc_info
        ):
            await convert_artifact(
                artifact_id="src-1",
                target_format=ArtifactFormat.markdown,
                user_id="u1",
                tenant_id="t1",
            )
        assert exc_info.value.error_code == "unsupported_conversion"

    async def test_same_format_raises_400(self) -> None:
        source = SimpleNamespace(
            uid="src-1",
            format=ArtifactFormat.markdown,
            title=None,
            original_name=None,
            language=None,
            type="document",
            workspace_id=None,
        )
        with patch(
            "apps.converter.services.get_artifact_for_user",
            AsyncMock(return_value=source),
        ), pytest.raises(BadRequestError) as exc_info:
            await convert_artifact(
                artifact_id="src-1",
                target_format=ArtifactFormat.markdown,
                user_id="u1",
                tenant_id="t1",
            )
        assert exc_info.value.error_code == "same_format"


@pytest.mark.unit
class TestConverterRoutes:
    async def test_convert_route_delegates(self) -> None:
        user = UserData(sub="u1", tenant_id="t1")
        child = MagicMock(uid="out-1")
        data = ConvertRequest(artifact_id="a1", target_format=ArtifactFormat.pdf)
        with patch(
            "apps.converter.routes.convert_artifact",
            AsyncMock(return_value=child),
        ) as convert_svc:
            result = await convert(data, user=user)
        assert result is child
        assert convert_svc.await_args.kwargs["artifact_id"] == "a1"
        assert convert_svc.await_args.kwargs["target_format"] == ArtifactFormat.pdf

    async def test_list_formats_returns_registered_edges(self) -> None:
        user = UserData(sub="u1", tenant_id="t1")
        edges = await list_conversion_formats(user=user)
        pairs = {(e.source_format, e.target_format) for e in edges}
        assert (ArtifactFormat.markdown, ArtifactFormat.docx) in pairs
        assert (ArtifactFormat.markdown, ArtifactFormat.pdf) in pairs
