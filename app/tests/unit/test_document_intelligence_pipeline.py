"""
Unit tests for the Document Intelligence orchestrator (full pipeline wiring).

Layout detection (PaddleOCR models) and the VLM HTTP call are mocked out —
this exercises the wiring between Loader -> Layout -> Elements -> Reading
Order -> AST -> Renderers -> Assets without needing GPU/network access.
"""

from io import BytesIO
from unittest.mock import patch

import pytest
from docx import Document as WordDocument
from PIL import Image

from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType
from apps.ocr.document_intelligence.pipeline import (
    DocumentIntelligencePipeline,
    PipelineResult,
    summarize_stats,
)
from apps.ocr.document_intelligence.qa import run_docx_qa


def _fake_detect_page(_self, image: Image.Image, page) -> list[LayoutElement]:
    w, _h = image.size

    def elem(
        suffix: str, elem_type: LayoutType, box: tuple[int, int, int, int]
    ) -> LayoutElement:
        return LayoutElement(
            id=f"{page.id}_{suffix}",
            page_id=page.id,
            page_number=page.page_number,
            type=elem_type,
            bbox=box,
            padded_bbox=box,
            confidence=0.9,
        )

    return [
        elem("title", LayoutType.title, (10, 10, w - 10, 60)),
        elem("para", LayoutType.paragraph, (10, 80, w - 10, 200)),
        elem("table", LayoutType.table, (10, 220, w - 10, 320)),
        elem("formula", LayoutType.formula, (10, 340, w - 10, 380)),
        elem("figure", LayoutType.figure, (10, 400, w - 10, 500)),
    ]


async def _fake_vlm_call(
    _self,
    _crop,
    system_prompt: str,
    _user_prompt: str,
    response_format=None,
    max_tokens: int = 1024,
) -> str:
    _self._last_tokens = 7
    if "table extraction engine" in system_prompt:
        return "<table><tr><td>A</td><td>B</td></tr></table>"
    if "LaTeX source of the formula" in system_prompt:
        return r"\frac{x^2}{y}"
    if "Respond in exactly two lines" in system_prompt:
        return "caption: a test figure\ndescription: shows a test"
    return "سلام دنیا؛ این یک متن آزمایشی است."


def _blank_image(width: int = 600, height: int = 800) -> BytesIO:
    img = Image.new("RGB", (width, height), "white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.mark.document_intelligence
class TestDocumentIntelligencePipeline:
    """End-to-end wiring test with layout/VLM mocked."""

    async def _run(self, mode: str = "semantic") -> PipelineResult:
        with (
            patch(
                "apps.ocr.document_intelligence.layout.LayoutDetector.detect_page",
                _fake_detect_page,
            ),
            patch(
                "apps.ocr.document_intelligence.elements.ElementProcessor._vlm_call",
                _fake_vlm_call,
            ),
        ):
            pipeline = DocumentIntelligencePipeline(dpi=100)
            try:
                return await pipeline.process(_blank_image(), "test.png", mode=mode)
            finally:
                pipeline.cleanup()

    async def test_produces_markdown_with_all_element_types(self) -> None:
        result = await self._run()

        assert "# " in result.markdown  # title
        assert "| A | B |" in result.markdown  # table
        assert "$$" in result.markdown  # formula
        assert "assets/figure_" in result.markdown  # figure asset link
        assert "سلام دنیا" in result.markdown  # OCR'd paragraph text

    async def test_default_mode_produces_semantic_docx_with_real_table_and_omath(
        self,
    ) -> None:
        """
        Default mode ("semantic") must go through the flow-based
        renderer (docx.py): the table is a real top-level body table
        (doc.tables sees it directly), not boxed inside w:txbxContent."""
        result = await self._run()

        doc = WordDocument(BytesIO(result.docx_bytes))
        assert "w:txbxContent" not in doc.element.xml
        assert len(doc.tables) == 1
        assert "oMath" in doc.element.xml

    async def test_visual_mode_produces_absolute_layout_docx(self) -> None:
        """
        mode="visual" is the optional, non-default absolute-layout
        renderer — tables there are boxed per-element, not top-level."""
        result = await self._run(mode="visual")

        doc = WordDocument(BytesIO(result.docx_bytes))
        assert "w:txbxContent" in doc.element.xml
        assert "<w:tbl>" in doc.element.xml
        assert "oMath" in doc.element.xml

    async def test_saves_visual_element_as_asset(self) -> None:
        result = await self._run()

        assert len(result.assets) == 1
        assert result.assets[0].type == "figure"

    async def test_stats_cover_every_page_and_element(self) -> None:
        result = await self._run()

        assert len(result.stats.pages) == 1
        page_stats = result.stats.pages[0]
        assert page_stats.layout_time >= 0
        assert page_stats.vlm_time >= 0
        assert len(page_stats.elements) == 5

        summary = summarize_stats(result.stats)
        assert summary["pages"][0]["element_count"] == 5
        assert "elements" not in summary["pages"][0]  # debug detail hidden by default

        summary_debug = summarize_stats(result.stats, include_elements=True)
        assert "elements" in summary_debug["pages"][0]

    async def test_writes_local_output_files(self) -> None:
        result = await self._run()

        assert (result.output_dir / "document.md").exists()
        assert (result.output_dir / "document.docx").exists()
        assert any((result.output_dir / "assets").iterdir())

    async def test_default_mode_output_passes_the_qa_gate(self) -> None:
        """
        CI gate: the default (semantic) pipeline output must clear every
        QA check — no text boxes, nothing dropped, real tables/OMML,
        correct RTL, header/footer handled correctly."""
        result = await self._run()

        report = run_docx_qa(result.document_ast, result.docx_bytes, mode="semantic")

        assert report.passed, report.failures()
