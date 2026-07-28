"""
Document Intelligence Pipeline orchestrator.

Wires the independent stages together:
Loader -> Layout Detection -> Element Processing (OCR/VLM) -> Reading Order ->
Document AST -> Markdown/Word Renderers -> Asset Manager.

Runs entirely on CPU today. The layout backend (LayoutDetector) and the VLM
client (ElementProcessor) are the two points designed to later be swapped for
calls to an external GPU service without touching the rest of the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from .assets import Asset, AssetManager
from .ast import DocumentAST, PageAST, build_ast, build_document_ast
from .elements import ElementProcessor, ProcessedElement
from .layout import VISUAL_TYPES, LayoutDetector, LayoutElement, LayoutType
from .loader import Document, Page, load_document
from .reading_order import ReadingOrderResolver
from .renderers.docx import render_docx
from .renderers.docx_absolute import render_docx_absolute
from .renderers.markdown import render_markdown
from .structure.paragraph_merge import merge_paragraphs
from .structure.table_continuation import merge_table_continuations

DocxMode = Literal["semantic", "visual"]

logger = logging.getLogger(__name__)


@dataclass
class PageStats:
    """Represent PageStats."""

    page_number: int
    layout_time: float
    vlm_time: float
    elements: list[dict] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Represent PipelineStats."""

    pages: list[PageStats] = field(default_factory=list)
    render_time: float = 0.0
    total_time: float = 0.0


@dataclass
class PipelineResult:
    """Represent PipelineResult."""

    markdown: str
    docx_bytes: bytes
    output_dir: Path
    assets: list[Asset]
    stats: PipelineStats
    document_ast: DocumentAST


def summarize_stats(stats: PipelineStats, include_elements: bool = False) -> dict:
    """
    Compact, Mongo-friendly summary of per-page timing/confidence.

    Full per-element detail is always available in logs; it's only embedded
    here when ``include_elements`` is set (Settings.ocr_di_output_debug).
    """
    pages_summary = []
    for p in stats.pages:
        confidences = [e["confidence"] for e in p.elements]
        entry = {
            "page_number": p.page_number,
            "layout_time": round(p.layout_time, 3),
            "vlm_time": round(p.vlm_time, 3),
            "element_count": len(p.elements),
            "avg_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences
            else None,
        }
        if include_elements:
            entry["elements"] = p.elements
        pages_summary.append(entry)
    return {
        "pages": pages_summary,
        "render_time": round(stats.render_time, 3),
        "total_time": round(stats.total_time, 3),
    }


class DocumentIntelligencePipeline:
    """Orchestrates the full Document Loader -> Renderers pipeline for one document."""

    def __init__(
        self,
        dpi: int | None = None,
        confidence_threshold: float | None = None,
        padding_ratio: float | None = None,
        iou_threshold: float | None = None,
        max_concurrent_vlm: int | None = None,
        vlm_model: str | None = None,
        openrouter_client: object | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize the instance."""
        from server.config import Settings

        self.dpi = dpi or Settings.ocr_pipeline_dpi
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(tempfile.mkdtemp(prefix="di_output_"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.layout_detector = LayoutDetector(
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else Settings.ocr_layout_confidence_threshold
            ),
            padding_ratio=(
                padding_ratio
                if padding_ratio is not None
                else Settings.ocr_di_crop_padding_ratio
            ),
            iou_threshold=(
                iou_threshold
                if iou_threshold is not None
                else Settings.ocr_di_iou_threshold
            ),
        )
        self.element_processor = ElementProcessor(
            vlm_model=vlm_model, openrouter_client=openrouter_client
        )
        self.reading_order = ReadingOrderResolver()
        self.asset_manager = AssetManager(self.output_dir)
        self.max_concurrent_vlm = (
            max_concurrent_vlm
            if max_concurrent_vlm is not None
            else Settings.ocr_di_max_concurrent_vlm
        )

    async def process(
        self, file_bytes: BytesIO, filename: str, mode: DocxMode = "semantic"
    ) -> PipelineResult:
        """
        Run the full pipeline on a PDF/image and return markdown + docx + assets.

        ``mode="semantic"`` (default) produces a flowing, fully-editable Word
        document built from real Paragraph/Heading/Table/Section objects —
        no Text Box or Shape for normal content. ``mode="visual"`` instead
        reproduces the source page's visual layout by placing each element
        in an absolutely-positioned floating text box; it trades editability
        for closer visual fidelity and is meant for forms/brochures, not as
        the default output.
        """
        t_start = time.time()
        file_bytes.seek(0)
        file_bytes.name = filename
        document: Document = load_document(file_bytes, dpi=self.dpi)

        page_asts: list[PageAST] = []
        page_stats: list[PageStats] = []
        for page in document.pages:
            page_ast, page_stat = await self._process_page(page)
            page_asts.append(page_ast)
            page_stats.append(page_stat)

        asset_map = {a.path: a.rel_path for a in self.asset_manager.get_assets()}
        document_ast = build_document_ast(page_asts, asset_map)
        # Merge adjacent same-page paragraph blocks that are really one
        # paragraph split by layout detection, before any renderer sees
        # them -- see structure/paragraph_merge.py.
        document_ast = merge_paragraphs(document_ast)
        # Merge a table cut off at the bottom of one page with its
        # continuation at the top of the next into one logical table --
        # see structure/table_continuation.py.
        document_ast = merge_table_continuations(document_ast)

        t_render = time.time()
        markdown = render_markdown(document_ast)
        docx_buf = (
            render_docx(document_ast, pdf_data=document.pdf_data)
            if mode == "semantic"
            else render_docx_absolute(document_ast, pdf_data=document.pdf_data)
        )
        render_time = time.time() - t_render

        docx_bytes = docx_buf.getvalue()
        (self.output_dir / "document.md").write_text(markdown, encoding="utf-8")
        (self.output_dir / "document.docx").write_bytes(docx_bytes)

        stats = PipelineStats(
            pages=page_stats, render_time=render_time, total_time=time.time() - t_start
        )
        self._log_stats(stats)

        return PipelineResult(
            markdown=markdown,
            docx_bytes=docx_bytes,
            output_dir=self.output_dir,
            assets=self.asset_manager.get_assets(),
            stats=stats,
            document_ast=document_ast,
        )

    async def _process_page(self, page: Page) -> tuple[PageAST, PageStats]:
        t0 = time.time()
        layout_elements = self.layout_detector.detect(page.image, page)
        layout_time = time.time() - t0

        if not layout_elements:
            logger.warning("Page %d: no layout elements detected", page.page_number)
            return (
                PageAST(page_number=page.page_number, nodes=[]),
                PageStats(
                    page_number=page.page_number, layout_time=layout_time, vlm_time=0.0
                ),
            )

        t1 = time.time()
        processed = await self._process_elements(layout_elements, page.image)
        vlm_time = time.time() - t1

        texts = {p.id: p.text for p in processed if p.text}
        ordered = self.reading_order.resolve(
            list(layout_elements), page.width, texts=texts
        )
        column_count = self.reading_order.detect_column_count(
            list(layout_elements), page.width
        )
        page_ast = build_ast(
            processed,
            ordered,
            page.page_number,
            page.width,
            page.height,
            page.dpi,
            column_count=column_count,
        )

        elem_stats = [
            {
                "id": p.id,
                "type": p.type.value,
                "confidence": p.confidence,
                "duration": round(p.vlm_duration, 3),
                "tokens": p.vlm_tokens,
            }
            for p in processed
        ]
        return (
            page_ast,
            PageStats(
                page_number=page.page_number,
                layout_time=layout_time,
                vlm_time=vlm_time,
                elements=elem_stats,
            ),
        )

    async def _process_elements(
        self, elements: list[LayoutElement], page_image: Image.Image
    ) -> list[ProcessedElement]:
        semaphore = asyncio.Semaphore(self.max_concurrent_vlm)

        async def _one(elem: LayoutElement) -> ProcessedElement:
            async with semaphore:
                processed = await self.element_processor.process(elem, page_image)
                if elem.type in VISUAL_TYPES:
                    crop = page_image.crop((
                        int(elem.padded_bbox[0]),
                        int(elem.padded_bbox[1]),
                        int(elem.padded_bbox[2]),
                        int(elem.padded_bbox[3]),
                    ))
                    asset_type = "chart" if elem.type == LayoutType.chart else "figure"
                    asset = self.asset_manager.save_image(
                        crop, elem.id, asset_type=asset_type
                    )
                    processed.asset_path = asset.path
                return processed

        return await asyncio.gather(*(_one(e) for e in elements))

    def _log_stats(self, stats: PipelineStats) -> None:
        for p in stats.pages:
            logger.info(
                "Page %d: layout=%.2fs vlm=%.2fs elements=%d",
                p.page_number,
                p.layout_time,
                p.vlm_time,
                len(p.elements),
            )
        logger.info("Render: %.2fs, total: %.2fs", stats.render_time, stats.total_time)

    def cleanup(self) -> None:
        """
        Remove the temp layout-crop dir. The output dir (md/docx/assets) is.

        left for the caller to upload/persist and clean up itself.
        """
        self.layout_detector.cleanup()
