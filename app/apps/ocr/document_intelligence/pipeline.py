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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from apps.ocr import checkpoint_store

from .assets import Asset, AssetManager
from .ast import ASTNode, DocumentAST, PageAST, build_ast, build_document_ast
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
ProgressCallback = Callable[[int, int], Awaitable[None]]

logger = logging.getLogger(__name__)


def _node_to_dict(node: ASTNode) -> dict:
    return {
        "type": node.type.value,
        "text": node.text,
        "html": node.html,
        "latex": node.latex,
        "caption": node.caption,
        "description": node.description,
        "chart_data": node.chart_data,
        "asset_path": node.asset_path,
        "children": [_node_to_dict(c) for c in node.children],
        "page_number": node.page_number,
        "level": node.level,
        "rows": node.rows,
        "cell_merges": [list(m) for m in node.cell_merges],
        "confidence": node.confidence,
        "bbox": list(node.bbox),
        "ordered": node.ordered,
        "repeat_header_row": node.repeat_header_row,
    }


def _node_from_dict(data: dict) -> ASTNode:
    return ASTNode(
        type=LayoutType(data["type"]),
        text=data.get("text", ""),
        html=data.get("html", ""),
        latex=data.get("latex", ""),
        caption=data.get("caption", ""),
        description=data.get("description", ""),
        chart_data=data.get("chart_data"),
        asset_path=data.get("asset_path", ""),
        children=[_node_from_dict(c) for c in data.get("children", [])],
        page_number=data.get("page_number", 1),
        level=data.get("level", 0),
        rows=data.get("rows", []),
        cell_merges=[tuple(m) for m in data.get("cell_merges", [])],
        confidence=data.get("confidence", 0.0),
        bbox=tuple(data.get("bbox", (0.0, 0.0, 0.0, 0.0))),
        ordered=data.get("ordered", False),
        repeat_header_row=data.get("repeat_header_row", False),
    )


def _page_to_checkpoint(page_ast: PageAST, page_stat: PageStats) -> dict:
    """
    Serialize one page's AST + stats for Redis.

    Only text/structure is checkpointed. Visual assets (figures/charts) are
    written to files under this pipeline instance's own temp output_dir via
    AssetManager, which doesn't survive a process crash -- a page resumed
    from checkpoint after a crash keeps its full text but loses any figure
    it contained. Fixing that would mean checkpointing asset image bytes
    too, not just page text/structure; left as a known, documented gap.
    """
    return {
        "page_ast": {
            "page_number": page_ast.page_number,
            "nodes": [_node_to_dict(n) for n in page_ast.nodes],
            "page_width": page_ast.page_width,
            "page_height": page_ast.page_height,
            "page_dpi": page_ast.page_dpi,
            "column_count": page_ast.column_count,
        },
        "page_stat": {
            "page_number": page_stat.page_number,
            "layout_time": page_stat.layout_time,
            "vlm_time": page_stat.vlm_time,
            "elements": page_stat.elements,
            "failed": page_stat.failed,
            "error": page_stat.error,
        },
    }


def _page_from_checkpoint(data: dict) -> tuple[PageAST, PageStats]:
    pa = data["page_ast"]
    ps = data["page_stat"]
    page_ast = PageAST(
        page_number=pa["page_number"],
        nodes=[_node_from_dict(n) for n in pa["nodes"]],
        page_width=pa.get("page_width", 0.0),
        page_height=pa.get("page_height", 0.0),
        page_dpi=pa.get("page_dpi", 300.0),
        column_count=pa.get("column_count", 1),
    )
    page_stat = PageStats(
        page_number=ps["page_number"],
        layout_time=ps["layout_time"],
        vlm_time=ps["vlm_time"],
        elements=ps.get("elements", []),
        failed=ps.get("failed", False),
        error=ps.get("error"),
    )
    return page_ast, page_stat


@dataclass
class PageStats:
    """Represent PageStats."""

    page_number: int
    layout_time: float
    vlm_time: float
    elements: list[dict] = field(default_factory=list)
    failed: bool = False
    error: str | None = None


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
    failed_pages: list[int] = []
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
            "failed": p.failed,
        }
        if p.failed:
            failed_pages.append(p.page_number)
        if include_elements:
            entry["elements"] = p.elements
        pages_summary.append(entry)
    return {
        "pages": pages_summary,
        "render_time": round(stats.render_time, 3),
        "total_time": round(stats.total_time, 3),
        "failed_pages": failed_pages,
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
        self,
        file_bytes: BytesIO,
        filename: str,
        mode: DocxMode = "semantic",
        on_page_done: ProgressCallback | None = None,
        task_uid: str | None = None,
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

        ``on_page_done``, if given, is awaited after every page (successful
        or not) with ``(completed_pages, total_pages)`` -- callers use it to
        report progress on long documents. A failing callback must not abort
        the document, so exceptions from it are only logged.

        ``task_uid``, if given, checkpoints each page's AST to Redis as it
        completes and reuses any checkpoint left by a previous, crashed run
        of the same task instead of reprocessing that page -- see
        ``_page_to_checkpoint`` for what is (and isn't) preserved.
        """
        t_start = time.time()
        file_bytes.seek(0)
        file_bytes.name = filename
        document: Document = load_document(file_bytes, dpi=self.dpi)
        total_pages = len(document.pages)
        checkpoints = (
            await checkpoint_store.load_pages(task_uid) if task_uid else {}
        )

        page_asts: list[PageAST] = []
        page_stats: list[PageStats] = []
        for page in document.pages:
            cached = checkpoints.get(page.page_number)
            if cached is not None:
                page_ast, page_stat = _page_from_checkpoint(cached)
            else:
                try:
                    page_ast, page_stat = await self._process_page(page)
                except Exception:
                    # Each element call already retries transient failures (see
                    # ElementProcessor._call_with_retry); reaching here means
                    # retries were exhausted. A single unlucky page must not
                    # discard every other page's already-completed work in a
                    # potentially hours-long, thousand-call job -- isolate the
                    # failure to this page and keep going.
                    logger.exception(
                        "Page %d failed after retries; skipping it, continuing",
                        page.page_number,
                    )
                    page_ast = PageAST(page_number=page.page_number, nodes=[])
                    page_stat = PageStats(
                        page_number=page.page_number,
                        layout_time=0.0,
                        vlm_time=0.0,
                        failed=True,
                        error="processing failed after retries",
                    )
                if task_uid:
                    await checkpoint_store.save_page(
                        task_uid,
                        page.page_number,
                        _page_to_checkpoint(page_ast, page_stat),
                    )
            page_asts.append(page_ast)
            page_stats.append(page_stat)

            if on_page_done is not None:
                try:
                    await on_page_done(len(page_asts), total_pages)
                except Exception:
                    logger.exception(
                        "Progress callback failed on page %d/%d; continuing",
                        len(page_asts),
                        total_pages,
                    )

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
