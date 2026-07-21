"""Modern document OCR pipeline orchestrator."""

import logging
import re
from io import BytesIO

from PIL import Image

from . import preprocessing
from .layout_detector import ElementType, LayoutBox, LayoutDetector
from .normalization import normalize_persian
from .reading_order import ReadingOrderResolver
from .renderer import render_pdf_bytes

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Multi-stage document pipeline: render → preprocess → detect → extract → normalize."""

    def __init__(
        self,
        dpi: int = 300,
        enable_preprocessing: bool = True,
        enable_layout: bool = True,
        enable_normalization: bool = True,
        pipeline_ocr_fn=None,
    ) -> None:
        self.dpi = dpi
        self.enable_preprocessing = enable_preprocessing
        self.enable_layout = enable_layout
        self.enable_normalization = enable_normalization
        self.pipeline_ocr_fn = pipeline_ocr_fn
        self.preprocessor = preprocessing.ImagePreprocessor()
        self.layout_detector = LayoutDetector()
        self.reading_order = ReadingOrderResolver()

    async def process_pdf(
        self, pdf_bytes: BytesIO, file_type: str = "application/pdf"
    ) -> str:
        """Process a PDF file through the full pipeline."""
        pages = render_pdf_bytes(pdf_bytes, dpi=self.dpi)
        return await self._process_images(pages)

    async def process_image(self, image: Image.Image) -> str:
        """Process a single image through the pipeline."""
        return await self._process_images([image])

    async def process_image_bytes(self, image_bytes: BytesIO) -> str:
        """Process an image from BytesIO."""
        image = Image.open(image_bytes)
        return await self.process_image(image)

    async def extract_assets(
        self, images: list[Image.Image], min_confidence: float = 0.3
    ) -> dict[int, list[LayoutBox]]:
        """Run layout detection on page images and return visual elements.

        Returns {page_number: [LayoutBox, ...]} for figures, charts, tables.
        """
        result: dict[int, list[LayoutBox]] = {}
        for page_num, img in enumerate(images, 1):
            processed = self.preprocessor.process(img) if self.enable_preprocessing else img
            try:
                elements = self.layout_detector.detect(processed, page_num)
            except Exception:
                continue
            visual = [
                e for e in elements
                if e.type in (ElementType.figure, ElementType.chart, ElementType.table)
                and e.confidence >= min_confidence
            ]
            if visual:
                result[page_num] = visual
        return result

    async def _process_images(self, images: list[Image.Image]) -> str:
        """Process a list of page images through the full pipeline."""
        all_text: list[str] = []

        for page_num, image in enumerate(images, 1):
            page_text = await self._process_page(image, page_num)
            if page_text:
                all_text.append(page_text)
            else:
                logger.warning("Page %d returned empty OCR result", page_num)
                all_text.append(f"\n> *[صفحه {page_num} — محتوای این صفحه قابل استخراج نبود]*\n")

        return "\n\n---\n\n".join(all_text)

    async def _process_page(self, image: Image.Image, page_number: int) -> str:
        """Process a single page through all stages."""
        if self.enable_preprocessing:
            processed = self.preprocessor.process(image)
        else:
            processed = image

        layout_elements: list[LayoutBox] = []
        if self.enable_layout:
            try:
                layout_elements = self.layout_detector.detect(processed, page_number)
                if layout_elements:
                    self.reading_order.resolve(layout_elements)
            except Exception:
                logger.warning("Layout detection failed for page %d", page_number)

        # Whiteout figures and charts so they never go to the VLM
        visual_markers: list[tuple[str, float]] = []
        cleaned = processed.copy()
        for elem in layout_elements:
            if elem.type not in (ElementType.figure, ElementType.chart):
                continue
            x1 = max(0, int(elem.x1) - 2)
            y1 = max(0, int(elem.y1) - 2)
            x2 = min(cleaned.width, int(elem.x2) + 2)
            y2 = min(cleaned.height, int(elem.y2) + 2)
            cleaned.paste((255, 255, 255), (x1, y1, x2, y2))
            marker_id = f"p{page_number:04d}-v{len(visual_markers) + 1:04d}"
            visual_markers.append((marker_id, (y1 + y2) / 2))
            logger.info("Whiteouted %s on page %d (y=%.0f)", elem.type.value, page_number, (y1 + y2) / 2)

        # Build layout hint for the VLM
        hint_lines: list[str] = []
        if layout_elements:
            for i, element in enumerate(layout_elements, 1):
                hint_lines.append(f"[{i}] ({element.type.value}) y={element.y1:.0f}–{element.y2:.0f}")
        hint_lines.append("Important: Figures, charts and images have been blanked out (white rectangles).")
        hint_lines.append("For each white rectangle, insert '![brief description in Persian](#)' at its position in the text.")
        hint_lines.append("Do NOT describe the content of white rectangles in prose — just place the image marker.")
        layout_hint = "Page layout detected:\n" + "\n".join(hint_lines)

        page_text = await self._extract_element(cleaned, layout_hint)

        if page_text:
            page_text = _postprocess_output(page_text)
            if self.enable_normalization:
                page_text = normalize_persian(page_text)

        return page_text

    async def _extract_element(self, image: Image.Image, layout_hint: str = "") -> str:
        """Extract text from a full page image, optionally with layout context."""
        if self.pipeline_ocr_fn:
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
            return await self.pipeline_ocr_fn(buf, layout_hint=layout_hint)

        return ""


def _postprocess_output(text: str) -> str:
    """Fix common OCR formatting issues."""
    text = re.sub(r"! \[", "![", text)
    text = re.sub(r"\(##?\)", "(#)", text)
    text = re.sub(r"https?://\s+", "", text)

    # Ensure display math $$...$$ is on its own line
    text = re.sub(r"([^\n])\$\$", r"\1\n$$", text)
    text = re.sub(r"\$\$([^\n])", r"$$\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
