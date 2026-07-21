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

        layout_hint = ""
        if self.enable_layout:
            try:
                elements = self.layout_detector.detect(processed, page_number)
                if elements:
                    self.reading_order.resolve(elements)
                    hints = [
                        f"[{i}] ({element.type.value}) y={element.y1:.0f}–{element.y2:.0f}"
                        for i, element in enumerate(elements, 1)
                    ]
                    layout_hint = "Page layout detected:\n" + "\n".join(hints)
            except Exception:
                logger.warning("Layout detection failed for page %d", page_number)

        page_text = await self._extract_element(processed, layout_hint)

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
    return text
