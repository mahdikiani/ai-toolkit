"""Modern document OCR pipeline — element-level processing with dual-model layout."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from PIL import Image

from . import preprocessing
from .layout_detector import ElementType, LayoutBox, LayoutDetector
from .normalization import normalize_persian
from .reading_order import ReadingOrderResolver
from .renderer import render_pdf_bytes

logger = logging.getLogger(__name__)

TEXT_TYPES = {
    ElementType.title,
    ElementType.section_heading,
    ElementType.paragraph,
    ElementType.list,
    ElementType.caption,
    ElementType.footnote,
    ElementType.abstract,
    ElementType.reference,
    ElementType.algorithm,
    ElementType.code,
}
VISUAL_TYPES = {ElementType.figure, ElementType.chart}
SPECIAL_TYPES = {ElementType.table, ElementType.formula}
SKIP_TYPES = {ElementType.header, ElementType.footer, ElementType.page_number}


class DocumentPipeline:
    """Element-level document pipeline.

    1. Layout detection (PP-DocLayoutV2 + V3 ensemble) → typed elements
    2. Each element cropped and routed:
       - Text → crop → VLM → extracted text
       - Visual → crop → saved as asset (uploaded later)
       - Table/Formula → crop → specialised extraction
    3. Reassemble in reading order with Markdown formatting
    4. Headers/Footers collected separately for DOCX generation
    """

    def __init__(
        self,
        dpi: int = 300,
        enable_preprocessing: bool = True,
        enable_layout: bool = True,
        enable_normalization: bool = True,
        include_headers: bool = False,
        include_footers: bool = False,
        pipeline_ocr_fn=None,
    ) -> None:
        self.dpi = dpi
        self.enable_preprocessing = enable_preprocessing
        self.enable_layout = enable_layout
        self.enable_normalization = enable_normalization
        self.include_headers = include_headers
        self.include_footers = include_footers
        self.pipeline_ocr_fn = pipeline_ocr_fn
        self.preprocessor = preprocessing.ImagePreprocessor()
        self.layout_detector = LayoutDetector()
        self.reading_order = ReadingOrderResolver()

        self.assets: list[dict[str, Any]] = []
        self._page_headers: dict[int, str] = {}
        self._page_footers: dict[int, str] = {}
        self._all_elements: list[LayoutBox] = []
        self._crops: dict[str, bytes] = {}

    def get_assets(self) -> list[dict[str, Any]]:
        return list(self.assets)

    def get_headers(self) -> dict[int, str]:
        return dict(self._page_headers)

    def get_footers(self) -> dict[int, str]:
        return dict(self._page_footers)

    def get_elements(self) -> list[LayoutBox]:
        return list(self._all_elements)

    def get_crop_bytes(self, element_id: str) -> bytes | None:
        return self._crops.get(element_id)

    def get_all_crops(self) -> dict[str, bytes]:
        return dict(self._crops)

    async def process_pdf(self, pdf_bytes: BytesIO) -> str:
        pages = render_pdf_bytes(pdf_bytes, dpi=self.dpi)
        self.assets.clear()
        self._page_headers.clear()
        self._page_footers.clear()
        self._all_elements.clear()
        self._crops.clear()
        return await self._process_images(pages)

    async def process_image(self, image: Image.Image) -> str:
        self.assets.clear()
        self._page_headers.clear()
        self._page_footers.clear()
        self._all_elements.clear()
        self._crops.clear()
        return await self._process_images([image])

    async def process_image_bytes(self, image_bytes: BytesIO) -> str:
        image = Image.open(image_bytes)
        return await self.process_image(image)

    async def _process_images(self, images: list[Image.Image]) -> str:
        all_text: list[str] = []
        for page_num, image in enumerate(images, 1):
            page_text, page_assets = await self._process_page(image, page_num)
            self.assets.extend(page_assets)
            if page_text:
                all_text.append(page_text)
            else:
                logger.warning("Page %d returned empty OCR result", page_num)
                all_text.append(f"\n> *[صفحه {page_num} — محتوای این صفحه قابل استخراج نبود]*\n")
        return "\n\n---\n\n".join(all_text)

    async def _process_page(
        self, image: Image.Image, page_number: int
    ) -> tuple[str, list[dict[str, Any]]]:
        if self.enable_preprocessing:
            processed = self.preprocessor.process(image)
        else:
            processed = image

        elements: list[LayoutBox] = []
        if self.enable_layout:
            try:
                elements = self.layout_detector.detect(processed, page_number)
                if elements:
                    self.reading_order.resolve(elements)
            except Exception:
                logger.warning("Layout detection failed for page %d", page_number)

        if not elements:
            elements = [self._full_page_element(processed, page_number)]

        self._all_elements.extend(elements)

        headers: list[str] = []
        footers: list[str] = []
        page_assets: list[dict[str, Any]] = []
        assembled: list[str] = []

        for elem in elements:
            if elem.type in SKIP_TYPES:
                crop = self._crop(processed, elem)
                if crop is not None:
                    buf = BytesIO()
                    crop.save(buf, format="PNG")
                    self._crops[elem.element_id] = buf.getvalue()
                continue

            crop = self._crop(processed, elem)
            if crop is None:
                continue

            if elem.type in VISUAL_TYPES:
                asset_id = f"asset_{elem.element_id}"
                buf = BytesIO()
                crop.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                self._crops[elem.element_id] = img_bytes
                page_assets.append({
                    "id": asset_id,
                    "page": page_number,
                    "element_id": elem.element_id,
                    "image_bytes": img_bytes,
                    "type": elem.type.value,
                })
                assembled.append(f"![{elem.type.value} در صفحه {page_number}]({asset_id})")
                continue

            buf = BytesIO()
            crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()
            buf.seek(0)
            self._crops[elem.element_id] = crop_bytes

            text = await self._ocr_crop(buf, elem)
            if text:
                formatted = self._format_element(text, elem)
                assembled.append(formatted)

        if self.enable_normalization:
            assembled = [normalize_persian(t) if t else t for t in assembled]

        if headers:
            self._page_headers[page_number] = "\n".join(headers)
        if footers:
            self._page_footers[page_number] = "\n".join(footers)

        return "\n\n".join(assembled), page_assets

    async def _ocr_crop(self, crop_buf: BytesIO, element: LayoutBox) -> str:
        if not self.pipeline_ocr_fn:
            return ""
        return await self.pipeline_ocr_fn(crop_buf, element)

    @staticmethod
    def _format_element(text: str, element: LayoutBox) -> str:
        if element.type == ElementType.title:
            return f"# {text}"
        if element.type == ElementType.section_heading:
            return f"## {text}"
        if element.type == ElementType.list:
            lines = text.split("\n")
            bulleted = [f"- {line.strip()}" for line in lines if line.strip()]
            return "\n".join(bulleted)
        return text

    @staticmethod
    def _crop(image: Image.Image, element: LayoutBox, padding: int = 8) -> Image.Image | None:
        x1 = max(0, int(element.x1) - padding)
        y1 = max(0, int(element.y1) - padding)
        x2 = min(image.width, int(element.x2) + padding)
        y2 = min(image.height, int(element.y2) + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _full_page_element(image: Image.Image, page_number: int) -> LayoutBox:
        from .layout_detector import LayoutBox as LB
        return LB(
            element_id=f"p{page_number:04d}-e0001",
            page_number=page_number,
            type=ElementType.paragraph,
            x1=0, y1=0,
            x2=float(image.width),
            y2=float(image.height),
            confidence=0.5,
            source="fallback",
        )
