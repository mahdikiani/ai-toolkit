"""Layout detection for document pages."""

import logging
from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

logger = logging.getLogger(__name__)


class ElementType(StrEnum):
    title = "title"
    section_heading = "section_heading"
    paragraph = "paragraph"
    list = "list"
    table = "table"
    figure = "figure"
    chart = "chart"
    formula = "formula"
    caption = "caption"
    header = "header"
    footer = "footer"
    page_number = "page_number"
    footnote = "footnote"
    unknown = "unknown"


@dataclass
class LayoutBox:
    """A single layout element detected on a page."""

    element_id: str
    page_number: int
    type: ElementType
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    text: str = ""
    source: str = ""


ELEMENT_TYPE_MAP: dict[str, ElementType] = {
    "title": ElementType.title,
    "section_heading": ElementType.section_heading,
    "paragraph": ElementType.paragraph,
    "text": ElementType.paragraph,
    "list": ElementType.list,
    "table": ElementType.table,
    "figure": ElementType.figure,
    "chart": ElementType.chart,
    "formula": ElementType.formula,
    "equation": ElementType.formula,
    "caption": ElementType.caption,
    "header": ElementType.header,
    "footer": ElementType.footer,
    "page_number": ElementType.page_number,
    "footnote": ElementType.footnote,
    "picture": ElementType.figure,
    "image": ElementType.figure,
    "table_caption": ElementType.caption,
    "figure_caption": ElementType.caption,
}


class LayoutDetector:
    """Detect document layout elements from a page image."""

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        self._model = None

    def detect(self, image: Image.Image, page_number: int) -> list[LayoutBox]:
        """Run layout detection on a page image and return detected elements."""
        return self._detect_pp_structure(image, page_number)

    def _detect_pp_structure(
        self, image: Image.Image, page_number: int
    ) -> list[LayoutBox]:
        """Detect layout using PaddleOCR's dedicated layout detector."""
        try:
            from paddleocr import LayoutDetection

            if self._model is None:
                self._model = LayoutDetection(
                    model_name="PP-DocLayout_plus-L",
                    engine_config={"enable_mkldnn": False, "cpu_threads": 2},
                )

            temp_path = None
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp, format="PNG")
                    temp_path = tmp.name

                result = next(iter(self._model.predict(temp_path)), None)
                if result is None:
                    raise RuntimeError("PaddleOCR returned no layout result")
                payload = getattr(result, "json", None)
                if callable(payload):
                    payload = payload()
                return self._convert_result(
                    payload, page_number, image.width, image.height
                )
            finally:
                if temp_path:
                    import os

                    os.unlink(temp_path)
        except ImportError as exc:
            msg = "paddleocr is required for document layout detection"
            raise RuntimeError(msg) from exc
        except Exception:
            logger.exception("PP-Structure inference error")
            raise

    def _convert_result(
        self, result: object, page_number: int, img_w: int, img_h: int
    ) -> list[LayoutBox]:
        """Convert PP-Structure output to LayoutBox list."""
        if not isinstance(result, dict):
            raise RuntimeError("PaddleOCR returned an invalid layout payload")
        values = result.get("res")
        if isinstance(values, dict):
            result = values
        items = (
            result.get("parsing_res_list")
            or result.get("boxes")
            or result.get("layout")
            or []
        )
        if not isinstance(items, list):
            raise RuntimeError("PaddleOCR returned no layout blocks")
        boxes: list[LayoutBox] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw_label = (
                item.get("block_label") or item.get("type") or item.get("label") or ""
            )
            elem_type = ELEMENT_TYPE_MAP.get(str(raw_label), ElementType.unknown)
            bbox = item.get("block_bbox") or item.get("bbox") or item.get("coordinate")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            elif isinstance(bbox, dict):
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                x2 = bbox.get("x2", 0)
                y2 = bbox.get("y2", 0)
            else:
                continue

            confidence = float(item.get("confidence", item.get("score", 0.5)))
            if confidence < self.confidence_threshold:
                continue

            text = ""
            res = item.get("res")
            if res and isinstance(res, list):
                text = res[0].get("text", "") if res[0] else ""

            box = LayoutBox(
                element_id=f"p{page_number:04d}-e{i + 1:04d}",
                page_number=page_number,
                type=elem_type,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=confidence,
                text=text,
                source="pp_structure_v3",
            )
            boxes.append(box)

        return boxes

    @staticmethod
    def _full_page_element(image: Image.Image, page_number: int) -> LayoutBox:
        """Fallback: treat entire page as one paragraph."""
        return LayoutBox(
            element_id=f"p{page_number:04d}-e0001",
            page_number=page_number,
            type=ElementType.paragraph,
            x1=0,
            y1=0,
            x2=float(image.width),
            y2=float(image.height),
            confidence=0.5,
            source="fallback",
        )
