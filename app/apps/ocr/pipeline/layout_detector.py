"""Layout detection for document pages using dual-model ensemble."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

logger = logging.getLogger(__name__)


class PaddleOcrMissingError(RuntimeError):
    """Raised when the optional paddleocr dependency is not installed."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__("paddleocr is required for document layout detection")


class ElementType(StrEnum):
    """Represent ElementType."""

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
    abstract = "abstract"
    algorithm = "algorithm"
    reference = "reference"
    code = "code"
    unknown = "unknown"


@dataclass
class LayoutBox:
    """Represent LayoutBox."""

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


# ── Label-to-type mapping for different models ────────────────────────────────
ELEMENT_TYPE_MAP: dict[str, ElementType] = {
    "title": ElementType.title,
    "doc_title": ElementType.title,
    "section_heading": ElementType.section_heading,
    "paragraph_title": ElementType.section_heading,
    "paragraph": ElementType.paragraph,
    "text": ElementType.paragraph,
    "plain_text": ElementType.paragraph,
    "abstract": ElementType.abstract,
    "reference": ElementType.reference,
    "reference_content": ElementType.reference,
    "list": ElementType.list,
    "table": ElementType.table,
    "figure": ElementType.figure,
    "chart": ElementType.chart,
    "image": ElementType.figure,
    "picture": ElementType.figure,
    "formula": ElementType.formula,
    "equation": ElementType.formula,
    "display_formula": ElementType.formula,
    "inline_formula": ElementType.formula,
    "isolate_formula": ElementType.formula,
    "caption": ElementType.caption,
    "figure_title": ElementType.caption,
    "figure_caption": ElementType.caption,
    "table_caption": ElementType.caption,
    "table_footnote": ElementType.footnote,
    "formula_caption": ElementType.caption,
    "formula_number": ElementType.footnote,
    "vision_footnote": ElementType.footnote,
    "header": ElementType.header,
    "header_image": ElementType.header,
    "footer": ElementType.footer,
    "footer_image": ElementType.footer,
    "page_number": ElementType.page_number,
    "number": ElementType.page_number,
    "footnote": ElementType.footnote,
    "aside_text": ElementType.paragraph,
    "vertical_text": ElementType.paragraph,
    "algorithm": ElementType.algorithm,
    "code": ElementType.code,
    "seal": ElementType.figure,
    "content": ElementType.paragraph,
    "phonetic": ElementType.paragraph,
    "abandon": ElementType.unknown,
}


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Compute Intersection over Union of two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def deduplicate_by_iou(
    boxes: list[LayoutBox], iou_threshold: float = 0.40
) -> list[LayoutBox]:
    """
    Remove duplicate detections where IOU > threshold, keeping the larger box.

    Two boxes that overlap significantly (IOU > threshold) are considered
    the same region; the larger one (by area) is kept.
    """
    if len(boxes) <= 1:
        return boxes

    # sort by area descending — larger boxes are preferred
    sorted_boxes = sorted(
        boxes,
        key=lambda b: -((b.x2 - b.x1) * (b.y2 - b.y1)),
    )

    kept: list[LayoutBox] = []
    for box in sorted_boxes:
        bbox = (box.x1, box.y1, box.x2, box.y2)
        is_duplicate = any(
            _iou(bbox, (k.x1, k.y1, k.x2, k.y2)) >= iou_threshold for k in kept
        )
        if not is_duplicate:
            kept.append(box)

    return kept


# ── Model names ──────────────────────────────────────────────────────────────
MODEL_V2 = "PP-DocLayoutV2"
MODEL_V3 = "PP-DocLayoutV3"


class LayoutDetector:
    """
    Detect document layout using an ensemble of PP-DocLayoutV2 + V3.

    The two models are run independently and their results are merged,
    then deduplicated by IOU (keep larger box when overlap > 40%).
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        """Initialize the instance."""
        self.confidence_threshold = confidence_threshold
        self._model_v2 = None
        self._model_v3 = None

    def detect(self, image: Image.Image, page_number: int) -> list[LayoutBox]:
        """Run dual-model detection and merge results."""
        boxes_v2 = self._detect_with(image, page_number, MODEL_V2, "_model_v2")
        boxes_v3 = self._detect_with(image, page_number, MODEL_V3, "_model_v3")

        all_boxes = boxes_v2 + boxes_v3
        if not all_boxes:
            return []

        result = deduplicate_by_iou(all_boxes, iou_threshold=0.40)
        logger.debug(
            "Layout: v2=%d, v3=%d, after dedup=%d",
            len(boxes_v2),
            len(boxes_v3),
            len(result),
        )
        return result

    def _detect_with(
        self,
        image: Image.Image,
        page_number: int,
        model_name: str,
        cache_attr: str,
    ) -> list[LayoutBox]:
        model = getattr(self, cache_attr, None)
        if model is None:
            try:
                from paddleocr import LayoutDetection

                model = LayoutDetection(
                    model_name=model_name,
                    engine_config={"enable_mkldnn": False, "cpu_threads": 2},
                )
                setattr(self, cache_attr, model)
            except ImportError as exc:
                raise PaddleOcrMissingError from exc

        import os
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                temp_path = tmp.name

            result = next(iter(model.predict(temp_path)), None)
            if result is None:
                return []
            payload = getattr(result, "json", None)
            if callable(payload):
                payload = payload()
            return self._convert_result(payload, page_number, image.width, image.height)
        except Exception:
            logger.debug(
                "%s detection failed for page %d",
                model_name,
                page_number,
                exc_info=True,
            )
            return []
        finally:
            if temp_path:
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)

    def _convert_result(
        self, result: object, page_number: int, img_w: int, img_h: int
    ) -> list[LayoutBox]:
        if not isinstance(result, dict):
            return []
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
            return []

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
