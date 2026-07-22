"""Layout Detection — ensemble PP-DocLayoutV2+V3 with padding crop."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image

from .loader import Page

logger = logging.getLogger(__name__)


class LayoutType(StrEnum):
    title = "title"
    heading = "heading"
    header = "header"
    footer = "footer"
    paragraph = "paragraph"
    list = "list"
    table = "table"
    table_caption = "table_caption"
    table_footnote = "table_footnote"
    figure = "figure"
    figure_caption = "figure_caption"
    chart = "chart"
    formula = "formula"
    code = "code"
    reference = "reference"
    page_number = "page_number"
    unknown = "unknown"


TEXT_TYPES = {
    LayoutType.title,
    LayoutType.heading,
    LayoutType.header,
    LayoutType.footer,
    LayoutType.paragraph,
    LayoutType.list,
    LayoutType.reference,
}
VISUAL_TYPES = {LayoutType.figure, LayoutType.chart}
TABLE_TYPES = {LayoutType.table, LayoutType.table_caption, LayoutType.table_footnote}
SPECIAL_TYPES = {LayoutType.formula, LayoutType.code}


@dataclass
class LayoutElement:
    id: str
    page_id: str
    page_number: int
    type: LayoutType
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (raw)
    padded_bbox: tuple[float, float, float, float]  # with 10% padding
    confidence: float
    crop_path: str | None = None


# ── Label mapping for PP-DocLayout models ───────────────────────────────────
LABEL_MAP: dict[str, LayoutType] = {
    "title": LayoutType.title,
    "doc_title": LayoutType.title,
    "section_heading": LayoutType.heading,
    "paragraph_title": LayoutType.heading,
    "heading": LayoutType.heading,
    "paragraph": LayoutType.paragraph,
    "text": LayoutType.paragraph,
    "plain_text": LayoutType.paragraph,
    "abstract": LayoutType.paragraph,
    "reference": LayoutType.reference,
    "reference_content": LayoutType.reference,
    "list": LayoutType.list,
    "table": LayoutType.table,
    "table_caption": LayoutType.table_caption,
    "table_footnote": LayoutType.table_footnote,
    "figure": LayoutType.figure,
    "chart": LayoutType.chart,
    "image": LayoutType.figure,
    "picture": LayoutType.figure,
    "formula": LayoutType.formula,
    "equation": LayoutType.formula,
    "display_formula": LayoutType.formula,
    "inline_formula": LayoutType.formula,
    "isolate_formula": LayoutType.formula,
    "caption": LayoutType.figure_caption,
    "figure_title": LayoutType.figure_caption,
    "figure_caption": LayoutType.figure_caption,
    "formula_caption": LayoutType.figure_caption,
    "formula_number": LayoutType.page_number,
    "vision_footnote": LayoutType.table_footnote,
    "header": LayoutType.header,
    "header_image": LayoutType.header,
    "footer": LayoutType.footer,
    "footer_image": LayoutType.footer,
    "page_number": LayoutType.page_number,
    "number": LayoutType.page_number,
    "footnote": LayoutType.footer,
    "aside_text": LayoutType.paragraph,
    "vertical_text": LayoutType.paragraph,
    "algorithm": LayoutType.code,
    "code": LayoutType.code,
    "seal": LayoutType.figure,
    "content": LayoutType.paragraph,
    "phonetic": LayoutType.paragraph,
    "abandon": LayoutType.unknown,
}

MODEL_NAMES = ["PP-DocLayoutV2", "PP-DocLayoutV3"]
CROP_PADDING_RATIO = 0.10  # fallback default; LayoutDetector reads Settings instead


def _intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    intersection = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _containment_ratio(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection area over the smaller box's area.

    Catches near-duplicate detections from the v2/v3 ensemble where one
    model's box is offset/padded differently from the other's — such pairs
    can have low IOU while one box is still almost entirely inside the
    other (plain IOU alone misses these and leaves duplicated text).
    """
    smaller_area = min(_box_area(a), _box_area(b))
    if smaller_area <= 0:
        return 0.0
    return _intersection_area(a, b) / smaller_area


def deduplicate_by_iou(
    elements: list[LayoutElement],
    iou_threshold: float = 0.40,
    containment_threshold: float = 0.70,
) -> list[LayoutElement]:
    if len(elements) <= 1:
        return elements
    sorted_elems = sorted(
        elements,
        key=lambda e: -((e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])),
    )
    kept: list[LayoutElement] = []
    for elem in sorted_elems:
        is_dup = any(
            _iou(elem.bbox, k.bbox) >= iou_threshold
            or _containment_ratio(elem.bbox, k.bbox) >= containment_threshold
            for k in kept
        )
        if not is_dup:
            kept.append(elem)
    return kept


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    padding_ratio: float = CROP_PADDING_RATIO,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


class LayoutDetector:
    """Dual-model ensemble layout detector with crop generation."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        padding_ratio: float = CROP_PADDING_RATIO,
        iou_threshold: float = 0.40,
        crop_dir: str | Path | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio
        self.iou_threshold = iou_threshold
        # Per-instance temp dir (defaults to a fresh mkdtemp) so concurrent tasks
        # never collide on crop paths and can be cleaned up as a unit.
        self.crop_dir = Path(crop_dir) if crop_dir else Path(tempfile.mkdtemp(prefix="di_crops_"))
        self._models: dict[str, object] = {}

        self.stats: dict[str, list[float]] = {
            "detect_time": [],
            "elements_per_page": [],
        }

    def detect_page(
        self, image: Image.Image, page: Page
    ) -> list[LayoutElement]:
        """Run detection and return layout elements with crops."""
        boxes_v2 = self._run_model(image, page, MODEL_NAMES[0])
        boxes_v3 = self._run_model(image, page, MODEL_NAMES[1])

        all_elems = boxes_v2 + boxes_v3
        all_elems = deduplicate_by_iou(all_elems, iou_threshold=self.iou_threshold)

        self.stats["elements_per_page"].append(len(all_elems))
        logger.debug(
            "Page %d: v2=%d, v3=%d, total=%d",
            page.page_number, len(boxes_v2), len(boxes_v3), len(all_elems),
        )

        for elem in all_elems:
            crop = image.crop(
                (int(elem.padded_bbox[0]), int(elem.padded_bbox[1]),
                 int(elem.padded_bbox[2]), int(elem.padded_bbox[3]))
            )
            page_crop_dir = self.crop_dir / elem.page_id
            page_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = page_crop_dir / f"{elem.id}.png"
            crop.save(crop_path, "PNG")
            elem.crop_path = str(crop_path)

        return all_elems

    def detect(self, image: Image.Image, page: Page) -> list[LayoutElement]:
        """Public API — detect elements on a single page."""
        return self.detect_page(image, page)

    def _run_model(
        self, image: Image.Image, page: Page, model_name: str
    ) -> list[LayoutElement]:
        """Run a single layout model on a page image."""
        model = self._get_model(model_name)
        if model is None:
            return []

        t0 = time.time()
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
            elements = self._parse_output(
                payload, page, model_name
            )
            self.stats["detect_time"].append(time.time() - t0)
            return elements
        except Exception:
            logger.debug("%s failed for page %d", model_name, page.page_number, exc_info=True)
            return []
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _get_model(self, model_name: str):
        """Lazy-init and cache layout models."""
        if model_name in self._models:
            return self._models[model_name]
        try:
            from paddleocr import LayoutDetection

            model = LayoutDetection(
                model_name=model_name,
                engine_config={"enable_mkldnn": False, "cpu_threads": 2},
            )
            self._models[model_name] = model
            return model
        except ImportError as exc:
            raise RuntimeError("paddleocr required for layout detection") from exc

    def _parse_output(
        self, result: object, page: Page, source: str
    ) -> list[LayoutElement]:
        """Convert model JSON output to LayoutElement list."""
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

        elements: list[LayoutElement] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw_label = (
                item.get("block_label") or item.get("type") or item.get("label") or ""
            )
            elem_type = LABEL_MAP.get(str(raw_label), LayoutType.unknown)
            bbox_raw = item.get("block_bbox") or item.get("bbox") or item.get("coordinate")
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                x1, y1, x2, y2 = bbox_raw
            elif isinstance(bbox_raw, dict):
                x1 = float(bbox_raw.get("x1", 0))
                y1 = float(bbox_raw.get("y1", 0))
                x2 = float(bbox_raw.get("x2", 0))
                y2 = float(bbox_raw.get("y2", 0))
            else:
                continue

            confidence = float(item.get("confidence", item.get("score", 0.5)))
            if confidence < self.confidence_threshold:
                continue
            bbox = (float(x1), float(y1), float(x2), float(y2))
            padded = _pad_bbox(bbox, page.width, page.height, self.padding_ratio)
            elem_id = f"{page.id}_e{i + 1:04d}"

            elements.append(
                LayoutElement(
                    id=elem_id,
                    page_id=page.id,
                    page_number=page.page_number,
                    type=elem_type,
                    bbox=bbox,
                    padded_bbox=padded,
                    confidence=confidence,
                )
            )
        return elements

    def log_stats(self) -> None:
        if self.stats["detect_time"]:
            logger.info(
                "Layout: avg=%.2fs, total elements=%d",
                sum(self.stats["detect_time"]) / len(self.stats["detect_time"]),
                sum(self.stats["elements_per_page"]),
            )

    def cleanup(self) -> None:
        """Remove this detector's temp crop directory."""
        import shutil

        shutil.rmtree(self.crop_dir, ignore_errors=True)


def load_layout_detector(confidence_threshold: float = 0.5) -> LayoutDetector:
    return LayoutDetector(confidence_threshold=confidence_threshold)
