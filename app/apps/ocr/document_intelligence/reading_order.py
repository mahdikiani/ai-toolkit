"""Reading Order Resolver — RTL-aware column detection for Persian/Arabic docs."""

from __future__ import annotations

import logging

from .layout import LayoutElement, LayoutType

logger = logging.getLogger(__name__)

FULL_WIDTH_TYPES = {
    LayoutType.title,
    LayoutType.heading,
    LayoutType.table,
    LayoutType.formula,
    LayoutType.code,
    LayoutType.header,
    LayoutType.footer,
}


class ReadingOrderResolver:
    """Resolve element order: detect columns → sort RTL → interleave full-width."""

    def resolve(
        self, elements: list[LayoutElement], page_width: float, is_rtl: bool = True
    ) -> list[LayoutElement]:
        if not elements:
            return elements

        full_width = self._detect_full_width(elements, page_width)
        column_candidates = [e for e in elements if e not in full_width]
        columns = self._detect_columns(column_candidates)

        if not columns:
            if is_rtl:
                elements.sort(key=lambda e: (e.bbox[1], -e.bbox[0]))
            else:
                elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
            return elements

        # Sort columns: RTL (rightmost first) or LTR (leftmost first)
        columns.sort(
            key=lambda c: (
                sum(e.bbox[0] for e in c) / max(len(c), 1),
            ),
            reverse=is_rtl,
        )
        ordered_cols: list[LayoutElement] = []
        for col in columns:
            col.sort(key=lambda e: e.bbox[1])
            ordered_cols.extend(col)

        # Interleave full-width elements with column content by vertical position
        elements[:] = sorted(
            [*full_width, *ordered_cols],
            key=lambda e: (
                e.bbox[1],
                0 if e in full_width else 1,
                -e.bbox[0] if is_rtl else e.bbox[0],
            ),
        )
        return elements

    @staticmethod
    def _detect_full_width(
        elements: list[LayoutElement], page_width: float
    ) -> list[LayoutElement]:
        return [
            e
            for e in elements
            if (e.bbox[2] - e.bbox[0]) >= page_width * 0.85
            or e.type in FULL_WIDTH_TYPES
        ]

    @staticmethod
    def _detect_columns(elements: list[LayoutElement]) -> list[list[LayoutElement]]:
        if len(elements) < 3:
            return []
        centers = [(e.bbox[0] + e.bbox[2]) / 2 for e in elements]
        span = max(centers) - min(centers)
        if span < 100:
            return []
        try:
            from sklearn.cluster import KMeans

            x_centers = [[(e.bbox[0] + e.bbox[2]) / 2] for e in elements]
            kmeans = KMeans(
                n_clusters=min(3, len(elements)), n_init=1, random_state=0
            )
            labels = kmeans.fit_predict(x_centers)
            cols: dict[int, list[LayoutElement]] = {}
            for elem, label in zip(elements, labels):
                cols.setdefault(int(label), []).append(elem)
            return list(cols.values())
        except Exception:
            return []
