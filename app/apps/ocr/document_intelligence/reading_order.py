"""Reading Order Resolver — RTL-aware column detection for Persian/Arabic docs."""

from __future__ import annotations

import itertools
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

RTL_RATIO_THRESHOLD = 0.3
FULL_WIDTH_MIN_RATIO = 0.85
FULL_WIDTH_TYPE_MIN_RATIO = 0.5
COLUMN_GAP_MIN_RATIO = 0.35


class ReadingOrderResolver:
    """Resolve element order: detect columns → sort RTL → interleave full-width."""

    def detect_column_count(
        self, elements: list[LayoutElement], page_width: float
    ) -> int:
        """
        Return how many text columns this page appears to have (1 = single column).

        Reuses the same detection ``resolve()`` uses internally to reorder
        elements, exposed separately so callers (Section Detection, for
        real Word column sections) can use it without resolve()'s
        reordering side effects.
        """
        if not elements:
            return 1
        full_width = self._detect_full_width(elements, page_width)
        column_candidates = [e for e in elements if e not in full_width]
        columns = self._detect_columns(column_candidates)
        return max(1, len(columns))

    def resolve(
        self,
        elements: list[LayoutElement],
        page_width: float,
        is_rtl: bool | None = None,
        texts: dict[str, str] | None = None,
    ) -> list[LayoutElement]:
        """
        Order elements for reading.

        If ``is_rtl`` is omitted, it is auto-detected from ``texts``
        (element_id -> extracted text) via script-range language
        detection, defaulting to RTL when no text is available yet (this
        pipeline targets Persian/Arabic documents).
        """
        if not elements:
            return elements

        if is_rtl is None:
            is_rtl = self.detect_is_rtl(elements, texts)

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
            key=lambda c: (sum(e.bbox[0] for e in c) / max(len(c), 1),),
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
    def detect_is_rtl(
        elements: list[LayoutElement], texts: dict[str, str] | None
    ) -> bool:
        """Auto-detect page direction from extracted text's script ratio."""
        if not texts:
            return True
        from ..pipeline.normalization import detect_rtl_ratio

        combined = " ".join(texts.get(e.id, "") for e in elements).strip()
        if not combined:
            return True
        return detect_rtl_ratio(combined) >= RTL_RATIO_THRESHOLD

    @staticmethod
    def _detect_full_width(
        elements: list[LayoutElement], page_width: float
    ) -> list[LayoutElement]:
        """
        Return elements that are full-width, or wide *enough* for their type.

        Type alone is not enough: a narrow heading sitting inside a
        side-by-side box next to another box must NOT be forced
        full-width, or it gets sorted purely by y and loses its real
        column position (e.g. it jumps ahead of a right-hand-side box it
        should read after in RTL).
        """
        return [
            e
            for e in elements
            if (e.bbox[2] - e.bbox[0]) >= page_width * FULL_WIDTH_MIN_RATIO
            or (
                e.type in FULL_WIDTH_TYPES
                and (e.bbox[2] - e.bbox[0]) >= page_width * FULL_WIDTH_TYPE_MIN_RATIO
            )
        ]

    @staticmethod
    def _detect_columns(elements: list[LayoutElement]) -> list[list[LayoutElement]]:
        if len(elements) < 3:
            return []
        centers = sorted((e.bbox[0] + e.bbox[2]) / 2 for e in elements)
        span = centers[-1] - centers[0]
        if span < 100:
            return []
        # KMeans with a fixed n_clusters always returns that many clusters,
        # even for an essentially single-column page with one stray element
        # pulling the spread up — fabricating false columns there scrambles
        # reading order. Only proceed if there's a genuinely large gap
        # separating groups of x-centers (real columns), not just noise.
        gaps = [b - a for a, b in itertools.pairwise(centers)]
        if max(gaps) < span * COLUMN_GAP_MIN_RATIO:
            return []
        try:
            from sklearn.cluster import KMeans

            # Ask only for as many clusters as large gaps imply (+1).
            # Requesting a fixed 3 on a clear 2-column page makes sklearn
            # warn (fewer distinct clusters than n_clusters) and can
            # fabricate a phantom third column.
            n_large_gaps = sum(1 for gap in gaps if gap >= span * COLUMN_GAP_MIN_RATIO)
            n_clusters = min(3, len(elements), n_large_gaps + 1)
            if n_clusters < 2:
                return []

            x_centers = [[(e.bbox[0] + e.bbox[2]) / 2] for e in elements]
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
            labels = kmeans.fit_predict(x_centers)
            cols: dict[int, list[LayoutElement]] = {}
            for elem, label in zip(elements, labels, strict=True):
                cols.setdefault(int(label), []).append(elem)
            return list(cols.values())
        except Exception:
            return []
