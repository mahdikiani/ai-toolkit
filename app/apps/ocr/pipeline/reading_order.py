"""Reading order reconstruction for Persian/RTL documents."""

import logging

from .layout_detector import ElementType, LayoutBox

logger = logging.getLogger(__name__)


class ReadingOrderResolver:
    """Resolve reading order for layout elements on a page."""

    def resolve(self, elements: list[LayoutBox]) -> list[LayoutBox]:
        """Assign reading_order to each element based on layout analysis."""
        if not elements:
            return elements

        page_map: dict[int, list[LayoutBox]] = {}
        for elem in elements:
            page_map.setdefault(elem.page_number, []).append(elem)

        ordered: list[LayoutBox] = []
        for page_elements in page_map.values():
            self._resolve_page(page_elements)
            ordered.extend(page_elements)
        elements[:] = ordered

        return elements

    def _resolve_page(self, elements: list[LayoutBox]) -> None:
        """Resolve order for a single page's elements."""
        page_width = max(element.x2 for element in elements)
        full_width = self._detect_full_width(elements, page_width)
        columns = self._detect_columns([e for e in elements if e not in full_width])

        if not columns:
            elements.sort(key=lambda element: (element.y1, -element.x1))
            return

        # Sort columns right-to-left for Persian
        columns.sort(key=lambda c: sum(e.x1 for e in c) / max(len(c), 1), reverse=True)
        ordered_columns: list[LayoutBox] = []
        for col in columns:
            col.sort(key=lambda e: e.y1)
            ordered_columns.extend(col)

        # Full-width headings and tables retain their vertical anchor.
        elements[:] = sorted(
            [*full_width, *ordered_columns],
            key=lambda element: (
                element.y1,
                0 if element in full_width else 1,
                -element.x1,
            ),
        )

    @staticmethod
    def _detect_full_width(
        elements: list[LayoutBox], page_width: float
    ) -> list[LayoutBox]:
        full: list[LayoutBox] = []
        for e in elements:
            w = e.x2 - e.x1
            if w >= page_width * 0.85 or e.type in (
                ElementType.title,
                ElementType.section_heading,
                ElementType.table,
            ):
                full.append(e)
        return full

    @staticmethod
    def _detect_columns(elements: list[LayoutBox]) -> list[list[LayoutBox]]:
        if len(elements) < 3:
            return []
        centers = [(e.x1 + e.x2) / 2 for e in elements]
        span = max(centers) - min(centers)
        if span < 100:
            return []
        try:
            from sklearn.cluster import KMeans

            x_centers = [[(e.x1 + e.x2) / 2] for e in elements]
            kmeans = KMeans(n_clusters=min(3, len(elements)), n_init=1, random_state=0)
            labels = kmeans.fit_predict(x_centers)
            cols: dict[int, list[LayoutBox]] = {}
            for elem, label in zip(elements, labels, strict=False):
                cols.setdefault(int(label), []).append(elem)
            return list(cols.values())
        except Exception:
            return []
