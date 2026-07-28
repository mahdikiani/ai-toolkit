"""
Section Detection -- split a document into Word sections by page size.

Word is section-based, not page-based; a new Word section is required
whenever the physical page size/orientation changes partway through a
document (e.g. a landscape page holding a wide table in an
otherwise-portrait report). Rendering everything at the first page's
dimensions, as a single-section document always does, silently discards
that.

Deliberately conservative and scoped: only page-size/orientation changes
are detected here. Header/footer/margin/column changes as independent
section triggers are plan items not yet implemented -- see the plan's
Section Detection section for the full trigger list this is scoped down
from.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ast import DocumentAST, PageAST

# Two pages count as the same section only if both physical dimensions are
# within this fraction of each other -- small enough to treat DPI rounding
# as noise but still catch a genuine landscape/portrait or paper-size change.
_DIMENSION_TOLERANCE_RATIO = 0.02


@dataclass(frozen=True)
class SectionSpan:
    """
    A contiguous run of pages sharing one physical page size and column count.

    Page size of 0 means "unknown for this span" -- the renderer falls
    back to the document-wide default in that case.
    """

    start_page_number: int
    end_page_number: int
    page_width_in: float
    page_height_in: float
    column_count: int = 1


def detect_sections(ast: DocumentAST) -> list[SectionSpan]:
    """
    Split the document into contiguous same-size, same-column-count page ranges.

    Ranges are in page order and share the same physical page size (and
    therefore orientation) and column count. Always returns at least one
    span, even for an empty document.
    """
    spans: list[SectionSpan] = []
    for page in ast.pages:
        width_in, height_in = _page_size_in(page)
        column_count = max(1, page.column_count)
        continues = spans and _continues_current_span(
            spans[-1], width_in, height_in, column_count
        )
        if continues:
            last = spans[-1]
            spans[-1] = SectionSpan(
                last.start_page_number,
                page.page_number,
                last.page_width_in,
                last.page_height_in,
                last.column_count,
            )
        else:
            spans.append(
                SectionSpan(
                    page.page_number,
                    page.page_number,
                    width_in,
                    height_in,
                    column_count,
                )
            )
    return spans or [SectionSpan(1, 1, 0.0, 0.0, 1)]


def _page_size_in(page: PageAST) -> tuple[float, float]:
    if page.page_width > 0 and page.page_height > 0 and page.page_dpi > 0:
        return page.page_width / page.page_dpi, page.page_height / page.page_dpi
    return 0.0, 0.0  # unknown -- caller falls back to the document default


def _continues_current_span(
    span: SectionSpan, width_in: float, height_in: float, column_count: int
) -> bool:
    if column_count != span.column_count:
        return False
    if width_in <= 0 or height_in <= 0:
        return True  # unknown page size -- assume it continues the current section
    if span.page_width_in <= 0 or span.page_height_in <= 0:
        return False  # the span itself was unknown-sized -- a known size starts fresh
    width_tol = span.page_width_in * _DIMENSION_TOLERANCE_RATIO
    height_tol = span.page_height_in * _DIMENSION_TOLERANCE_RATIO
    return (
        abs(span.page_width_in - width_in) <= width_tol
        and abs(span.page_height_in - height_in) <= height_tol
    )
