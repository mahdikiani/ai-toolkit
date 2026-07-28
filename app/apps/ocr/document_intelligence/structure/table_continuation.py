"""
Table Continuation — merge a table split across a page boundary.

A table cut off by the layout pipeline becomes two independent AST table
nodes today, each rendered as its own separate Word table (usually with the
continuation missing a header row, and no structural link between the two
halves).

This step merges a table that is the LAST node on page N with a table that
is the FIRST node on page N+1, when they share the same column count and
both sit near the relevant page edge (the first near the bottom of page N,
the continuation near the top of page N+1) -- the only reliable, low-risk
signals available without deeper table-structure understanding (no
column-width comparison: bbox-derived widths aren't stable across a
page/DPI boundary). If the continuation's first row duplicates the
original header row, it's dropped rather than appended a second time. The
merged table is marked to repeat its header row on every printed page via
Word's own repeat-header-row mechanism (``w:tblHeader``), so a reader
never sees a page break drop the header.

Deliberately conservative: a mismatched column count, or either table not
sitting near its expected page edge, never merges -- and no attempt is
made to detect a *mid-page* split.
"""

from __future__ import annotations

from ..ast import ASTNode, DocumentAST, PageAST
from ..layout import LayoutType

# A table must reach at least this far down page N (as a fraction of page
# height) to plausibly be cut off by the page boundary, and a continuation
# must start within this fraction of the top of page N+1.
_BOTTOM_ZONE_RATIO = 0.85
_TOP_ZONE_RATIO = 0.15


def merge_table_continuations(document_ast: DocumentAST) -> DocumentAST:
    """Merge adjacent-page table continuations into single logical tables."""
    pages = [
        PageAST(
            page_number=p.page_number,
            nodes=list(p.nodes),
            page_width=p.page_width,
            page_height=p.page_height,
            page_dpi=p.page_dpi,
        )
        for p in document_ast.pages
    ]
    for i in range(len(pages) - 1):
        page, next_page = pages[i], pages[i + 1]
        if not page.nodes or not next_page.nodes:
            continue
        last, first = page.nodes[-1], next_page.nodes[0]
        if not _looks_like_continuation(last, page, first, next_page):
            continue
        page.nodes[-1] = _merge_tables(last, first)
        next_page.nodes.pop(0)
    return DocumentAST(
        title=document_ast.title, pages=pages, assets=document_ast.assets
    )


def _looks_like_continuation(
    last: ASTNode, page: PageAST, first: ASTNode, next_page: PageAST
) -> bool:
    if last.type != LayoutType.table or first.type != LayoutType.table:
        return False
    if not last.rows or not first.rows:
        return False
    if max(len(r) for r in last.rows) != max(len(r) for r in first.rows):
        return False
    if page.page_height and last.bbox[3] < page.page_height * _BOTTOM_ZONE_RATIO:
        return False
    next_top_limit = next_page.page_height * _TOP_ZONE_RATIO
    return not (next_page.page_height and first.bbox[1] > next_top_limit)


def _merge_tables(first: ASTNode, continuation: ASTNode) -> ASTNode:
    header_repeats = _rows_equal(first.rows[0], continuation.rows[0])
    drop = 1 if header_repeats else 0
    continuation_rows = continuation.rows[drop:]
    row_offset = len(first.rows)

    merged_merges = list(first.cell_merges)
    for r1, c1, r2, c2 in continuation.cell_merges:
        if r1 < drop:
            continue  # the merge lived entirely inside the dropped duplicate header row
        merged_merges.append((r1 - drop + row_offset, c1, r2 - drop + row_offset, c2))

    x1, y1, x2, _y2 = first.bbox
    _x1, _y1, _x2, y2 = continuation.bbox
    return ASTNode(
        type=LayoutType.table,
        rows=first.rows + continuation_rows,
        cell_merges=merged_merges,
        repeat_header_row=True,
        bbox=(x1, y1, x2, y2),
        page_number=first.page_number,
        confidence=min(first.confidence, continuation.confidence),
    )


def _rows_equal(row_a: list[str], row_b: list[str]) -> bool:
    return [c.strip().lower() for c in row_a] == [c.strip().lower() for c in row_b]
