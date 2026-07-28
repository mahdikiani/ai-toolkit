"""
Paragraph Reconstruction.

Merges adjacent AST paragraph blocks that are
really one continuous paragraph split by layout detection, instead of
handing every detected block to the renderer as an independent Paragraph.

PP-DocLayout already detects paragraph-sized blocks, not individual lines,
so most blocks already correspond 1:1 to a real paragraph and need no
merging. This step only catches the narrower case where a single paragraph
was split into two adjacent blocks — and is deliberately conservative: it
merges only when several independent signals agree, never on vertical
proximity alone (see the plan's Paragraph Reconstruction section).

Scoped to same-page merging only; a paragraph continuing across a page
break is a separate, harder problem (interacts with where the page break
itself gets emitted) left for a follow-up.
"""

from __future__ import annotations

import re

from ..ast import ASTNode, DocumentAST, PageAST
from ..layout import LayoutType

_SENTENCE_END_RE = re.compile(r"[.!؟?][\"'”’)\]]*\s*$")  # ruff: ignore[ambiguous-unicode-character-string]
_LIST_MARKER_RE = re.compile(r"^\s*([-•●▪]|\d+[.)]|[۰-۹]+[.)])\s+")  # ruff: ignore[ambiguous-unicode-character-string]

# How close two blocks' left edges must be (relative to page width) to
# count as the same text column.
COLUMN_TOLERANCE_RATIO = 0.03
# How similar two blocks' widths must be (relative to the wider one) --
# guards against merging a full-width paragraph into an unrelated narrow
# block (e.g. a caption) that merely happens to align on the left edge.
WIDTH_SIMILARITY_RATIO = 0.3
# Vertical gap between blocks, as a fraction of the shorter block's own
# height, below which the gap looks like ordinary line spacing inside one
# paragraph rather than the space *between* two separate paragraphs.
MAX_GAP_TO_HEIGHT_RATIO = 0.5


def merge_paragraphs(document_ast: DocumentAST) -> DocumentAST:
    """
    Merge adjacent same-page paragraph blocks where they look like one.

    Return a new DocumentAST with adjacent same-page paragraph blocks
    merged where they look like one paragraph split by layout detection.
    """
    return DocumentAST(
        title=document_ast.title,
        pages=[_merge_page(page) for page in document_ast.pages],
        assets=document_ast.assets,
    )


def _merge_page(page: PageAST) -> PageAST:
    merged: list[ASTNode] = []
    for node in page.nodes:
        if (
            merged
            and merged[-1].type == LayoutType.paragraph
            and node.type == LayoutType.paragraph
            and _should_merge(merged[-1], node, page.page_width)
        ):
            merged[-1] = _merge_nodes(merged[-1], node)
        else:
            merged.append(node)
    return PageAST(
        page_number=page.page_number,
        nodes=merged,
        page_width=page.page_width,
        page_height=page.page_height,
        page_dpi=page.page_dpi,
    )


def _should_merge(prev: ASTNode, nxt: ASTNode, page_width: float) -> bool:
    if not prev.text.strip() or not nxt.text.strip():
        return False
    if _SENTENCE_END_RE.search(prev.text.strip()):
        return False
    if _LIST_MARKER_RE.match(nxt.text.strip()):
        return False

    empty_bbox = (0.0, 0.0, 0.0, 0.0)
    if prev.bbox == empty_bbox or nxt.bbox == empty_bbox:
        return False

    px1, py1, px2, py2 = prev.bbox
    nx1, ny1, nx2, ny2 = nxt.bbox
    if ny1 < py2:  # nxt isn't actually below prev -- don't guess at overlap
        return False

    ref_width = page_width or max(px2 - px1, nx2 - nx1, 1.0)
    if abs(px1 - nx1) > ref_width * COLUMN_TOLERANCE_RATIO:
        return False

    prev_width = max(1.0, px2 - px1)
    next_width = max(1.0, nx2 - nx1)
    wider = max(prev_width, next_width)
    if abs(prev_width - next_width) / wider > WIDTH_SIMILARITY_RATIO:
        return False

    prev_height = max(1.0, py2 - py1)
    next_height = max(1.0, ny2 - ny1)
    gap = ny1 - py2
    return gap <= min(prev_height, next_height) * MAX_GAP_TO_HEIGHT_RATIO


def _merge_nodes(prev: ASTNode, nxt: ASTNode) -> ASTNode:
    separator = "" if prev.text.rstrip().endswith("-") else " "
    merged_text = f"{prev.text.rstrip()}{separator}{nxt.text.lstrip()}"
    px1, py1, px2, _py2 = prev.bbox
    nx1, _ny1, nx2, ny2 = nxt.bbox
    merged_bbox = (min(px1, nx1), py1, max(px2, nx2), ny2)
    return ASTNode(
        type=LayoutType.paragraph,
        text=merged_text,
        confidence=min(prev.confidence, nxt.confidence),
        bbox=merged_bbox,
        page_number=prev.page_number,
    )
