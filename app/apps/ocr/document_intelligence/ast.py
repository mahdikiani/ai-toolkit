"""Document AST — internal tree model for multi-format rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .elements import ProcessedElement
from .layout import LayoutElement, LayoutType


@dataclass
class ASTNode:
    """A single node in the document AST (paragraph, heading, table, ...)."""

    type: LayoutType
    text: str = ""
    html: str = ""
    latex: str = ""
    caption: str = ""
    description: str = ""
    chart_data: dict | None = None
    asset_path: str = ""
    children: list[ASTNode] = field(default_factory=list)
    page_number: int = 1
    level: int = 0  # for headings
    rows: list[list[str]] = field(default_factory=list)
    # Merge spans within `rows`, as (row_start, col_start, row_end, col_end)
    # 0-indexed inclusive ranges -- reconstructed from the source table's
    # rowspan/colspan attributes so the renderer can call cell.merge().
    cell_merges: list[tuple[int, int, int, int]] = field(default_factory=list)
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ordered: bool = False  # for list items: numbered ("1.") vs bulleted
    repeat_header_row: bool = False  # for tables: repeat row 0 on every printed page


@dataclass
class PageAST:
    """AST nodes for a single rendered page, plus its pixel dimensions."""

    page_number: int
    nodes: list[ASTNode]
    # Pixel dimensions at the DPI the page was rendered at (same coordinate
    # space as ASTNode.bbox / LayoutElement.bbox, so ratios between the two
    # are unit-consistent). Divide by page_dpi to get inches for page setup.
    page_width: float = 0.0
    page_height: float = 0.0
    page_dpi: float = 300.0
    column_count: int = 1


@dataclass
class DocumentAST:
    """The full document AST: title, pages, and shared asset map."""

    title: str = ""
    pages: list[PageAST] = field(default_factory=list)
    assets: dict[str, str] = field(default_factory=dict)  # id -> path


def build_ast(
    processed: list[ProcessedElement],
    ordered: list[LayoutElement],
    page_number: int,
    page_width: float = 0.0,
    page_height: float = 0.0,
    page_dpi: float = 300.0,
    column_count: int = 1,
) -> PageAST:
    """Convert processed elements + reading order to a PageAST."""
    order_map = {e.id: i for i, e in enumerate(ordered)}
    sorted_elems = sorted(processed, key=lambda p: order_map.get(p.id, 9999))

    nodes: list[ASTNode] = []
    for elem in sorted_elems:
        node = ASTNode(
            type=elem.type,
            text=elem.text,
            html=elem.html,
            latex=elem.latex,
            caption=elem.caption or "",
            description=elem.description or "",
            chart_data=elem.chart_data,
            asset_path=elem.asset_path,
            page_number=elem.page_number,
            level=_heading_level(elem.type),
            confidence=elem.confidence,
            bbox=elem.bbox,
        )

        if elem.type == LayoutType.table and elem.html:
            node.rows, node.cell_merges = _parse_html_table(elem.html)

        if elem.type == LayoutType.list:
            node.children = _split_list_items(elem.text)

        nodes.append(node)

    return PageAST(
        page_number=page_number,
        nodes=nodes,
        page_width=page_width,
        page_height=page_height,
        page_dpi=page_dpi,
        column_count=column_count,
    )


def build_document_ast(
    pages: list[PageAST], asset_map: dict[str, str] | None = None
) -> DocumentAST:
    """Combine per-page ASTs into a full DocumentAST and pull out a title."""
    title = ""
    for page in pages:
        for node in page.nodes:
            if node.type == LayoutType.title and node.text.strip():
                title = node.text.strip()
                break
        if title:
            break
    return DocumentAST(title=title, pages=pages, assets=dict(asset_map or {}))


def _heading_level(t: LayoutType) -> int:
    if t == LayoutType.title:
        return 1
    if t == LayoutType.heading:
        return 2
    return 0


class _TableHTMLParser(HTMLParser):
    """
    Extract row/cell structure (including rowspan/colspan) from a VLM table.

    Uses a real parser instead of a regex that strips all tag attributes
    -- a regex approach would silently destroy rowspan/colspan before the
    AST was even built, so no renderer could ever recover merged cells
    regardless of its own capabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self.raw_rows: list[list[tuple[str, int, int]]] = []
        self.current_row: list[tuple[str, int, int]] | None = None
        self._cell_parts: list[str] = []
        self._cell_attrs: dict[str, str | None] = {}
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_parts = []
            self._cell_attrs = dict(attrs)
        elif tag == "br" and self._in_cell:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            self._close_cell()
        elif tag == "tr" and self.current_row is not None:
            self.raw_rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def _close_cell(self) -> None:
        text = "".join(self._cell_parts).strip()
        rowspan = _safe_int(self._cell_attrs.get("rowspan"), 1)
        colspan = _safe_int(self._cell_attrs.get("colspan"), 1)
        if self.current_row is None:
            self.current_row = []
        self.current_row.append((text, rowspan, colspan))
        self._in_cell = False

    def finish(self) -> list[list[tuple[str, int, int]]]:
        """Flush a trailing row left open by a missing closing ``</tr>``."""
        if self.current_row:
            self.raw_rows.append(self.current_row)
        return self.raw_rows


def _safe_int(value: str | None, default: int) -> int:
    try:
        n = int(value) if value else default
    except ValueError:
        return default
    else:
        return n if n > 0 else default


def _place_cell(
    grid: list[list[str]],
    occupied: list[set[int]],
    merges: list[tuple[int, int, int, int]],
    n_rows: int,
    r: int,
    c: int,
    text: str,
    rowspan: int,
    colspan: int,
) -> int:
    """
    Place one cell into ``grid`` at row ``r``, starting at the first free column >= c.

    Returns the next unoccupied column to try in this row.
    """
    while c in occupied[r]:
        c += 1
    row_end = min(r + rowspan, n_rows) - 1
    col_end = c + colspan - 1
    for rr in range(r, row_end + 1):
        for cc in range(c, col_end + 1):
            occupied[rr].add(cc)
    if row_end > r or col_end > c:
        merges.append((r, c, row_end, col_end))
    while len(grid[r]) <= col_end:
        grid[r].append("")
    grid[r][c] = text
    return col_end + 1


def _parse_html_table(
    html: str,
) -> tuple[list[list[str]], list[tuple[int, int, int, int]]]:
    """
    Parse an HTML table into a rectangular grid plus a list of merge spans.

    Correctly accounts for rowspan/colspan (standard HTML table
    grid-placement algorithm: cells fill the first unoccupied column in
    their row, occupying subsequent rows/columns per their span).
    """
    parser = _TableHTMLParser()
    parser.feed(html)
    raw_rows = parser.finish()
    n_rows = len(raw_rows)
    if not n_rows:
        return [], []

    grid: list[list[str]] = [[] for _ in range(n_rows)]
    occupied: list[set[int]] = [set() for _ in range(n_rows)]
    merges: list[tuple[int, int, int, int]] = []

    for r, row_cells in enumerate(raw_rows):
        c = 0
        for text, rowspan, colspan in row_cells:
            c = _place_cell(
                grid, occupied, merges, n_rows, r, c, text, rowspan, colspan
            )

    col_count = max((len(row) for row in grid), default=0)
    for row in grid:
        while len(row) < col_count:
            row.append("")
    return grid, merges


_BULLET_MARKER_RE = re.compile(r"^\s*[-•●▪‣◦*]\s+")
_ORDERED_MARKER_RE = re.compile(r"^\s*(?:\d+|[۰-۹]+|[a-zA-Z])[.)]\s+")  # ruff: ignore[ambiguous-unicode-character-string]


def _split_list_items(text: str) -> list[ASTNode]:
    """
    Split raw list text into one child ASTNode per item.

    Strips the OCR'd bullet/number marker (so the renderer's own "List
    Bullet"/"List Number" style glyph isn't duplicated alongside a
    leftover "• " or "1. " in the text) and records whether the item was
    numbered, so the renderer can pick a real ordered-list style instead
    of always bulleting.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    items = []
    for line in lines:
        ordered = bool(_ORDERED_MARKER_RE.match(line))
        marker_re = _ORDERED_MARKER_RE if ordered else _BULLET_MARKER_RE
        stripped = marker_re.sub("", line, count=1).strip()
        items.append(
            ASTNode(type=LayoutType.list, text=stripped or line, ordered=ordered)
        )
    return items
