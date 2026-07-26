"""Document AST — internal tree model for multi-format rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


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
            node.rows = _html_table_to_rows(elem.html)

        if elem.type == LayoutType.list:
            node.children = _split_list_items(elem.text)

        nodes.append(node)

    return PageAST(
        page_number=page_number,
        nodes=nodes,
        page_width=page_width,
        page_height=page_height,
        page_dpi=page_dpi,
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


def _html_table_to_rows(html: str) -> list[list[str]]:
    """Convert a simple HTML table into a list of row cell-text lists."""
    import re

    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.DOTALL)
        if cells:
            rows.append([c.strip() for c in cells])
    return rows


def _split_list_items(text: str) -> list[ASTNode]:
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        if line:
            items.append(ASTNode(type=LayoutType.list, text=line))
    return items
