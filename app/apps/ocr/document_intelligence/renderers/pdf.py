"""PDF renderer for the shared document intelligence AST."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path

from weasyprint import HTML

from ..ast import ASTNode, DocumentAST
from ..inline_markdown import parse_inline_segments
from ..layout import LayoutType

_FONT_DIR = Path(__file__).with_name("fonts")
_REGULAR_FONT = _FONT_DIR / "IRNazanin-Regular.ttf"
_BOLD_FONT = _FONT_DIR / "IRNazanin-Bold.ttf"

_CAPTION_TYPES = {
    LayoutType.table_caption,
    LayoutType.table_footnote,
    LayoutType.figure_caption,
}
_TEXT_TYPES = {
    LayoutType.paragraph,
    LayoutType.reference,
    LayoutType.unknown,
}
_RTL_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic (includes Persian core letters)
    (0x0700, 0x074F),  # Syriac
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0780, 0x07BF),  # Thaana
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB1D, 0xFDFF),  # Hebrew / Arabic presentation forms A
    (0xFE70, 0xFEFF),  # Arabic presentation forms B
)


def _is_rtl_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _RTL_RANGES)


def _base_dir(text: str) -> str:
    """Approximate HTML5 ``dir="auto"`` using its first-strong heuristic."""
    for ch in text or "":
        if _is_rtl_char(ch):
            return "rtl"
        if ch.isalpha():
            return "ltr"
    return "rtl"


def _font_data_uri(path: Path) -> str:
    """Return a self-contained font URL understood by WeasyPrint."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/ttf;base64,{encoded}"


def _render_inline(text: str) -> str:
    """Render already-parsed inline Markdown segments as safe HTML."""
    fragments: list[str] = []
    for segment in parse_inline_segments(text or ""):
        value = escape(segment.text)
        if segment.url:
            url = escape(segment.url, quote=True)
            value = f'<a href="{url}">{value}</a>'
        elif segment.code:
            value = f'<code class="inline-code" dir="ltr">{value}</code>'
        elif segment.math:
            value = f'<span class="inline-formula" dir="ltr">{value}</span>'
        if segment.bold:
            value = f"<strong>{value}</strong>"
        if segment.italic:
            value = f"<em>{value}</em>"
        fragments.append(value)
    return "".join(fragments)


def _render_list(node: ASTNode) -> str:
    tag = "ol" if node.ordered else "ul"
    items = "".join(
        f'<li dir="{_base_dir(child.text)}">{_render_inline(child.text)}</li>'
        for child in node.children
    )
    return f"<{tag}>{items}</{tag}>"


def _table_spans(
    node: ASTNode, row_count: int, column_count: int
) -> tuple[dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]]]:
    """Resolve valid merge anchors and cells covered by their spans."""
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    covered: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    for row_start, col_start, row_end, col_end in node.cell_merges:
        valid = (
            0 <= row_start <= row_end < row_count
            and 0 <= col_start <= col_end < column_count
            and (row_start != row_end or col_start != col_end)
        )
        if not valid:
            continue
        span_cells = {
            (row, column)
            for row in range(row_start, row_end + 1)
            for column in range(col_start, col_end + 1)
        }
        anchor = (row_start, col_start)
        if any(cell in occupied for cell in span_cells):
            continue
        anchors[anchor] = (row_end - row_start + 1, col_end - col_start + 1)
        occupied.update(span_cells)
        covered.update(span_cells - {anchor})
    return anchors, covered


def _render_table(node: ASTNode) -> str:
    """Render an AST table, including rowspan/colspan merge metadata."""
    if not node.rows:
        return ""
    column_count = max((len(row) for row in node.rows), default=0)
    if not column_count:
        return ""
    anchors, covered = _table_spans(node, len(node.rows), column_count)

    rendered_rows: list[str] = []
    for row_index, row in enumerate(node.rows):
        cell_tag = "th" if row_index == 0 else "td"
        cells: list[str] = []
        for column_index in range(column_count):
            coordinate = (row_index, column_index)
            if coordinate in covered:
                continue
            rowspan, colspan = anchors.get(coordinate, (1, 1))
            value = row[column_index] if column_index < len(row) else ""
            value = str(value)
            attributes = f' dir="{_base_dir(value)}"'
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            cells.append(f"<{cell_tag}{attributes}>{_render_inline(value)}</{cell_tag}>")
        rendered_rows.append(f"<tr>{''.join(cells)}</tr>")

    head = f"<thead>{rendered_rows[0]}</thead>"
    body = f"<tbody>{''.join(rendered_rows[1:])}</tbody>"
    return f"<table>{head}{body}</table>"


def _render_node(node: ASTNode) -> str:
    """Map one AST node type to one HTML fragment."""
    if node.type == LayoutType.title:
        return f'<h1 dir="{_base_dir(node.text)}">{_render_inline(node.text)}</h1>'
    if node.type == LayoutType.heading:
        level = max(1, min(node.level or 1, 6))
        return (
            f'<h{level} dir="{_base_dir(node.text)}">'
            f"{_render_inline(node.text)}</h{level}>"
        )
    if node.type in _TEXT_TYPES:
        return (
            f'<p dir="{_base_dir(node.text)}">{_render_inline(node.text)}</p>'
            if node.text.strip()
            else ""
        )
    if node.type in _CAPTION_TYPES:
        return (
            f'<p class="caption" dir="{_base_dir(node.text)}">'
            f"<em>{_render_inline(node.text)}</em></p>"
            if node.text.strip()
            else ""
        )
    if node.type == LayoutType.list:
        return _render_list(node)
    if node.type == LayoutType.table:
        return _render_table(node)
    if node.type == LayoutType.formula:
        formula = node.latex or node.html or node.text
        return (
            f'<div class="formula" dir="ltr">{escape(formula)}</div>'
            if formula.strip()
            else ""
        )
    if node.type == LayoutType.code:
        return f'<pre dir="ltr"><code>{escape(node.text)}</code></pre>'
    if node.type in {LayoutType.figure, LayoutType.chart}:
        description = node.caption or node.text
        return (
            f'<p class="caption" dir="{_base_dir(description)}">'
            f"<em>{_render_inline(description)}</em></p>"
            if description.strip()
            else ""
        )
    # Header, footer, and page number are deliberately absent from the plain v1 PDF.
    return ""


def _document_html(ast: DocumentAST) -> str:
    """Build the self-contained A4 HTML document passed to WeasyPrint."""
    regular_font = _font_data_uri(_REGULAR_FONT)
    bold_font = _font_data_uri(_BOLD_FONT)
    body = "".join(_render_node(node) for page in ast.pages for node in page.nodes)
    return f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <style>
    @font-face {{
      font-family: "IRNazanin";
      src: url("{regular_font}") format("truetype");
      font-style: normal;
      font-weight: 400;
    }}
    @font-face {{
      font-family: "IRNazanin";
      src: url("{bold_font}") format("truetype");
      font-style: normal;
      font-weight: 700;
    }}
    @page {{ size: A4; margin: 2cm; }}
    html {{ direction: rtl; }}
    body {{
      direction: rtl;
      font-family: "IRNazanin", "Liberation Serif", serif;
      font-size: 13pt;
      line-height: 1.65;
      color: #171717;
    }}
    h1, h2, h3, h4, h5, h6 {{
      font-weight: 700;
      line-height: 1.3;
      margin: 0.8em 0 0.35em;
      break-after: avoid;
    }}
    h1 {{ font-size: 24pt; }}
    h2 {{ font-size: 20pt; }}
    h3 {{ font-size: 17pt; }}
    h4 {{ font-size: 15pt; }}
    h5 {{ font-size: 14pt; }}
    h6 {{ font-size: 13pt; }}
    p {{ margin: 0.45em 0; text-align: right; }}
    ul, ol {{ margin: 0.4em 0; padding-right: 1.6em; padding-left: 0; }}
    li {{ margin: 0.15em 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0.8em 0;
      break-inside: avoid;
    }}
    th, td {{ border: 1px solid #777; padding: 0.35em 0.5em; text-align: right; }}
    th {{ background: #eeeeee; font-weight: 700; }}
    .caption {{ color: #444; }}
    pre, .formula {{
      direction: ltr;
      text-align: left;
      unicode-bidi: isolate;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: "Liberation Mono", monospace;
      background: #f4f4f4;
      border-right: 3px solid #888;
      padding: 0.65em 0.8em;
      margin: 0.7em 0;
    }}
    .formula {{ font-style: italic; text-align: center; }}
    .inline-code, .inline-formula {{
      direction: ltr;
      unicode-bidi: isolate;
      font-family: "Liberation Mono", monospace;
    }}
    a {{ color: inherit; text-decoration: underline; }}
  </style>
</head>
<body>{body}</body>
</html>"""


def render_pdf(ast: DocumentAST) -> BytesIO:
    """Render a ``DocumentAST`` as an A4, RTL-aware PDF buffer."""
    pdf_bytes = HTML(string=_document_html(ast)).write_pdf()
    return BytesIO(pdf_bytes)
