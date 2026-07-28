"""Markdown Renderer — from Document AST to standard Markdown."""

from __future__ import annotations

from ..ast import DocumentAST
from ..layout import LayoutType


def render_markdown(ast: DocumentAST) -> str:
    """
    Convert DocumentAST to clean Markdown string.

    ``ast.title`` is not re-rendered here: the title element already appears
    in its natural reading-order position within ``ast.pages`` (as a ``# ``
    heading). It is used instead for document metadata (see renderers/docx.py).
    """
    lines: list[str] = []
    for i, page in enumerate(ast.pages):
        if i > 0:
            lines.append("")
            lines.append("---")
            lines.append("")
        for node in page.nodes:
            text = _render_node(node, ast.assets)
            if text:
                lines.append(text)
    return "\n".join(lines) + "\n"


def _render_figure_or_chart(node: object, assets: dict[str, str]) -> str | None:
    asset_rel = assets.get(node.asset_path, node.asset_path) if node.asset_path else ""
    md = ""
    if asset_rel:
        md += f"![{node.caption}]({asset_rel})\n"
    if node.caption:
        md += f"*{node.caption}*"
    if node.description and node.description != node.caption:
        md += f"\n{node.description}"
    return md.strip() or None


def _render_list(node: object) -> str:
    parts = (
        [f"- {c.text}" for c in node.children] if node.children else [f"- {node.text}"]
    )
    return "\n".join(parts)


def _render_simple_node(node: object) -> str | None:
    if node.type == LayoutType.title:
        return f"# {node.text}"
    if node.type == LayoutType.heading:
        return f"## {node.text}"
    if node.type in (LayoutType.header, LayoutType.footer):
        return f"*{node.text}*"
    if node.type == LayoutType.paragraph:
        return node.text
    if node.type == LayoutType.reference:
        return f"> {node.text}"
    if node.type == LayoutType.formula:
        return f"$$\n{node.latex}\n$$"
    if node.type == LayoutType.code:
        return f"```\n{node.text}\n```"
    return None


def _render_node(node: object, assets: dict[str, str]) -> str | None:
    simple = _render_simple_node(node)
    if simple is not None:
        return simple
    if node.type == LayoutType.list:
        return _render_list(node)
    if node.type == LayoutType.table:
        return _render_table(node)
    if node.type in (LayoutType.figure, LayoutType.chart):
        return _render_figure_or_chart(node, assets)
    return node.text or None


def rewrite_asset_links(markdown: str, url_map: dict[str, str]) -> str:
    """
    Replace local asset paths in rendered Markdown with public URLs.

    ``url_map`` maps the local relative path (e.g. ``assets/image_001.png``,
    as produced by AssetManager) to a publicly reachable URL after upload.
    """
    for local_path, url in url_map.items():
        markdown = markdown.replace(f"]({local_path})", f"]({url})")
    return markdown


def _render_table(node: object) -> str:
    if not node.rows:
        return ""
    lines: list[str] = []
    header = node.rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in node.rows[1:])
    return "\n".join(lines)
