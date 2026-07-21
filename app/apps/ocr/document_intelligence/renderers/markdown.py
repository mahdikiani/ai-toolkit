"""Markdown Renderer — from Document AST to standard Markdown."""

from __future__ import annotations

from pathlib import Path

from .ast import DocumentAST, PageAST
from .layout import LayoutType


def render_markdown(ast: DocumentAST) -> str:
    """Convert DocumentAST to clean Markdown string."""
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


def _render_node(node, assets: dict[str, str]) -> str | None:
    if node.type == LayoutType.title:
        return f"# {node.text}"
    if node.type == LayoutType.heading:
        return f"## {node.text}"
    if node.type == LayoutType.header or node.type == LayoutType.footer:
        return f"*{node.text}*"
    if node.type == LayoutType.paragraph:
        return node.text
    if node.type == LayoutType.reference:
        return f"> {node.text}"
    if node.type == LayoutType.list:
        parts = [f"- {c.text}" for c in node.children] if node.children else [f"- {node.text}"]
        return "\n".join(parts)
    if node.type == LayoutType.table:
        return _render_table(node)
    if node.type == LayoutType.formula:
        return f"$$\n{node.latex}\n$$"
    if node.type == LayoutType.figure or node.type == LayoutType.chart:
        asset_rel = assets.get(node.asset_path, node.asset_path) if node.asset_path else ""
        md = ""
        if asset_rel:
            md += f"![{node.caption}]({asset_rel})\n"
        if node.caption:
            md += f"*{node.caption}*"
        if node.description and node.description != node.caption:
            md += f"\n{node.description}"
        return md.strip() or None
    if node.type == LayoutType.code:
        return f"```\n{node.text}\n```"
    return node.text or None


def _render_table(node) -> str:
    if not node.rows:
        return ""
    lines: list[str] = []
    header = node.rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in node.rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
