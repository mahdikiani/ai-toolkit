"""Plain-text Markdown -> DocumentAST parser (no OCR/layout involved).

Turns LLM-generated Markdown (Promptic results, meeting minutes, chat
summaries, ...) into the same DocumentAST the OCR pipeline builds, so it can
be rendered through the exact same high-quality DOCX renderer (real tables,
OMML formulas, heading styles, inline bold/italic) instead of a separate,
cruder implementation living elsewhere.
"""

from __future__ import annotations

import re

from .ast import ASTNode, DocumentAST, PageAST
from .layout import LayoutType

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_FORMULA_BLOCK_RE = re.compile(r"^\s*\$\$\s*$")


def parse_markdown(text: str, title: str = "") -> DocumentAST:
    """Parse a raw Markdown string into a single-page DocumentAST."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    nodes: list[ASTNode] = []
    quote_buffer: list[str] = []
    i = 0

    def flush_quote() -> None:
        if quote_buffer:
            nodes.append(ASTNode(type=LayoutType.reference, text="\n".join(quote_buffer)))
            quote_buffer.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_quote()
            i += 1
            continue

        if _FENCE_RE.match(stripped):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not _FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            i += 1
            nodes.append(ASTNode(type=LayoutType.code, text="\n".join(code_lines)))
            continue

        if _FORMULA_BLOCK_RE.match(stripped):
            i += 1
            formula_lines: list[str] = []
            while i < len(lines) and not _FORMULA_BLOCK_RE.match(lines[i].strip()):
                formula_lines.append(lines[i])
                i += 1
            i += 1
            nodes.append(ASTNode(type=LayoutType.formula, latex="\n".join(formula_lines).strip()))
            continue

        quote_match = _QUOTE_RE.match(line)
        if quote_match:
            quote_buffer.append(quote_match.group(1))
            i += 1
            continue
        flush_quote()

        if _HR_RE.match(stripped):
            i += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            node_type = LayoutType.title if level == 1 else LayoutType.heading
            nodes.append(
                ASTNode(type=node_type, text=heading_match.group(2).strip(), level=level)
            )
            i += 1
            continue

        if (
            _TABLE_ROW_RE.match(stripped)
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            rows = [_parse_table_row(stripped)]
            i += 2
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                rows.append(_parse_table_row(lines[i].strip()))
                i += 1
            nodes.append(ASTNode(type=LayoutType.table, rows=rows))
            continue

        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            children: list[ASTNode] = []
            while i < len(lines):
                match = _BULLET_RE.match(lines[i]) or _NUMBERED_RE.match(lines[i])
                if not match:
                    break
                children.append(ASTNode(type=LayoutType.list, text=match.group(1).strip()))
                i += 1
            nodes.append(ASTNode(type=LayoutType.list, children=children))
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_special_line(lines[i]):
            para_lines.append(lines[i].strip())
            i += 1
        nodes.append(ASTNode(type=LayoutType.paragraph, text=" ".join(para_lines)))

    flush_quote()
    resolved_title = title or next(
        (n.text for n in nodes if n.type == LayoutType.title), ""
    )
    return DocumentAST(title=resolved_title, pages=[PageAST(page_number=1, nodes=nodes)])


def _parse_table_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _is_special_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _HEADING_RE.match(stripped)
        or _BULLET_RE.match(line)
        or _NUMBERED_RE.match(line)
        or _QUOTE_RE.match(line)
        or _HR_RE.match(stripped)
        or _FENCE_RE.match(stripped)
        or _FORMULA_BLOCK_RE.match(stripped)
        or _TABLE_ROW_RE.match(stripped)
    )
