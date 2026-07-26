"""
Cross-page header/footer detection with real PAGE/NUMPAGES field recognition.

Phase 1 promoted a header/footer only when its text repeated verbatim
across pages — which can never match a footer like "صفحه 3 از 10", since
the digits differ on every page. This module normalizes digit runs before
comparing, so a footer that repeats *except for a page-number pattern*
still gets recognized as one repeating footer, with the varying digit run
identified as a real PAGE (or NUMPAGES) field rather than left as literal
OCR'd text — see acceptance criterion #6 in the plan.

Standalone ``page_number`` nodes (a bare digit block with no surrounding
footer text) are handled separately: if they form a clean 1:1 sequence with
the page number, they get promoted to a footer paragraph containing only a
PAGE field.

Deliberately conservative: a digit run is only ever treated as PAGE/NUMPAGES
when it matches on *every* occurrence, never guessed from a partial match —
per the plan's requirement that BBox/OCR text must never be silently
reinterpreted with unverified confidence.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..ast import DocumentAST
from ..layout import LayoutType

_DIGIT_RUN_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class RegionSegment:
    """Represent RegionSegment."""

    kind: str  # "text" | "page" | "numpages"
    text: str = ""  # literal text (kind="text") or cached display value


@dataclass(frozen=True)
class PromotedRegion:
    """Represent PromotedRegion."""

    segments: tuple[RegionSegment, ...]

    @property
    def plain_text(self) -> str:
        """Perform plain text."""
        return "".join(seg.text for seg in self.segments)

    @property
    def has_page_field(self) -> bool:
        """Perform has page field."""
        return any(seg.kind == "page" for seg in self.segments)


def detect_header_footer_regions(
    ast: DocumentAST,
) -> tuple[list[PromotedRegion], list[PromotedRegion]]:
    """
    Return header and footer paragraph lists.

    PromotedRegion, one per paragraph the promoted header/footer should
    contain (usually 0 or 1, occasionally 2 when a repeating footer and an
    independent page-number sequence are both present).
    """
    total_pages = len(ast.pages)
    header_region = _build_region_if_repeated(
        _collect_occurrences(ast, LayoutType.header), total_pages
    )
    footer_region = _build_region_if_repeated(
        _collect_occurrences(ast, LayoutType.footer), total_pages
    )
    page_number_region = _page_number_only_region(
        _collect_occurrences(ast, LayoutType.page_number), total_pages
    )

    headers = [header_region] if header_region else []
    footers = [footer_region] if footer_region else []
    already_has_page_field = footer_region is not None and footer_region.has_page_field
    if page_number_region and not already_has_page_field:
        footers.append(page_number_region)
    return headers, footers


def _collect_occurrences(
    ast: DocumentAST, layout_type: LayoutType
) -> list[tuple[str, int]]:
    return [
        (node.text.strip(), page.page_number)
        for page in ast.pages
        for node in page.nodes
        if node.type == layout_type and node.text.strip()
    ]


def _digit_template(text: str) -> str:
    return _DIGIT_RUN_RE.sub("\x00", text)


def _min_required(total_pages: int) -> int:
    return max(2, round(total_pages * 0.5))


def _build_region_if_repeated(
    occurrences: list[tuple[str, int]], total_pages: int
) -> PromotedRegion | None:
    if not occurrences or total_pages < 2:
        return None
    templates = [_digit_template(text) for text, _ in occurrences]
    template, count = Counter(templates).most_common(1)[0]
    if count < _min_required(total_pages):
        return None
    pairs = zip(occurrences, templates, strict=True)
    matching = [occ for occ, tmpl in pairs if tmpl == template]
    return _build_region(matching, total_pages)


def _build_region(
    occurrences: list[tuple[str, int]], total_pages: int
) -> PromotedRegion:
    occurrences = sorted(occurrences, key=lambda o: o[1])
    template_source = occurrences[0][0]
    parts = _DIGIT_RUN_RE.split(template_source)
    template_digits = _DIGIT_RUN_RE.findall(template_source)
    n_slots = len(template_digits)

    segments: list[RegionSegment] = []
    if parts[0]:
        segments.append(RegionSegment("text", parts[0]))

    for slot in range(n_slots):
        pairs: list[tuple[int, int]] = []
        for raw, page_num in occurrences:
            matches = _DIGIT_RUN_RE.findall(raw)
            if len(matches) == n_slots:
                pairs.append((int(matches[slot]), page_num))
        cached = template_digits[slot]
        if pairs and all(value == page_num for value, page_num in pairs):
            segments.append(RegionSegment("page", cached))
        elif pairs and all(value == total_pages for value, _ in pairs):
            segments.append(RegionSegment("numpages", cached))
        else:
            segments.append(RegionSegment("text", cached))
        if parts[slot + 1]:
            segments.append(RegionSegment("text", parts[slot + 1]))

    return PromotedRegion(tuple(segments))


def _page_number_only_region(
    occurrences: list[tuple[str, int]], total_pages: int
) -> PromotedRegion | None:
    if total_pages < 2:
        return None
    numeric = [
        (int(text), page_num) for text, page_num in occurrences if text.isdigit()
    ]
    if len(numeric) < _min_required(total_pages):
        return None
    if not all(value == page_num for value, page_num in numeric):
        return None  # not a clean 1:1 sequence -- don't guess at an offset
    return PromotedRegion((RegionSegment("page", str(numeric[0][0])),))
