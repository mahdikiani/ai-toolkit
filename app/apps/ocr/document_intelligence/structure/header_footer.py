"""
Cross-page header/footer detection: position-aware, multi-slot, with fields.

Real PAGE/NUMPAGES field recognition and Different-First-Page support.
Text-repetition alone promoted only a single "most common" header/footer
per document, in effect discarding a second independent repeating element
(e.g. a logo box above a separate banner line, or a Different First Page
title) since only the single most-frequent template ever won. This module
also groups occurrences by their normalized vertical position on the page
("slot") -- each slot is checked for repetition independently, so multiple
simultaneously-recurring header/footer paragraphs are all promoted, not
just whichever one happens to be most frequent overall.

Digit runs are normalized before comparing text, so a footer like "صفحه 3
از 10" (different digits every page) is still recognized as one repeating
pattern rather than never matching at all -- see acceptance criterion #6
in the plan. Standalone ``page_number`` nodes (a bare digit block with no
surrounding footer text) are handled separately: if they form a clean 1:1
sequence with the page number, they get promoted to a footer paragraph
containing only a PAGE field.

Deliberately conservative throughout: a digit run is only ever treated as
PAGE/NUMPAGES when it matches on *every* occurrence, and "different first
page" is only inferred when pages 2..N have an independently-verified
repeating pattern that page 1 demonstrably doesn't match -- never guessed
from a partial or ambiguous signal.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..ast import DocumentAST
from ..layout import LayoutType

_DIGIT_RUN_RE = re.compile(r"\d+")

# Occurrences within this fraction of page height of each other are treated
# as the same recurring header/footer element (the same "slot"), so two
# genuinely distinct elements (e.g. a logo line and a banner line below it)
# are detected and promoted independently rather than only one winning.
_SLOT_BUCKET_RATIO = 0.08


@dataclass(frozen=True)
class RegionSegment:
    """
    One piece of a promoted header/footer paragraph.

    Either literal text, or a real PAGE/NUMPAGES field (``text`` then
    holds the cached display value).
    """

    kind: str  # "text" | "page" | "numpages"
    text: str = ""


@dataclass(frozen=True)
class PromotedRegion:
    """One paragraph's worth of a promoted header/footer."""

    segments: tuple[RegionSegment, ...]

    @property
    def plain_text(self) -> str:
        """Flattened text of all segments (fields shown as their cached value)."""
        return "".join(seg.text for seg in self.segments)

    @property
    def has_page_field(self) -> bool:
        """Whether any segment is a real PAGE field."""
        return any(seg.kind == "page" for seg in self.segments)


@dataclass(frozen=True)
class HeaderFooterPlan:
    """
    Everything needed to write a section's header or footer.

    ``regions`` are the regular (all-pages, or all-but-first-page when
    ``different_first_page``) paragraphs; ``first_page_regions`` is --
    only when genuinely detected -- a distinct set of paragraphs for page
    1 alone.
    """

    regions: tuple[PromotedRegion, ...] = field(default_factory=tuple)
    first_page_regions: tuple[PromotedRegion, ...] | None = None

    @property
    def promoted(self) -> bool:
        """Whether any regular header/footer region was detected."""
        return bool(self.regions)

    @property
    def different_first_page(self) -> bool:
        """Whether a genuine Different First Page variant was detected."""
        return self.first_page_regions is not None


@dataclass(frozen=True)
class _Occurrence:
    text: str
    page_number: int
    rel_y: float  # bbox top / page height -- for position-slot grouping


def detect_header_footer_regions(
    ast: DocumentAST,
) -> tuple[HeaderFooterPlan, HeaderFooterPlan]:
    """Detect the header plan and footer plan for the whole document."""
    total_pages = len(ast.pages)
    header_occ = _collect_occurrences(ast, LayoutType.header)
    footer_occ = _collect_occurrences(ast, LayoutType.footer)
    header_plan = _detect_plan(header_occ, total_pages)
    footer_plan = _detect_plan(footer_occ, total_pages)

    page_number_occ = _collect_occurrences(ast, LayoutType.page_number)
    page_number_region = _page_number_only_region(
        [(o.text, o.page_number) for o in page_number_occ], total_pages
    )
    already_has_page_field = any(r.has_page_field for r in footer_plan.regions)
    if page_number_region and not already_has_page_field:
        footer_plan = HeaderFooterPlan(
            regions=(*footer_plan.regions, page_number_region),
            first_page_regions=footer_plan.first_page_regions,
        )

    return header_plan, footer_plan


def _collect_occurrences(
    ast: DocumentAST, layout_type: LayoutType
) -> list[_Occurrence]:
    result: list[_Occurrence] = []
    for page in ast.pages:
        height = page.page_height or 0.0
        for node in page.nodes:
            if node.type != layout_type or not node.text.strip():
                continue
            rel_y = (node.bbox[1] / height) if height else 0.0
            result.append(_Occurrence(node.text.strip(), page.page_number, rel_y))
    return result


def _slot_key(rel_y: float) -> int:
    return round(rel_y / _SLOT_BUCKET_RATIO)


def _digit_template(text: str) -> str:
    return _DIGIT_RUN_RE.sub("\x00", text)


def _min_required(page_count: int) -> int:
    return max(2, round(page_count * 0.5))


def _detect_plan(occurrences: list[_Occurrence], total_pages: int) -> HeaderFooterPlan:
    if not occurrences or total_pages < 2:
        return HeaderFooterPlan()
    regular_regions = _promote_all_slots(occurrences, total_pages)
    first_page_regions = _detect_first_page_variant(
        occurrences, total_pages, regular_regions
    )
    return HeaderFooterPlan(
        regions=tuple(regular_regions), first_page_regions=first_page_regions
    )


def _promote_all_slots(
    occurrences: list[_Occurrence], total_pages: int
) -> list[PromotedRegion]:
    """
    Promote every vertical slot that clears the repetition bar.

    Occurrences are grouped by vertical slot and independently promoted,
    in top-to-bottom order.
    """
    slots: dict[int, list[_Occurrence]] = {}
    for occ in occurrences:
        slots.setdefault(_slot_key(occ.rel_y), []).append(occ)

    positioned: list[tuple[float, PromotedRegion]] = []
    for occs in slots.values():
        pairs = [(o.text, o.page_number) for o in occs]
        region = _build_region_if_repeated(pairs, total_pages)
        if region is not None:
            mean_y = sum(o.rel_y for o in occs) / len(occs)
            positioned.append((mean_y, region))
    positioned.sort(key=lambda item: item[0])
    return [region for _, region in positioned]


def _detect_first_page_variant(
    occurrences: list[_Occurrence],
    total_pages: int,
    regular_regions: list[PromotedRegion],
) -> tuple[PromotedRegion, ...] | None:
    """
    Detect a genuine Different First Page variant, or return None.

    Page 1 counts as a genuine "different first page" only when pages
    2..N already establish a verified regular pattern (``regular_regions``)
    *and* page 1's own header/footer text doesn't match that pattern.
    Without an established regular pattern there is nothing for page 1 to
    differ from, so no variant is reported (never invented from a single
    page alone).
    """
    if not regular_regions:
        return None
    page1_occs = [o for o in occurrences if o.page_number == 1]
    if not page1_occs:
        return None

    regular_templates = {_digit_template(r.plain_text) for r in regular_regions}
    page1_templates = {_digit_template(o.text) for o in page1_occs}
    if page1_templates & regular_templates:
        return None  # page 1 already matches the regular pattern

    slots: dict[int, list[_Occurrence]] = {}
    for occ in page1_occs:
        slots.setdefault(_slot_key(occ.rel_y), []).append(occ)

    positioned: list[tuple[float, PromotedRegion]] = []
    for occs in slots.values():
        region = _build_region([(o.text, o.page_number) for o in occs], total_pages)
        mean_y = sum(o.rel_y for o in occs) / len(occs)
        positioned.append((mean_y, region))
    positioned.sort(key=lambda item: item[0])
    return tuple(region for _, region in positioned)


def _build_region_if_repeated(
    occurrences: list[tuple[str, int]], total_pages: int
) -> PromotedRegion | None:
    if not occurrences:
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
