"""
Document Structure Understanding — cross-page/cross-node analysis that runs.

between the raw AST and the renderers (Phase 2 of the Semantic DOCX plan).
"""

from .header_footer import (
    HeaderFooterPlan,
    PromotedRegion,
    RegionSegment,
    detect_header_footer_regions,
)
from .paragraph_merge import merge_paragraphs
from .sections import SectionSpan, detect_sections
from .table_continuation import merge_table_continuations

__all__ = [
    "HeaderFooterPlan",
    "PromotedRegion",
    "RegionSegment",
    "SectionSpan",
    "detect_header_footer_regions",
    "detect_sections",
    "merge_paragraphs",
    "merge_table_continuations",
]
