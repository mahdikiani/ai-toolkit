"""
Document Structure Understanding — cross-page/cross-node analysis that runs.

between the raw AST and the renderers (Phase 2 of the Semantic DOCX plan).
"""

from .header_footer import PromotedRegion, RegionSegment, detect_header_footer_regions

__all__ = ["PromotedRegion", "RegionSegment", "detect_header_footer_regions"]
