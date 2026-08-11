"""Conversion strategy registry — (source_format, target_format) → callable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from apps.artifacts.enums import ArtifactFormat

ConversionFn = Callable[..., bytes]


@dataclass(frozen=True, slots=True)
class ConversionEdge:
    """A registered conversion edge in the Artifact format graph."""

    source_format: ArtifactFormat
    target_format: ArtifactFormat
    strategy: ConversionFn
    name: str


_REGISTRY: dict[tuple[ArtifactFormat, ArtifactFormat], ConversionEdge] = {}


def register(
    source_format: ArtifactFormat,
    target_format: ArtifactFormat,
    strategy: ConversionFn,
    *,
    name: str | None = None,
) -> ConversionFn:
    """Register a conversion strategy for an explicit format edge."""
    key = (source_format, target_format)
    edge = ConversionEdge(
        source_format=source_format,
        target_format=target_format,
        strategy=strategy,
        name=name or f"{source_format.value}->{target_format.value}",
    )
    _REGISTRY[key] = edge
    return strategy


def get_edge(
    source_format: ArtifactFormat,
    target_format: ArtifactFormat,
) -> ConversionEdge | None:
    """Return the registered edge for a direct conversion, if any."""
    return _REGISTRY.get((source_format, target_format))


def list_edges() -> list[ConversionEdge]:
    """List all registered conversion edges."""
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """Clear all edges (test helper)."""
    _REGISTRY.clear()


def ensure_builtin_strategies() -> None:
    """
    Ensure built-in conversion edges are registered.

    Safe to call repeatedly (including after ``clear_registry``): re-imports
    alone are not enough because Python caches modules after the first import.
    """
    from .strategies.markdown_docx import markdown_bytes_to_docx
    from .strategies.markdown_pdf import markdown_bytes_to_pdf

    register(
        ArtifactFormat.markdown,
        ArtifactFormat.docx,
        markdown_bytes_to_docx,
        name="markdown_docx",
    )
    register(
        ArtifactFormat.markdown,
        ArtifactFormat.pdf,
        markdown_bytes_to_pdf,
        name="markdown_pdf",
    )
