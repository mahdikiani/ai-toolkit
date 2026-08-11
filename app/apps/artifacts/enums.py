"""Artifact format enumeration."""

from enum import StrEnum


class ArtifactFormat(StrEnum):
    """Supported Artifact content formats."""

    markdown = "markdown"
    docx = "docx"
    pdf = "pdf"
    html = "html"


MIME_BY_FORMAT: dict[ArtifactFormat, str] = {
    ArtifactFormat.markdown: "text/markdown",
    ArtifactFormat.docx: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ArtifactFormat.pdf: "application/pdf",
    ArtifactFormat.html: "text/html",
}

EXTENSION_BY_FORMAT: dict[ArtifactFormat, str] = {
    ArtifactFormat.markdown: ".md",
    ArtifactFormat.docx: ".docx",
    ArtifactFormat.pdf: ".pdf",
    ArtifactFormat.html: ".html",
}
