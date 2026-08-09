"""Integration tests for Markdown document conversion API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestMarkdownToDocxRoutes:
    """Integration tests for /document-convert endpoints."""

    async def test_markdown_to_docx_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /document-convert/markdown-to-docx without auth should return 401."""
        response = await client.post(
            "/document-convert/markdown-to-docx",
            json={"markdown": "# hello", "title": "test"},
        )
        assert response.status_code in (401, 403)

    async def test_markdown_to_docx_missing_markdown_field(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST without the required markdown field should return 401/403/422."""
        response = await client.post("/document-convert/markdown-to-docx", json={})
        assert response.status_code in (401, 403, 422)

    async def test_markdown_to_docx_upload_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /document-convert/markdown-to-docx/upload without auth should 401."""
        response = await client.post(
            "/document-convert/markdown-to-docx/upload",
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
        assert response.status_code in (401, 403)

    async def test_markdown_to_docx_upload_without_file(
        self, client: httpx.AsyncClient
    ) -> None:
        """
        POST /document-convert/markdown-to-docx/upload without a file should
        return 401/403/422 (auth is checked, but a missing required file must
        never be silently accepted either)."""
        response = await client.post("/document-convert/markdown-to-docx/upload")
        assert response.status_code in (401, 403, 422)


@pytest.mark.integration
class TestMarkdownToPdfRoutes:
    """Auth and application wiring tests for the PDF conversion endpoints."""

    async def test_markdown_to_pdf_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/document-convert/markdown-to-pdf",
            json={"markdown": "# hello", "title": "test"},
        )
        assert response.status_code in (401, 403)

    async def test_markdown_to_pdf_upload_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/document-convert/markdown-to-pdf/upload",
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
        assert response.status_code in (401, 403)
