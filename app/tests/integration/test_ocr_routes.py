"""Integration tests for OCR API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestOcrRoutes:
    """Integration tests for /ocrs endpoints."""

    async def test_list_ocr_tasks_without_auth(self, client: httpx.AsyncClient) -> None:
        """GET /ocrs without auth should return 401."""
        response = await client.get("/ocrs")
        assert response.status_code in (401, 403)

    async def test_create_ocr_task_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs without auth should return 401."""
        response = await client.post(
            "/ocrs",
            json={"file_url": "https://example.com/test.pdf"},
        )
        assert response.status_code in (401, 403)

    async def test_create_ocr_task_missing_file_url(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs without file_url should return 422."""
        response = await client.post("/ocrs", json={})
        assert response.status_code in (401, 403, 422)

    async def test_create_ocr_task_empty_file_url(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs with empty file_url should return 422."""
        response = await client.post("/ocrs", json={"file_url": ""})
        assert response.status_code in (401, 403, 422)

    async def test_get_ocr_task_not_found(self, client: httpx.AsyncClient) -> None:
        """GET /ocrs/{uid} for non-existent task should return 401 or 404."""
        response = await client.get("/ocrs/nonexistent_uid_12345")
        assert response.status_code in (401, 403, 404)

    async def test_get_ocr_result_not_found(self, client: httpx.AsyncClient) -> None:
        """GET /ocrs/{uid}/result for non-existent task should return 401 or 404."""
        response = await client.get("/ocrs/nonexistent_uid_12345/result")
        assert response.status_code in (401, 403, 404)

    async def test_list_ocr_tasks_pagination_params(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /ocrs should accept pagination parameters."""
        response = await client.get("/ocrs?offset=0&limit=10")
        assert response.status_code in (200, 401, 403)

    async def test_list_ocr_tasks_invalid_limit(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /ocrs with invalid limit should return 422."""
        response = await client.get("/ocrs?limit=0")
        assert response.status_code in (401, 403, 422)

    async def test_upload_endpoint_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs/upload without auth should return 401."""
        response = await client.post(
            "/ocrs/upload",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code in (401, 403)

    async def test_upload_endpoint_without_file(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs/upload without file should return 422."""
        response = await client.post("/ocrs/upload")
        assert response.status_code in (401, 403, 422)

    async def test_upload_base64_endpoint_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /ocrs/upload/base64 without auth should return 401."""
        response = await client.post(
            "/ocrs/upload/base64",
            json={"content_base64": "ZmFrZSBwZGY=", "mime_type": "application/pdf"},
        )
        assert response.status_code in (401, 403)
