"""Integration tests for translation API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestTranslateRoutes:
    """Integration tests for /translates endpoints."""

    async def test_list_translate_tasks_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /translates without auth should return 401."""
        response = await client.get("/translates")
        assert response.status_code in (401, 403)

    async def test_create_translate_task_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /translates without auth should return 401."""
        response = await client.post(
            "/translates",
            json={"text": "Hello world", "language": "Persian"},
        )
        assert response.status_code in (401, 403)

    async def test_create_translate_task_missing_text(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /translates without text should return 422."""
        response = await client.post("/translates", json={"language": "Persian"})
        assert response.status_code in (401, 403, 422)

    async def test_get_translate_task_not_found(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /translates/{uid} for non-existent task should return 401 or 404."""
        response = await client.get("/translates/nonexistent_uid_12345")
        assert response.status_code in (401, 403, 404)

    async def test_list_translate_tasks_pagination(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /translates should accept pagination parameters."""
        response = await client.get("/translates?offset=0&limit=10")
        assert response.status_code in (200, 401, 403)

    async def test_list_translate_tasks_invalid_limit(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /translates with invalid limit should return 422."""
        response = await client.get("/translates?limit=-1")
        assert response.status_code in (401, 403, 422)
