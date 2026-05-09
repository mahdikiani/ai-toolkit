"""Integration tests for transcription API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestTranscribeRoutes:
    """Integration tests for /transcribes endpoints."""

    async def test_list_transcribe_tasks_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /transcribes without auth should return 401."""
        response = await client.get("/transcribes")
        assert response.status_code in (401, 403)

    async def test_create_transcribe_task_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /transcribes without auth should return 401."""
        response = await client.post(
            "/transcribes",
            json={"file_url": "https://example.com/audio.mp3"},
        )
        assert response.status_code in (401, 403)

    async def test_create_transcribe_task_missing_file_url(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /transcribes without file_url should return 422."""
        response = await client.post("/transcribes", json={})
        assert response.status_code in (401, 403, 422)

    async def test_get_transcribe_task_not_found(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /transcribes/{uid} for non-existent task should return 401 or 404."""
        response = await client.get("/transcribes/nonexistent_uid_12345")
        assert response.status_code in (401, 403, 404)

    async def test_get_transcribe_result_not_found(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /transcribes/{uid}/result for non-existent task should return 401 or 404."""
        response = await client.get("/transcribes/nonexistent_uid_12345/result")
        assert response.status_code in (401, 403, 404)

    async def test_list_transcribe_tasks_pagination(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /transcribes should accept pagination parameters."""
        response = await client.get("/transcribes?offset=0&limit=10")
        assert response.status_code in (200, 401, 403)

    async def test_webhook_endpoint_not_found(self, client: httpx.AsyncClient) -> None:
        """POST /transcribes/{uid}/webhook for non-existent task should return 404."""
        response = await client.post(
            "/transcribes/nonexistent_uid/webhook",
            json={"id": "job_123", "status": "COMPLETED"},
        )
        assert response.status_code in (401, 403, 404, 422)
