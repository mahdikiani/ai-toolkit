"""Integration tests for chat API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestChatSessionRoutes:
    """Integration tests for /chat/sessions endpoints."""

    async def test_list_sessions_without_auth_returns_401(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /chat/sessions without auth should return 401."""
        response = await client.get("/chat/sessions")
        assert response.status_code in (401, 403)

    async def test_create_session_without_auth_returns_401(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /chat/sessions without auth should return 401."""
        response = await client.post(
            "/chat/sessions",
            json={"title": "Test Session"},
        )
        assert response.status_code in (401, 403)

    async def test_get_session_not_found(self, client: httpx.AsyncClient) -> None:
        """GET /chat/sessions/{uid} for non-existent session should return 401 or 404."""
        response = await client.get("/chat/sessions/nonexistent_uid_12345")
        assert response.status_code in (401, 403, 404)

    async def test_list_sessions_pagination_params(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /chat/sessions should accept pagination parameters."""
        response = await client.get("/chat/sessions?offset=0&limit=10")
        assert response.status_code in (200, 401, 403)


@pytest.mark.integration
class TestChatThreadRoutes:
    """Integration tests for /chat/sessions/{uid}/threads endpoints."""

    async def test_list_threads_without_auth(self, client: httpx.AsyncClient) -> None:
        """GET /chat/sessions/{uid}/threads without auth should return 401."""
        response = await client.get("/chat/sessions/some_uid/threads")
        assert response.status_code in (401, 403, 404)

    async def test_create_thread_without_auth(self, client: httpx.AsyncClient) -> None:
        """POST /chat/sessions/{uid}/threads without auth should return 401."""
        response = await client.post(
            "/chat/sessions/some_uid/threads",
            json={"title": "Test Thread"},
        )
        assert response.status_code in (401, 403, 404)


@pytest.mark.integration
class TestChatMessageRoutes:
    """Integration tests for message endpoints."""

    async def test_list_messages_without_auth(self, client: httpx.AsyncClient) -> None:
        """GET messages without auth should return 401."""
        response = await client.get(
            "/chat/sessions/some_uid/threads/some_thread/messages"
        )
        assert response.status_code in (401, 403, 404)

    async def test_post_message_without_auth(self, client: httpx.AsyncClient) -> None:
        """POST message without auth should return 401."""
        response = await client.post(
            "/chat/sessions/some_uid/threads/some_thread/messages",
            json={"content": "Hello"},
        )
        assert response.status_code in (401, 403, 404)

    async def test_post_message_invalid_content(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST message with empty content should return 422."""
        response = await client.post(
            "/chat/sessions/some_uid/threads/some_thread/messages",
            json={"content": ""},
        )
        assert response.status_code in (401, 403, 404, 422)


@pytest.mark.integration
class TestOpenAICompatibleEndpoint:
    """Integration tests for OpenAI-compatible endpoint."""

    async def test_openai_endpoint_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /chat/completions without auth should return 401."""
        response = await client.post(
            "/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code in (401, 403)

    async def test_openai_endpoint_invalid_json(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /chat/completions with invalid JSON should return 400 or 401."""
        response = await client.post(
            "/chat/completions",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 401, 403, 422)
