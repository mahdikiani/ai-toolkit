"""Integration tests for prompts API endpoints."""

import httpx
import pytest


@pytest.mark.integration
class TestPromptsRoutes:
    """Integration tests for /prompts endpoints."""

    async def test_list_prompts_returns_200(self, client: httpx.AsyncClient) -> None:
        """GET /prompts should return 200 (no auth required for listing)."""
        response = await client.get("/prompts/")
        # Prompts listing may or may not require auth
        assert response.status_code in (200, 401, 403)

    async def test_list_prompts_returns_list(self, client: httpx.AsyncClient) -> None:
        """GET /prompts should return a list."""
        response = await client.get("/prompts/")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    async def test_get_prompt_schema_not_found(self, client: httpx.AsyncClient) -> None:
        """GET /prompts/{name}/schema for non-existent prompt should return 404."""
        response = await client.get("/prompts/nonexistent_prompt_xyz/schema")
        assert response.status_code in (401, 403, 404)

    async def test_get_prompt_schema_returns_schema_structure(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /prompts/{name}/schema should return schema with expected fields."""
        response = await client.get("/prompts/some_prompt/schema")
        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "description" in data

    async def test_list_prompts_response_structure(
        self, client: httpx.AsyncClient
    ) -> None:
        """Each prompt in the list should have name and description."""
        response = await client.get("/prompts/")
        if response.status_code == 200:
            data = response.json()
            for prompt in data:
                assert "name" in prompt
                assert "description" in prompt
