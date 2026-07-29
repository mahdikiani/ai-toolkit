# Unit tests for OpenRouter client helpers.

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from utils.integrations import openrouter as or_client


@pytest.mark.unit
class TestOpenRouterHelpers:
    def test_chat_completions_url(self) -> None:
        assert or_client.chat_completions_url().endswith("/chat/completions")

    def test_resolve_api_key_from_settings(self) -> None:
        with patch.object(or_client.Settings, "openrouter_api_key", "secret"):
            assert or_client.resolve_api_key() == "secret"

    def test_resolve_api_key_missing_raises(self) -> None:
        with (
            patch.object(or_client.Settings, "openrouter_api_key", None),
            pytest.raises(or_client.OpenRouterConfigurationError),
        ):
            or_client.resolve_api_key()

    def test_build_headers(self) -> None:
        with patch.object(or_client.Settings, "openrouter_api_key", "secret"):
            headers = or_client.build_headers()
        assert headers["Authorization"] == "Bearer secret"
        assert headers["Content-Type"] == "application/json"

    def test_extract_provider_meta(self) -> None:
        meta = or_client.extract_provider_meta(
            {"id": "1", "model": "m", "usage": {"total_tokens": 3}, "cost": 0.1},
            provider="openrouter",
        )
        assert meta["provider"] == "openrouter"
        assert meta["raw_cost"] == pytest.approx(0.1)

    def test_parse_sse_delta_line(self) -> None:
        payload = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
        assert or_client.parse_sse_delta_line(f"data: {payload}") == "hi"
        assert or_client.parse_sse_delta_line("data: [DONE]") == "[DONE]"
        assert or_client.parse_sse_delta_line("") is None


@pytest.mark.unit
class TestOpenRouterHTTP:
    async def test_complete_chat_json_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "x"}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "utils.integrations.openrouter.httpx.AsyncClient",
        ) as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = None
            client.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = client

            data = await or_client.complete_chat_json({"model": "m", "messages": []})

        assert data["id"] == "x"

    async def test_complete_chat_json_http_error(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(500, request=request, text="fail")
        error = httpx.HTTPStatusError("fail", request=request, response=response)

        with (
            patch(
                "utils.integrations.openrouter.httpx.AsyncClient",
            ) as client_cls,
            pytest.raises(or_client.OpenRouterRequestError),
        ):
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = None
            client.post = AsyncMock(side_effect=error)
            client_cls.return_value = client

            await or_client.complete_chat_json({"model": "m", "messages": []})

    async def test_post_chat_completion_unchecked(self) -> None:
        mock_resp = MagicMock(status_code=200)
        with patch(
            "utils.integrations.openrouter.httpx.AsyncClient",
        ) as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = None
            client.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = client

            resp = await or_client.post_chat_completion_unchecked({"model": "m"})

        assert resp.status_code == 200

    async def test_stream_chat_deltas_yields_content(self) -> None:
        async def fake_iter(*args, **kwargs):
            yield "hello"

        with patch(
            "utils.integrations.openrouter._iter_chat_deltas",
            side_effect=fake_iter,
        ):
            chunks = [
                chunk async for chunk in or_client.stream_chat_deltas({"model": "m"})
            ]

        assert chunks == ["hello"]
