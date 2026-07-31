"""Unit tests for the fal.ai integration client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.integrations import fal


def _mock_client(responses: list[MagicMock]) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    if len(responses) == 1:
        client.post = AsyncMock(return_value=responses[0])
        client.get = AsyncMock(return_value=responses[0])
    else:
        client.post = AsyncMock(side_effect=responses[:1])
        client.get = AsyncMock(side_effect=responses[1:])
    return client


@pytest.mark.unit
class TestResolveApiKey:
    """Tests for _resolve_api_key."""

    def test_raises_when_unset(self) -> None:
        with (
            patch.object(fal.Settings, "fal_api_key", None),
            pytest.raises(fal.FalError),
        ):
            fal._resolve_api_key()

    def test_returns_configured_key(self) -> None:
        with patch.object(fal.Settings, "fal_api_key", "k"):
            assert fal._resolve_api_key() == "k"


@pytest.mark.unit
class TestRunSync:
    """Tests for run_sync."""

    async def test_returns_json_on_success(self) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"ok": True}
        client = _mock_client([resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch("utils.integrations.fal.httpx.AsyncClient", return_value=client),
        ):
            result = await fal.run_sync("/endpoint", {"a": 1})

        assert result == {"ok": True}

    async def test_raises_on_http_error(self) -> None:
        resp = MagicMock(status_code=500, text="boom")
        client = _mock_client([resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch("utils.integrations.fal.httpx.AsyncClient", return_value=client),
            pytest.raises(fal.FalError),
        ):
            await fal.run_sync("/endpoint", {"a": 1})


@pytest.mark.unit
class TestCreateAsyncRequest:
    """Tests for create_async_request."""

    async def test_returns_immediately_without_status_url(self) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"result": "done"}
        client = _mock_client([resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch("utils.integrations.fal.httpx.AsyncClient", return_value=client),
        ):
            result = await fal.create_async_request("/endpoint", {"a": 1})

        assert result == {"result": "done"}

    async def test_raises_on_submit_error(self) -> None:
        resp = MagicMock(status_code=400, text="bad request")
        client = _mock_client([resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch("utils.integrations.fal.httpx.AsyncClient", return_value=client),
            pytest.raises(fal.FalError),
        ):
            await fal.create_async_request("/endpoint", {"a": 1})

    async def test_polls_until_completed(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"status_url": "https://x/status"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "COMPLETED", "output": "x"}

        submit_client = _mock_client([submit_resp])
        poll_client = _mock_client([poll_resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch(
                "utils.integrations.fal.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            patch("utils.integrations.fal.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await fal.create_async_request("/endpoint", {"a": 1})

        assert result == {"status": "COMPLETED", "output": "x"}

    async def test_raises_on_failed_status(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"status_url": "https://x/status"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "FAILED", "error": "boom"}

        submit_client = _mock_client([submit_resp])
        poll_client = _mock_client([poll_resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch(
                "utils.integrations.fal.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            patch("utils.integrations.fal.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(fal.FalError),
        ):
            await fal.create_async_request("/endpoint", {"a": 1})

    async def test_raises_on_poll_http_error(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"status_url": "https://x/status"}
        poll_resp = MagicMock(status_code=500, text="boom")

        submit_client = _mock_client([submit_resp])
        poll_client = _mock_client([poll_resp])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch(
                "utils.integrations.fal.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            patch("utils.integrations.fal.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(fal.FalError),
        ):
            await fal.create_async_request("/endpoint", {"a": 1})

    async def test_raises_on_timeout(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"status_url": "https://x/status"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "IN_PROGRESS"}

        submit_client = _mock_client([submit_resp])
        poll_client = AsyncMock()
        poll_client.__aenter__.return_value = poll_client
        poll_client.__aexit__.return_value = None
        poll_client.get = AsyncMock(return_value=poll_resp)

        # Deadline already elapsed on the first check.
        times = iter([1000.0, 1000.0, 2000.0])

        with (
            patch.object(fal.Settings, "fal_api_key", "k"),
            patch(
                "utils.integrations.fal.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            patch("utils.integrations.fal.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "utils.integrations.fal.asyncio.get_event_loop",
                return_value=MagicMock(time=lambda: next(times)),
            ),
            pytest.raises(fal.FalError),
        ):
            await fal.create_async_request(
                "/endpoint", {"a": 1}, timeout_secs=0.0
            )
