"""Unit tests for the Replicate integration client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.integrations import replicate


def _client(resp: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    return client


@pytest.mark.unit
class TestResolveApiKey:
    """Tests for _resolve_api_key."""

    def test_raises_when_unset(self) -> None:
        with (
            patch.object(replicate.Settings, "replicate_api_key", None),
            pytest.raises(replicate.ReplicateError),
        ):
            replicate._resolve_api_key()

    def test_returns_configured_key(self) -> None:
        with patch.object(replicate.Settings, "replicate_api_key", "k"):
            assert replicate._resolve_api_key() == "k"


@pytest.mark.unit
class TestCreatePrediction:
    """Tests for create_prediction (submit + poll)."""

    async def test_raises_on_submit_error(self) -> None:
        resp = MagicMock(status_code=400, text="bad request")
        client = _client(resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                return_value=client,
            ),
            pytest.raises(replicate.ReplicateError),
        ):
            await replicate.create_prediction("owner/model", {"a": 1})

    async def test_splits_model_and_version(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "succeeded", "output": "x"}

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
        ):
            result = await replicate.create_prediction(
                "owner/model:abc123", {"a": 1}
            )

        assert result == {"status": "succeeded", "output": "x"}
        sent_payload = submit_client.post.call_args.kwargs["json"]
        assert sent_payload["version"] == "abc123"
        assert "model" not in sent_payload

    async def test_polls_until_succeeded(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "succeeded", "output": "x"}

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
        ):
            result = await replicate.create_prediction("owner/model", {"a": 1})

        assert result["output"] == "x"

    async def test_raises_on_poll_http_error(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=500, text="boom")

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            pytest.raises(replicate.ReplicateError),
        ):
            await replicate.create_prediction("owner/model", {"a": 1})

    async def test_raises_on_failed_status(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "failed", "error": "oops"}

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            pytest.raises(replicate.ReplicateError),
        ):
            await replicate.create_prediction("owner/model", {"a": 1})

    async def test_raises_on_canceled_status(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "canceled"}

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            pytest.raises(replicate.ReplicateError),
        ):
            await replicate.create_prediction("owner/model", {"a": 1})

    async def test_raises_on_timeout(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"id": "pred_1"}
        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {"status": "processing"}

        submit_client = _client(submit_resp)
        poll_client = _client(poll_resp)

        times = iter([1000.0, 1000.0, 2000.0])

        with (
            patch.object(replicate.Settings, "replicate_api_key", "k"),
            patch(
                "utils.integrations.replicate.httpx.AsyncClient",
                side_effect=[submit_client, poll_client],
            ),
            patch(
                "utils.integrations.replicate.asyncio.get_event_loop",
                return_value=MagicMock(time=lambda: next(times)),
            ),
            patch(
                "utils.integrations.replicate.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(replicate.ReplicateError),
        ):
            await replicate.create_prediction(
                "owner/model", {"a": 1}, timeout_secs=0.0
            )
