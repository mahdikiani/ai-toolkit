"""Mock fixtures for external dependencies."""

from collections.abc import AsyncIterator, Callable, Iterator
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_openrouter_complete() -> Iterator[MagicMock]:
    """Mock OpenRouter non-streaming API calls."""
    with patch("utils.integrations.openrouter.complete_chat_json") as mock_complete:
        mock_complete.return_value = {
            "choices": [
                {"message": {"content": "Mocked AI response", "role": "assistant"}}
            ],
            "model": "openai/gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        yield mock_complete


@pytest.fixture
def mock_openrouter_stream() -> Iterator[Callable[..., AsyncIterator[str]]]:
    """Mock OpenRouter streaming API calls."""

    async def _mock_stream_gen(*args: object, **kwargs: object) -> AsyncIterator[str]:
        await AsyncMock()()
        for chunk in ["Mocked ", "streaming ", "response"]:
            yield chunk

    with patch(
        "utils.integrations.openrouter.stream_chat_deltas", side_effect=_mock_stream_gen
    ):
        yield _mock_stream_gen


@pytest.fixture
def mock_openrouter(
    mock_openrouter_complete: MagicMock,
    mock_openrouter_stream: Callable[..., AsyncIterator[str]],
) -> dict[str, object]:
    """Provide combined streaming and non-streaming OpenRouter mocks."""
    return {
        "complete": mock_openrouter_complete,
        "stream": mock_openrouter_stream,
    }


@pytest.fixture
def mock_finance_no_key() -> Iterator[None]:
    """Mock finance service when no API key is configured (returns inf quota)."""
    with patch("server.config.Settings.finance_api_key", None):
        yield


@pytest.fixture
def mock_finance() -> Iterator[dict[str, MagicMock]]:
    """Mock finance service (quota and metering)."""
    mock_usage = MagicMock()
    mock_usage.uid = "usage_test_123"
    mock_usage.amount = Decimal("10.0")

    with (
        patch("utils.billing.finance.get_quota", new_callable=AsyncMock) as mock_get,
        patch("utils.billing.finance.meter_cost", new_callable=AsyncMock) as mock_meter,
        patch(
            "utils.billing.finance.check_quota", new_callable=AsyncMock
        ) as mock_check,
    ):
        mock_get.return_value = Decimal("1000.0")
        mock_check.return_value = Decimal("1000.0")
        mock_meter.return_value = mock_usage

        yield {
            "get_quota": mock_get,
            "check_quota": mock_check,
            "meter_cost": mock_meter,
            "usage": mock_usage,
        }


@pytest.fixture
def mock_finance_insufficient() -> Iterator[AsyncMock]:
    """Mock finance service with insufficient quota."""
    from ufaas import exceptions

    with patch(
        "utils.billing.finance.check_quota", new_callable=AsyncMock
    ) as mock_check:
        mock_check.side_effect = exceptions.InsufficientFundsError(
            "You have only 0 coins, while you need 1 coins."
        )
        yield mock_check


@pytest.fixture
def mock_media() -> Iterator[dict[str, AsyncMock]]:
    """Mock media service (file storage and retrieval)."""
    with (
        patch(
            "utils.integrations.media.upload_file", new_callable=AsyncMock
        ) as mock_upload,
        patch(
            "utils.integrations.media.download_file", new_callable=AsyncMock
        ) as mock_download,
    ):
        mock_upload.return_value = "https://mock-storage.example.com/file123.pdf"
        mock_download.return_value = b"Mock file content"

        yield {
            "upload": mock_upload,
            "download": mock_download,
        }


@pytest.fixture
def mock_file_system(tmp_path: Path) -> Path:
    """Provide a temporary file system for testing."""
    return tmp_path
