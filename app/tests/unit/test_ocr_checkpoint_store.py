"""Unit tests for the Redis-backed per-page OCR checkpoint store."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.ocr import checkpoint_store


@pytest.mark.unit
class TestCheckpointStoreWithoutRedis:
    """Redis unconfigured (get_redis() -> None) must degrade to safe no-ops."""

    async def test_save_page_is_a_noop(self) -> None:
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=None):
            await checkpoint_store.save_page("task-1", 1, {"text": "hi"})

    async def test_load_pages_returns_empty(self) -> None:
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=None):
            result = await checkpoint_store.load_pages("task-1")
        assert result == {}

    async def test_clear_is_a_noop(self) -> None:
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=None):
            await checkpoint_store.clear("task-1")


@pytest.mark.unit
class TestCheckpointStoreWithRedis:
    def _fake_redis(self) -> MagicMock:
        redis = MagicMock()
        pipe = AsyncMock()
        pipe.__aenter__.return_value = pipe
        pipe.__aexit__.return_value = None
        pipe.hset = MagicMock()
        pipe.expire = MagicMock()
        redis.pipeline = MagicMock(return_value=pipe)
        redis.hgetall = AsyncMock()
        redis.delete = AsyncMock()
        return redis

    async def test_save_page_writes_hash_field_with_ttl(self) -> None:
        redis = self._fake_redis()
        pipe = redis.pipeline.return_value
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            await checkpoint_store.save_page("task-1", 3, {"text": "hello"})

        pipe.hset.assert_called_once_with(
            "ocr:pages:task-1", "3", json.dumps({"text": "hello"})
        )
        pipe.expire.assert_called_once_with(
            "ocr:pages:task-1", checkpoint_store._TTL_SECONDS
        )
        pipe.execute.assert_awaited_once()

    async def test_save_page_failure_is_logged_not_raised(self) -> None:
        redis = self._fake_redis()
        redis.pipeline.side_effect = RuntimeError("connection lost")
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            await checkpoint_store.save_page("task-1", 1, {"text": "x"})

    async def test_load_pages_parses_stored_json(self) -> None:
        redis = self._fake_redis()
        redis.hgetall.return_value = {
            "1": json.dumps({"text": "page one"}),
            "2": json.dumps({"text": "page two"}),
        }
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            result = await checkpoint_store.load_pages("task-1")

        assert result == {1: {"text": "page one"}, 2: {"text": "page two"}}

    async def test_load_pages_skips_malformed_entries(self) -> None:
        redis = self._fake_redis()
        redis.hgetall.return_value = {
            "1": json.dumps({"text": "good"}),
            "not-a-number": json.dumps({"text": "bad key"}),
            "2": "{not valid json",
        }
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            result = await checkpoint_store.load_pages("task-1")

        assert result == {1: {"text": "good"}}

    async def test_load_pages_failure_returns_empty(self) -> None:
        redis = self._fake_redis()
        redis.hgetall.side_effect = RuntimeError("connection lost")
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            result = await checkpoint_store.load_pages("task-1")
        assert result == {}

    async def test_clear_deletes_the_hash_key(self) -> None:
        redis = self._fake_redis()
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            await checkpoint_store.clear("task-1")
        redis.delete.assert_awaited_once_with("ocr:pages:task-1")

    async def test_clear_failure_is_logged_not_raised(self) -> None:
        redis = self._fake_redis()
        redis.delete.side_effect = RuntimeError("connection lost")
        with patch("apps.ocr.checkpoint_store.get_redis", return_value=redis):
            await checkpoint_store.clear("task-1")
