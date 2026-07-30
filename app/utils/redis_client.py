"""Shared async Redis client for optional, best-effort features (OCR checkpointing)."""

from __future__ import annotations

from redis.asyncio import Redis

from server.config import Settings


def _build_redis_client(redis_uri: str | None) -> Redis | None:
    """
    Build the async Redis client, or None if Redis isn't configured.

    decode_responses=True keeps hash field/value access as plain str
    (checkpoint_store.py indexes them directly); fastapi_mongo_base's own
    shared client doesn't set this, so a dedicated client is built here
    instead of reusing it.
    """
    if not redis_uri:
        return None
    return Redis.from_url(
        redis_uri,
        socket_connect_timeout=1,
        socket_timeout=1,
        decode_responses=True,
    )


redis: Redis | None = _build_redis_client(Settings().redis_uri)


def get_redis() -> Redis | None:
    """Return the shared async Redis client, or None if Redis isn't configured."""
    return redis
