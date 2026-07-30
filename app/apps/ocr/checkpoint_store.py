"""
Redis-backed per-page checkpoint store for resumable OCR jobs.

OCR tasks run as in-process FastAPI background tasks, not a separate
worker queue -- if the process crashes or restarts mid-document, all
in-memory progress is normally lost and the job has to restart from page
1. Persisting each page's result to Redis as soon as it's done lets a
resumed run skip pages it already finished.

Degrades to "no resumability" (not an error) when Redis isn't configured:
every function is a safe no-op/empty-result in that case, matching how
this whole feature is optional infra layered on top of an already-working
pipeline, not a hard dependency.
"""

from __future__ import annotations

import json
import logging

from utils.redis_client import get_redis

logger = logging.getLogger(__name__)

_TTL_SECONDS = 24 * 3600


def _key(task_uid: str) -> str:
    return f"ocr:pages:{task_uid}"


async def save_page(task_uid: str, page_number: int, data: dict) -> None:
    """Persist one page's completed result so a crashed job can resume past it."""
    redis = get_redis()
    if redis is None:
        return
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(_key(task_uid), str(page_number), json.dumps(data))
            pipe.expire(_key(task_uid), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        logger.exception(
            "Failed to checkpoint page %d for task %s", page_number, task_uid
        )


async def load_pages(task_uid: str) -> dict[int, dict]:
    """Load any previously-checkpointed pages for a task (empty if none/unavailable)."""
    redis = get_redis()
    if redis is None:
        return {}
    try:
        raw = await redis.hgetall(_key(task_uid))
    except Exception:
        logger.exception("Failed to load checkpoints for task %s", task_uid)
        return {}

    pages: dict[int, dict] = {}
    for page_str, data_str in raw.items():
        try:
            pages[int(page_str)] = json.loads(data_str)
        except (ValueError, json.JSONDecodeError):
            logger.warning(
                "Skipping malformed checkpoint page=%s for task %s",
                page_str,
                task_uid,
            )
    return pages


async def clear(task_uid: str) -> None:
    """Drop checkpoints once a task reaches a terminal state (completed or error)."""
    redis = get_redis()
    if redis is None:
        return
    try:
        await redis.delete(_key(task_uid))
    except Exception:
        logger.exception("Failed to clear checkpoints for task %s", task_uid)
