"""YouTube transcription services using youtube-transcript.io API."""

import logging

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance

from .models import YoutubeTranscriptTask
from .video_id import parse_youtube_video_id

logger = logging.getLogger(__name__)


def _video_title(item: dict) -> str | None:
    """Read a title from known youtube-transcript.io response shapes."""
    candidates = [
        item.get("title"),
        item.get("video_title"),
        item.get("videoTitle"),
    ]
    for container_key in ("metadata", "video"):
        container = item.get(container_key)
        if isinstance(container, dict):
            candidates.extend(
                [
                    container.get("title"),
                    container.get("video_title"),
                    container.get("videoTitle"),
                ]
            )
    return next(
        (str(value).strip() for value in candidates if str(value or "").strip()),
        None,
    )


async def _youtube_oembed_title(video_id: str) -> str | None:
    """Fetch the public YouTube title when the transcript provider omits it."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://www.youtube.com/oembed",
                params={
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "format": "json",
                },
            )
            response.raise_for_status()
            title = str(response.json().get("title") or "").strip()
            return title or None
    except (httpx.HTTPError, ValueError, TypeError):
        logger.warning("Could not resolve YouTube title for video %s", video_id)
        return None


async def process_youtube(task: YoutubeTranscriptTask) -> YoutubeTranscriptTask:
    """Fetch transcript from youtube-transcript.io and save the result."""
    api_key = Settings.youtube_transcript_api_key
    if not api_key:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result="YouTube Transcript API key is not configured",
        )
        return task

    amount = finance.estimate_youtube_cost()
    quota = await finance.check_quota(
        task.user_id, amount, raise_exception=False, workspace_id=task.workspace_id
    )
    if quota < amount:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result="insufficient_quota",
        )
        return task

    task.video_id = parse_youtube_video_id(task.video_id)

    auth_header = f"Basic {api_key}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://www.youtube-transcript.io/api/transcripts",
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                json={"ids": [task.video_id]},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result=(
                f"YouTube Transcript API error:"
                f" {e.response.status_code} {e.response.text}"
            ),
        )
        return task
    except httpx.RequestError as e:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result=f"Request failed: {e}",
        )
        return task

    if not isinstance(data, list) or not data:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result=f"No transcript found for video ID: {task.video_id}",
        )
        return task

    video_data = data[0]
    tracks = video_data.get("tracks", [])
    if not tracks:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result=f"No transcript found for video ID: {task.video_id}",
        )
        return task

    text_parts = [
        item.get("text", "") for track in tracks for item in track.get("transcript", [])
    ]

    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "youtube",
                "provider": "youtube-transcript.io",
                "video_id": task.video_id,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter youtube usage for task %s", task.uid)

    provider_meta = {
        "provider": "youtube-transcript.io",
        "video_id": task.video_id,
    }
    title = _video_title(video_data) or await _youtube_oembed_title(task.video_id)
    if title:
        provider_meta["title"] = title

    await task.update_and_emit(
        task_status=TaskStatusEnum.completed,
        result=" ".join(text_parts),
        provider_meta=provider_meta,
        usage_amount=float(usage.amount) if usage else amount,
        usage_id=usage.uid if usage else None,
    )
    return task
