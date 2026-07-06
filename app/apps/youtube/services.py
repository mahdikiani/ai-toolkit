"""YouTube transcription services using youtube-transcript.io API."""

import base64
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import finance

from .models import YoutubeTask


async def process_youtube(task: YoutubeTask) -> YoutubeTask:
    """Fetch transcript from youtube-transcript.io and save the result."""
    api_key = Settings.youtube_transcript_api_key
    if not api_key:
        task.task_status = TaskStatusEnum.error
        task.result = "YouTube Transcript API key is not configured"
        return await task.save()

    task.video_id = normalize_video_id(task.video_id)

    auth_header = "Basic " + base64.b64encode(
        f"{api_key}:".encode()
    ).decode()

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
        task.task_status = TaskStatusEnum.error
        task.result = (
            f"YouTube Transcript API error: {e.response.status_code} {e.response.text}"
        )
        return await task.save()
    except httpx.RequestError as e:
        task.task_status = TaskStatusEnum.error
        task.result = f"Request failed: {e}"
        return await task.save()

    transcripts = data.get("transcripts", [])
    if not transcripts:
        task.task_status = TaskStatusEnum.error
        task.result = f"No transcript found for video ID: {task.video_id}"
        return await task.save()

    text_parts = [item.get("text", "") for item in transcripts[0].get("transcript", [])]

    amount = finance.estimate_youtube_cost()
    usage = await finance.meter_cost(
        task.user_id,
        amount,
        meta_data={
            "service": "youtube",
            "provider": "youtube-transcript.io",
            "video_id": task.video_id,
        },
    )

    task.task_status = TaskStatusEnum.completed
    task.result = " ".join(text_parts)
    task.provider_meta = {
        "provider": "youtube-transcript.io",
        "video_id": task.video_id,
    }
    task.usage_amount = float(usage.amount) if usage else amount
    task.usage_id = usage.uid if usage else None
    return await task.save()


def normalize_video_id(value: str) -> str:
    """Extract a YouTube video id from an id or common YouTube URL."""
    candidate = value.strip()
    parsed = urlparse(candidate)
    if not parsed.netloc:
        return candidate
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    query_video = parse_qs(parsed.query).get("v")
    if query_video:
        return query_video[0]
    if "/shorts/" in parsed.path:
        return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
    if "/embed/" in parsed.path:
        return parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
    return candidate
