"""YouTube transcription services using youtube-transcript.io API."""

import base64

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings

from .models import YoutubeTask


async def process_youtube(task: YoutubeTask) -> YoutubeTask:
    """Fetch transcript from youtube-transcript.io and save the result."""
    api_key = Settings.youtube_transcript_api_key
    if not api_key:
        task.task_status = TaskStatusEnum.failed
        task.result = "YouTube Transcript API key is not configured"
        return await task.save()

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
        task.task_status = TaskStatusEnum.failed
        task.result = (
            f"YouTube Transcript API error: {e.response.status_code} {e.response.text}"
        )
        return await task.save()
    except httpx.RequestError as e:
        task.task_status = TaskStatusEnum.failed
        task.result = f"Request failed: {e}"
        return await task.save()

    transcripts = data.get("transcripts", [])
    if not transcripts:
        task.task_status = TaskStatusEnum.failed
        task.result = f"No transcript found for video ID: {task.video_id}"
        return await task.save()

    text_parts = [item.get("text", "") for item in transcripts[0].get("transcript", [])]

    task.task_status = TaskStatusEnum.completed
    task.result = " ".join(text_parts)
    return await task.save()
