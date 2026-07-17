"""YouTube transcription services using youtube-transcript.io API."""

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance

from .models import YoutubeTranscriptTask
from .video_id import parse_youtube_video_id


async def process_youtube(task: YoutubeTranscriptTask) -> YoutubeTranscriptTask:
    """Fetch transcript from youtube-transcript.io and save the result."""
    api_key = Settings.youtube_transcript_api_key
    if not api_key:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result="YouTube Transcript API key is not configured",
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

    tracks = data[0].get("tracks", [])
    if not tracks:
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            result=f"No transcript found for video ID: {task.video_id}",
        )
        return task

    text_parts = [
        item.get("text", "")
        for track in tracks
        for item in track.get("transcript", [])
    ]

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

    await task.update_and_emit(
        task_status=TaskStatusEnum.completed,
        result=" ".join(text_parts),
        provider_meta={
            "provider": "youtube-transcript.io",
            "video_id": task.video_id,
        },
        usage_amount=float(usage.amount) if usage else amount,
        usage_id=usage.uid if usage else None,
    )
    return task
