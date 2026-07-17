"""Webpage extraction services using Jina Reader."""

import logging

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from .models import WebpageTask

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"


async def process_webpage(task: WebpageTask) -> WebpageTask:
    """Fetch readable page content via Jina Reader and save the result."""
    reader_url = f"{JINA_READER_BASE}{task.url}"
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(reader_url)
            response.raise_for_status()
            content = response.text
    except httpx.HTTPStatusError as exc:
        task.task_status = TaskStatusEnum.error
        task.result = f"Jina Reader error: {exc.response.status_code}"
        await task.save_report(task.result)
        return task
    except httpx.RequestError as exc:
        task.task_status = TaskStatusEnum.error
        task.result = f"Request failed: {exc}"
        await task.save_report(task.result)
        return task

    if not content.strip():
        task.task_status = TaskStatusEnum.error
        task.result = "No content extracted from webpage"
        await task.save_report(task.result)
        return task

    task.task_status = TaskStatusEnum.completed
    task.result = content
    task.provider_meta = {"provider": "jina-reader", "url": task.url}
    await task.save_report("Task processed successfully")
    return task
