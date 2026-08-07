"""Webpage extraction services using Jina Reader."""

import logging
import re
from urllib.parse import urlparse

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from utils.billing import finance

from .models import WebpageTask

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"
_TITLE_RE = re.compile(r"^Title:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _page_title(content: str, url: str) -> str:
    """Return Jina's page title, with a stable URL-derived fallback."""
    match = _TITLE_RE.search(content)
    if match and (title := match.group(1).strip()):
        return title
    parsed = urlparse(url)
    path_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return path_name.replace("-", " ").replace("_", " ").strip() or parsed.netloc


async def process_webpage(task: WebpageTask) -> WebpageTask:
    """Fetch readable page content via Jina Reader and save the result."""
    amount = finance.estimate_fixed_cost("webpage", "per_request")
    quota = await finance.check_quota(
        task.user_id,
        amount,
        raise_exception=False,
        workspace_id=task.workspace_id,
    )
    if quota < amount:
        task.task_status = TaskStatusEnum.error
        await task.save_report("insufficient_quota")
        return task

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
    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "webpage",
                "provider": "jina-reader",
                "task_uid": task.uid,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter webpage usage for task %s", task.uid)
    task.usage_amount = float(usage.amount) if usage else amount
    task.usage_id = usage.uid if usage else None
    task.provider_meta = {
        "provider": "jina-reader",
        "url": task.url,
        "title": _page_title(content, task.url),
        "usage": {"amount": amount},
    }
    await task.save_report("Task processed successfully")
    return task
