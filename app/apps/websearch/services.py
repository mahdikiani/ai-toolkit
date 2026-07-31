"""Web search services backed by Exa."""

import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from utils.billing import finance
from utils.integrations.exa import exa_search

from .models import WebSearchTask

logger = logging.getLogger(__name__)


def _estimate_search_cost(task: WebSearchTask) -> float:
    pricing = finance.pricing_config().get("web_search") or {}
    per_search = float(pricing.get("default_per_search", 1.0))
    markup = float(pricing.get("markup", 1.0))
    return per_search * markup


async def process_search(task: WebSearchTask) -> WebSearchTask:
    """Run a web search and record its (fixed, per-search) usage cost."""
    estimated = _estimate_search_cost(task)
    quota = await finance.check_quota(
        task.user_id, estimated, raise_exception=False, workspace_id=task.workspace_id
    )
    if quota < estimated:
        task.task_status = TaskStatusEnum.error
        await task.save_report("insufficient_quota")
        return task

    try:
        result = await exa_search(
            query=task.query,
            num_results=task.num_results,
            include_domains=task.include_domains,
            exclude_domains=task.exclude_domains,
        )
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(f"Search failed: {exc}")
        return task

    task.result = result

    try:
        await finance.meter_cost(
            task.user_id,
            estimated,
            meta_data={
                "service": "web_search",
                "provider": "exa",
                "query": task.query[:200],
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter web search usage")

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Search completed successfully")
    return task
