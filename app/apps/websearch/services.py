"""Web search services backed by Exa."""

import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from utils.billing import finance
from utils.integrations.exa import exa_search

from .models import WebSearchTask

logger = logging.getLogger(__name__)


def _estimate_search_cost(task: WebSearchTask) -> float:
    return finance.estimate_fixed_cost("web_search", "default_per_search")


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

    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            estimated,
            meta_data={
                "service": "web_search",
                "provider": "exa",
                "query": task.query[:200],
                "task_uid": task.uid,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter web search usage")

    task.usage_amount = float(usage.amount) if usage else estimated
    task.usage_id = usage.uid if usage else None

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Search completed successfully")
    return task
