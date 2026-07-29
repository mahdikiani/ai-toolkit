"""Exa (exa.ai) web search API client."""

import logging

import httpx

from server.config import Settings

logger = logging.getLogger(__name__)

EXA_BASE_URL = "https://api.exa.ai"


class ExaError(Exception):
    """Exa API error."""


def _resolve_api_key() -> str:
    key = Settings.exa_api_key
    if not key:
        error = ExaError("EXA_API_KEY is not configured")
        raise error
    return key


async def exa_search(
    query: str,
    *,
    num_results: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict:
    """
    Search the web via Exa.

    Args:
        query: Search query string.
        num_results: Number of results to return (1-50).
        include_domains: Only return results from these domains.
        exclude_domains: Exclude results from these domains.
        start_published_date: ISO date string for earliest publish date.
        end_published_date: ISO date string for latest publish date.

    Returns:
        The full Exa search response dict with ``results`` list.
    """
    api_key = _resolve_api_key()
    url = f"{EXA_BASE_URL}/search"

    payload: dict = {
        "query": query,
        "numResults": num_results,
        "useAutoprompt": True,
    }
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains
    if start_published_date:
        payload["startPublishedDate"] = start_published_date
    if end_published_date:
        payload["endPublishedDate"] = end_published_date

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code >= 400:
        error = ExaError(f"Exa API error {resp.status_code}: {resp.text[:500]}")
        raise error

    return resp.json()
