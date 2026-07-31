"""fal.ai API client for running synchronous and asynchronous inferences."""

import asyncio
import logging

import httpx

from server.config import Settings

logger = logging.getLogger(__name__)

FAL_BASE_URL = "https://fal.run"


class FalError(Exception):
    """fal.ai API error."""


def _resolve_api_key() -> str:
    key = Settings.fal_api_key
    if not key:
        error = FalError("FAL_API_KEY is not configured")
        raise error
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Key {_resolve_api_key()}",
        "Content-Type": "application/json",
    }


async def run_sync(
    endpoint: str, input_data: dict, *, timeout_secs: float = 120.0
) -> dict:
    """POST to a fal.ai sync endpoint and return the parsed JSON result."""
    url = f"{FAL_BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=timeout_secs) as client:
        resp = await client.post(url, json=input_data, headers=_headers())

    if resp.status_code >= 400:
        error = FalError(
            f"fal.ai API error {resp.status_code} at {endpoint}: {resp.text[:500]}"
        )
        raise error
    return resp.json()


async def create_async_request(
    endpoint: str,
    input_data: dict,
    *,
    poll_interval: float = 1.0,
    timeout_secs: float = 300.0,
) -> dict:
    """Submit a fal.ai async request and poll until it completes or fails."""
    url = f"{FAL_BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        submit_resp = await client.post(url, json=input_data, headers=_headers())

    if submit_resp.status_code >= 400:
        error = FalError(
            f"fal.ai submit error {submit_resp.status_code}: "
            f"{submit_resp.text[:500]}"
        )
        raise error

    result = submit_resp.json()

    status_url = result.get("status_url")
    response_url = result.get("response_url")

    if not status_url and not response_url:
        return result

    deadline = asyncio.get_event_loop().time() + timeout_secs
    async with httpx.AsyncClient(timeout=30.0) as client:
        poll_url = status_url or response_url
        while True:
            await asyncio.sleep(poll_interval)
            poll_resp = await client.get(poll_url, headers=_headers())
            if poll_resp.status_code >= 400:
                error = FalError(
                    f"fal.ai poll error {poll_resp.status_code}: "
                    f"{poll_resp.text[:500]}"
                )
                raise error
            data = poll_resp.json()
            status = data.get("status", "completed")
            if status == "COMPLETED":
                return data
            if status == "FAILED":
                error = FalError(
                    f"fal.ai request failed: {data.get('error', 'Unknown')}"
                )
                raise error
            if asyncio.get_event_loop().time() > deadline:
                error = FalError("fal.ai request timed out")
                raise error
