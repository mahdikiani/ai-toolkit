"""Replicate API client for running predictions asynchronously."""

import asyncio
import logging

import httpx

from server.config import Settings

logger = logging.getLogger(__name__)

REPLICATE_BASE_URL = "https://api.replicate.com/v1"


class ReplicateError(Exception):
    """Replicate API error."""


def _resolve_api_key() -> str:
    key = Settings.replicate_api_key
    if not key:
        error = ReplicateError("REPLICATE_API_KEY is not configured")
        raise error
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_resolve_api_key()}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }


async def create_prediction(
    model: str,
    input_data: dict,
    *,
    timeout_secs: float = 600.0,
) -> dict:
    """Create a Replicate prediction and poll it until it finishes."""
    version = None
    if ":" in model:
        model, version = model.split(":", 1)

    url = f"{REPLICATE_BASE_URL}/predictions"
    payload: dict = {
        "input": input_data,
    }
    if version:
        payload["version"] = version
    else:
        payload["model"] = f"{model}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=_headers())

    if resp.status_code >= 400:
        error = ReplicateError(
            f"Replicate API error {resp.status_code}: {resp.text[:500]}"
        )
        raise error

    prediction = resp.json()
    return await _poll_prediction(prediction["id"], timeout_secs=timeout_secs)


async def _poll_prediction(
    prediction_id: str,
    *,
    timeout_secs: float = 600.0,
    poll_interval: float = 2.0,
) -> dict:
    url = f"{REPLICATE_BASE_URL}/predictions/{prediction_id}"
    deadline = asyncio.get_event_loop().time() + timeout_secs

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            resp = await client.get(url, headers=_headers())
            if resp.status_code >= 400:
                error = ReplicateError(
                    f"Replicate poll error {resp.status_code}: {resp.text[:500]}"
                )
                raise error
            data = resp.json()
            status = data.get("status")
            if status == "succeeded":
                return data
            if status == "failed":
                err_detail = data.get("error", "Unknown error")
                error = ReplicateError(f"Replicate prediction failed: {err_detail}")
                raise error
            if status == "canceled":
                error = ReplicateError("Replicate prediction was canceled")
                raise error
            if asyncio.get_event_loop().time() + poll_interval > deadline:
                error = ReplicateError("Replicate prediction timed out")
                raise error
            await asyncio.sleep(poll_interval)
