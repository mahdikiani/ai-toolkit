"""
Provider proxy services for completion routes.

Canonical OpenAI-compatible surface is ``/openai/v1`` (``apps.openai_compat``).
These helpers wrap the shared openai_compat services for leftover callers
and for the unmounted ``language.completion`` routes module.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from apps.openai_compat.services import (
    estimate_chat_cost,
    handle_non_stream_chat,
    handle_stream_chat,
    meter_chat_usage,
)
from server.config import Settings
from utils.billing import finance
from utils.integrations import openrouter as openrouter_client


def _estimate_input_cost(payload: dict) -> float:
    """Estimate max cost from input messages for quota pre-check."""
    return estimate_chat_cost(payload)


async def meter_completion_response(raw: bytes, *, user_id: str) -> None:
    """Record completion usage from a provider JSON response when available."""
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    provider_meta = openrouter_client.extract_provider_meta(data, provider="openrouter")
    await meter_chat_usage(
        user_id=user_id,
        model=str(provider_meta.get("model") or ""),
        usage=provider_meta.get("usage")
        if isinstance(provider_meta.get("usage"), dict)
        else None,
        provider_meta=provider_meta,
        service="completion",
    )


async def proxy_chat_completions(
    payload: dict,
    *,
    user_id: str,
) -> tuple[bytes, str | None, int]:
    """Forward a non-streaming chat completion via shared openai_compat path."""
    if not payload.get("model"):
        payload = {**payload, "model": Settings.default_model}
    response = await handle_non_stream_chat(
        payload,
        user_id=user_id,
        model=payload["model"],
        service="completion",
    )
    return response.body, "application/json", response.status_code


async def proxy_chat_completions_raw_stream(
    payload: dict,
    *,
    user_id: str = "anonymous",
) -> AsyncIterator[bytes]:
    """Forward streaming chat completions using shared openai_compat metering."""
    if not payload.get("model"):
        payload = {**payload, "model": Settings.default_model}
    streaming = await handle_stream_chat(payload, user_id=user_id, service="completion")
    async for chunk in streaming.body_iterator:
        if isinstance(chunk, bytes):
            yield chunk
        else:
            yield str(chunk).encode()


__all__ = [
    "_estimate_input_cost",
    "estimate_chat_cost",
    "finance",
    "handle_non_stream_chat",
    "handle_stream_chat",
    "meter_chat_usage",
    "meter_completion_response",
    "proxy_chat_completions",
    "proxy_chat_completions_raw_stream",
]
