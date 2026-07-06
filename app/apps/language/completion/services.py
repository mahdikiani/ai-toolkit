"""Provider proxy services for completion routes."""

import json
from collections.abc import AsyncIterator

from fastapi import HTTPException

from utils import finance
from utils import openrouter as openrouter_client


async def proxy_chat_completions(
    payload: dict,
    *,
    user_id: str,
) -> tuple[bytes, str | None, int]:
    """Forward a non-streaming chat completion request to OpenRouter."""
    try:
        resp = await openrouter_client.post_chat_completion_unchecked(payload)
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    ctype = resp.headers.get("content-type")
    if resp.status_code < 400:
        await meter_completion_response(resp.content, user_id=user_id)
    return resp.content, ctype, resp.status_code


async def meter_completion_response(raw: bytes, *, user_id: str) -> None:
    """Record completion usage from a provider JSON response when available."""
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    provider_meta = openrouter_client.extract_provider_meta(data, provider="openrouter")
    provider_usage = provider_meta.get("usage")
    amount = finance.estimate_text_cost(
        model=str(provider_meta.get("model") or ""),
        usage=provider_usage if isinstance(provider_usage, dict) else None,
        raw_cost=provider_meta.get("raw_cost"),
    )
    await finance.meter_cost(
        user_id,
        amount,
        meta_data={
            "service": "completion",
            "provider_meta": provider_meta,
        },
    )


async def proxy_chat_completions_raw_stream(payload: dict) -> AsyncIterator[bytes]:
    """Forward streaming chat completions and yield raw SSE bytes."""
    try:
        async for chunk in openrouter_client.stream_chat_completion_bytes(payload):
            yield chunk
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    except openrouter_client.OpenRouterError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
