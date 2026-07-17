"""Provider proxy services for completion routes."""

import json
from collections.abc import AsyncIterator

from apps.language.shared.exceptions import (
    OpenRouterHttpError,
    OpenRouterInsufficientCreditsError,
    OpenRouterNotConfiguredError,
)
from server.config import Settings
from utils.billing import finance
from utils.integrations import openrouter as openrouter_client


def _estimate_input_cost(payload: dict) -> float:
    """Estimate max cost from input messages for quota pre-check."""
    pricing = finance.pricing_config()["text"]
    markup = float(pricing.get("markup", 1.0))
    model = payload.get("model", Settings.default_model)

    total_chars = 0
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                total_chars += (
                    len(part.get("text", "")) if isinstance(part, dict) else 0
                )

    estimated_tokens = max(100, total_chars // 4)
    models_cfg = pricing.get("models", {})
    if isinstance(models_cfg, dict):
        model_cfg = models_cfg.get(model, {})
        if isinstance(model_cfg, dict):
            per_1k = float(
                model_cfg.get("per_1k_tokens")
                or pricing.get("default_per_1k_tokens", 1.0)
            )
        else:
            per_1k = float(pricing.get("default_per_1k_tokens", 1.0))
    else:
        per_1k = float(pricing.get("default_per_1k_tokens", 1.0))
    return (estimated_tokens / 1000) * per_1k * markup


async def proxy_chat_completions(
    payload: dict,
    *,
    user_id: str,
) -> tuple[bytes, str | None, int]:
    """Forward a non-streaming chat completion request to OpenRouter."""
    if not payload.get("model"):
        payload = {**payload, "model": Settings.default_model}
    estimated = _estimate_input_cost(payload)
    await finance.check_quota(user_id, estimated, raise_exception=True)
    try:
        resp = await openrouter_client.post_chat_completion_unchecked(payload)
    except ValueError:
        raise OpenRouterNotConfiguredError() from None
    ctype = resp.headers.get("content-type")
    if resp.status_code < 400:
        await meter_completion_response(resp.content, user_id=user_id)
    elif resp.status_code == 402:
        raise OpenRouterInsufficientCreditsError()
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
    amount = finance.estimate_text_cost(
        model=str(provider_meta.get("model") or ""),
        usage=provider_meta.get("usage"),
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
    if not payload.get("model"):
        payload = {**payload, "model": Settings.default_model}
    if not Settings.openrouter_api_key:
        raise OpenRouterNotConfiguredError()
    try:
        async for chunk in openrouter_client.stream_chat_completion_bytes(payload):
            yield chunk
    except openrouter_client.OpenRouterError as e:
        raise OpenRouterHttpError(e.status_code, e.detail) from e
