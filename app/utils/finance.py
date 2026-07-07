"""Financial utilities for quota management and configurable usage metering."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
from ufaas import exceptions

from server.config import Settings

from .saas import QuotaSchema, UsageCreateSchema, UsageSchema

resource_variant = getattr(Settings, "UFAAS_RESOURCE_VARIANT", "")


def _insufficient_funds_error(message: str) -> exceptions.InsufficientFundsError:
    """Build an InsufficientFundsError across compatible ufaas versions."""
    try:
        return exceptions.InsufficientFundsError(message)
    except TypeError:
        error = exceptions.InsufficientFundsError.__new__(
            exceptions.InsufficientFundsError
        )
        Exception.__init__(error, 402, message)
        error.status_code = 402
        error.error_code = "insufficient_funds"
        error.detail = message
        error.message = {"en": message}
        error.data = {}
        return error


DEFAULT_PRICING: dict[str, Any] = {
    "text": {
        "markup": 1.0,
        "default_per_1k_tokens": 1.0,
        "models": {},
    },
    "ocr": {
        "default_per_page": 1.0,
        "engines": {},
    },
    "transcribe": {
        "providers": {
            "soniox": {"per_minute": 1.0},
        },
    },
    "youtube": {
        "per_request": 1.0,
    },
}


def pricing_config() -> dict[str, Any]:
    """Return configured pricing rules with safe defaults."""
    configured = getattr(Settings, "pricing", None)
    if isinstance(configured, dict):
        return {**DEFAULT_PRICING, **configured}
    return DEFAULT_PRICING


def estimate_text_cost(
    *,
    model: str | None = None,
    usage: dict | None = None,
    raw_cost: float | int | str | None = None,
) -> float:
    """Estimate text model cost from provider usage metadata."""
    pricing = pricing_config()["text"]
    markup = float(pricing.get("markup", 1.0))
    if raw_cost is not None:
        return float(raw_cost) * markup

    total_tokens = 0
    if usage:
        total_tokens = int(usage.get("total_tokens") or 0)
        if not total_tokens:
            total_tokens = int(usage.get("prompt_tokens") or 0) + int(
                usage.get("completion_tokens") or 0
            )
    model_pricing = pricing.get("models", {}).get(model or "", {})
    per_1k = float(
        model_pricing.get("per_1k_tokens")
        or pricing.get("default_per_1k_tokens", 1.0)
    )
    return (total_tokens / 1000) * per_1k * markup


def estimate_ocr_cost(*, pages: int, engine: str | None = None) -> float:
    """Estimate OCR cost by page count and engine."""
    pricing = pricing_config()["ocr"]
    engine_pricing = pricing.get("engines", {}).get(engine or "", {})
    per_page = float(engine_pricing.get("per_page") or pricing["default_per_page"])
    return max(0, pages) * per_page


def estimate_transcribe_cost(*, minutes: float, provider: str = "soniox") -> float:
    """Estimate transcription cost by duration and provider."""
    pricing = pricing_config()["transcribe"]
    provider_pricing = pricing.get("providers", {}).get(provider, {})
    per_minute = float(provider_pricing.get("per_minute", 1.0))
    return max(0.0, minutes) * per_minute


def estimate_youtube_cost() -> float:
    """Estimate one YouTube transcript API request cost."""
    return float(pricing_config()["youtube"].get("per_request", 1.0))


@asynccontextmanager
async def get_ufaas_client() -> AsyncGenerator[httpx.AsyncClient]:
    """
    Create an async HTTP client configured for the uFaaS API.

    Yields:
        Configured httpx.AsyncClient for uFaaS API calls.
    """
    async with httpx.AsyncClient(
        base_url=Settings.finance_base_url or "https://saas.uln.me/api/saas/v1/",
        headers={"x-api-key": Settings.finance_api_key or ""},
    ) as client:
        yield client


async def meter_cost(
    user_id: str, amount: float, meta_data: dict | None = None
) -> UsageSchema:
    """
    Record usage cost for a user.

    Args:
        user_id: The user's unique identifier.
        amount: The cost amount to meter.
        meta_data: Optional metadata to attach to the usage record.

    Returns:
        The created usage record schema.
    """
    if not Settings.finance_api_key:
        return None
    async with get_ufaas_client() as ufaas_client:
        usage_schema = UsageCreateSchema(
            user_id=user_id,
            asset="coin",
            amount=Decimal(str(amount)),
            variant=resource_variant,
            meta_data=meta_data,
        )
        usage_response = await ufaas_client.post(
            "/usages", json=usage_schema.model_dump(mode="json")
        )
        usage_response.raise_for_status()
        usage = UsageSchema.model_validate(usage_response.json())
        return usage


async def get_quota(user_id: str) -> Decimal:
    """
    Retrieve the remaining quota for a user.

    Args:
        user_id: The user's unique identifier.

    Returns:
        The user's remaining quota, or infinity if finance is disabled.
    """
    if not Settings.finance_api_key:
        return Decimal("inf")
    async with get_ufaas_client() as ufaas_client:
        quotas_response = await ufaas_client.get(
            "/enrollments/quotas",
            params={"user_id": user_id, "asset": "coin", "variant": resource_variant},
        )
        quotas_response.raise_for_status()
        quotas = QuotaSchema.model_validate(quotas_response.json())
    return quotas.quota


async def cancel_usage(usage_id: str) -> None:
    """
    Cancel a previously recorded usage.

    Args:
        usage_id: The ID of the usage record to cancel.
    """
    if usage_id is None:
        return

    if not Settings.finance_api_key:
        return

    async with get_ufaas_client() as ufaas_client:
        await ufaas_client.post(f"/usages/{usage_id}/cancel")


async def check_quota(
    user_id: str, coin: float, *, raise_exception: bool = True
) -> Decimal:
    """
    Check if a user has sufficient quota.

    Args:
        user_id: The user's unique identifier.
        coin: The required coin amount.
        raise_exception: Whether to raise an exception on insufficient funds.

    Returns:
        The user's current quota.

    Raises:
        InsufficientFundsError: If quota is insufficient and raise_exception is True.
    """
    quota = await get_quota(user_id)
    if raise_exception and (quota is None or quota < coin):
        raise _insufficient_funds_error(
            f"You have only {quota} coins, while you need {coin} coins."
        )
    return quota
