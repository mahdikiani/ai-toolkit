"""Financial utilities for quota management and configurable usage metering."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from ufaas import exceptions

from server.config import Settings

from .saas import QuotaSchema, UsageCreateSchema, UsageSchema

resource_variant = getattr(Settings, "UFAAS_RESOURCE_VARIANT", "")
DEFAULT_MARKUP = 1.2


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


PricingSection = dict[str, object]
PricingConfig = dict[str, PricingSection]

DEFAULT_PRICING: PricingConfig = {
    "text": {
        "markup": DEFAULT_MARKUP,
        "default_per_1k_tokens": 1.0,
        "models": {},
    },
    "speech": {
        "markup": DEFAULT_MARKUP,
        "default_per_1k_chars": 0.5,
    },
    "ocr": {
        "markup": DEFAULT_MARKUP,
        "default_per_page": 1.0,
        "engines": {},
    },
    "transcribe": {
        "markup": DEFAULT_MARKUP,
        "providers": {
            "soniox": {"per_minute": 1.0},
        },
    },
    "youtube": {
        "markup": DEFAULT_MARKUP,
        "per_request": 1.0,
    },
    "webpage": {
        "markup": DEFAULT_MARKUP,
        "per_request": 1.0,
    },
    "image": {
        "markup": DEFAULT_MARKUP,
        "default_per_image": 1.0,
        "models": {},
    },
    "web_search": {
        "markup": DEFAULT_MARKUP,
        "default_per_search": 1.0,
    },
    "video": {
        "markup": DEFAULT_MARKUP,
        "default_per_video": 1.0,
    },
    "voice_morph": {
        "markup": DEFAULT_MARKUP,
        "default_per_request": 1.0,
    },
}


def _pricing_section(value: object) -> PricingSection:
    """Normalize an untrusted pricing section to string-keyed values."""
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def pricing_config() -> PricingConfig:
    """Return configured pricing rules with safe defaults."""
    configured = getattr(Settings, "pricing", None)
    if isinstance(configured, dict):
        configured_sections = {
            key: _pricing_section(value)
            for key, value in configured.items()
            if isinstance(key, str)
        }
        return {
            key: {**default_section, **configured_sections.get(key, {})}
            for key, default_section in DEFAULT_PRICING.items()
        } | {
            key: section
            for key, section in configured_sections.items()
            if key not in DEFAULT_PRICING
        }
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
    model_pricing = _pricing_section(pricing.get("models")).get(model or "", {})
    model_pricing = _pricing_section(model_pricing)
    per_1k = float(
        model_pricing.get("per_1k_tokens") or pricing.get("default_per_1k_tokens", 1.0)
    )
    return (total_tokens / 1000) * per_1k * markup


def estimate_ocr_cost(*, pages: int, engine: str | None = None) -> float:
    """Estimate OCR cost by page count and engine."""
    pricing = pricing_config()["ocr"]
    engine_pricing = _pricing_section(pricing.get("engines")).get(engine or "", {})
    engine_pricing = _pricing_section(engine_pricing)
    per_page = float(engine_pricing.get("per_page") or pricing["default_per_page"])
    markup = float(pricing.get("markup", 1.0))
    return max(0, pages) * per_page * markup


def estimate_transcribe_cost(*, minutes: float, provider: str = "soniox") -> float:
    """Estimate transcription cost by duration and provider."""
    pricing = pricing_config()["transcribe"]
    provider_pricing = _pricing_section(pricing.get("providers")).get(provider, {})
    provider_pricing = _pricing_section(provider_pricing)
    per_minute = float(provider_pricing.get("per_minute", 1.0))
    markup = float(pricing.get("markup", 1.0))
    return max(0.0, minutes) * per_minute * markup


def estimate_youtube_cost() -> float:
    """Estimate one YouTube transcript API request cost."""
    pricing = pricing_config()["youtube"]
    return float(pricing.get("per_request", 1.0)) * float(
        pricing.get("markup", 1.0)
    )


def estimate_fixed_cost(section: str, price_key: str) -> float:
    """Estimate a fixed-price operation with its configured markup."""
    pricing = pricing_config().get(section) or {}
    return float(pricing.get(price_key, 1.0)) * float(pricing.get("markup", 1.0))


def estimate_speech_cost(*, chars: int) -> float:
    """Estimate text-to-speech cost from input character count."""
    pricing = pricing_config()["speech"]
    base = (max(0, chars) / 1000) * float(
        pricing.get("default_per_1k_chars", 0.5)
    )
    return max(0.01, base * float(pricing.get("markup", 1.0)))


def estimate_image_cost(*, count: int = 1, model: str | None = None) -> float:
    """Estimate image generation/edit cost with model override and markup."""
    pricing = pricing_config()["image"]
    model_pricing = _pricing_section(pricing.get("models")).get(model or "", {})
    model_pricing = _pricing_section(model_pricing)
    per_image = float(
        model_pricing.get("per_image") or pricing.get("default_per_image", 1.0)
    )
    return max(0, count) * per_image * float(pricing.get("markup", 1.0))


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
    user_id: str,
    amount: float,
    meta_data: dict | None = None,
    *,
    workspace_id: str | None = None,
    idempotency_key: str | None = None,
) -> UsageSchema | None:
    """
    Record usage cost for a user.

    Args:
        user_id: The user's unique identifier.
        amount: The cost amount to meter.
        meta_data: Optional metadata to attach to the usage record.
        workspace_id: When set, the cost is drawn from the workspace's
            shared quota pool instead of the user's personal balance,
            while user_id still records who incurred it.
        idempotency_key: Stable operation key used by Finance to make retries safe.

    Returns:
        The created usage record schema.
    """
    if not Settings.finance_api_key:
        return None
    async with get_ufaas_client() as ufaas_client:
        if idempotency_key is None and meta_data:
            service = meta_data.get("service")
            task_uid = meta_data.get("task_uid")
            if service and task_uid:
                idempotency_key = f"ai-toolkit:{service}:{task_uid}"
        usage_schema = UsageCreateSchema(
            user_id=user_id,
            workspace_id=workspace_id,
            asset="coin",
            amount=Decimal(str(amount)),
            variant=resource_variant,
            meta_data=meta_data,
            idempotency_key=idempotency_key,
        )
        attempts = 3 if idempotency_key else 1
        for attempt in range(attempts):
            try:
                usage_response = await ufaas_client.post(
                    "/usages", json=usage_schema.model_dump(mode="json")
                )
                usage_response.raise_for_status()
                return UsageSchema.model_validate(usage_response.json())
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable_status = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code >= 500
                )
                if attempt + 1 >= attempts or not retryable_status:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
    return None


async def get_quota(user_id: str, *, workspace_id: str | None = None) -> Decimal:
    """
    Retrieve the remaining quota for a user or a workspace.

    Args:
        user_id: The user's unique identifier.
        workspace_id: When set, the pooled quota of the workspace is
            returned instead of the user's personal quota.

    Returns:
        The remaining quota, or infinity if finance is disabled.
    """
    if not Settings.finance_api_key:
        return Decimal("inf")
    params = {"user_id": user_id, "asset": "coin", "variant": resource_variant}
    if workspace_id:
        params["workspace_id"] = workspace_id
    async with get_ufaas_client() as ufaas_client:
        quotas_response = await ufaas_client.get("/enrollments/quotas", params=params)
        quotas_response.raise_for_status()
        quotas = QuotaSchema.model_validate(quotas_response.json())
    return quotas.quota


async def cancel_usage(usage_id: str | None) -> None:
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
    user_id: str,
    coin: float,
    *,
    raise_exception: bool = True,
    workspace_id: str | None = None,
) -> Decimal:
    """
    Check if a user (or their workspace) has sufficient quota.

    Args:
        user_id: The user's unique identifier.
        coin: The required coin amount.
        raise_exception: Whether to raise an exception on insufficient funds.
        workspace_id: When set, checks the workspace's pooled quota instead
            of the user's personal quota.

    Returns:
        The current quota.

    Raises:
        InsufficientFundsError: If quota is insufficient and raise_exception is True.
    """
    quota = await get_quota(user_id, workspace_id=workspace_id)
    if raise_exception and (quota is None or quota < coin):
        error = _insufficient_funds_error(
            f"You have only {quota} coins, while you need {coin} coins."
        )
        raise error
    return quota


async def check_quota_or_error(
    user_id: str, coin: float, *, workspace_id: str | None = None
) -> Decimal:
    """
    Pre-flight quota check for direct request/response handlers.

    Raises a clean ``BaseHTTPException`` (402) instead of letting
    ``InsufficientFundsError`` propagate -- that exception isn't a
    ``BaseHTTPException``, so it isn't recognized by any registered
    handler and falls through to the generic 500 handler, which callers
    (e.g. mirza-bot's CompletionClient) can't tell apart from a real
    server fault.
    """
    try:
        return await check_quota(
            user_id, coin, raise_exception=True, workspace_id=workspace_id
        )
    except exceptions.InsufficientFundsError as exc:
        raise BaseHTTPException(
            status_code=402,
            error_code="insufficient_quota",
            detail=exc.detail,
            message={"en": exc.detail},
        ) from exc
