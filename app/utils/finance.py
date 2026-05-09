"""Financial utilities for quota management and usage metering via uFaaS."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
from ufaas import exceptions

from server.config import Settings

from .saas import QuotaSchema, UsageCreateSchema, UsageSchema

resource_variant = getattr(Settings, "UFAAS_RESOURCE_VARIANT", "")


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
        raise exceptions.InsufficientFundsError(
            f"You have only {quota} coins, while you need {coin} coins."
        )
    return quota
