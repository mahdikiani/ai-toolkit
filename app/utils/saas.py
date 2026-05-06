"""Provide module functionality."""
from decimal import Decimal
from typing import Self

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.utils.bsontools import decimal_amount
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Bundle(BaseModel):
    """Represent Bundle."""

    asset: str
    quota: Decimal
    unit: str | None = None

    model_config = ConfigDict(allow_inf_nan=True)

    @field_validator("quota", mode="before")
    @classmethod
    def validate_quota(cls, value: Decimal) -> Decimal:
        """Run validate quota."""
        return decimal_amount(value)


class UsageConsumption(BaseModel):
    """Represent UsageConsumption."""

    enrollment_id: str
    amount: Decimal
    leftover_bundles: list[Bundle] = []

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """Run validate amount."""
        return decimal_amount(value)


class UsageCreateSchema(BaseModel):
    """Represent UsageCreateSchema."""

    user_id: str | None = None
    enrollment_id: str | None = None
    asset: str
    amount: Decimal = Decimal(1)
    variant: str | None = None
    meta_data: dict | None = None

    @model_validator(mode="after")
    def validate_enrollment_id(self) -> Self:
        """Run validate enrollment id."""
        if not self.user_id and not self.enrollment_id:
            raise ValueError("Either user_id or enrollment_id must be provided")
        return self

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """Run validate amount."""
        if value <= 0:
            raise ValueError("Amount must be greater than 0")
        return value


class UsageSchema(TenantUserEntitySchema):
    # enrollment_id: str
    # asset: str
    # amount: Decimal
    """Represent UsageSchema."""

    consumptions: list[UsageConsumption]
    asset: str
    amount: Decimal
    variant: str | None = None

    # @classmethod
    # def search_field_set(cls) -> list[str]:
    #     return list(set(super().search_field_set() + ["asset", "variant"]))

    @classmethod
    def search_exclude_set(cls) -> list[str]:
        """Run search exclude set."""
        return list({*super().search_field_set(), "consumptions"})

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """Run validate amount."""
        return decimal_amount(value)


class QuotaSchema(BaseModel):
    """Represent QuotaSchema."""

    user_id: str | None = None
    asset: str
    quota: Decimal
    unit: str | None = None
    variant: str | None = None
    _quota: Decimal | None = None

    model_config = ConfigDict(allow_inf_nan=True)
