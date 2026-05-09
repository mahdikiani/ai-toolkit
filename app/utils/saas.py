"""SaaS schema definitions for usage, quota, and bundle management."""

from decimal import Decimal
from typing import Self

from fastapi_mongo_base.schemas import TenantUserEntitySchema
from fastapi_mongo_base.utils.bsontools import decimal_amount
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Bundle(BaseModel):
    """Represents a resource bundle with asset and quota information."""

    asset: str
    quota: Decimal
    unit: str | None = None

    model_config = ConfigDict(allow_inf_nan=True)

    @field_validator("quota", mode="before")
    @classmethod
    def validate_quota(cls, value: Decimal) -> Decimal:
        """
        Validate and normalize the quota amount.

        Args:
            value: Raw quota value.

        Returns:
            Normalized Decimal amount.
        """
        return decimal_amount(value)


class UsageConsumption(BaseModel):
    """Represents a usage consumption record with enrollment and amount."""

    enrollment_id: str
    amount: Decimal
    leftover_bundles: list[Bundle] = []

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """
        Validate and normalize the consumption amount.

        Args:
            value: Raw amount value.

        Returns:
            Normalized Decimal amount.
        """
        return decimal_amount(value)


class UsageCreateSchema(BaseModel):
    """Schema for creating a new usage record."""

    user_id: str | None = None
    enrollment_id: str | None = None
    asset: str
    amount: Decimal = Decimal(1)
    variant: str | None = None
    meta_data: dict | None = None

    @model_validator(mode="after")
    def validate_enrollment_id(self) -> Self:
        """
        Validate that either user_id or enrollment_id is provided.

        Returns:
            The validated schema instance.

        Raises:
            ValueError: If neither user_id nor enrollment_id is provided.
        """
        if not self.user_id and not self.enrollment_id:
            raise ValueError("Either user_id or enrollment_id must be provided")
        return self

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """
        Validate that the amount is positive.

        Args:
            value: Amount to validate.

        Returns:
            The validated amount.

        Raises:
            ValueError: If amount is not greater than 0.
        """
        if value <= 0:
            raise ValueError("Amount must be greater than 0")
        return value


class UsageSchema(TenantUserEntitySchema):
    """Schema representing a usage record with consumption details."""

    # enrollment_id: str
    # asset: str
    # amount: Decimal

    consumptions: list[UsageConsumption]
    asset: str
    amount: Decimal
    variant: str | None = None

    # @classmethod
    # def search_field_set(cls) -> list[str]:
    #     return list(set(super().search_field_set() + ["asset", "variant"]))

    @classmethod
    def search_exclude_set(cls) -> list[str]:
        """
        Return the set of fields to exclude from search results.

        Returns:
            List of field names to exclude.
        """
        return list({*super().search_field_set(), "consumptions"})

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """
        Validate and normalize the usage amount.

        Args:
            value: Raw amount value.

        Returns:
            Normalized Decimal amount.
        """
        return decimal_amount(value)


class QuotaSchema(BaseModel):
    """Schema representing a user's quota information."""

    user_id: str | None = None
    asset: str
    quota: Decimal
    unit: str | None = None
    variant: str | None = None
    _quota: Decimal | None = None

    model_config = ConfigDict(allow_inf_nan=True)
