"""Unit tests for finance utilities."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestGetQuota:
    """Tests for get_quota function."""

    async def test_returns_infinity_when_no_api_key(self) -> None:
        """get_quota should return infinity when finance API key is not set."""
        from utils.finance import get_quota

        with patch("utils.finance.Settings") as mock_settings:
            mock_settings.finance_api_key = None

            result = await get_quota("user_123")

        assert result == Decimal("inf")

    async def test_returns_quota_from_api(self) -> None:
        """get_quota should return quota from the finance API."""
        from utils.finance import get_quota

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"quota": "500.00"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("utils.finance.Settings") as mock_settings,
            patch("utils.finance.get_ufaas_client") as mock_get_client,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)

            # Mock QuotaSchema
            with patch("utils.finance.QuotaSchema") as mock_schema:
                mock_schema.model_validate.return_value = MagicMock(
                    quota=Decimal("500.00")
                )
                result = await get_quota("user_123")

        assert result == Decimal("500.00")


@pytest.mark.unit
class TestCheckQuota:
    """Tests for check_quota function."""

    async def test_returns_quota_when_sufficient(self) -> None:
        """check_quota should return quota when it's sufficient."""
        from utils.finance import check_quota

        with patch(
            "utils.finance.get_quota",
            new_callable=AsyncMock,
            return_value=Decimal("100"),
        ):
            result = await check_quota("user_123", 10.0)

        assert result == Decimal("100")

    async def test_raises_when_insufficient_and_raise_exception_true(self) -> None:
        """check_quota should raise InsufficientFundsError when quota is insufficient."""
        from ufaas import exceptions

        from utils.finance import check_quota

        with patch(
            "utils.finance.get_quota", new_callable=AsyncMock, return_value=Decimal("5")
        ), pytest.raises(exceptions.InsufficientFundsError):
            await check_quota("user_123", 10.0, raise_exception=True)

    async def test_returns_quota_when_insufficient_and_raise_exception_false(
        self,
    ) -> None:
        """check_quota should return quota without raising when raise_exception=False."""
        from utils.finance import check_quota

        with patch(
            "utils.finance.get_quota", new_callable=AsyncMock, return_value=Decimal("5")
        ):
            result = await check_quota("user_123", 10.0, raise_exception=False)

        assert result == Decimal("5")


@pytest.mark.unit
class TestMeterCost:
    """Tests for meter_cost function."""

    async def test_returns_none_when_no_api_key(self) -> None:
        """meter_cost should return None when finance API key is not set."""
        from utils.finance import meter_cost

        with patch("utils.finance.Settings") as mock_settings:
            mock_settings.finance_api_key = None

            result = await meter_cost("user_123", 5.0)

        assert result is None

    async def test_records_usage_when_api_key_set(self) -> None:
        """meter_cost should record usage when API key is configured."""
        from utils.finance import meter_cost

        mock_usage = MagicMock()
        mock_usage.uid = "usage_123"
        mock_usage.amount = Decimal("5.0")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"uid": "usage_123", "amount": "5.0"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("utils.finance.Settings") as mock_settings,
            patch("utils.finance.get_ufaas_client") as mock_get_client,
            patch("utils.finance.UsageSchema") as mock_schema,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_schema.model_validate.return_value = mock_usage

            result = await meter_cost("user_123", 5.0)

        assert result == mock_usage
