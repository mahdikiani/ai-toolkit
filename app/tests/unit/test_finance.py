"""Unit tests for finance utilities."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ufaas import exceptions


@pytest.mark.unit
class TestGetQuota:
    """Tests for get_quota function."""

    async def test_returns_infinity_when_no_api_key(self) -> None:
        """get_quota should return infinity when finance API key is not set."""
        from utils.billing.finance import get_quota

        with patch("utils.billing.finance.Settings") as mock_settings:
            mock_settings.finance_api_key = None

            result = await get_quota("user_123")

        assert result == Decimal("inf")

    async def test_returns_quota_from_api(self) -> None:
        """get_quota should return quota from the finance API."""
        from utils.billing.finance import get_quota

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"quota": "500.00"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("utils.billing.finance.Settings") as mock_settings,
            patch("utils.billing.finance.get_ufaas_client") as mock_get_client,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)

            # Mock QuotaSchema
            with patch("utils.billing.finance.QuotaSchema") as mock_schema:
                mock_schema.model_validate.return_value = MagicMock(
                    quota=Decimal("500.00")
                )
                result = await get_quota("user_123")

        assert result == Decimal("500.00")

    async def test_forwards_workspace_id_when_provided(self) -> None:
        """get_quota should pass workspace_id through as a query param."""
        from utils.billing.finance import get_quota

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"quota": "500.00"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("utils.billing.finance.Settings") as mock_settings,
            patch("utils.billing.finance.get_ufaas_client") as mock_get_client,
            patch("utils.billing.finance.QuotaSchema") as mock_schema,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_schema.model_validate.return_value = MagicMock(
                quota=Decimal("500.00")
            )

            await get_quota("user_123", workspace_id="workspace_1")

        assert mock_client.get.call_args.kwargs["params"]["workspace_id"] == (
            "workspace_1"
        )

    async def test_omits_workspace_id_when_not_provided(self) -> None:
        """get_quota should not send workspace_id when it's absent."""
        from utils.billing.finance import get_quota

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"quota": "500.00"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("utils.billing.finance.Settings") as mock_settings,
            patch("utils.billing.finance.get_ufaas_client") as mock_get_client,
            patch("utils.billing.finance.QuotaSchema") as mock_schema,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_schema.model_validate.return_value = MagicMock(
                quota=Decimal("500.00")
            )

            await get_quota("user_123")

        assert "workspace_id" not in mock_client.get.call_args.kwargs["params"]


@pytest.mark.unit
class TestCheckQuota:
    """Tests for check_quota function."""

    async def test_returns_quota_when_sufficient(self) -> None:
        """check_quota should return quota when it's sufficient."""
        from utils.billing.finance import check_quota

        with patch(
            "utils.billing.finance.get_quota",
            new_callable=AsyncMock,
            return_value=Decimal("100"),
        ):
            result = await check_quota("user_123", 10.0)

        assert result == Decimal("100")

    async def test_raises_when_insufficient_and_raise_exception_true(self) -> None:
        """Raise InsufficientFundsError for insufficient quota."""
        from ufaas import exceptions

        from utils.billing.finance import check_quota

        with (
            patch(
                "utils.billing.finance.get_quota",
                new_callable=AsyncMock,
                return_value=Decimal("5"),
            ),
            pytest.raises(exceptions.InsufficientFundsError),
        ):
            await check_quota("user_123", 10.0, raise_exception=True)

    async def test_returns_quota_when_insufficient_and_raise_exception_false(
        self,
    ) -> None:
        """Return quota without raising when raise_exception is false."""
        from utils.billing.finance import check_quota

        with patch(
            "utils.billing.finance.get_quota",
            new_callable=AsyncMock,
            return_value=Decimal("5"),
        ):
            result = await check_quota("user_123", 10.0, raise_exception=False)

        assert result == Decimal("5")

    async def test_forwards_workspace_id_to_get_quota(self) -> None:
        """check_quota should forward workspace_id to get_quota."""
        from utils.billing.finance import check_quota

        with patch(
            "utils.billing.finance.get_quota",
            new_callable=AsyncMock,
            return_value=Decimal("100"),
        ) as mock_get_quota:
            await check_quota("user_123", 10.0, workspace_id="workspace_1")

        mock_get_quota.assert_awaited_once_with("user_123", workspace_id="workspace_1")


@pytest.mark.unit
class TestCheckQuotaOrError:
    """Tests for check_quota_or_error function."""

    async def test_forwards_workspace_id_to_check_quota(self) -> None:
        """check_quota_or_error should forward workspace_id to check_quota."""
        from utils.billing.finance import check_quota_or_error

        with patch(
            "utils.billing.finance.check_quota",
            new_callable=AsyncMock,
            return_value=Decimal("100"),
        ) as mock_check_quota:
            result = await check_quota_or_error(
                "user_123", 10.0, workspace_id="workspace_1"
            )

        assert result == Decimal("100")
        mock_check_quota.assert_awaited_once_with(
            "user_123", 10.0, raise_exception=True, workspace_id="workspace_1"
        )

    async def test_raises_clean_402_on_insufficient_funds(self) -> None:
        """check_quota_or_error should convert InsufficientFundsError to a 402."""
        from fastapi_mongo_base.core.exceptions import BaseHTTPException

        from utils.billing.finance import (
            _insufficient_funds_error,
            check_quota_or_error,
        )

        with (
            patch(
                "utils.billing.finance.check_quota",
                new_callable=AsyncMock,
                side_effect=_insufficient_funds_error("not enough"),
            ),
            pytest.raises(BaseHTTPException) as exc_info,
        ):
            await check_quota_or_error("user_123", 10.0)

        assert exc_info.value.status_code == 402


@pytest.mark.unit
class TestMeterCost:
    """Tests for meter_cost function."""

    async def test_returns_none_when_no_api_key(self) -> None:
        """meter_cost should return None when finance API key is not set."""
        from utils.billing.finance import meter_cost

        with patch("utils.billing.finance.Settings") as mock_settings:
            mock_settings.finance_api_key = None

            result = await meter_cost("user_123", 5.0)

        assert result is None

    async def test_records_usage_when_api_key_set(self) -> None:
        """meter_cost should record usage when API key is configured."""
        from utils.billing.finance import meter_cost

        mock_usage = MagicMock()
        mock_usage.uid = "usage_123"
        mock_usage.amount = Decimal("5.0")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"uid": "usage_123", "amount": "5.0"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("utils.billing.finance.Settings") as mock_settings,
            patch("utils.billing.finance.get_ufaas_client") as mock_get_client,
            patch("utils.billing.finance.UsageSchema") as mock_schema,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_schema.model_validate.return_value = mock_usage

            result = await meter_cost("user_123", 5.0)

        assert result == mock_usage

    async def test_forwards_workspace_id_in_usage_schema(self) -> None:
        """meter_cost should stamp workspace_id onto the created usage record."""
        from utils.billing.finance import meter_cost

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"uid": "usage_123", "amount": "5.0"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("utils.billing.finance.Settings") as mock_settings,
            patch("utils.billing.finance.get_ufaas_client") as mock_get_client,
            patch("utils.billing.finance.UsageSchema") as mock_schema,
        ):
            mock_settings.finance_api_key = "test_key"
            mock_get_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_get_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_schema.model_validate.return_value = MagicMock()

            await meter_cost("user_123", 5.0, workspace_id="workspace_1")

        sent_body = mock_client.post.call_args.kwargs["json"]
        assert sent_body["workspace_id"] == "workspace_1"


@pytest.mark.unit
class TestInsufficientFundsError:
    """Tests for _insufficient_funds_error function."""

    def test_constructs_error_normally(self) -> None:
        """Should construct InsufficientFundsError with message."""
        from utils.billing.finance import _insufficient_funds_error

        error = _insufficient_funds_error("test error")
        assert isinstance(error, exceptions.InsufficientFundsError)

    def test_fallback_construct_on_type_error(self) -> None:
        """Should fallback when constructor raises TypeError."""
        from utils.billing.finance import _insufficient_funds_error

        def raise_type_error(self: object, msg: str) -> None:
            raise TypeError

        with patch.object(
            exceptions.InsufficientFundsError, "__init__", raise_type_error
        ):
            error = _insufficient_funds_error("test error")
            assert error.status_code == 402
            assert error.__dict__["error_code"] == "insufficient_funds"
            assert error.detail == "test error"


@pytest.mark.unit
class TestPricingConfig:
    """Tests for pricing_config function."""

    def test_returns_defaults_when_no_pricing_configured(self) -> None:
        """Should return DEFAULT_PRICING when Settings has no pricing."""
        from utils.billing.finance import pricing_config

        with patch("utils.billing.finance.Settings") as mock_settings:
            mock_settings.pricing = None
            result = pricing_config()

        assert "text" in result
        assert "ocr" in result
        assert "transcribe" in result
        assert "youtube" in result

    def test_merges_with_configured_pricing(self) -> None:
        """Should merge DEFAULT_PRICING with configured pricing."""
        from utils.billing.finance import pricing_config

        custom = {"text": {"markup": 2.0}}
        with patch("utils.billing.finance.Settings") as mock_settings:
            mock_settings.pricing = custom
            result = pricing_config()

        assert result["text"]["markup"] == pytest.approx(2.0)


@pytest.mark.unit
class TestEstimateTextCost:
    """Tests for estimate_text_cost function."""

    def test_uses_raw_cost(self) -> None:
        """Should use raw_cost when provided."""
        from utils.billing.finance import estimate_text_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={"text": {"markup": 1.0}},
        ):
            result = estimate_text_cost(raw_cost=10.0)

        assert result == pytest.approx(10.0)

    def test_uses_total_tokens_from_usage(self) -> None:
        """Should use total_tokens from usage dict."""
        from utils.billing.finance import estimate_text_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={
                "text": {"markup": 1.0, "default_per_1k_tokens": 2.0, "models": {}}
            },
        ):
            result = estimate_text_cost(usage={"total_tokens": 1000})

        assert result == pytest.approx(2.0)

    def test_falls_back_to_prompt_completion_tokens(self) -> None:
        """Should sum prompt and completion tokens when total_tokens missing."""
        from utils.billing.finance import estimate_text_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={
                "text": {"markup": 1.0, "default_per_1k_tokens": 2.0, "models": {}}
            },
        ):
            result = estimate_text_cost(
                usage={"prompt_tokens": 400, "completion_tokens": 600}
            )

        assert result == pytest.approx(2.0)

    def test_uses_model_specific_pricing(self) -> None:
        """Should use model-specific per_1k_tokens."""
        from utils.billing.finance import estimate_text_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={
                "text": {
                    "markup": 1.0,
                    "default_per_1k_tokens": 1.0,
                    "models": {"gpt-4": {"per_1k_tokens": 3.0}},
                }
            },
        ):
            result = estimate_text_cost(usage={"total_tokens": 1000}, model="gpt-4")

        assert result == pytest.approx(3.0)


@pytest.mark.unit
class TestEstimateOcrCost:
    """Tests for estimate_ocr_cost function."""

    def test_default_per_page(self) -> None:
        """Should use default per_page when no engine specified."""
        from utils.billing.finance import estimate_ocr_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={"ocr": {"default_per_page": 5.0, "engines": {}}},
        ):
            result = estimate_ocr_cost(pages=3)

        assert result == pytest.approx(15.0)

    def test_engine_specific_pricing(self) -> None:
        """Should use engine-specific per_page."""
        from utils.billing.finance import estimate_ocr_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={
                "ocr": {
                    "default_per_page": 1.0,
                    "engines": {"paddle": {"per_page": 2.0}},
                }
            },
        ):
            result = estimate_ocr_cost(pages=3, engine="paddle")

        assert result == pytest.approx(6.0)


@pytest.mark.unit
class TestEstimateTranscribeCost:
    """Tests for estimate_transcribe_cost function."""

    def test_default_provider_pricing(self) -> None:
        """Should use default provider pricing."""
        from utils.billing.finance import estimate_transcribe_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={"transcribe": {"providers": {"soniox": {"per_minute": 1.0}}}},
        ):
            result = estimate_transcribe_cost(minutes=5.0)

        assert result == pytest.approx(5.0)

    def test_custom_provider_pricing(self) -> None:
        """Should use specified provider pricing."""
        from utils.billing.finance import estimate_transcribe_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={"transcribe": {"providers": {"custom": {"per_minute": 2.0}}}},
        ):
            result = estimate_transcribe_cost(minutes=3.0, provider="custom")

        assert result == pytest.approx(6.0)


@pytest.mark.unit
class TestEstimateYoutubeCost:
    """Tests for estimate_youtube_cost function."""

    def test_returns_per_request_cost(self) -> None:
        """Should return per_request from youtube pricing."""
        from utils.billing.finance import estimate_youtube_cost

        with patch(
            "utils.billing.finance.pricing_config",
            return_value={"youtube": {"per_request": 2.5}},
        ):
            result = estimate_youtube_cost()

        assert result == pytest.approx(2.5)


@pytest.mark.unit
class TestCancelUsage:
    """Tests for cancel_usage function."""

    async def test_returns_none_when_usage_id_is_empty(self) -> None:
        """Should return None when usage_id is empty."""
        from utils.billing.finance import cancel_usage

        result = await cancel_usage("")
        assert result is None

    async def test_returns_none_when_no_api_key(self) -> None:
        """Should return None when finance API key is not set."""
        from utils.billing.finance import cancel_usage

        with patch("utils.billing.finance.Settings") as mock_settings:
            mock_settings.finance_api_key = None
            result = await cancel_usage("usage_123")

        assert result is None
