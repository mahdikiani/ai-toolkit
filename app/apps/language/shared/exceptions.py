"""Shared language-app API errors (fastapi_mongo_base format)."""

from fastapi_mongo_base import errors


class LanguageAppError(errors.BaseHTTPException):
    """Base for language-app service errors."""

    status_code: int = 500
    error_code: str = "language_app_error"
    message_en: str = "A language service error occurred"
    message_fa: str | None = "یک خطا در ارتباط با مدل‌های زبانی رخ داده است"

    def __init__(
        self,
        *,
        error_code: str | None = None,
        detail: str | None = None,
        message: dict | None = None,
        status_code: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize with optional overrides for status, code, and message."""
        super().__init__(
            status_code=status_code or self.status_code,
            error_code=error_code or self.error_code,
            detail=detail or self.message_en,
            message=message,
            **kwargs,
        )


class ItemNotFoundError(errors.NotFoundError):
    """Generic missing resource."""

    def __init__(self, detail: str) -> None:
        """Initialize with a custom not-found detail message."""
        super().__init__(detail=detail)


class ThreadNotFoundError(errors.NotFoundError):
    """Thread missing or not owned by the session."""

    message_en = "Thread not found"
    message_fa: str | None = "رشته‌ی گفت‌وگویی یافت نشد"


class OpenRouterNotConfiguredError(LanguageAppError):
    """OpenRouter API key is missing."""

    status_code = 503
    error_code = "service_unavailable"
    message_en = "OPENROUTER_API_KEY is not configured"
    message_fa: str | None = "OPENROUTER_API_KEY مجوز OpenRouter تنظیم نشده است"


class OpenRouterUpstreamError(LanguageAppError):
    """OpenRouter returned an error or the request failed upstream."""

    status_code = 502
    error_code = "upstream_error"
    message_en = "OpenRouter upstream error"
    message_fa: str | None = "خطا در ارتباط با OpenRouter"

    def __init__(self, detail: str | None = None) -> None:
        """Initialize with optional upstream error detail from OpenRouter."""
        super().__init__(detail=detail or self.message_en)


class OpenRouterInsufficientCreditsError(OpenRouterUpstreamError):
    """OpenRouter rejected the request due to insufficient provider credits."""

    status_code = 402
    error_code = "insufficient_credits"
    message_en = "Insufficient credits on upstream provider"
    message_fa: str | None = "اعتبار کافی برای استفاده از OpenRouter ندارید"

    def __init__(self) -> None:
        """Initialize with the fixed insufficient-credits message."""
        super().__init__()


class OpenRouterHttpError(LanguageAppError):
    """Forward a non-success OpenRouter HTTP status."""

    error_code = "upstream_error"
    message_en = "OpenRouter returned an error"
    message_fa: str | None = "خطا در ارتباط با OpenRouter"

    def __init__(self, status_code: int, detail: str) -> None:
        """Initialize with the forwarded OpenRouter HTTP status and detail."""
        super().__init__(status_code=status_code, detail=detail)


class ThreadHasNoMessagesError(LanguageAppError):
    """Completion requested on an empty thread."""

    status_code = 400
    error_code = "bad_request"
    message_en = "Thread has no messages yet"
    message_fa: str | None = "رشته‌ی گفت‌وگو هیچ پیامی ندارد"
