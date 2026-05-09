"""FastAPI server configuration."""

import dataclasses
import os
from pathlib import Path
from typing import ClassVar

import dotenv
from fastapi_mongo_base.core import config

dotenv.load_dotenv()


@dataclasses.dataclass
class Settings(config.Settings):
    """Server config settings."""

    project_name: str = os.getenv("PROJECT_NAME", "pishrun ai")
    base_dir: Path = Path(__file__).resolve().parent.parent
    prompts_dir: ClassVar[Path] = (
        Path(__file__).resolve().parent.parent / "apps" / "executions" / "prompts"
    )
    base_path: str = "/api/ai/v1"
    storage_path: str = os.getenv("STORAGE_PATH", str(base_dir / "storage"))

    coverage_dir: Path = base_dir / "htmlcov"
    currency: str = "IRR"

    finance_api_key: str | None = os.getenv("FINANCE_API_KEY")
    finance_base_url: str | None = os.getenv("FINANCE_BASE_URL")
    media_api_key: str | None = os.getenv("MEDIA_API_KEY")
    media_base_url: str | None = os.getenv("MEDIA_BASE_URL")

    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    soniox_api_key: str | None = os.getenv("SONIOX_API_KEY")

    ocr_engine: str = os.getenv("OCR_ENGINE", "llm")
    paddle_pipeline_name: str = os.getenv("PADDLE_PIPELINE_NAME", "PaddleOCR-VL-1.5")
    paddle_device: str = os.getenv("PADDLE_DEVICE", "cpu")

    minutes_price: float = float(os.getenv("MINUTES_PRICE", 1))

    transcribe_enable_chunking: bool = (
        os.getenv("TRANSCRIBE_ENABLE_CHUNKING", "1") == "1"
    )
    transcribe_chunk_min_minutes: int = int(
        os.getenv("TRANSCRIBE_CHUNK_MIN_MINUTES", 5)
    )
    transcribe_chunk_max_minutes: int = int(
        os.getenv("TRANSCRIBE_CHUNK_MAX_MINUTES", 10)
    )
    transcribe_chunk_min_silence_ms: int = int(
        os.getenv("TRANSCRIBE_CHUNK_MIN_SILENCE_MS", 750)
    )
    transcribe_chunk_silence_threshold: int = int(
        os.getenv("TRANSCRIBE_CHUNK_SILENCE_THRESHOLD", -40)
    )
    transcribe_chunk_format: str = os.getenv("TRANSCRIBE_CHUNK_FORMAT", "wav")
    transcribe_max_parallel_requests: int = int(
        os.getenv("TRANSCRIBE_MAX_PARALLEL_REQUESTS", 3)
    )
    transcribe_poll_interval_seconds: float = float(
        os.getenv("TRANSCRIBE_POLL_INTERVAL_SECONDS", 5)
    )
    transcribe_chunking_fallback_single: bool = (
        os.getenv("TRANSCRIBE_CHUNKING_FALLBACK_SINGLE", "1") == "1"
    )

    @classmethod
    def get_log_config(cls, console_level: str = "INFO", **kwargs: object) -> dict:
        """Return logging configuration dictionary."""
        # Use absolute path for log file
        log_file_path = cls.base_dir / "logs" / "app.log"
        # Ensure logs directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        log_config = {
            "formatters": {
                "standard": {
                    "format": "[{levelname} {name} : {filename}:{lineno} : {asctime} -> {funcName:10}] {message}",  # noqa: E501
                    "style": "{",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": console_level,
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "filename": str(log_file_path),
                },
            },
            "loggers": {
                "": {
                    "handlers": [
                        "console",
                        "file",
                    ],
                    "level": console_level,
                    "propagate": True,
                },
            },
            "version": 1,
        }
        return log_config
