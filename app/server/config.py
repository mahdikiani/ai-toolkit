"""FastAPI server configuration."""

import dataclasses
import os
from pathlib import Path

import dotenv
from fastapi_mongo_base.core import config

dotenv.load_dotenv()


@dataclasses.dataclass
class Settings(config.Settings):
    """Server config settings."""

    project_name: str = os.getenv("PROJECT_NAME", "pishrun ai")
    base_dir: Path = Path(__file__).resolve().parent.parent
    base_path: str = "/api/ai/v1"
    storage_path: str = os.getenv("STORAGE_PATH", str(base_dir / "storage"))

    coverage_dir: Path = base_dir / "htmlcov"
    currency: str = "IRR"

    finance_api_key: str | None = os.getenv("FINANCE_API_KEY")
    finance_base_url: str | None = os.getenv("FINANCE_BASE_URL")
    media_api_key: str | None = os.getenv("MEDIA_API_KEY")
    media_base_url: str | None = os.getenv("MEDIA_BASE_URL")

    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    metis_api_key: str | None = os.getenv("METIS_API_KEY")
    pishrun_api_key: str | None = os.getenv("PISHRUN_API_KEY")
    dify_api_key: str | None = os.getenv("DIFY_API_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    soniox_api_key: str | None = os.getenv("SONIOX_API_KEY")

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
                    "filename": "logs/app.log",
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
