"""FastAPI server configuration."""

import dataclasses
import json
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
        Path(__file__).resolve().parent.parent
        / "apps"
        / "language"
        / "promptic"
        / "prompts"
    )
    base_path: str = "/api/ai/v1"
    storage_path: str = os.getenv("STORAGE_PATH", str(base_dir / "storage"))

    coverage_dir: Path = base_dir / "htmlcov"
    currency: str = "IRR"

    finance_api_key: str | None = os.getenv("FINANCE_API_KEY")
    finance_base_url: str | None = os.getenv("FINANCE_BASE_URL")
    media_api_key: str | None = os.getenv("MEDIA_API_KEY")
    media_base_url: str | None = os.getenv("MEDIA_BASE_URL")

    youtube_transcript_api_key: str | None = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    title_model: str = os.getenv("OPENROUTER_TITLE_MODEL", "openai/gpt-5.6-terra")
    soniox_api_key: str | None = os.getenv("SONIOX_API_KEY")
    soniox_webhook_secret: str | None = os.getenv("SONIOX_WEBHOOK_SECRET")
    soniox_ws_url: str = os.getenv(
        "SONIOX_WS_URL", "wss://stt-rt.soniox.com/transcribe-websocket"
    )
    soniox_rt_model: str = os.getenv("SONIOX_RT_MODEL", "stt-rt-v5")
    soniox_rt_language_hints: ClassVar[list[str]] = json.loads(
        os.getenv("SONIOX_RT_LANGUAGE_HINTS", '["fa", "en"]')
    )
    soniox_rt_max_buffer_bytes: int = int(
        os.getenv("SONIOX_RT_MAX_BUFFER_BYTES", str(50 * 1024 * 1024))
    )
    openai_compat_models: str = os.getenv("OPENAI_COMPAT_MODELS", "")
    openai_stream_max_bytes: int = int(
        os.getenv("OPENAI_STREAM_MAX_BYTES", str(8 * 1024 * 1024))
    )
    default_tts_model: str = os.getenv("DEFAULT_TTS_MODEL", "openai/gpt-4o-mini-tts")

    exa_api_key: str | None = os.getenv("EXA_API_KEY")
    replicate_api_key: str | None = os.getenv("REPLICATE_API_KEY")
    fal_api_key: str | None = os.getenv("FAL_API_KEY")

    ocr_engine: str = os.getenv("OCR_ENGINE", "pipeline")
    ocr_pipeline_dpi: int = int(os.getenv("OCR_PIPELINE_DPI", "300"))
    ocr_pipeline_enable_layout: bool = (
        os.getenv("OCR_PIPELINE_ENABLE_LAYOUT", "true").lower() == "true"
    )
    ocr_pipeline_enable_preprocessing: bool = (
        os.getenv("OCR_PIPELINE_ENABLE_PREPROCESSING", "true").lower() == "true"
    )
    ocr_vlm_model: str = os.getenv("OCR_VLM_MODEL", "google/gemini-3.1-flash-lite")
    ocr_layout_confidence_threshold: float = float(
        os.getenv("OCR_LAYOUT_CONFIDENCE_THRESHOLD", "0.5")
    )
    paddle_pipeline_name: str = os.getenv("PADDLE_PIPELINE_NAME", "PaddleOCR-VL-1.5")
    paddle_device: str = os.getenv("PADDLE_DEVICE", "cpu")

    ocr_di_crop_padding_ratio: float = float(
        os.getenv("OCR_DI_CROP_PADDING_RATIO", "0.10")
    )
    ocr_di_iou_threshold: float = float(os.getenv("OCR_DI_IOU_THRESHOLD", "0.40"))
    ocr_di_max_concurrent_vlm: int = int(os.getenv("OCR_DI_MAX_CONCURRENT_VLM", "5"))
    ocr_di_output_debug: bool = (
        os.getenv("OCR_DI_OUTPUT_DEBUG", "false").lower() == "true"
    )

    minutes_price: float = float(os.getenv("MINUTES_PRICE", 1))
    pricing: ClassVar[dict] = json.loads(os.getenv("AI_TOOLKIT_PRICING", "{}"))

    transcribe_enable_chunking: bool = (
        os.getenv("TRANSCRIBE_ENABLE_CHUNKING", "0") == "1"
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
                    "format": (
                        "[{levelname} {name} : {filename}:{lineno} : {asctime} "
                        "-> {funcName:10}] {message}"
                    ),
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
