"""Fixtures for creating test tasks."""

import contextlib

import pytest_asyncio

from apps.language.promptic.models import PrompticTask
from apps.language.translate.models import TranslateTask
from apps.ocr.models import OcrTask
from apps.transcribe.models import TranscribeTask


@pytest_asyncio.fixture
async def ocr_task(mock_user: dict) -> OcrTask:
    """Create a test OCR task."""
    task = await OcrTask.create_item(
        {
            "user_id": mock_user["user_id"],
            "tenant_id": mock_user["tenant_id"],
            "file_url": "https://example.com/test.pdf",
            "task_status": "init",
        }
    )
    yield task
    with contextlib.suppress(Exception):
        await task.delete()


@pytest_asyncio.fixture
async def transcribe_task(mock_user: dict) -> TranscribeTask:
    """Create a test transcription task."""
    task = await TranscribeTask.create_item(
        {
            "user_id": mock_user["user_id"],
            "tenant_id": mock_user["tenant_id"],
            "file_url": "https://example.com/audio.mp3",
            "task_status": "init",
        }
    )
    yield task
    with contextlib.suppress(Exception):
        await task.delete()


@pytest_asyncio.fixture
async def translate_task(mock_user: dict) -> TranslateTask:
    """Create a test translation task."""
    task = await TranslateTask.create_item(
        {
            "user_id": mock_user["user_id"],
            "tenant_id": mock_user["tenant_id"],
            "text": "Hello world",
            "language": "Persian",
            "task_status": "init",
        }
    )
    yield task
    with contextlib.suppress(Exception):
        await task.delete()


@pytest_asyncio.fixture
async def execution_task(mock_user: dict) -> PrompticTask:
    """Create a test execution task."""
    import hashlib

    idempotency_key = hashlib.sha256(b"test_prompt:test_input").hexdigest()
    task = await PrompticTask.create_item(
        {
            "user_id": mock_user["user_id"],
            "tenant_id": mock_user["tenant_id"],
            "prompt_name": "test_prompt",
            "input_variables": {"text": "test input"},
            "idempotency_key": idempotency_key,
            "task_status": "init",
        }
    )
    yield task
    with contextlib.suppress(Exception):
        await task.delete()
