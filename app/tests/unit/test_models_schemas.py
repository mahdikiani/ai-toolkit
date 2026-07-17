"""Unit tests for Pydantic schemas and data models."""

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import ValidationError

from apps.language.chat.schemas import (
    ChatMessageCreate,
    ChatMessageSchema,
    ChatQuickStartCreate,
    ChatQuickThreadCreate,
    ChatSessionCreate,
    ChatThreadCreate,
    ChatThreadSchema,
)
from apps.language.promptic.schemas import (
    PrompticCreate as ExecutionTaskCreate,
)
from apps.language.promptic.schemas import (
    PrompticSchema as ExecutionTaskSchema,
)
from apps.language.translate.schemas import TranslateSchemaCreate
from apps.ocr.schemas import OcrEngineType, OcrTaskSchemaCreate
from apps.transcribe.schemas import TranscribeTaskSchemaCreate


@pytest.mark.unit
class TestExecutionTaskCreate:
    """Tests for ExecutionTaskCreate schema."""

    def test_valid_minimal_creation(self) -> None:
        """Should create task with minimal required fields."""
        task = ExecutionTaskCreate(input_variables={"key": "value"})
        assert task.input_variables == {"key": "value"}
        assert task.webhook_url is None
        assert task.idempotency_key is None
        assert task.meta_data == {}

    def test_default_empty_input_variables(self) -> None:
        """Should default input_variables to empty dict."""
        task = ExecutionTaskCreate()
        assert task.input_variables == {}

    def test_full_creation(self) -> None:
        """Should create task with all fields."""
        task = ExecutionTaskCreate(
            input_variables={"text": "hello"},
            webhook_url="https://example.com/webhook",
            idempotency_key="custom_key",
            meta_data={"source": "test"},
        )
        assert task.webhook_url == "https://example.com/webhook"
        assert task.idempotency_key == "custom_key"
        assert task.meta_data == {"source": "test"}

    def test_serialization(self) -> None:
        """Should serialize to dict correctly."""
        task = ExecutionTaskCreate(input_variables={"key": "value"})
        data = task.model_dump()
        assert "input_variables" in data
        assert data["input_variables"] == {"key": "value"}

    def test_deserialization(self) -> None:
        """Should deserialize from dict correctly."""
        data = {"input_variables": {"key": "value"}, "webhook_url": None}
        task = ExecutionTaskCreate.model_validate(data)
        assert task.input_variables == {"key": "value"}


@pytest.mark.unit
class TestExecutionTaskSchema:
    """Tests for ExecutionTaskSchema."""

    def test_requires_prompt_name(self) -> None:
        """Should require prompt_name field."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionTaskSchema(
                idempotency_key="key_123",
                user_id="user_123",
                uid="550e8400-e29b-41d4-a716-446655440000",
                task_status=TaskStatusEnum.init,
            )
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("prompt_name",) for e in errors)

    def test_requires_idempotency_key(self) -> None:
        """Should require idempotency_key field."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionTaskSchema(
                prompt_name="test",
                user_id="user_123",
                uid="550e8400-e29b-41d4-a716-446655440000",
                task_status=TaskStatusEnum.init,
            )
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("idempotency_key",) for e in errors)

    def test_result_defaults_to_none(self) -> None:
        """Result field should default to None."""
        task = ExecutionTaskSchema(
            prompt_name="test",
            idempotency_key="key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.init,
        )
        assert task.result is None

    def test_webhook_failed_defaults_to_false(self) -> None:
        """webhook_failed should default to False."""
        task = ExecutionTaskSchema(
            prompt_name="test",
            idempotency_key="key_123",
            user_id="user_123",
            uid="550e8400-e29b-41d4-a716-446655440000",
            task_status=TaskStatusEnum.init,
        )
        assert task.webhook_failed is False


@pytest.mark.unit
class TestOcrTaskSchemaCreate:
    """Tests for OcrTaskSchemaCreate schema."""

    def test_requires_file_url(self) -> None:
        """Should require file_url field."""
        with pytest.raises(ValidationError):
            OcrTaskSchemaCreate.model_validate({})

    def test_rejects_empty_file_url(self) -> None:
        """Should reject empty file_url."""
        with pytest.raises(ValidationError):
            OcrTaskSchemaCreate(file_url="")

    def test_valid_url(self) -> None:
        """Should accept valid file URL."""
        task = OcrTaskSchemaCreate(file_url="https://example.com/file.pdf")
        assert task.file_url == "https://example.com/file.pdf"

    def test_valid_base64_url(self) -> None:
        """Should accept base64 data URL."""
        task = OcrTaskSchemaCreate(file_url="data:image/png;base64,abc123")
        assert task.file_url.startswith("data:")

    def test_ocr_engine_defaults_to_none(self) -> None:
        """ocr_engine should default to None."""
        task = OcrTaskSchemaCreate(file_url="https://example.com/file.pdf")
        assert task.ocr_engine is None

    def test_valid_ocr_engine_llm(self) -> None:
        """Should accept 'llm' as ocr_engine."""
        task = OcrTaskSchemaCreate(
            file_url="https://example.com/file.pdf",
            ocr_engine=OcrEngineType.llm,
        )
        assert task.ocr_engine == OcrEngineType.llm


@pytest.mark.unit
class TestTranscribeTaskSchemaCreate:
    """Tests for TranscribeTaskSchemaCreate schema."""

    def test_requires_file_url(self) -> None:
        """Should require file_url field."""
        with pytest.raises(ValidationError):
            TranscribeTaskSchemaCreate.model_validate({})

    def test_valid_creation(self) -> None:
        """Should create schema with valid file URL."""
        task = TranscribeTaskSchemaCreate(file_url="https://example.com/audio.mp3")
        assert task.file_url == "https://example.com/audio.mp3"
        assert task.user_id is None
        assert task.webhook_url is None


@pytest.mark.unit
class TestTranslateSchemaCreate:
    """Tests for TranslateSchemaCreate schema."""

    def test_requires_text(self) -> None:
        """Should require text field."""
        with pytest.raises(ValidationError):
            TranslateSchemaCreate.model_validate({})

    def test_defaults_language_to_persian(self) -> None:
        """Language should default to 'Persian'."""
        task = TranslateSchemaCreate(text="Hello")
        assert task.language == "Persian"

    def test_custom_language(self) -> None:
        """Should accept custom language."""
        task = TranslateSchemaCreate(text="Hello", language="English")
        assert task.language == "English"


@pytest.mark.unit
class TestChatSchemas:
    """Tests for chat-related schemas."""

    def test_chat_session_create_optional_fields(self) -> None:
        """ChatSessionCreate should have all optional fields."""
        session = ChatSessionCreate()
        assert session.title is None
        assert session.initial_thread_title is None
        assert session.initial_chat_model is None

    def test_chat_thread_create_optional_fields(self) -> None:
        """ChatThreadCreate should have all optional fields."""
        thread = ChatThreadCreate()
        assert thread.title is None
        assert thread.chat_model is None

    def test_chat_message_create_requires_content(self) -> None:
        """ChatMessageCreate should require content field."""
        with pytest.raises(ValidationError):
            ChatMessageCreate()

    def test_chat_message_create_defaults(self) -> None:
        """ChatMessageCreate should have correct defaults."""
        msg = ChatMessageCreate(content="Hello")
        assert msg.role == "user"
        assert msg.generate_reply is True
        assert msg.stream is False
        assert msg.reply_to_uid is None

    def test_chat_quick_start_create_defaults(self) -> None:
        """ChatQuickStartCreate should enable title suggestion by default."""
        req = ChatQuickStartCreate(content="Hello")
        assert req.suggest_title is True
        assert req.title is None
        assert req.chat_model is None
        assert req.generate_reply is True

    def test_chat_quick_thread_create_defaults(self) -> None:
        """ChatQuickThreadCreate should enable thread title suggestion by default."""
        req = ChatQuickThreadCreate(content="Hello")
        assert req.suggest_thread_title is True
        assert req.thread_title is None
        assert req.chat_model is None
        assert req.generate_reply is True

    def test_chat_message_create_valid_roles(self) -> None:
        """ChatMessageCreate should accept valid roles."""
        for role in ("user", "assistant", "system"):
            msg = ChatMessageCreate(content="Hello", role=role)
            assert msg.role == role

    def test_chat_message_create_invalid_role(self) -> None:
        """ChatMessageCreate should reject invalid roles."""
        with pytest.raises(ValidationError):
            ChatMessageCreate.model_validate({
                "content": "Hello",
                "role": "invalid_role",
            })

    def test_chat_message_schema_requires_thread_uid(self) -> None:
        """ChatMessageSchema should require thread_uid."""
        with pytest.raises(ValidationError):
            ChatMessageSchema.model_validate({
                "role": "user",
                "content": "Hello",
                "uid": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
            })

    def test_chat_thread_schema_requires_session_uid(self) -> None:
        """ChatThreadSchema should require session_uid."""
        with pytest.raises(ValidationError):
            ChatThreadSchema.model_validate({
                "uid": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
            })
