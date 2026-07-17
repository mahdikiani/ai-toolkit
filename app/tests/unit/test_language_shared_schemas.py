"""
Unit tests for shared language service schemas.

Tests for Role, ContentType, ContentPart, and MessageBlock schemas
defined in apps/language/shared/schemas.py.
"""

import pytest
from pydantic import ValidationError

from apps.language.shared.schemas import (
    ContentPart,
    ContentType,
    MessageBlock,
    Role,
)


def content_parts(message: MessageBlock) -> list[ContentPart]:
    """Return the normalized content after verifying its documented shape."""
    assert all(isinstance(part, ContentPart) for part in message.content)
    return [part for part in message.content if isinstance(part, ContentPart)]


@pytest.mark.unit
class TestRoleEnum:
    """Tests for Role enum values."""

    def test_role_system_value(self) -> None:
        """Role.SYSTEM should have value 'system'."""
        assert Role.SYSTEM == "system"
        assert Role.SYSTEM.value == "system"

    def test_role_user_value(self) -> None:
        """Role.USER should have value 'user'."""
        assert Role.USER == "user"
        assert Role.USER.value == "user"

    def test_role_assistant_value(self) -> None:
        """Role.ASSISTANT should have value 'assistant'."""
        assert Role.ASSISTANT == "assistant"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_enum_members(self) -> None:
        """Role enum should have exactly three members."""
        members = list(Role)
        assert len(members) == 3
        assert Role.SYSTEM in members
        assert Role.USER in members
        assert Role.ASSISTANT in members

    def test_role_string_comparison(self) -> None:
        """Role enum values should compare equal to strings."""
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.SYSTEM == "system"


@pytest.mark.unit
class TestContentTypeEnum:
    """Tests for ContentType enum values."""

    def test_content_type_text_value(self) -> None:
        """ContentType.TEXT should have value 'text'."""
        assert ContentType.TEXT == "text"
        assert ContentType.TEXT.value == "text"

    def test_content_type_image_value(self) -> None:
        """ContentType.IMAGE should have value 'image'."""
        assert ContentType.IMAGE == "image"
        assert ContentType.IMAGE.value == "image"

    def test_content_type_document_value(self) -> None:
        """ContentType.DOCUMENT should have value 'document'."""
        assert ContentType.DOCUMENT == "document"
        assert ContentType.DOCUMENT.value == "document"

    def test_content_type_enum_members(self) -> None:
        """ContentType enum should have exactly three members."""
        members = list(ContentType)
        assert len(members) == 3
        assert ContentType.TEXT in members
        assert ContentType.IMAGE in members
        assert ContentType.DOCUMENT in members

    def test_content_type_string_comparison(self) -> None:
        """ContentType enum values should compare equal to strings."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.DOCUMENT == "document"


@pytest.mark.unit
class TestContentPartValidation:
    """Tests for ContentPart validation logic."""

    def test_text_content_part_valid(self) -> None:
        """Should create valid text ContentPart with text field."""
        part = ContentPart(type=ContentType.TEXT, text="Hello world")
        assert part.type == ContentType.TEXT
        assert part.text == "Hello world"
        assert part.file_url is None

    def test_text_content_part_missing_text(self) -> None:
        """Should reject text ContentPart without text field."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.TEXT, text=None)
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(
            "text is required when type=text" in str(e["ctx"]["error"]) for e in errors
        )

    def test_text_content_part_with_file_url_ignored(self) -> None:
        """Should accept text ContentPart with file_url (file_url ignored)."""
        part = ContentPart(type=ContentType.TEXT, text="Hello", file_url="ignored.jpg")
        assert part.type == ContentType.TEXT
        assert part.text == "Hello"
        # file_url is present but validation only checks required fields

    def test_image_content_part_valid(self) -> None:
        """Should create valid image ContentPart with file_url."""
        part = ContentPart(
            type=ContentType.IMAGE, file_url="https://example.com/image.jpg"
        )
        assert part.type == ContentType.IMAGE
        assert part.file_url == "https://example.com/image.jpg"
        assert part.text is None

    def test_image_content_part_missing_file_url(self) -> None:
        """Should reject image ContentPart without file_url."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.IMAGE, file_url=None)
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(
            "file_url is required when type=image or document" in str(e["ctx"]["error"])
            for e in errors
        )

    def test_document_content_part_valid(self) -> None:
        """Should create valid document ContentPart with file_url."""
        part = ContentPart(
            type=ContentType.DOCUMENT, file_url="https://example.com/doc.pdf"
        )
        assert part.type == ContentType.DOCUMENT
        assert part.file_url == "https://example.com/doc.pdf"
        assert part.text is None

    def test_document_content_part_missing_file_url(self) -> None:
        """Should reject document ContentPart without file_url."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.DOCUMENT, file_url=None)
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(
            "file_url is required when type=image or document" in str(e["ctx"]["error"])
            for e in errors
        )

    def test_content_part_default_type(self) -> None:
        """ContentPart should default to TEXT type."""
        part = ContentPart(text="Hello")
        assert part.type == ContentType.TEXT
        assert part.text == "Hello"

    def test_content_part_serialization(self) -> None:
        """ContentPart should serialize to dict correctly."""
        part = ContentPart(type=ContentType.TEXT, text="Hello")
        data = part.model_dump()
        assert data["type"] == "text"
        assert data["text"] == "Hello"
        assert data["file_url"] is None

    def test_content_part_deserialization(self) -> None:
        """ContentPart should deserialize from dict correctly."""
        data = {"type": "image", "file_url": "https://example.com/img.png"}
        part = ContentPart.model_validate(data)
        assert part.type == ContentType.IMAGE
        assert part.file_url == "https://example.com/img.png"


@pytest.mark.unit
class TestMessageBlockNormalization:
    """Tests for MessageBlock string-to-ContentPart normalization."""

    def test_string_content_normalization(self) -> None:
        """Should normalize string content to list of ContentPart."""
        msg = MessageBlock(role=Role.USER, content="Hello world")
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ContentPart)
        assert msg.content[0].type == ContentType.TEXT
        assert msg.content[0].text == "Hello world"

    def test_list_content_passthrough(self) -> None:
        """Should pass through list of ContentPart without modification."""
        parts = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(type=ContentType.IMAGE, file_url="image.jpg"),
        ]
        msg = MessageBlock(role=Role.USER, content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].text == "Hello"
        assert msg.content[1].file_url == "image.jpg"

    def test_empty_string_normalization(self) -> None:
        """Should normalize empty string to ContentPart with empty text."""
        msg = MessageBlock(role=Role.USER, content="")
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0].text == ""

    def test_multiline_string_normalization(self) -> None:
        """Should normalize multiline string to single ContentPart."""
        text = "Line 1\nLine 2\nLine 3"
        msg = MessageBlock(role=Role.USER, content=text)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0].text == text

    def test_message_block_default_role(self) -> None:
        """MessageBlock should default to SYSTEM role."""
        msg = MessageBlock(content="Hello")
        assert msg.role == Role.SYSTEM

    def test_message_block_requires_content(self) -> None:
        """MessageBlock should require content field."""
        with pytest.raises(ValidationError):
            MessageBlock(role=Role.USER)


@pytest.mark.unit
class TestRoundTripPreservation:
    """Tests for round-trip preservation (string → ContentPart → string)."""

    def test_simple_string_round_trip(self) -> None:
        """Simple string should preserve through round-trip."""
        original = "Hello world"
        msg = MessageBlock(role=Role.USER, content=original)
        # Extract text back from ContentPart
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original

    def test_multiline_string_round_trip(self) -> None:
        """Multiline string should preserve through round-trip."""
        original = "Line 1\nLine 2\nLine 3"
        msg = MessageBlock(role=Role.USER, content=original)
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original

    def test_string_with_special_chars_round_trip(self) -> None:
        """String with special characters should preserve through round-trip."""
        original = "Hello! @#$%^&*() 你好 مرحبا"
        msg = MessageBlock(role=Role.USER, content=original)
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original

    def test_empty_string_round_trip(self) -> None:
        """Empty string should preserve through round-trip."""
        original = ""
        msg = MessageBlock(role=Role.USER, content=original)
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original

    def test_whitespace_string_round_trip(self) -> None:
        """String with whitespace should preserve through round-trip."""
        original = "  Hello  \n  World  "
        msg = MessageBlock(role=Role.USER, content=original)
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original

    def test_unicode_string_round_trip(self) -> None:
        """Unicode string should preserve through round-trip."""
        original = "Hello 世界 🌍 مرحبا بالعالم"
        msg = MessageBlock(role=Role.USER, content=original)
        reconstructed = content_parts(msg)[0].text
        assert reconstructed == original


@pytest.mark.unit
class TestValidationErrorMessages:
    """Tests for validation error messages."""

    def test_text_content_part_error_message(self) -> None:
        """Should provide clear error message for missing text in TEXT type."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.TEXT, text=None)
        error_msg = str(exc_info.value)
        assert "text is required when type=text" in error_msg

    def test_image_content_part_error_message(self) -> None:
        """Should provide clear error message for missing file_url in IMAGE type."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.IMAGE, file_url=None)
        error_msg = str(exc_info.value)
        assert "file_url is required when type=image or document" in error_msg

    def test_document_content_part_error_message(self) -> None:
        """Should provide clear error message for missing file_url in DOCUMENT type."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type=ContentType.DOCUMENT, file_url=None)
        error_msg = str(exc_info.value)
        assert "file_url is required when type=image or document" in error_msg

    def test_message_block_missing_content_error(self) -> None:
        """Should provide clear error message for missing content."""
        with pytest.raises(ValidationError) as exc_info:
            MessageBlock(role=Role.USER)
        assert isinstance(exc_info.value, ValidationError)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)
        assert any(e["type"] == "missing" for e in errors)


@pytest.mark.unit
class TestMessageBlockSerialization:
    """Tests for MessageBlock serialization and deserialization."""

    def test_serialize_string_content(self) -> None:
        """Should serialize MessageBlock with string content."""
        msg = MessageBlock(role=Role.USER, content="Hello")
        data = msg.model_dump()
        assert data["role"] == "user"
        assert isinstance(data["content"], list)
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello"

    def test_serialize_list_content(self) -> None:
        """Should serialize MessageBlock with list content."""
        parts = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(type=ContentType.IMAGE, file_url="image.jpg"),
        ]
        msg = MessageBlock(role=Role.ASSISTANT, content=parts)
        data = msg.model_dump()
        assert data["role"] == "assistant"
        assert len(data["content"]) == 2
        assert data["content"][0]["text"] == "Hello"
        assert data["content"][1]["file_url"] == "image.jpg"

    def test_deserialize_string_content(self) -> None:
        """Should deserialize MessageBlock from dict with string content."""
        data = {"role": "user", "content": "Hello world"}
        msg = MessageBlock.model_validate(data)
        assert msg.role == Role.USER
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello world"

    def test_deserialize_list_content(self) -> None:
        """Should deserialize MessageBlock from dict with list content."""
        data = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here's an image:"},
                {"type": "image", "file_url": "https://example.com/img.png"},
            ],
        }
        msg = MessageBlock.model_validate(data)
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 2
        assert msg.content[0].text == "Here's an image:"
        assert msg.content[1].file_url == "https://example.com/img.png"


@pytest.mark.unit
class TestMixedContentScenarios:
    """Tests for mixed content scenarios (text + attachments)."""

    def test_text_and_image_content(self) -> None:
        """Should handle message with text and image."""
        parts = [
            ContentPart(type=ContentType.TEXT, text="Check this out:"),
            ContentPart(type=ContentType.IMAGE, file_url="photo.jpg"),
        ]
        msg = MessageBlock(role=Role.USER, content=parts)
        assert len(msg.content) == 2
        assert content_parts(msg)[0].type == ContentType.TEXT
        assert content_parts(msg)[1].type == ContentType.IMAGE

    def test_text_and_document_content(self) -> None:
        """Should handle message with text and document."""
        parts = [
            ContentPart(type=ContentType.TEXT, text="Please review:"),
            ContentPart(type=ContentType.DOCUMENT, file_url="report.pdf"),
        ]
        msg = MessageBlock(role=Role.USER, content=parts)
        assert len(msg.content) == 2
        assert content_parts(msg)[0].type == ContentType.TEXT
        assert content_parts(msg)[1].type == ContentType.DOCUMENT

    def test_multiple_attachments(self) -> None:
        """Should handle message with multiple attachments."""
        parts = [
            ContentPart(type=ContentType.TEXT, text="Multiple files:"),
            ContentPart(type=ContentType.IMAGE, file_url="image1.jpg"),
            ContentPart(type=ContentType.IMAGE, file_url="image2.png"),
            ContentPart(type=ContentType.DOCUMENT, file_url="doc.pdf"),
        ]
        msg = MessageBlock(role=Role.USER, content=parts)
        assert len(msg.content) == 4
        assert all(isinstance(part, ContentPart) for part in msg.content)

    def test_image_only_content(self) -> None:
        """Should handle message with only image (no text)."""
        parts = [ContentPart(type=ContentType.IMAGE, file_url="photo.jpg")]
        msg = MessageBlock(role=Role.USER, content=parts)
        assert len(msg.content) == 1
        assert content_parts(msg)[0].type == ContentType.IMAGE
        assert content_parts(msg)[0].text is None
