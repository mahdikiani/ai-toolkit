"""Verification test for Task 1.2: Shared schema models."""

import pytest

from apps.language.shared.schemas import ContentPart, ContentType, MessageBlock, Role


def test_role_enum() -> None:
    """Test Role enum values."""
    assert Role.SYSTEM == "system"
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"


def test_content_type_enum() -> None:
    """Test ContentType enum values."""
    assert ContentType.TEXT == "text"
    assert ContentType.IMAGE == "image"
    assert ContentType.DOCUMENT == "document"


def test_content_part_text() -> None:
    """Test ContentPart with text content."""
    part = ContentPart(type=ContentType.TEXT, text="Hello world")
    assert part.type == ContentType.TEXT
    assert part.text == "Hello world"
    assert part.file_url is None


def test_content_part_image() -> None:
    """Test ContentPart with image content."""
    part = ContentPart(type=ContentType.IMAGE, file_url="s3://bucket/image.jpg")
    assert part.type == ContentType.IMAGE
    assert part.file_url == "s3://bucket/image.jpg"
    assert part.text is None


def test_content_part_validation_text_missing() -> None:
    """Test ContentPart validation fails when text is missing for TEXT type."""
    with pytest.raises(ValueError, match="text is required when type=text"):
        ContentPart(type=ContentType.TEXT, file_url="some_url")


def test_content_part_validation_file_url_missing() -> None:
    """Test ContentPart validation fails when file_url is missing for IMAGE type."""
    with pytest.raises(
        ValueError, match="file_url is required when type=image or document"
    ):
        ContentPart(type=ContentType.IMAGE, text="some text")


def test_message_block_string_normalization() -> None:
    """Test MessageBlock normalizes string content to ContentPart list."""
    msg = MessageBlock(role=Role.USER, content="Hello world")
    assert msg.role == Role.USER
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert msg.content[0].type == ContentType.TEXT
    assert msg.content[0].text == "Hello world"


def test_message_block_list_passthrough() -> None:
    """Test MessageBlock accepts list of ContentPart."""
    parts = [
        ContentPart(type=ContentType.TEXT, text="Check this image:"),
        ContentPart(type=ContentType.IMAGE, file_url="s3://bucket/photo.jpg"),
    ]
    msg = MessageBlock(role=Role.USER, content=parts)
    assert msg.role == Role.USER
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0].text == "Check this image:"
    assert msg.content[1].file_url == "s3://bucket/photo.jpg"


def test_round_trip_preservation() -> None:
    """Test round-trip: string → ContentPart → string preserves text."""
    original_text = "This is my message"
    msg = MessageBlock(role=Role.USER, content=original_text)
    # Extract text back from ContentPart
    assert isinstance(msg.content[0], ContentPart)
    extracted_text = msg.content[0].text
    assert extracted_text == original_text


if __name__ == "__main__":
    test_role_enum()
    test_content_type_enum()
    test_content_part_text()
    test_content_part_image()
    test_content_part_validation_text_missing()
    test_content_part_validation_file_url_missing()
    test_message_block_string_normalization()
    test_message_block_list_passthrough()
    test_round_trip_preservation()
