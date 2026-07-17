"""Final comprehensive test for shared schemas implementation."""

import pytest

from apps.language.shared.schemas import ContentPart, ContentType, MessageBlock, Role


def content_parts(message: MessageBlock) -> list[ContentPart]:
    """Return the normalized content after verifying its documented shape."""
    assert all(isinstance(part, ContentPart) for part in message.content)
    return [part for part in message.content if isinstance(part, ContentPart)]


def test_all() -> None:
    """Run all verification tests."""

    # Test 1: Role enum (Requirement 5.2)
    assert Role.SYSTEM == "system"
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"
    assert len(Role) == 3

    # Test 2: ContentType enum (Requirement 5.3)
    assert ContentType.TEXT == "text"
    assert ContentType.IMAGE == "image"
    assert ContentType.DOCUMENT == "document"
    assert len(ContentType) == 3

    # Test 3: ContentPart with text (Requirements 3.2, 5.4)
    text_part = ContentPart(type=ContentType.TEXT, text="Hello world")
    assert text_part.type == ContentType.TEXT
    assert text_part.text == "Hello world"
    assert text_part.file_url is None

    # Test 4: ContentPart with image (Requirements 3.3, 5.4)
    image_part = ContentPart(type=ContentType.IMAGE, file_url="s3://bucket/image.jpg")
    assert image_part.type == ContentType.IMAGE
    assert image_part.file_url == "s3://bucket/image.jpg"
    assert image_part.text is None

    # Test 5: ContentPart with document (Requirements 3.3, 5.4)
    doc_part = ContentPart(type=ContentType.DOCUMENT, file_url="s3://bucket/doc.pdf")
    assert doc_part.type == ContentType.DOCUMENT
    assert doc_part.file_url == "s3://bucket/doc.pdf"

    # Test 6: ContentPart validation - text without text field (Requirements 3.4, 3.5)
    with pytest.raises(ValueError, match="text is required when type=text"):
        ContentPart(type=ContentType.TEXT)

    # Test 7: ContentPart validation - image without file_url (Requirements 3.4, 3.5)
    with pytest.raises(ValueError, match="file_url is required"):
        ContentPart(type=ContentType.IMAGE)

    # Test 8: ContentPart validation - document without file_url (Requirements 3.4, 3.5)
    with pytest.raises(ValueError, match="file_url is required"):
        ContentPart(type=ContentType.DOCUMENT)

    # Test 9: MessageBlock with string content (Requirements 5.5, 5.6, 7.3)
    msg = MessageBlock(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert content_parts(msg)[0].type == ContentType.TEXT
    assert content_parts(msg)[0].text == "Hello"

    # Test 10: MessageBlock with ContentPart list (Requirement 5.5)
    parts = [
        ContentPart(type=ContentType.TEXT, text="Check this image:"),
        ContentPart(type=ContentType.IMAGE, file_url="s3://bucket/photo.jpg"),
    ]
    msg = MessageBlock(role=Role.USER, content=parts)
    assert len(msg.content) == 2
    assert content_parts(msg)[0].text == "Check this image:"
    assert content_parts(msg)[1].file_url == "s3://bucket/photo.jpg"

    # Test 11: Round-trip preservation (Requirement 5.9)
    original_text = "This is a test message with special chars: !@#$%^&*()"
    msg = MessageBlock(role=Role.USER, content=original_text)
    reconstructed_text = content_parts(msg)[0].text
    assert reconstructed_text == original_text

    # Test 12: Multiline text preservation (Requirement 5.9)
    multiline_text = "Line 1\nLine 2\nLine 3"
    msg = MessageBlock(role=Role.USER, content=multiline_text)
    assert content_parts(msg)[0].text == multiline_text

    # Test 13: Empty string handling
    msg = MessageBlock(role=Role.SYSTEM, content="")
    assert content_parts(msg)[0].text == ""

    # Test 14: Unicode text preservation
    unicode_text = "Hello 世界 مرحبا 🌍"
    msg = MessageBlock(role=Role.USER, content=unicode_text)
    assert content_parts(msg)[0].text == unicode_text

    # Test 15: Mixed content message
    mixed_parts = [
        ContentPart(type=ContentType.TEXT, text="Here's a document:"),
        ContentPart(type=ContentType.DOCUMENT, file_url="s3://bucket/report.pdf"),
        ContentPart(type=ContentType.TEXT, text="And an image:"),
        ContentPart(type=ContentType.IMAGE, file_url="s3://bucket/chart.png"),
    ]
    msg = MessageBlock(role=Role.ASSISTANT, content=mixed_parts)
    assert len(msg.content) == 4
    assert content_parts(msg)[0].type == ContentType.TEXT
    assert content_parts(msg)[1].type == ContentType.DOCUMENT
    assert content_parts(msg)[2].type == ContentType.TEXT
    assert content_parts(msg)[3].type == ContentType.IMAGE


if __name__ == "__main__":
    test_all()
