"""
Shared schemas for language services.

This module defines common data models used across all language services
(chat, prompts, executions, translate) to maintain consistency and reduce
code duplication.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class MissingContentFieldError(ValueError):
    """Raised when content lacks the field required by its type."""


class Role(StrEnum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(StrEnum):
    """Type of content in a message part."""

    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class ContentPart(BaseModel):
    """
    A piece of message content (text, image, or document).

    Each ContentPart represents a single piece of content with a specific type.
    Text content requires a text field, while image and document content
    require a file_url field.
    """

    type: ContentType = Field(
        default=ContentType.TEXT, description="The type of the content part"
    )
    text: str | None = Field(
        default=None, description="Text content (required when type=text)"
    )
    file_url: str | None = Field(
        default=None,
        description="File reference (required when type=image or document)",
    )

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        """
        Ensure either text or file_url is provided based on type.

        Raises:
            ValueError: If text is missing when type=text, or if file_url
                is missing when type=image or document.
        """
        if self.type == ContentType.TEXT:
            if self.text is None:
                error = MissingContentFieldError("text is required when type=text")
                raise error
        else:  # IMAGE or DOCUMENT
            if self.file_url is None:
                error = MissingContentFieldError(
                    "file_url is required when type=image or document"
                )
                raise error
        return self


class MessageBlock(BaseModel):
    """
    A message with role and content (text or structured parts).

    MessageBlock supports both legacy string content and new structured
    ContentPart lists. String content is automatically normalized to a
    single-element ContentPart list for internal consistency.
    """

    role: Role = Field(
        default=Role.SYSTEM, description="The role of the message sender"
    )
    content: str | list[ContentPart] = Field(
        ..., description="Message content as string or list of parts"
    )

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: str | list[ContentPart]) -> list[ContentPart]:
        """
        Normalize string content to list of ContentPart for internal storage.

        This validator ensures backward compatibility by automatically converting
        string content to the new structured format.

        Args:
            v: Content as string or list of ContentPart objects

        Returns:
            List of ContentPart objects
        """
        if isinstance(v, str):
            return [ContentPart(type=ContentType.TEXT, text=v)]
        return v
