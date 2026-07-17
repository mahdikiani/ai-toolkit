"""Schemas for execution task management."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class MissingContentError(ValueError):
    """Raised when a content part has neither text nor a file URL."""


class Role(StrEnum):
    """Enum for message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(StrEnum):
    """Enum for content types."""

    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class ModelConfig(BaseModel):
    """Schema for model configuration parameters."""

    temperature: float = 0.2
    top_p: float = 1


class ContentPart(BaseModel):
    """Schema for content parts in messages."""

    type: ContentType = Field(
        ContentType.TEXT, description="The type of the content part"
    )
    text: str | None = None
    file_url: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        """Validate that either text or file_url is provided."""
        if self.text is None and self.file_url is None:
            error = MissingContentError("Either text or file_url must be provided")
            raise error
        return self


class MessageBlock(BaseModel):
    """Schema for message blocks."""

    role: Role = Field(Role.SYSTEM, description="The role of the message")
    content: str | list[ContentPart]

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: str | list[ContentPart]) -> list[ContentPart]:
        """Normalize content to a list of ContentParts."""
        if isinstance(v, str):
            return [ContentPart(type=ContentType.TEXT, text=v)]
        return v


class PromptListResponse(BaseModel):
    """Schema for prompt response."""

    name: str
    description: str | None = Field(None)
    tags: list[str] = Field(default_factory=list)
    config: ModelConfig = ModelConfig()
    model: str = "google/gemini-3.0-flash-preview"


class PromptSchemaResponse(PromptListResponse):
    """A prompt to be executed."""

    input_fields: list = Field(default_factory=list)
    output_schema: dict | list | None = None


class PromptDetailSchema(PromptSchemaResponse):
    """Schema for detailed prompt information."""

    messages: list[MessageBlock] = Field(default_factory=list)
