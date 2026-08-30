"""Pydantic request and response schemas for the backend API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .memory import (
    MAX_MEMORY_KEY_CHARS,
    MAX_MEMORY_VALUE_CHARS,
    normalize_memory_key,
    validate_memory_category,
    validate_memory_value,
)


def _required_trimmed(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must not be empty")
    return trimmed


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        return None if value is None else _required_trimmed(value)


class ConversationRename(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=200)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _required_trimmed(value)


class MessageMetadataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    validator: dict[str, Any] | None = Field(default=None, validation_alias="validator_json")
    tools: list[Any] | None = Field(default=None, validation_alias="tool_calls_json")
    memory: list[Any] | None = Field(default=None, validation_alias="memory_refs_json")


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    metadata: MessageMetadataRead | None = Field(
        default=None,
        validation_alias="metadata_record",
    )


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    archived: bool


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None
    message: str = Field(max_length=100_000)

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        return None if value is None else _required_trimmed(value)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class ValidatorMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    retry_attempted: bool
    retry_passed: bool | None


class ChatMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str | None
    latency_ms: int | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    validator: ValidatorMetadata
    tools: list[Any]
    memory: list[Any]


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conversation_id: str
    message_id: str
    response: str
    metadata: ChatMetadata


class MemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(max_length=32)
    key: str = Field(max_length=MAX_MEMORY_KEY_CHARS)
    value: str = Field(max_length=MAX_MEMORY_VALUE_CHARS)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        return validate_memory_category(value)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return normalize_memory_key(value)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return validate_memory_value(value)


class MemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=MAX_MEMORY_VALUE_CHARS)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return validate_memory_value(value)


class MemorySourceRead(BaseModel):
    conversation_id: str | None = None
    message_id: str | None = None


class MemoryRead(BaseModel):
    id: str
    operation: str
    category: str
    key: str
    value: str | None = None
    status: str
    source: MemorySourceRead
    updated_at: datetime
