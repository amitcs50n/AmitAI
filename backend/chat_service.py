"""Transactional chat orchestration with an isolated mock generator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from sqlalchemy.orm import Session

from .models import utc_now
from .repositories import ConversationRepository, MessageRepository

MOCK_RESPONSE = "This is a mocked AmitAI response."


@dataclass(frozen=True)
class GenerationMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatGenerationResult:
    response: str
    model: str | None = "mock"
    latency_ms: int | None = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    validator: dict[str, Any] = field(
        default_factory=lambda: {"retry_attempted": False, "retry_passed": None}
    )
    tools: list[Any] = field(default_factory=list)
    memory: list[Any] = field(default_factory=list)


class ResponseGenerator(Protocol):
    def generate_response(
        self, messages: Sequence[GenerationMessage]
    ) -> ChatGenerationResult: ...


GenerationCallable = Callable[[Sequence[GenerationMessage]], ChatGenerationResult]


def generate_response(messages: Sequence[GenerationMessage]) -> ChatGenerationResult:
    """Default deterministic generator; replace this at the service boundary later."""

    del messages
    return ChatGenerationResult(response=MOCK_RESPONSE)


class ConversationNotFoundError(LookupError):
    pass


class ChatGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    message_id: str
    response: str
    metadata: ChatGenerationResult


def _deterministic_title(message: str) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= 60:
        return normalized
    return f"{normalized[:57].rstrip()}..."


def _timestamp_after(previous: datetime | None) -> datetime:
    timestamp = utc_now()
    if previous is not None and timestamp <= previous:
        return previous + timedelta(microseconds=1)
    return timestamp


class ChatService:
    def __init__(
        self,
        session: Session,
        generator: ResponseGenerator | GenerationCallable | None = None,
    ) -> None:
        self.session = session
        self.generator = generator
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)

    def _generate(self, messages: Sequence[GenerationMessage]) -> ChatGenerationResult:
        if self.generator is None:
            result = generate_response(messages)
        elif callable(self.generator):
            result = self.generator(messages)
        else:
            result = self.generator.generate_response(messages)

        if not isinstance(result, ChatGenerationResult):
            raise TypeError("Generator must return ChatGenerationResult")
        if not isinstance(result.response, str) or not result.response.strip():
            raise ValueError("Generator returned an empty response")
        if result.model is not None and not isinstance(result.model, str):
            raise TypeError("Generator model metadata must be a string or null")

        for field_name in ("latency_ms", "input_tokens", "output_tokens"):
            value = getattr(result, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise TypeError(
                    f"Generator {field_name} metadata must be a nonnegative integer or null"
                )

        if not isinstance(result.validator, dict):
            raise TypeError("Generator validator metadata must be an object")
        retry_attempted = result.validator.get("retry_attempted")
        retry_passed = result.validator.get("retry_passed")
        if not isinstance(retry_attempted, bool):
            raise TypeError("Generator retry_attempted metadata must be boolean")
        if retry_passed is not None and not isinstance(retry_passed, bool):
            raise TypeError("Generator retry_passed metadata must be boolean or null")
        if not isinstance(result.tools, list):
            raise TypeError("Generator tools metadata must be a list")
        if not isinstance(result.memory, list):
            raise TypeError("Generator memory metadata must be a list")
        return result

    def chat(self, *, conversation_id: str | None, message: str) -> ChatResult:
        try:
            with self.session.begin():
                if conversation_id is None:
                    conversation = self.conversations.create(_deterministic_title(message))
                else:
                    conversation = self.conversations.get(conversation_id)
                    if conversation is None:
                        raise ConversationNotFoundError(conversation_id)

                history = self.messages.list_for_conversation(conversation.id)
                previous_timestamp = history[-1].created_at if history else None
                user_message = self.messages.create(
                    conversation,
                    role="user",
                    content=message,
                    created_at=_timestamp_after(previous_timestamp),
                )
                generation_messages = [
                    GenerationMessage(role=item.role, content=item.content) for item in history
                ]
                generation_messages.append(
                    GenerationMessage(role=user_message.role, content=user_message.content)
                )

                try:
                    generation = self._generate(generation_messages)
                except Exception as exc:
                    raise ChatGenerationError("Assistant generation failed") from exc

                assistant_message = self.messages.create(
                    conversation,
                    role="assistant",
                    content=generation.response,
                    created_at=_timestamp_after(user_message.created_at),
                )
                self.messages.add_metadata(
                    assistant_message,
                    model=generation.model,
                    latency_ms=generation.latency_ms,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    validator=generation.validator,
                    tools=generation.tools,
                    memory=generation.memory,
                )
                self.conversations.touch(
                    conversation,
                    now=_timestamp_after(assistant_message.created_at),
                )

            return ChatResult(
                conversation_id=conversation.id,
                message_id=assistant_message.id,
                response=generation.response,
                metadata=generation,
            )
        except Exception:
            self.session.rollback()
            raise
