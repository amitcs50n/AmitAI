"""Persistent chat orchestration with an isolated generation boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from threading import Event
from typing import Any, Protocol

from sqlalchemy.orm import Session

from .asset_storage import AssetStorage
from .assets import VISION_NOT_ENABLED, AssetError, AssetService, validate_asset_ids
from .memory import (
    LOCAL_MEMORY_OWNER_ID,
    MemoryCommandDecision,
    MemoryService,
    StagedMemoryMutation,
    format_memory_command_context,
    format_memory_context,
    memory_metadata_reference,
    parse_memory_command,
)
from .models import Message, utc_now
from .repositories import ConversationRepository, MessageRepository

MOCK_RESPONSE = "This is a mocked AmitAI response."


@dataclass(frozen=True)
class RemoteProjection:
    """Request-local replacement; content=None omits the message remotely."""

    content: str | None


@dataclass(frozen=True)
class GenerationMessage:
    role: str
    content: str
    remote_projection: RemoteProjection | None = None


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


@dataclass(frozen=True)
class ChatGenerationDelta:
    delta: str


class ResponseGenerator(Protocol):
    def generate_response(
        self, messages: Sequence[GenerationMessage]
    ) -> ChatGenerationResult: ...


class StreamingResponseGenerator(Protocol):
    def stream_response(
        self,
        messages: Sequence[GenerationMessage],
        *,
        cancel_event: Event,
    ) -> Iterator[ChatGenerationDelta | ChatGenerationResult]: ...


GenerationCallable = Callable[[Sequence[GenerationMessage]], ChatGenerationResult]


def generate_response(messages: Sequence[GenerationMessage]) -> ChatGenerationResult:
    """Default deterministic generator used when no runtime is explicitly selected."""

    del messages
    return ChatGenerationResult(response=MOCK_RESPONSE)


class ConversationNotFoundError(LookupError):
    pass


class ChatGenerationError(RuntimeError):
    pass


class ChatPrivacyError(ChatGenerationError):
    def __init__(self) -> None:
        super().__init__("Remote inference blocked by local privacy policy")


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    message_id: str
    response: str
    metadata: ChatGenerationResult


@dataclass(frozen=True)
class ChatStreamEvent:
    event: str
    data: dict[str, Any]


@dataclass(frozen=True)
class _PreparedChat:
    conversation_id: str | None
    message: str
    title: str
    request_timestamp: datetime
    user_created_at: datetime
    generation_messages: tuple[GenerationMessage, ...]
    retrieved_memory: tuple[dict[str, Any], ...]
    staged_memory: StagedMemoryMutation | None
    asset_ids: tuple[str, ...] = ()


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


def _command_projection(decision: MemoryCommandDecision) -> RemoteProjection | None:
    if not decision.intent_detected:
        return None
    if decision.command is None:
        return RemoteProjection("The local memory command was not applied. Acknowledge briefly.")
    return RemoteProjection(
        f"A local memory {decision.command.operation} operation is staged. "
        "Acknowledge briefly; it becomes durable only if this chat succeeds."
    )


def _history_context(history: Sequence[Message]) -> list[GenerationMessage]:
    """Protect legacy commands and their paired acknowledgments without rewriting storage."""

    result: list[GenerationMessage] = []
    previous_was_command = False
    for message in history:
        projection = None
        if message.role == "user":
            decision = parse_memory_command(message.content)
            previous_was_command = decision.intent_detected
            if previous_was_command:
                projection = RemoteProjection("A local memory command was requested.")
        else:
            if message.role == "assistant" and previous_was_command:
                projection = RemoteProjection("The local memory command was acknowledged.")
            previous_was_command = False
        result.append(GenerationMessage(message.role, message.content, projection))
    return result


class ChatService:
    def __init__(
        self,
        session: Session,
        generator: (
            ResponseGenerator | StreamingResponseGenerator | GenerationCallable | None
        ) = None,
        *,
        memory_owner_id: str = LOCAL_MEMORY_OWNER_ID,
        asset_storage: AssetStorage | None = None,
    ) -> None:
        self.session = session
        self.generator = generator
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.memory = MemoryService(session, owner_id=memory_owner_id)
        self.assets = AssetService(session, asset_storage) if asset_storage is not None else None

    def _validate_generation(self, result: object) -> ChatGenerationResult:
        """Validate the generator boundary before any response can be persisted."""

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

    def _generate(self, messages: Sequence[GenerationMessage]) -> ChatGenerationResult:
        if self.generator is None:
            result = generate_response(messages)
        elif callable(self.generator):
            result = self.generator(messages)
        else:
            result = self.generator.generate_response(messages)

        return self._validate_generation(result)

    def _stream_generation(
        self,
        messages: Sequence[GenerationMessage],
        *,
        cancel_event: Event,
    ) -> Iterator[ChatGenerationDelta | ChatGenerationResult]:
        stream_method = getattr(self.generator, "stream_response", None)
        if not callable(stream_method):
            result = self._generate(messages)
            if cancel_event.is_set():
                return
            yield ChatGenerationDelta(delta=result.response)
            yield result
            return

        stream = iter(stream_method(messages, cancel_event=cancel_event))
        deltas: list[str] = []
        result: ChatGenerationResult | None = None
        try:
            for item in stream:
                if cancel_event.is_set():
                    return
                if isinstance(item, ChatGenerationDelta):
                    if result is not None:
                        raise TypeError("Generator emitted text after final metadata")
                    if not isinstance(item.delta, str) or not item.delta:
                        raise TypeError("Generator stream deltas must be non-empty strings")
                    deltas.append(item.delta)
                    yield item
                    continue
                if isinstance(item, ChatGenerationResult):
                    if result is not None:
                        raise TypeError("Generator emitted final metadata more than once")
                    result = self._validate_generation(item)
                    continue
                raise TypeError(
                    "Generator stream must yield ChatGenerationDelta or ChatGenerationResult"
                )
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if cancel_event.is_set():
            return
        if result is None:
            raise TypeError("Generator stream ended without final metadata")
        if "".join(deltas) != result.response:
            raise ValueError("Generator stream deltas do not reconstruct the final response")
        yield result

    def _prepare_chat(
        self, *, conversation_id: str | None, message: str, asset_ids: tuple[str, ...] = (),
    ) -> _PreparedChat:
        request_timestamp = utc_now()
        validate_asset_ids(asset_ids)
        with self.session.begin():
            if conversation_id is None:
                title = _deterministic_title(message)
                previous_timestamp = request_timestamp
                history_messages: list[GenerationMessage] = []
            else:
                conversation = self.conversations.get_fresh(conversation_id)
                if conversation is None:
                    raise ConversationNotFoundError(conversation_id)
                history = self.messages.list_for_conversation(conversation.id)
                previous_timestamp = (
                    history[-1].created_at if history else conversation.created_at
                )
                history_messages = _history_context(history)
                title = ""

            if asset_ids:
                if self.assets is None:
                    raise AssetError("Image attachments are unavailable", 503)
                self.assets.validate_links(asset_ids, conversation_id)
                # No image content/filename/ID, user prompt, retrieved memory or
                # automatic memory mutation enters inference for this stub turn.
                return _PreparedChat(
                    conversation_id, message, title, request_timestamp,
                    _timestamp_after(previous_timestamp), (), (), None, asset_ids,
                )

            decision = parse_memory_command(message)
            staged_memory = self.memory.stage_chat_command(decision)
            if decision.command is not None and staged_memory is None:
                decision = MemoryCommandDecision(
                    intent_detected=True,
                    reason="Memory target was not found or is not active",
                )
            excluded_memory_ids = (
                frozenset({staged_memory.memory_id})
                if staged_memory is not None and staged_memory.operation == "deleted"
                else frozenset()
            )
            retrieved_memory = self.memory.retrieve(
                message,
                exclude_memory_ids=excluded_memory_ids,
            )

        user_created_at = _timestamp_after(previous_timestamp)
        generation_messages: list[GenerationMessage] = []
        if retrieved_memory:
            # Filter only the already-selected bounded records. Command acknowledgments
            # need no remote memory context, even if the target was previously opted in.
            remote_memory = [
                record for record in retrieved_memory
                if record["sensitivity"] == "remote_allowed" and not decision.intent_detected
            ]
            generation_messages.append(
                GenerationMessage(
                    role="system",
                    content=format_memory_context(retrieved_memory),
                    remote_projection=RemoteProjection(
                        format_memory_context(remote_memory) if remote_memory else None
                    ),
                )
            )
        command_context = format_memory_command_context(decision)
        if command_context is not None:
            generation_messages.append(
                GenerationMessage(
                    role="system", content=command_context,
                    remote_projection=RemoteProjection(None),
                )
            )
        generation_messages.extend(history_messages)
        generation_messages.append(GenerationMessage(
            role="user", content=message, remote_projection=_command_projection(decision)
        ))
        return _PreparedChat(
            conversation_id=conversation_id,
            message=message,
            title=title,
            request_timestamp=request_timestamp,
            user_created_at=user_created_at,
            generation_messages=tuple(generation_messages),
            retrieved_memory=tuple(retrieved_memory),
            staged_memory=staged_memory,
        )

    def _persist_chat(
        self,
        prepared: _PreparedChat,
        generation: ChatGenerationResult,
    ) -> ChatResult:
        assistant_created_at = _timestamp_after(prepared.user_created_at)
        conversation_updated_at = _timestamp_after(assistant_created_at)
        with self.session.begin():
            if prepared.conversation_id is None:
                conversation = self.conversations.create(
                    prepared.title,
                    now=prepared.request_timestamp,
                )
            else:
                conversation = self.conversations.get_fresh(prepared.conversation_id)
                if conversation is None:
                    raise ConversationNotFoundError(prepared.conversation_id)

            user_message = self.messages.create(
                conversation,
                role="user",
                content=prepared.message,
                created_at=prepared.user_created_at,
            )
            if prepared.asset_ids:
                if self.assets is None:
                    raise AssetError("Image attachments are unavailable", 503)
                self.assets.attach(prepared.asset_ids, user_message)
            memory_metadata = [
                *generation.memory,
                *(
                    memory_metadata_reference(record)
                    for record in prepared.retrieved_memory
                ),
            ]
            if prepared.staged_memory is not None:
                memory_metadata.append(
                    memory_metadata_reference(
                        self.memory.apply(
                            prepared.staged_memory,
                            source_message=user_message,
                        )
                    )
                )
            committed_generation = replace(generation, memory=memory_metadata)
            assistant_message = self.messages.create(
                conversation,
                role="assistant",
                content=committed_generation.response,
                created_at=assistant_created_at,
            )
            self.messages.add_metadata(
                assistant_message,
                model=committed_generation.model,
                latency_ms=committed_generation.latency_ms,
                input_tokens=committed_generation.input_tokens,
                output_tokens=committed_generation.output_tokens,
                validator=committed_generation.validator,
                tools=committed_generation.tools,
                memory=committed_generation.memory,
            )
            self.conversations.touch(
                conversation,
                now=conversation_updated_at,
            )

        return ChatResult(
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            response=committed_generation.response,
            metadata=committed_generation,
        )

    @staticmethod
    def _final_event_data(result: ChatResult) -> dict[str, Any]:
        metadata = result.metadata
        return {
            "conversation_id": result.conversation_id,
            "message_id": result.message_id,
            "response": result.response,
            "metadata": {
                "model": metadata.model,
                "latency_ms": metadata.latency_ms,
                "input_tokens": metadata.input_tokens,
                "output_tokens": metadata.output_tokens,
                "validator": metadata.validator,
                "tools": metadata.tools,
                "memory": metadata.memory,
            },
        }

    def chat(
        self, *, conversation_id: str | None, message: str, asset_ids: tuple[str, ...] = (),
    ) -> ChatResult:
        try:
            prepared = self._prepare_chat(
                conversation_id=conversation_id,
                message=message,
                asset_ids=asset_ids,
            )
            try:
                generation = (
                    ChatGenerationResult(response=VISION_NOT_ENABLED, model="media-not-enabled")
                    if prepared.asset_ids else self._generate(prepared.generation_messages)
                )
            except ChatPrivacyError:
                raise
            except Exception as exc:
                raise ChatGenerationError("Assistant generation failed") from exc
            return self._persist_chat(prepared, generation)
        except Exception:
            self.session.rollback()
            raise

    def stream_chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        cancel_event: Event | None = None,
        asset_ids: tuple[str, ...] = (),
    ) -> Iterator[ChatStreamEvent]:
        signal = cancel_event or Event()
        persisted = False
        generation_stream: Iterator[ChatGenerationDelta | ChatGenerationResult] | None = None
        try:
            prepared = self._prepare_chat(
                conversation_id=conversation_id,
                message=message,
                asset_ids=asset_ids,
            )
            yield ChatStreamEvent(
                event="start",
                data={"conversation_id": prepared.conversation_id},
            )
            if signal.is_set():
                return

            generation: ChatGenerationResult | None = None
            generation_stream = iter((
                ChatGenerationDelta(VISION_NOT_ENABLED),
                ChatGenerationResult(response=VISION_NOT_ENABLED, model="media-not-enabled"),
            )) if prepared.asset_ids else self._stream_generation(
                prepared.generation_messages,
                cancel_event=signal,
            )
            try:
                for item in generation_stream:
                    if signal.is_set():
                        return
                    if isinstance(item, ChatGenerationDelta):
                        yield ChatStreamEvent(event="text", data={"delta": item.delta})
                    else:
                        generation = item
            except ChatPrivacyError:
                raise
            except Exception as exc:
                raise ChatGenerationError("Assistant generation failed") from exc

            if signal.is_set():
                return
            if generation is None:
                raise ChatGenerationError("Assistant generation failed")

            result = self._persist_chat(prepared, generation)
            persisted = True
            yield ChatStreamEvent(event="final", data=self._final_event_data(result))
            yield ChatStreamEvent(event="done", data={})
        except Exception:
            self.session.rollback()
            raise
        finally:
            if not persisted:
                signal.set()
            if generation_stream is not None:
                close = getattr(generation_stream, "close", None)
                if callable(close):
                    close()
