"""Small persistence repositories used by the service and route layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import (
    Conversation,
    MemoryRevision,
    MemorySlot,
    Message,
    MessageMetadata,
    utc_now,
)

VALID_MESSAGE_ROLES = frozenset({"user", "assistant", "system", "tool"})


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, title: str, *, now: datetime | None = None) -> Conversation:
        timestamp = now or utc_now()
        conversation = Conversation(
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def get_fresh(self, conversation_id: str) -> Conversation | None:
        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .execution_options(populate_existing=True)
        )
        return self.session.scalar(statement)

    def get_with_messages(self, conversation_id: str) -> Conversation | None:
        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(
                selectinload(Conversation.messages).selectinload(Message.metadata_record)
            )
        )
        return self.session.scalar(statement)

    def list(self) -> list[Conversation]:
        statement = select(Conversation).order_by(
            Conversation.updated_at.desc(),
            Conversation.created_at.desc(),
        )
        return list(self.session.scalars(statement))

    def rename(self, conversation: Conversation, title: str) -> Conversation:
        conversation.title = title
        conversation.updated_at = utc_now()
        self.session.flush()
        return conversation

    def touch(self, conversation: Conversation, *, now: datetime | None = None) -> None:
        conversation.updated_at = now or utc_now()
        self.session.flush()

    def delete(self, conversation: Conversation) -> None:
        self.session.delete(conversation)
        self.session.flush()


class MessageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        conversation: Conversation,
        *,
        role: str,
        content: str,
        created_at: datetime | None = None,
    ) -> Message:
        if role not in VALID_MESSAGE_ROLES:
            raise ValueError(f"Unsupported message role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must not be empty")

        message = Message(
            conversation=conversation,
            role=role,
            content=content,
            created_at=created_at or utc_now(),
        )
        self.session.add(message)
        self.session.flush()
        return message

    def list_for_conversation(self, conversation_id: str) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        return list(self.session.scalars(statement))

    def add_metadata(
        self,
        message: Message,
        *,
        model: str | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        validator: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        memory: list[Any] | None = None,
    ) -> MessageMetadata:
        metadata = MessageMetadata(
            message=message,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            validator_json=validator,
            tool_calls_json=tools,
            memory_refs_json=memory,
        )
        self.session.add(metadata)
        self.session.flush()
        return metadata


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_slot(self, memory_id: str) -> MemorySlot | None:
        return self.session.get(MemorySlot, memory_id)

    def get_slot_fresh(self, memory_id: str) -> MemorySlot | None:
        statement = (
            select(MemorySlot)
            .where(MemorySlot.id == memory_id)
            .execution_options(populate_existing=True)
        )
        return self.session.scalar(statement)

    def get_slot_by_key(
        self,
        *,
        owner_id: str,
        category: str,
        key: str,
        fresh: bool = False,
    ) -> MemorySlot | None:
        statement = select(MemorySlot).where(
            MemorySlot.owner_id == owner_id,
            MemorySlot.category == category,
            MemorySlot.key == key,
        )
        if fresh:
            statement = statement.execution_options(populate_existing=True)
        return self.session.scalar(statement)

    def get_current_revision(self, memory: MemorySlot) -> MemoryRevision | None:
        statement = select(MemoryRevision).where(
            MemoryRevision.memory_id == memory.id,
            MemoryRevision.revision == memory.current_revision,
        )
        return self.session.scalar(statement)

    def list_current(
        self,
        *,
        owner_id: str,
        status: str = "active",
        category: str | None = None,
    ) -> list[tuple[MemorySlot, MemoryRevision]]:
        statement = (
            select(MemorySlot, MemoryRevision)
            .join(
                MemoryRevision,
                (MemoryRevision.memory_id == MemorySlot.id)
                & (MemoryRevision.revision == MemorySlot.current_revision),
            )
            .where(
                MemorySlot.owner_id == owner_id,
                MemorySlot.status == status,
                MemoryRevision.status == status,
            )
        )
        if category is not None:
            statement = statement.where(MemorySlot.category == category)
        statement = statement.order_by(
            MemorySlot.updated_at.desc(),
            MemorySlot.id.asc(),
        )
        return list(self.session.execute(statement).tuples())

    def list_revisions(self, memory_id: str) -> list[MemoryRevision]:
        statement = (
            select(MemoryRevision)
            .where(MemoryRevision.memory_id == memory_id)
            .order_by(MemoryRevision.revision.asc())
        )
        return list(self.session.scalars(statement))
