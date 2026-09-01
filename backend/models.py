"""SQLAlchemy models for conversations, messages, and message metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid4())


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC timestamps and restore tzinfo lost by SQLite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTC timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: (Message.created_at, Message.id),
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="ck_messages_valid_role",
        ),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    metadata_record: Mapped["MessageMetadata | None"] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        single_parent=True,
        uselist=False,
    )


class MessageMetadata(Base):
    __tablename__ = "message_metadata"

    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validator_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tool_calls_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    memory_refs_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)

    message: Mapped[Message] = relationship(back_populates="metadata_record")


class MemorySlot(Base):
    __tablename__ = "memory_slots"
    __table_args__ = (
        CheckConstraint(
            "sensitivity IN ('local_only', 'remote_allowed')",
            name="ck_memory_slots_valid_sensitivity",
        ),
        CheckConstraint(
            "status IN ('active', 'deleted')",
            name="ck_memory_slots_valid_status",
        ),
        CheckConstraint(
            "current_revision >= 1",
            name="ck_memory_slots_positive_revision",
        ),
        UniqueConstraint(
            "owner_id",
            "category",
            "key",
            name="uq_memory_slots_owner_category_key",
        ),
        Index(
            "ix_memory_slots_owner_status_category_updated",
            "owner_id",
            "status",
            "category",
            "updated_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    current_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    sensitivity: Mapped[str] = mapped_column(
        String(16), default="local_only", server_default="local_only", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    revisions: Mapped[list["MemoryRevision"]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: MemoryRevision.revision,
    )


class MemoryRevision(Base):
    __tablename__ = "memory_revisions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'stale', 'deleted')",
            name="ck_memory_revisions_valid_status",
        ),
        CheckConstraint(
            "revision >= 1",
            name="ck_memory_revisions_positive_revision",
        ),
        CheckConstraint(
            "status != 'deleted' OR value IS NULL",
            name="ck_memory_revisions_deleted_value_redacted",
        ),
        UniqueConstraint(
            "memory_id",
            "revision",
            name="uq_memory_revisions_memory_revision",
        ),
        Index("ix_memory_revisions_memory_status", "memory_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    memory_id: Mapped[str] = mapped_column(
        ForeignKey("memory_slots.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    memory: Mapped[MemorySlot] = relationship(back_populates="revisions")
