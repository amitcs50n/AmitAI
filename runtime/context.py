"""Deterministic minimization of model-visible generation context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from backend.chat_service import RemoteProjection

from .privacy import InferenceExecutionScope

MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CONTEXT_CHARS = 20_000
HISTORY_OMISSION_NOTICE = (
    "Earlier conversation turns were omitted from the available context. "
    "Do not infer or reconstruct their contents."
)

_TRUSTED_CONTEXT_PREFIXES = (
    "MEMORY_CONTEXT_V1\n",
    "MEMORY_COMMAND_V1\n",
)


class ContextMessage(Protocol):
    role: str
    content: str


@dataclass(frozen=True)
class _ProjectedMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CompiledModelContext:
    """Available provider input plus content-free conversational availability.

    Counts exclude trusted frames and the current user. Latest-turn flags refer
    to source positions, so privacy projection cannot substitute an older turn.
    No omitted message, summary, or raw source is retained here.
    """

    messages: list[dict[str, str]]
    history_truncated: bool
    retained_history_count: int
    retained_user_turn_count: int
    latest_prior_turn_retained: bool
    latest_prior_user_turn_retained: bool
    trusted_context_count: int


def _project(
    messages: Sequence[ContextMessage], scope: InferenceExecutionScope
) -> list[ContextMessage]:
    projected: list[ContextMessage] = []
    for message in messages:
        projection = getattr(message, "remote_projection", None)
        if scope is InferenceExecutionScope.REMOTE and projection is not None:
            if not isinstance(projection, RemoteProjection):
                raise TypeError("Invalid remote context projection")
            if projection.content is not None:
                projected.append(_ProjectedMessage(message.role, projection.content))
        else:
            projected.append(message)
    return projected


def _is_trusted_runtime_context(message: ContextMessage) -> bool:
    return message.role == "system" and message.content.startswith(
        _TRUSTED_CONTEXT_PREFIXES
    )


def _select_recent_history(
    history: Sequence[ContextMessage],
) -> tuple[list[ContextMessage], bool]:
    selected_newest_first: list[ContextMessage] = []
    content_chars = 0
    truncated = False
    for message in reversed(history):
        if len(selected_newest_first) == MAX_HISTORY_MESSAGES:
            truncated = True
            break
        message_chars = len(message.content)
        if content_chars + message_chars > MAX_HISTORY_CONTEXT_CHARS:
            truncated = True
            break
        selected_newest_first.append(message)
        content_chars += message_chars

    selected = list(reversed(selected_newest_first))
    while selected and selected[0].role == "assistant":
        selected.pop(0)
    return selected, truncated


def compile_model_context(
    messages: Sequence[ContextMessage],
    *,
    runtime_system_prompt: str,
    tool_instructions: str,
    execution_scope: InferenceExecutionScope,
) -> CompiledModelContext:
    """Compile the minimum deterministic context shared by every provider."""

    if not isinstance(execution_scope, InferenceExecutionScope):
        raise TypeError("Invalid inference execution scope")
    if not messages or messages[-1].role != "user":
        raise ValueError("Runtime chat messages must end with the current user turn")

    prior_messages = messages[:-1]
    trusted_context: list[ContextMessage] = []
    history_start = 0
    for index, message in enumerate(prior_messages):
        if not _is_trusted_runtime_context(message):
            history_start = index
            break
        trusted_context.append(message)
    else:
        history_start = len(prior_messages)

    trusted_context = _project(trusted_context, execution_scope)
    # Decide only from projected history: private omissions and orphan cleanup
    # must not manufacture a limit-truncation signal or disclose removed content.
    history = prior_messages[history_start:]
    projected_history: list[ContextMessage] = []
    projected_positions: list[int] = []
    for index, message in enumerate(history):
        projected = _project([message], execution_scope)
        if projected:
            projected_history.extend(projected)
            projected_positions.append(index)
    recent_history, truncated = _select_recent_history(projected_history)
    retained_positions = set(projected_positions[-len(recent_history):]) if recent_history else set()
    # Only roles/positions are consulted outside the bounded projected content.
    prior_turns = [i for i, message in enumerate(history) if message.role in ("user", "assistant")]
    prior_users = [i for i, message in enumerate(history) if message.role == "user"]
    projected_current = _project(messages[-1:], execution_scope)
    if not projected_current:
        raise ValueError("Current user context must not be omitted")
    current_user = projected_current[0]
    system_content = f"{runtime_system_prompt}\n\n{tool_instructions}"
    if truncated:
        system_content += f"\n\nCONTEXT_AVAILABILITY\n{HISTORY_OMISSION_NOTICE}"
    compiled = [
        {
            "role": "system",
            "content": system_content,
        },
        *(
            {"role": message.role, "content": message.content}
            for message in trusted_context
        ),
        *(
            {"role": message.role, "content": message.content}
            for message in recent_history
        ),
        {"role": current_user.role, "content": current_user.content},
    ]
    return CompiledModelContext(
        messages=compiled,
        history_truncated=truncated,
        retained_history_count=sum(m.role in ("user", "assistant") for m in recent_history),
        retained_user_turn_count=sum(m.role == "user" for m in recent_history),
        latest_prior_turn_retained=bool(prior_turns and prior_turns[-1] in retained_positions),
        latest_prior_user_turn_retained=bool(prior_users and prior_users[-1] in retained_positions),
        trusted_context_count=len(trusted_context),
    )


def compile_model_messages(
    messages: Sequence[ContextMessage],
    *,
    runtime_system_prompt: str,
    tool_instructions: str,
    execution_scope: InferenceExecutionScope,
) -> list[dict[str, str]]:
    """Compatibility wrapper: preserve the V4 production message list exactly."""
    return compile_model_context(
        messages, runtime_system_prompt=runtime_system_prompt,
        tool_instructions=tool_instructions, execution_scope=execution_scope,
    ).messages
