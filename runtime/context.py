"""Deterministic minimization of model-visible generation context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CONTEXT_CHARS = 20_000

_TRUSTED_CONTEXT_PREFIXES = (
    "MEMORY_CONTEXT_V1\n",
    "MEMORY_COMMAND_V1\n",
)


class ContextMessage(Protocol):
    role: str
    content: str


def _is_trusted_runtime_context(message: ContextMessage) -> bool:
    return message.role == "system" and message.content.startswith(
        _TRUSTED_CONTEXT_PREFIXES
    )


def _select_recent_history(
    history: Sequence[ContextMessage],
) -> list[ContextMessage]:
    selected_newest_first: list[ContextMessage] = []
    content_chars = 0
    for message in reversed(history):
        if len(selected_newest_first) == MAX_HISTORY_MESSAGES:
            break
        message_chars = len(message.content)
        if content_chars + message_chars > MAX_HISTORY_CONTEXT_CHARS:
            break
        selected_newest_first.append(message)
        content_chars += message_chars

    selected = list(reversed(selected_newest_first))
    while selected and selected[0].role == "assistant":
        selected.pop(0)
    return selected


def compile_model_messages(
    messages: Sequence[ContextMessage],
    *,
    runtime_system_prompt: str,
    tool_instructions: str,
) -> list[dict[str, str]]:
    """Compile the minimum deterministic context shared by every provider."""

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

    recent_history = _select_recent_history(prior_messages[history_start:])
    current_user = messages[-1]
    return [
        {
            "role": "system",
            "content": f"{runtime_system_prompt}\n\n{tool_instructions}",
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
