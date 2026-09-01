from dataclasses import dataclass

from runtime.context import (
    MAX_HISTORY_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
    compile_model_messages,
)
from runtime.privacy import InferenceExecutionScope


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def _compile(messages: list[Message]) -> list[dict[str, str]]:
    return compile_model_messages(
        messages,
        runtime_system_prompt="Runtime rules",
        tool_instructions="Tool rules",
        execution_scope=InferenceExecutionScope.LOCAL,
    )


def test_context_compiler_keeps_only_newest_complete_history_messages() -> None:
    old_canary = "OLD_PRIVATE_HISTORY_CANARY_918273"
    very_old_canary = "VERY_OLD_PRIVATE_HISTORY_CANARY_817263"
    recent_canary = "RECENT_HISTORY_CANARY_192837"
    history = [
        Message("user", very_old_canary),
        Message("assistant", old_canary),
        *(
            Message("user" if index % 2 == 0 else "assistant", f"history-{index}")
            for index in range(MAX_HISTORY_MESSAGES)
        ),
        Message("assistant", recent_canary),
    ]
    current = "CURRENT_USER_REQUEST_445566"

    compiled = _compile([*history, Message("user", current)])
    serialized = repr(compiled)

    assert very_old_canary not in serialized
    assert old_canary not in serialized
    assert recent_canary in serialized
    assert compiled[-1] == {"role": "user", "content": current}
    assert len(compiled[1:-1]) <= MAX_HISTORY_MESSAGES


def test_context_compiler_omits_history_when_newest_message_exceeds_char_budget() -> None:
    oversized = "x" * (MAX_HISTORY_CONTEXT_CHARS + 1)
    current = "Current request remains intact " + ("y" * 25_000)

    compiled = _compile(
        [
            Message("user", "Older history"),
            Message("assistant", oversized),
            Message("user", current),
        ]
    )

    assert compiled == [
        {"role": "system", "content": "Runtime rules\n\nTool rules"},
        {"role": "user", "content": current},
    ]


def test_context_compiler_drops_orphan_assistant_at_history_window_start() -> None:
    compiled = _compile(
        [
            Message("assistant", "orphaned by prior omission"),
            Message("user", "retained user"),
            Message("assistant", "retained assistant"),
            Message("user", "current"),
        ]
    )

    assert compiled[1:] == [
        {"role": "user", "content": "retained user"},
        {"role": "assistant", "content": "retained assistant"},
        {"role": "user", "content": "current"},
    ]


def test_trusted_runtime_context_is_ordered_before_budgeted_history() -> None:
    memory = "MEMORY_CONTEXT_V1\n<memory_context>{}</memory_context>"
    command = "MEMORY_COMMAND_V1\n<memory_command>{}</memory_command>"

    compiled = _compile(
        [
            Message("system", memory),
            Message("system", command),
            Message("user", "historical user"),
            Message("assistant", "historical assistant"),
            Message("user", "current user"),
        ]
    )

    assert compiled == [
        {"role": "system", "content": "Runtime rules\n\nTool rules"},
        {"role": "system", "content": memory},
        {"role": "system", "content": command},
        {"role": "user", "content": "historical user"},
        {"role": "assistant", "content": "historical assistant"},
        {"role": "user", "content": "current user"},
    ]
