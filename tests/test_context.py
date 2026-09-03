from dataclasses import dataclass

import pytest

from backend.chat_service import GenerationMessage, RemoteProjection
from backend.memory import format_memory_context
from runtime.context import (
    HISTORY_OMISSION_NOTICE,
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
        {"role": "system", "content": (
            "Runtime rules\n\nTool rules\n\nCONTEXT_AVAILABILITY\n" + HISTORY_OMISSION_NOTICE
        )},
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

    assert HISTORY_OMISSION_NOTICE not in compiled[0]["content"]
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


@pytest.mark.parametrize("count", [0, MAX_HISTORY_MESSAGES, MAX_HISTORY_MESSAGES + 1])
def test_history_notice_only_when_count_limit_drops_a_turn(count):
    history = [Message("user" if i % 2 == 0 else "assistant", f"turn-{i}")
               for i in range(count)]
    compiled = _compile([*history, Message("user", "Current")])
    assert (HISTORY_OMISSION_NOTICE in compiled[0]["content"]) is (count > MAX_HISTORY_MESSAGES)
    assert len(compiled[1:-1]) <= MAX_HISTORY_MESSAGES


@pytest.mark.parametrize("size", [MAX_HISTORY_CONTEXT_CHARS, MAX_HISTORY_CONTEXT_CHARS + 1])
def test_history_notice_only_when_character_limit_drops_a_turn(size):
    history = "CHAR_LIMIT_CANARY " + "x" * (size - len("CHAR_LIMIT_CANARY "))
    compiled = _compile([Message("user", history), Message("user", "Current")])
    truncated = size > MAX_HISTORY_CONTEXT_CHARS
    assert (HISTORY_OMISSION_NOTICE in compiled[0]["content"]) is truncated
    assert (history in repr(compiled)) is not truncated


def test_omission_notice_is_content_free_and_does_not_summarize_dropped_turns():
    retained = [Message("user" if i % 2 == 0 else "assistant", f"retained-{i}")
                for i in range(MAX_HISTORY_MESSAGES)]
    compiled = [
        _compile([Message("user", old), *retained, Message("user", "Current")])
        for old in ("DROPPED_PRIVATE_CANARY_A", "Different missing topic: CANARY_B")
    ]
    assert compiled[0] == compiled[1]
    assert compiled[0][0]["content"] == (
        "Runtime rules\n\nTool rules\n\nCONTEXT_AVAILABILITY\n" + HISTORY_OMISSION_NOTICE
    )
    assert "CANARY" not in repr(compiled)


@pytest.mark.parametrize("projected_count", [0, MAX_HISTORY_MESSAGES + 1])
def test_notice_uses_only_projected_history_not_privacy_omissions(projected_count):
    source = [
        GenerationMessage("user", "PRIVATE_HISTORY " * 2000, RemoteProjection(
            f"public turn {i}" if i < projected_count else None,
        ))
        for i in range(MAX_HISTORY_MESSAGES + 1)
    ]
    source.append(GenerationMessage("user", "Current"))
    compiled = compile_model_messages(
        source, runtime_system_prompt="Rules", tool_instructions="Tools",
        execution_scope=InferenceExecutionScope.REMOTE,
    )
    assert (HISTORY_OMISSION_NOTICE in compiled[0]["content"]) is (
        projected_count > MAX_HISTORY_MESSAGES
    )
    assert "PRIVATE_HISTORY" not in repr(compiled)


def test_trusted_memory_projection_and_user_correction_stay_separate_from_assistant_guess():
    private = format_memory_context([{"category": "project", "key": "note", "value": "PRIVATE"}])
    visible = format_memory_context([
        {"category": "project", "key": "database", "value": "PostgreSQL"},
    ])
    source = [
        GenerationMessage("system", private, RemoteProjection(visible)),
        GenerationMessage("user", "Which database?"),
        GenerationMessage("assistant", "I guessed MariaDB."),
        GenerationMessage("user", "Correction: use SQLite for this task."),
    ]
    compiled = compile_model_messages(
        source, runtime_system_prompt="Rules", tool_instructions="Tools",
        execution_scope=InferenceExecutionScope.REMOTE,
    )
    assert compiled[1] == {"role": "system", "content": visible}
    assert compiled[-2] == {"role": "assistant", "content": source[-2].content}
    assert compiled[-1] == {"role": "user", "content": source[-1].content}
    assert "PRIVATE" not in repr(compiled)
    assert HISTORY_OMISSION_NOTICE not in repr(compiled)
