"""V4 compiler output fingerprints captured before the V5 implementation."""

import hashlib
import json
from pathlib import Path

import pytest

from backend.chat_service import GenerationMessage as Message
from backend.chat_service import RemoteProjection
from backend.memory import format_memory_context
from runtime.context import compile_model_messages
from runtime.privacy import InferenceExecutionScope

SNAPSHOT = Path(__file__).parent / "fixtures/context_compiler_v4.json"


def sources():
    memory = format_memory_context([
        {"category": "project", "key": "dispatch.database", "value": "PostgreSQL"},
    ])
    normal = [Message("user", "A retained task"), Message("assistant", "Acknowledged")]
    return {
        "empty": [],
        "normal": normal,
        "count": [Message("user" if i % 2 == 0 else "assistant", f"turn {i}")
                  for i in range(45)],
        "chars": [Message("user" if i % 2 == 0 else "assistant", "x" * 1200)
                  for i in range(19)],
        "oversized": [*normal, Message("assistant", "x" * 20_001)],
        "orphan": [Message("assistant", "Orphan"), *normal],
        "memory": [Message("system", memory), *normal],
        "command": [Message("system", memory), Message(
            "system", 'MEMORY_COMMAND_V1\n<memory_command>{"status":"staged"}</memory_command>',
        ), *normal],
        "projection": [Message("system", memory, RemoteProjection(None)),
                       Message("user", "private" * 4000, RemoteProjection("Public task")),
                       Message("assistant", "private ack", RemoteProjection(None))],
        "privacy_only": [Message("user", "private" * 4000, RemoteProjection(None))] * 25,
    }


def compile_source(history, scope):
    return compile_model_messages(
        [*history, Message("user", "Current request")],
        runtime_system_prompt="Production rules", tool_instructions="Tool rules",
        execution_scope=scope,
    )


def fingerprint(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize("scope", list(InferenceExecutionScope))
@pytest.mark.parametrize("name", list(sources()))
def test_compiler_messages_equal_starting_revision(name, scope):
    baseline = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert baseline["revision"] == "75702d68148248cec7efb4451fa78edad150641e"
    assert fingerprint(compile_source(sources()[name], scope)) == baseline[scope.value][name]
