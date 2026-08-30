import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.app import create_app
from backend.chat_service import (
    ChatGenerationDelta,
    ChatGenerationError,
    ChatGenerationResult,
    ChatService,
)
from backend.database import Database
from backend.models import Conversation, Message, MessageMetadata
from backend.repositories import ConversationRepository, MessageRepository
from evaluation.hf_backend import GenerationOutput
from runtime.config import load_runtime_config
from runtime.generator import TransformersChatGenerator


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _counts(application) -> tuple[int, int, int]:
    with application.state.database.session_factory() as session:
        return (
            session.scalar(select(func.count()).select_from(Conversation)),
            session.scalar(select(func.count()).select_from(Message)),
            session.scalar(select(func.count()).select_from(MessageMetadata)),
        )


def _parse_sse(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    for line in [*lines, ""]:
        if line == "":
            if event_name is not None:
                raw_data = "\n".join(data_lines)
                events.append(
                    {
                        "event": event_name,
                        "data": json.loads(raw_data) if raw_data else None,
                    }
                )
            event_name = None
            data_lines = []
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    return events


def _post_stream(
    client: TestClient,
    payload: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        headers = dict(response.headers)
        events = _parse_sse(response.iter_lines())
    return headers, events


class RecordingStreamingGenerator:
    def __init__(
        self,
        responses: list[tuple[list[str], ChatGenerationResult]],
    ) -> None:
        self.responses = iter(responses)
        self.histories = []
        self.cancel_events: list[threading.Event] = []

    def stream_response(self, messages, *, cancel_event):
        self.histories.append(list(messages))
        self.cancel_events.append(cancel_event)
        deltas, result = next(self.responses)
        for delta in deltas:
            if cancel_event.is_set():
                return
            yield ChatGenerationDelta(delta=delta)
        if not cancel_event.is_set():
            yield result


def test_streaming_chat_emits_multiple_exact_deltas_and_final_metadata(
    tmp_path: Path,
) -> None:
    expected_response = "Aevon streams exactly."
    expected_metadata = {
        "model": "fake/stream-model",
        "latency_ms": 37,
        "input_tokens": 19,
        "output_tokens": 4,
        "validator": {
            "retry_attempted": False,
            "retry_passed": None,
            "retry_count": 0,
        },
        "tools": [{"name": "calculator"}],
        "memory": [{"id": "memory-1"}],
    }
    generator = RecordingStreamingGenerator(
        [
            (
                ["Aevon", " streams", " exactly."],
                ChatGenerationResult(
                    response=expected_response,
                    **expected_metadata,
                ),
            )
        ]
    )
    application = create_app(
        _database_url(tmp_path / "stream-api.sqlite3"),
        generator=generator,
    )

    with TestClient(application) as client:
        headers, events = _post_stream(client, {"message": "Stream this response"})

        assert headers["content-type"].startswith("text/event-stream")
        assert events[0]["event"] == "start"
        assert [event["event"] for event in events[-2:]] == ["final", "done"]

        deltas = [
            event["data"]["delta"]
            for event in events
            if event["event"] == "text"
        ]
        assert len(deltas) == 3
        assert "".join(deltas) == expected_response

        final = next(event["data"] for event in events if event["event"] == "final")
        assert UUID(final["conversation_id"])
        assert UUID(final["message_id"])
        assert final == {
            "conversation_id": final["conversation_id"],
            "message_id": final["message_id"],
            "response": expected_response,
            "metadata": expected_metadata,
        }

        detail = client.get(f"/api/conversations/{final['conversation_id']}").json()
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
        ]
        assert [message["content"] for message in detail["messages"]] == [
            "Stream this response",
            expected_response,
        ]
        assert detail["messages"][-1]["id"] == final["message_id"]
        assert detail["messages"][-1]["metadata"] == expected_metadata

    assert _counts(application) == (1, 2, 1)
    assert len(generator.cancel_events) == 1


def test_streaming_tool_call_markup_is_hidden_and_only_final_answer_persists(
    tmp_path: Path,
) -> None:
    tool_call = (
        '<tool_call>{"arguments":{"expression":"2 + 3 * 4"},'
        '"name":"calculator"}</tool_call>'
    )

    class ToolStreamingEngine:
        def __init__(self) -> None:
            self.outputs = iter(
                [
                    [
                        "<tool_",
                        tool_call.removeprefix("<tool_"),
                        GenerationOutput(tool_call, 10, 10),
                    ],
                    [
                        "The answer",
                        " is 14.",
                        GenerationOutput("The answer is 14.", 20, 5),
                    ],
                ]
            )

        def generate_detailed_stream(
            self,
            _messages,
            _generation_config,
            *,
            cancel_event,
        ):
            assert cancel_event.is_set() is False
            yield from next(self.outputs)

    engine = ToolStreamingEngine()
    generator = TransformersChatGenerator(
        load_runtime_config(),
        engine_factory=lambda _model, _seed: engine,
    )
    application = create_app(
        _database_url(tmp_path / "stream-tool.sqlite3"),
        generator=generator,
    )

    with TestClient(application) as client:
        _, events = _post_stream(client, {"message": "What is 2 + 3 * 4?"})

        assert [event["event"] for event in events] == [
            "start",
            "text",
            "text",
            "final",
            "done",
        ]
        assert tool_call not in json.dumps(events)
        assert "<tool_result>" not in json.dumps(events)
        final = next(event["data"] for event in events if event["event"] == "final")
        assert final["response"] == "The answer is 14."
        assert final["metadata"]["tools"] == [
            {
                "attempt": 1,
                "name": "calculator",
                "arguments": {"expression": "2 + 3 * 4"},
                "success": True,
                "result": "14",
            }
        ]
        detail = client.get(f"/api/conversations/{final['conversation_id']}").json()
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
        ]
        assert [message["content"] for message in detail["messages"]] == [
            "What is 2 + 3 * 4?",
            "The answer is 14.",
        ]

    assert _counts(application) == (1, 2, 1)


def test_streaming_late_tool_markup_is_sanitized_and_visible_text_persists(
    tmp_path: Path,
) -> None:
    malformed = (
        "Sure, I'll calculate it. "
        '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}'
        "</tool_call>"
    )
    final_response = "Sure, I'll calculate it."

    class MalformedToolStreamingEngine:
        def __init__(self) -> None:
            self.outputs = iter(
                [
                    [
                        "Sure, I'll calculate it. ",
                        malformed.removeprefix("Sure, I'll calculate it. "),
                        GenerationOutput(malformed, 10, 12),
                    ]
                ]
            )

        def generate_detailed_stream(
            self,
            _messages,
            _generation_config,
            *,
            cancel_event,
        ):
            assert cancel_event.is_set() is False
            yield from next(self.outputs)

    engine = MalformedToolStreamingEngine()
    generator = TransformersChatGenerator(
        load_runtime_config(),
        engine_factory=lambda _model, _seed: engine,
    )
    application = create_app(
        _database_url(tmp_path / "stream-malformed-tool.sqlite3"),
        generator=generator,
    )

    with TestClient(application) as client:
        _, events = _post_stream(client, {"message": "What is 2 + 2?"})

        serialized = json.dumps(events)
        assert malformed not in serialized
        assert "Sure, I'll calculate it." in serialized
        assert "<tool_call" not in serialized
        assert "<tool_result" not in serialized
        assert [event["event"] for event in events] == [
            "start",
            "text",
            "final",
            "done",
        ]
        final = next(event["data"] for event in events if event["event"] == "final")
        assert final["response"] == final_response
        assert final["metadata"]["tools"] == []
        detail = client.get(f"/api/conversations/{final['conversation_id']}").json()
        assert malformed not in json.dumps(detail)
        assert [message["content"] for message in detail["messages"]] == [
            "What is 2 + 2?",
            final_response,
        ]

    assert _counts(application) == (1, 2, 1)


def test_streaming_chat_passes_complete_ordered_conversation_history(
    tmp_path: Path,
) -> None:
    generator = RecordingStreamingGenerator(
        [
            (
                ["First", " answer"],
                ChatGenerationResult(response="First answer"),
            ),
            (
                ["Second", " answer"],
                ChatGenerationResult(response="Second answer"),
            ),
        ]
    )
    application = create_app(
        _database_url(tmp_path / "stream-history.sqlite3"),
        generator=generator,
    )

    with TestClient(application) as client:
        _, first_events = _post_stream(client, {"message": "First question"})
        conversation_id = next(
            event["data"]["conversation_id"]
            for event in first_events
            if event["event"] == "final"
        )
        _post_stream(
            client,
            {"conversation_id": conversation_id, "message": "Second question"},
        )

    assert [[item.content for item in history] for history in generator.histories] == [
        ["First question"],
        ["First question", "First answer", "Second question"],
    ]
    assert [[item.role for item in history] for history in generator.histories] == [
        ["user"],
        ["user", "assistant", "user"],
    ]


class PartiallyFailingStreamingGenerator:
    def stream_response(self, messages, *, cancel_event):
        assert messages[-1].role == "user"
        assert cancel_event.is_set() is False
        yield ChatGenerationDelta(delta="partial text that must not persist")
        raise RuntimeError("private streaming generator failure")


class BufferedValidationFailureGenerator:
    failed_candidates = (
        "Paris is the capital",
        "France has Paris capital",
        "Paris remains the capital",
    )

    def stream_response(self, messages, *, cancel_event):
        assert messages[-1].role == "user"
        assert cancel_event.is_set() is False
        raise ChatGenerationError("Assistant generation failed")
        yield  # pragma: no cover - keeps this method on the streaming protocol


def test_streaming_final_validation_failure_emits_only_terminal_error(
    tmp_path: Path,
) -> None:
    generator = BufferedValidationFailureGenerator()
    application = create_app(
        _database_url(tmp_path / "stream-validation-failure.sqlite3"),
        generator=generator,
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        _, events = _post_stream(
            client,
            {
                "message": (
                    "What is the capital of France? Answer in exactly 3 words."
                )
            },
        )

        assert [event["event"] for event in events] == ["start", "error"]
        assert events[-1]["data"] == {"detail": "Assistant generation failed"}
        assert not any(
            event["event"] in {"text", "final", "done"} for event in events
        )
        assert all(
            candidate not in json.dumps(events)
            for candidate in generator.failed_candidates
        )
        assert client.get("/api/conversations").json() == []

    assert _counts(application) == (0, 0, 0)


def test_streaming_final_validation_failure_preserves_existing_conversation(
    tmp_path: Path,
) -> None:
    application = create_app(
        _database_url(tmp_path / "stream-validation-existing.sqlite3"),
        generator=BufferedValidationFailureGenerator(),
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        existing = client.post(
            "/api/conversations",
            json={"title": "Keep unchanged"},
        ).json()
        before = client.get(f"/api/conversations/{existing['id']}").json()

        _, events = _post_stream(
            client,
            {
                "conversation_id": existing["id"],
                "message": "Answer in exactly 3 words.",
            },
        )
        after = client.get(f"/api/conversations/{existing['id']}").json()

        assert [event["event"] for event in events] == ["start", "error"]
        assert after == before

    assert _counts(application) == (1, 0, 0)


def test_streaming_generator_failure_never_persists_partial_output(
    tmp_path: Path,
) -> None:
    application = create_app(
        _database_url(tmp_path / "stream-failure.sqlite3"),
        generator=PartiallyFailingStreamingGenerator(),
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        _, events = _post_stream(client, {"message": "Fail after a delta"})

        assert any(
            event["event"] == "text"
            and event["data"] == {"delta": "partial text that must not persist"}
            for event in events
        )
        error = next(event["data"] for event in events if event["event"] == "error")
        assert error == {"detail": "Assistant generation failed"}
        assert "private streaming generator failure" not in json.dumps(events)
        assert not any(event["event"] == "final" for event in events)
        assert client.get("/api/conversations").json() == []

    assert _counts(application) == (0, 0, 0)


def test_streaming_generation_and_iteration_run_outside_sql_transactions(
    tmp_path: Path,
) -> None:
    database = Database.from_url(_database_url(tmp_path / "stream-transaction.sqlite3"))
    database.create_schema()
    try:
        with database.session_factory() as session:
            with session.begin():
                conversation = ConversationRepository(session).create("Existing")
                conversation_id = conversation.id

            transaction_states: list[bool] = []

            class TransactionCheckingGenerator:
                def stream_response(self, messages, *, cancel_event):
                    del cancel_event
                    transaction_states.append(session.in_transaction())
                    assert [item.content for item in messages] == ["Next question"]
                    yield ChatGenerationDelta(delta="Generated")
                    transaction_states.append(session.in_transaction())
                    yield ChatGenerationDelta(delta=" outside SQL")
                    transaction_states.append(session.in_transaction())
                    yield ChatGenerationResult(response="Generated outside SQL")

            events = list(
                ChatService(session, generator=TransactionCheckingGenerator()).stream_chat(
                    conversation_id=conversation_id,
                    message="Next question",
                    cancel_event=threading.Event(),
                )
            )

            assert events
            assert transaction_states == [False, False, False]
            assert session.in_transaction() is False
            assert [
                item.content
                for item in MessageRepository(session).list_for_conversation(
                    conversation_id
                )
            ] == ["Next question", "Generated outside SQL"]
    finally:
        database.engine.dispose()


def test_stream_cancellation_does_not_persist_a_partial_assistant_turn(
    tmp_path: Path,
) -> None:
    database = Database.from_url(_database_url(tmp_path / "stream-cancel.sqlite3"))
    database.create_schema()
    try:
        with database.session_factory() as session:
            with session.begin():
                conversation = ConversationRepository(session).create("Existing")
                conversation_id = conversation.id

            delta_emitted = False

            class CancelledGenerator:
                def stream_response(self, _messages, *, cancel_event):
                    nonlocal delta_emitted
                    delta_emitted = True
                    yield ChatGenerationDelta(delta="partial")
                    if cancel_event.is_set():
                        return
                    yield ChatGenerationResult(response="partial but complete")

            cancel_event = threading.Event()
            stream = ChatService(session, generator=CancelledGenerator()).stream_chat(
                conversation_id=conversation_id,
                message="Cancel this",
                cancel_event=cancel_event,
            )
            for _ in range(3):
                next(stream)
                if delta_emitted:
                    break
            assert delta_emitted is True
            cancel_event.set()
            list(stream)

            assert session.in_transaction() is False
            assert MessageRepository(session).list_for_conversation(conversation_id) == []
    finally:
        database.engine.dispose()
