from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.chat_service import ChatGenerationResult
from backend.models import Conversation


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


@pytest.fixture
def application(tmp_path: Path):
    return create_app(_database_url(tmp_path / "api.sqlite3"))


@pytest.fixture
def client(application):
    with TestClient(application) as test_client:
        yield test_client


def _assert_utc(timestamp: str) -> None:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)


def _create_conversation(client: TestClient, title: str = "AmitAI dev") -> dict:
    response = client.post("/api/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_list_conversations(client: TestClient) -> None:
    created = _create_conversation(client, "  AmitAI dev  ")
    default_title = client.post("/api/conversations")

    assert UUID(created["id"])
    assert created["title"] == "AmitAI dev"
    assert created["archived"] is False
    _assert_utc(created["created_at"])
    _assert_utc(created["updated_at"])
    assert default_title.status_code == 201
    assert default_title.json()["title"] == "New conversation"

    listed = client.get("/api/conversations")
    assert listed.status_code == 200
    assert {item["id"] for item in listed.json()} == {
        created["id"],
        default_title.json()["id"],
    }


def test_conversations_are_ordered_by_updated_at_descending(
    client: TestClient,
    application,
) -> None:
    first = _create_conversation(client, "First")
    second = _create_conversation(client, "Second")

    with application.state.database.session_factory() as session, session.begin():
        session.get(Conversation, first["id"]).updated_at = datetime(
            2025, 1, 1, tzinfo=timezone.utc
        )
        session.get(Conversation, second["id"]).updated_at = datetime(
            2025, 1, 2, tzinfo=timezone.utc
        )

    listed = client.get("/api/conversations")

    assert [item["id"] for item in listed.json()] == [second["id"], first["id"]]


def test_read_and_rename_conversation(client: TestClient) -> None:
    created = _create_conversation(client)

    renamed = client.patch(
        f"/api/conversations/{created['id']}",
        json={"title": "  Backend work  "},
    )
    read = client.get(f"/api/conversations/{created['id']}")

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Backend work"
    assert renamed.json()["created_at"] == created["created_at"]
    assert read.status_code == 200
    assert read.json()["title"] == "Backend work"
    assert read.json()["messages"] == []


@pytest.mark.parametrize("payload", [{}, {"title": ""}, {"title": "   "}, {"extra": 1}])
def test_invalid_rename_is_rejected(client: TestClient, payload: dict) -> None:
    created = _create_conversation(client)

    response = client.patch(f"/api/conversations/{created['id']}", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Explain this Python error"},
        {"conversation_id": None, "message": "Explain this Python error"},
    ],
)
def test_chat_without_an_id_creates_and_persists_a_conversation(
    client: TestClient,
    payload: dict,
) -> None:
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"conversation_id", "message_id", "response", "metadata"}
    assert UUID(body["conversation_id"])
    assert UUID(body["message_id"])
    assert body["response"] == "This is a mocked AmitAI response."
    assert body["metadata"] == {
        "model": "mock",
        "latency_ms": 0,
        "input_tokens": None,
        "output_tokens": None,
        "validator": {"retry_attempted": False, "retry_passed": None},
        "tools": [],
        "memory": [],
    }

    conversation = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert conversation["title"] == "Explain this Python error"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert [message["content"] for message in conversation["messages"]] == [
        "Explain this Python error",
        "This is a mocked AmitAI response.",
    ]
    assert conversation["messages"][1]["id"] == body["message_id"]
    assert conversation["messages"][1]["metadata"]["model"] == "mock"


def test_chat_uses_an_existing_conversation_and_passes_ordered_history(
    tmp_path: Path,
) -> None:
    histories = []

    def recording_generator(messages):
        histories.append(list(messages))
        return ChatGenerationResult(response=f"Mock answer {len(histories)}")

    application = create_app(
        _database_url(tmp_path / "history.sqlite3"),
        generator=recording_generator,
    )
    with TestClient(application) as client:
        created = _create_conversation(client, "Existing")
        first = client.post(
            "/api/chat",
            json={"conversation_id": created["id"], "message": "First question"},
        )
        second = client.post(
            "/api/chat",
            json={"conversation_id": created["id"], "message": "Second question"},
        )

        assert first.status_code == second.status_code == 200
        assert len(client.get("/api/conversations").json()) == 1
        detail = client.get(f"/api/conversations/{created['id']}").json()
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [item.content for item in histories[1]] == [
            "First question",
            "Mock answer 1",
            "Second question",
        ]
        created_times = [
            datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            for item in detail["messages"]
        ]
        assert created_times == sorted(created_times)
        assert len(set(created_times)) == len(created_times)


def test_chat_exposes_and_persists_token_and_validator_metadata(tmp_path: Path) -> None:
    def metadata_generator(_messages):
        return ChatGenerationResult(
            response="Real response",
            model="fake/real-model",
            latency_ms=42,
            input_tokens=355,
            output_tokens=47,
            validator={
                "retry_attempted": True,
                "retry_passed": True,
                "retry_count": 2,
                "parsed_constraints": [{"type": "exact_words", "count": 5}],
                "final_validation": {"passed": True},
            },
        )

    application = create_app(
        _database_url(tmp_path / "metadata.sqlite3"),
        generator=metadata_generator,
    )
    with TestClient(application) as client:
        response = client.post("/api/chat", json={"message": "Write exactly 5 words."})

        assert response.status_code == 200
        metadata = response.json()["metadata"]
        assert metadata["input_tokens"] == 355
        assert metadata["output_tokens"] == 47
        assert metadata["validator"]["retry_count"] == 2
        assert metadata["validator"]["final_validation"] == {"passed": True}

        detail = client.get(
            f"/api/conversations/{response.json()['conversation_id']}"
        ).json()
        persisted = detail["messages"][-1]["metadata"]
        assert persisted["input_tokens"] == 355
        assert persisted["output_tokens"] == 47
        assert persisted["validator"] == metadata["validator"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": None},
        {"message": ""},
        {"message": "   "},
        {"message": "Hello", "role": "assistant"},
    ],
)
def test_invalid_chat_payload_is_rejected(client: TestClient, payload: dict) -> None:
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422


def test_unknown_conversations_return_404_without_creating_data(client: TestClient) -> None:
    unknown_id = str(uuid4())

    assert client.get(f"/api/conversations/{unknown_id}").status_code == 404
    assert (
        client.patch(
            f"/api/conversations/{unknown_id}",
            json={"title": "Unknown"},
        ).status_code
        == 404
    )
    assert client.delete(f"/api/conversations/{unknown_id}").status_code == 404
    chat = client.post(
        "/api/chat",
        json={"conversation_id": unknown_id, "message": "Hello"},
    )
    assert chat.status_code == 404
    assert client.get("/api/conversations").json() == []
