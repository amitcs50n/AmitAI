import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.chat_service import (
    ChatGenerationResult,
    ChatService,
    ConversationNotFoundError,
)
from backend.database import Database
from backend.models import Conversation, Message, MessageMetadata
from backend.repositories import ConversationRepository, MessageRepository
from evaluation.hf_backend import GenerationOutput
from runtime.config import load_runtime_config
from runtime.generator import TransformersChatGenerator
from runtime.tooling import ToolDefinition, ToolRegistry
from tests.app_factory import create_test_app as create_app


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _counts(application) -> tuple[int, int, int]:
    with application.state.database.session_factory() as session:
        return (
            session.scalar(select(func.count()).select_from(Conversation)),
            session.scalar(select(func.count()).select_from(Message)),
            session.scalar(select(func.count()).select_from(MessageMetadata)),
        )


def test_deleting_a_conversation_cascades_messages_and_metadata(tmp_path: Path) -> None:
    application = create_app(_database_url(tmp_path / "cascade.sqlite3"))

    with TestClient(application) as client:
        chat = client.post("/api/chat", json={"message": "Persist this turn"})
        assert chat.status_code == 200
        conversation_id = chat.json()["conversation_id"]
        assert _counts(application) == (1, 2, 1)
        with application.state.database.session_factory() as session:
            metadata = session.get(MessageMetadata, chat.json()["message_id"])
            assert metadata.model == "mock"
            assert metadata.validator_json == {
                "retry_attempted": False,
                "retry_passed": None,
            }
            assert metadata.tool_calls_json == []
            assert metadata.memory_refs_json == []

        deleted = client.delete(f"/api/conversations/{conversation_id}")

        assert deleted.status_code == 204
        assert deleted.content == b""
        assert _counts(application) == (0, 0, 0)
        assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_conversation_history_survives_an_application_restart(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "persistent.sqlite3")
    first_application = create_app(database_url)

    with TestClient(first_application) as client:
        chat = client.post("/api/chat", json={"message": "Keep this conversation"})
        assert chat.status_code == 200
        conversation_id = chat.json()["conversation_id"]

    restarted_application = create_app(database_url)
    with TestClient(restarted_application) as client:
        conversation = client.get(f"/api/conversations/{conversation_id}")

        assert conversation.status_code == 200
        assert [message["role"] for message in conversation.json()["messages"]] == [
            "user",
            "assistant",
        ]


class FailingGenerator:
    def generate_response(self, messages):
        assert messages[-1].role == "user"
        raise RuntimeError("private generator failure")


def test_generation_failure_rolls_back_an_automatic_conversation(tmp_path: Path) -> None:
    application = create_app(
        _database_url(tmp_path / "failed-new.sqlite3"),
        generator=FailingGenerator(),
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/api/chat", json={"message": "This must roll back"})

        assert response.status_code == 500
        assert response.json() == {"detail": "Assistant generation failed"}
        assert "private generator failure" not in response.text
        assert _counts(application) == (0, 0, 0)


def test_generation_failure_preserves_an_existing_conversation(tmp_path: Path) -> None:
    application = create_app(
        _database_url(tmp_path / "failed-existing.sqlite3"),
        generator=FailingGenerator(),
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/conversations",
            json={"title": "Keep me unchanged"},
        ).json()
        before = client.get(f"/api/conversations/{created['id']}").json()

        response = client.post(
            "/api/chat",
            json={"conversation_id": created["id"], "message": "This must roll back"},
        )
        after = client.get(f"/api/conversations/{created['id']}").json()

        assert response.status_code == 500
        assert after == before
        assert _counts(application) == (1, 0, 0)


def test_invalid_generator_metadata_rolls_back_before_response_validation(
    tmp_path: Path,
) -> None:
    def invalid_generator(_messages):
        return ChatGenerationResult(response="Invalid metadata", validator=None)

    application = create_app(
        _database_url(tmp_path / "invalid-metadata.sqlite3"),
        generator=invalid_generator,
    )

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/api/chat", json={"message": "This must also roll back"})

        assert response.status_code == 500
        assert response.json() == {"detail": "Assistant generation failed"}
        assert _counts(application) == (0, 0, 0)


def test_existing_conversation_generation_runs_outside_a_transaction(
    tmp_path: Path,
) -> None:
    database = Database.from_url(
        _database_url(tmp_path / "transaction.sqlite3"),
        encrypted=False,
    )
    database.create_schema()
    try:
        with database.session_factory() as session:
            with session.begin():
                conversation = ConversationRepository(session).create("Existing")
                conversation_id = conversation.id

            def asserting_generator(messages):
                assert session.in_transaction() is False
                assert [item.content for item in messages] == ["Next question"]
                return ChatGenerationResult(response="Generated outside SQL")

            result = ChatService(session, generator=asserting_generator).chat(
                conversation_id=conversation_id,
                message="Next question",
            )

            assert result.response == "Generated outside SQL"
            assert session.in_transaction() is False
            assert [
                item.content
                for item in MessageRepository(session).list_for_conversation(
                    conversation_id
                )
            ] == ["Next question", "Generated outside SQL"]
    finally:
        database.engine.dispose()


def test_tool_execution_runs_outside_a_transaction(tmp_path: Path) -> None:
    database = Database.from_url(
        _database_url(tmp_path / "tool-transaction.sqlite3"),
        encrypted=False,
    )
    database.create_schema()
    try:
        with database.session_factory() as session:
            with session.begin():
                conversation = ConversationRepository(session).create("Existing")
                conversation_id = conversation.id

            transaction_states: list[bool] = []

            class TransactionProbeTool:
                definition = ToolDefinition(
                    name="transaction_probe",
                    description="Test transaction boundaries",
                    arguments={},
                )

                def validate_arguments(self, arguments):
                    assert arguments == {}
                    return {}

                def execute(self, arguments):
                    assert arguments == {}
                    transaction_states.append(session.in_transaction())
                    return "outside"

            class SequenceEngine:
                def __init__(self) -> None:
                    self.outputs = iter(
                        [
                            GenerationOutput(
                                '<tool_call>{"arguments":{},"name":"transaction_probe"}'
                                "</tool_call>",
                                5,
                                5,
                            ),
                            GenerationOutput("Completed outside SQL", 8, 3),
                        ]
                    )

                def generate_detailed(self, _messages, _generation_config):
                    transaction_states.append(session.in_transaction())
                    return next(self.outputs)

            generator = TransformersChatGenerator(
                load_runtime_config(),
                engine_factory=lambda _model, _seed: SequenceEngine(),
                tool_registry=ToolRegistry([TransactionProbeTool()]),
            )
            result = ChatService(session, generator=generator).chat(
                conversation_id=conversation_id,
                message="Use the transaction probe",
            )

            assert transaction_states == [False, False, False]
            assert result.response == "Completed outside SQL"
            assert result.metadata.tools == [
                {
                    "attempt": 1,
                    "name": "transaction_probe",
                    "arguments": {},
                    "success": True,
                    "result": "outside",
                }
            ]
            assert [
                item.content
                for item in MessageRepository(session).list_for_conversation(
                    conversation_id
                )
            ] == ["Use the transaction probe", "Completed outside SQL"]
    finally:
        database.engine.dispose()


def test_existing_conversation_is_freshly_refetched_before_persistence(
    tmp_path: Path,
) -> None:
    database = Database.from_url(
        _database_url(tmp_path / "deleted-during-generation.sqlite3"),
        encrypted=False,
    )
    database.create_schema()
    try:
        with database.session_factory() as session:
            with session.begin():
                conversation = ConversationRepository(session).create("Delete during generation")
                conversation_id = conversation.id

            def deleting_generator(_messages):
                assert session.in_transaction() is False
                with database.session_factory() as other_session, other_session.begin():
                    other_conversation = ConversationRepository(other_session).get(
                        conversation_id
                    )
                    assert other_conversation is not None
                    ConversationRepository(other_session).delete(other_conversation)
                return ChatGenerationResult(response="Must not persist")

            with pytest.raises(ConversationNotFoundError):
                ChatService(session, generator=deleting_generator).chat(
                    conversation_id=conversation_id,
                    message="Race with deletion",
                )

        with database.session_factory() as verification_session:
            assert ConversationRepository(verification_session).get(conversation_id) is None
            assert MessageRepository(verification_session).list_for_conversation(
                conversation_id
            ) == []
    finally:
        database.engine.dispose()


def test_repository_rejects_invalid_roles_before_persistence(tmp_path: Path) -> None:
    application = create_app(_database_url(tmp_path / "roles.sqlite3"))

    with TestClient(application):
        with application.state.database.session_factory() as session:
            with session.begin():
                conversation = Conversation(title="Role validation")
                session.add(conversation)
                session.flush()
                with pytest.raises(ValueError, match="Unsupported message role"):
                    MessageRepository(session).create(
                        conversation,
                        role="hacker",
                        content="Invalid",
                    )

        assert _counts(application) == (1, 0, 0)


def test_backend_has_no_model_runtime_imports() -> None:
    forbidden_roots = {
        "transformers",
        "torch",
        "unsloth",
        "peft",
        "bitsandbytes",
        "evaluation",
        "runtime",
    }
    imported_roots: set[str] = set()

    for path in Path("backend").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(forbidden_roots)
