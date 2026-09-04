"""CPU-only checks for explicit memory UX and commit-before-acknowledgment."""

import json
from threading import Event

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

from backend.app import create_app
from backend.chat_service import ChatGenerationResult, ChatService
from backend.database import Database
from backend.memory import MemoryConflictError, MemoryService, parse_memory_command
from backend.models import MemoryRevision, MemorySlot, Message
from tests.app_factory import create_test_app
from tests.test_remote_privacy import RemoteHarness, assert_success, chat


class NoInference:
    def generate_response(self, *_args, **_kwargs):
        pytest.fail("Explicit memory commands must not invoke a model")

    stream_response = generate_response


@pytest.mark.parametrize("text,operation,category,key,value", [
    ("Remember that my favourite color is black.", "remember", "preference", "favourite_color", "black"),
    ("Remember my dog's name is Bruno.", "remember", "profile", "dog_name", "Bruno"),
    ("Remember my dog’s name is Bruno.", "remember", "profile", "dog_name", "Bruno"),
    ("Forget my favourite color.", "forget", "preference", "favourite_color", None),
    ("Actually, update my favourite color to blue.", "update", "preference", "favourite_color", "blue"),
    ("Remember my favorite colour is black and white.", "remember", "preference", "favourite_color", "black and white"),
    ("Remember my name is José.", "remember", "profile", "name", "José"),
    ("Remember preference favourite_color: black", "remember", "preference", "favourite_color", "black"),
    ("Actually, update preference favourite_color: blue", "update", "preference", "favourite_color", "blue"),
    ("Forget preference favourite_color", "forget", "preference", "favourite_color", None),
])
def test_explicit_grammar(text, operation, category, key, value):
    decision = parse_memory_command(text)
    assert decision.intent_detected and decision.command is not None
    command = decision.command
    assert (command.operation, command.category, command.key, command.value) == (operation, category, key, value)


@pytest.mark.parametrize("text", [
    "My favourite color is black.", "My dog's name is Bruno.",
    "Actually, I like blue now.", "I remember my favourite color is black.",
    "Don't remember my favourite color is black.",
    'Translate "Remember my favourite color is black."',
    '"Forget my favourite color."', "If I ask later, forget my favourite color.",
])
def test_ordinary_statements_are_not_commands(text):
    assert not parse_memory_command(text).intent_detected


@pytest.mark.parametrize("text", [
    "Remember that.", "Remember my favourite color is", "Update my favourite color.",
    "Remember my favourite color is black. Forget my dog's name.",
    "Remember my favourite color is black and my dog's name is Bruno.",
    "Remember my favourite color is black if I confirm later.",
    "Remember my favourite color is black\nForget my dog's name.",
    "Forget my favourite color and my dog's name.",
    "Remember my password is CREDENTIAL_CANARY.",
    "Remember my API key is CREDENTIAL_CANARY.",
    "Remember my favourite color is password: CREDENTIAL_CANARY.",
])
def test_ambiguous_and_sensitive_commands_cannot_mutate_or_claim_success(text):
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=NoInference())) as client:
        for streaming in (False, True):
            response, events = chat(client, text, streaming)
            result = assert_success(response, events, streaming)
            assert "No memory was changed" in result["response"]
            assert "CREDENTIAL_CANARY" not in result["response"]
            assert result["metadata"]["memory"] == []
        assert client.get("/api/memory").json() == []


@pytest.mark.parametrize("streaming", [False, True])
def test_natural_create_update_forget_and_missing_target(streaming):
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=NoInference())) as client:
        def send(text):
            response, events = chat(client, text, streaming)
            result = assert_success(response, events, streaming)
            assert result["metadata"]["model"] == "local-memory"
            assert result["metadata"]["tools"] == []
            return result

        first = send("Remember that my favourite color is black.")
        assert "Memory saved" in first["response"] and "local only" in first["response"]
        record = client.get("/api/memory").json()[0]
        assert (record["key"], record["value"], record["sensitivity"]) == ("favourite_color", "black", "local_only")
        assert record["source"]["conversation_id"] == first["conversation_id"]
        send("Remember my dog's name is Bruno.")
        updated = send("Actually, update my favorite colour to blue.")
        assert updated["metadata"]["memory"][0]["id"] == record["id"]
        records = {item["key"]: item for item in client.get("/api/memory").json()}
        assert records["favourite_color"]["value"] == "blue"
        assert records["favourite_color"]["sensitivity"] == "local_only"
        assert records["dog_name"]["value"] == "Bruno"
        assert "Forgotten" in send("Forget my favourite color.")["response"]
        for text in ("Forget my favourite color.", "Update my favourite color to red."):
            assert "No memory was changed" in send(text)["response"]
        assert [item["key"] for item in client.get("/api/memory").json()] == ["dog_name"]
        assert client.get("/api/memory", params={"status": "deleted"}).json()[0]["value"] is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["apply", "conflict", "commit"])
def test_failed_mutation_never_exposes_success_or_partial_history(tmp_path, monkeypatch, streaming, failure):
    database = Database.from_url(f"sqlite+pysqlite:///{(tmp_path / 'failure.db').as_posix()}", encrypted=False)
    database.create_schema()
    try:
        with database.session_factory() as session:
            service = ChatService(session, generator=NoInference())
            def fail(*_args, **_kwargs):
                raise MemoryConflictError("test conflict") if failure == "conflict" else RuntimeError("test failure")

            if failure == "commit":
                # The preparation read also commits; fail only after pending writes were flushed.
                @event.listens_for(session, "before_commit")
                def fail_write_commit(target):
                    if target.scalar(select(func.count()).select_from(Message)):
                        fail()
            else:
                monkeypatch.setattr(service.memory, "apply", fail)
            kwargs = {"conversation_id": None, "message": "Remember my favourite color is black."}
            if streaming:
                stream = service.stream_chat(**kwargs)
                assert next(stream).event == "start"
                with pytest.raises((RuntimeError, MemoryConflictError)):
                    next(stream)  # No text/final acknowledgment can precede a failed commit.
            else:
                with pytest.raises((RuntimeError, MemoryConflictError)):
                    service.chat(**kwargs)
            assert not session.in_transaction()
        with database.session_factory() as reader:
            assert reader.scalar(select(func.count()).select_from(Message)) == 0
            assert reader.scalar(select(func.count()).select_from(MemorySlot)) == 0
            assert reader.scalar(select(func.count()).select_from(MemoryRevision)) == 0
    finally:
        database.engine.dispose()


def test_stream_success_is_already_durable_and_stop_before_commit_is_empty(tmp_path):
    database = Database.from_url(f"sqlite+pysqlite:///{(tmp_path / 'stream.db').as_posix()}", encrypted=False)
    database.create_schema()
    try:
        with database.session_factory() as session:
            cancel = Event()
            stream = ChatService(session, generator=NoInference()).stream_chat(
                conversation_id=None, message="Remember my favourite color is black.", cancel_event=cancel,
            )
            assert next(stream).event == "start"
            cancel.set()
            assert list(stream) == []
            with database.session_factory() as reader:
                assert reader.scalar(select(func.count()).select_from(MemorySlot)) == 0
            stream = ChatService(session, generator=NoInference()).stream_chat(
                conversation_id=None, message="Remember my favourite color is black.",
            )
            assert next(stream).event == "start"
            assert next(stream).event == "text"
            with database.session_factory() as reader:
                assert MemoryService(reader).list_memories()[0]["value"] == "black"
                assert reader.scalar(select(func.count()).select_from(Message)) == 2
            stream.close()  # Disconnecting after success cannot undo the already committed write.
    finally:
        database.engine.dispose()


def test_encrypted_memory_survives_restart_and_is_retrieved_in_new_chat(tmp_path):
    path = tmp_path / "encrypted.db"
    kwargs = {"database_url": f"sqlite+pysqlite:///{path.as_posix()}", "encrypted_storage": True,
              "database_key": "31" * 32, "enforce_local_auth": False}
    with TestClient(create_app(**kwargs, generator=NoInference())) as client:
        first = client.post("/api/chat", json={"message": "Remember my dog's name is Bruno."}).json()
    calls = []
    def recall(messages):
        calls.append(messages)
        return ChatGenerationResult(response="Your dog's name is Bruno.")
    with TestClient(create_app(**kwargs, generator=recall)) as client:
        second = client.post("/api/chat", json={"message": "What is my dog's name?"}).json()
        assert second["conversation_id"] != first["conversation_id"]
        assert second["metadata"]["memory"][0]["operation"] == "retrieved"
        assert '"value":"Bruno"' in calls[0][0].content
    assert not path.read_bytes().startswith(b"SQLite format 3")
    assert b"Bruno" not in path.read_bytes()


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_opt_in_and_revocation_are_explicit_and_history_stays_projected(streaming):
    harness = RemoteHarness()
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)) as client:
        response, events = chat(client, "Remember my favourite color is BLACK_VALUE_CANARY.", streaming)
        first = assert_success(response, events, streaming)
        assert harness.calls == []
        record = client.get("/api/memory").json()[0]
        for conversation_id in (None, first["conversation_id"]):
            assert_success(*chat(client, "What is my favourite color?", streaming, conversation_id), streaming)
            assert "BLACK_VALUE_CANARY" not in json.dumps(harness.calls[-1])
        route = f"/api/memory/{record['id']}"
        assert client.patch(route, json={"sensitivity": "remote_allowed"}).status_code == 200
        response, events = chat(client, "Actually, update my favourite color to BLUE_VALUE_CANARY.", streaming)
        updated = assert_success(response, events, streaming)
        assert "allowed remote Aevon" in updated["response"]
        assert client.get("/api/memory").json()[0]["sensitivity"] == "remote_allowed"
        assert_success(*chat(client, "What is my favourite color?", streaming), streaming)
        assert "BLUE_VALUE_CANARY" in json.dumps(harness.calls[-1])
        assert "BLACK_VALUE_CANARY" not in json.dumps(harness.calls[-1])
        assert client.patch(route, json={"sensitivity": "local_only"}).status_code == 200
        assert_success(*chat(client, "What is my favourite color?", streaming, updated["conversation_id"]), streaming)
        assert "BLUE_VALUE_CANARY" not in json.dumps(harness.calls[-1])
    harness.provider.close()
