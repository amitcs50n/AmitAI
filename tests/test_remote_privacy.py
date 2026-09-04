import json
import logging
from collections import deque
from dataclasses import replace
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.chat_service import (
    ChatGenerationResult,
    ChatPrivacyError,
    ChatService,
    GenerationMessage,
    RemoteProjection,
)
from backend.memory import (
    MAX_MEMORY_CONTEXT_CHARS,
    MAX_RETRIEVED_MEMORIES,
    MemoryService,
    MemoryValidationError,
    validate_memory_value,
)
from backend.models import Conversation, MemoryRevision, MemorySlot, Message, MessageMetadata
from backend.secret_detection import contains_credential_like_text
from evaluation.hf_backend import GenerationOutput
from runtime.config import DEFAULT_RUNTIME_CONFIG_PATH, EXPECTED_MODEL_NAME, load_runtime_config
from runtime.context import MAX_HISTORY_CONTEXT_CHARS, MAX_HISTORY_MESSAGES, compile_model_messages
from runtime.generator import ProviderChatGenerator
from runtime.privacy import InferenceExecutionScope, RemoteDisclosureBlockedError
from runtime.providers import LocalTransformersInferenceProvider, RemoteInferenceProvider
from tests.app_factory import create_test_app
from tests.test_secret_detection import BENIGN_TEXT

REMOTE_TOKEN = 'REMOTE_TRANSPORT_CANARY_92731_"quoted\\token'
PRIVATE = "LOCAL_ONLY_VALUE_CANARY_981237"
ALLOWED = "REMOTE_ALLOWED_VALUE_CANARY_192837"
IRRELEVANT = "IRRELEVANT_PRIVATE_CANARY_982736"
COMMAND_KEY = "private.command.target"
COMMAND_VALUE = "MEMORY_COMMAND_VALUE_CANARY_563412"
ERROR = "Remote inference blocked by local privacy policy"
CONFIG = load_runtime_config(DEFAULT_RUNTIME_CONFIG_PATH)


class RemoteHarness:
    def __init__(self, outputs=("A safe answer",), *, config=CONFIG,
                 resolver=lambda _hostname, _port: ["8.8.8.8"]):
        self.outputs = deque(outputs)
        self.calls: list[dict] = []
        self.paths: list[str] = []
        self.bodies: list[bytes] = []
        self.provider = RemoteInferenceProvider(
            "https://inference.invalid", REMOTE_TOKEN, EXPECTED_MODEL_NAME,
            allowed_origins=["https://inference.invalid"], resolver=resolver,
            transport=httpx.MockTransport(self.handle),
        )
        self.generator = ProviderChatGenerator(config, provider=self.provider)

    def handle(self, request):
        assert request.headers["Authorization"] == f"Bearer {REMOTE_TOKEN}"
        assert request.headers["Content-Type"] == "application/json"
        payload = json.loads(request.content)
        self.calls.append(payload)
        self.bodies.append(request.content)
        self.paths.append(request.url.path)
        answer = self.outputs.popleft() if self.outputs else "A safe answer"
        final = {
            "request_id": payload["request_id"], "model": EXPECTED_MODEL_NAME,
            "text": answer, "input_tokens": 11, "output_tokens": 3,
        }
        if request.url.path.endswith("/stream"):
            middle = max(1, len(answer) // 2)
            events = [f"event: delta\ndata: {json.dumps({'delta': chunk})}\n\n"
                      for chunk in (answer[:middle], answer[middle:]) if chunk]
            events.append(f"event: final\ndata: {json.dumps(final)}\n\n")
            return httpx.Response(200, text="".join(events), headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=final)


def memory(client, key, value, *, sensitivity="local_only", category="project"):
    response = client.post("/api/memory", json={
        "category": category, "key": key, "value": value, "sensitivity": sensitivity,
    })
    assert response.status_code == 201
    return response.json()


def chat(client, prompt, streaming, conversation_id=None):
    response = client.post("/api/chat/stream" if streaming else "/api/chat", json={
        "message": prompt, "conversation_id": conversation_id,
    })
    events = []
    if streaming:
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return response, events


def assert_success(response, events, streaming):
    assert response.status_code == 200
    if not streaming:
        return response.json()
    assert events[0][0] == "start"
    assert events[-1][0] == "done"
    result = next(data for name, data in events if name == "final")
    assert "".join(data["delta"] for name, data in events if name == "text") == result["response"]
    return result


def assert_blocked(response, events, streaming):
    if streaming:
        assert response.status_code == 200
        assert [name for name, _ in events] == ["start", "error"]
        assert events[-1][1] == {"detail": ERROR}
    else:
        assert response.status_code == 422
        assert response.json() == {"detail": ERROR}


def durable_snapshot(app):
    with app.state.database.engine.connect() as connection:
        return {table.name: connection.execute(select(table)).all()
                for table in (Conversation.__table__, Message.__table__, MessageMetadata.__table__)}


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_discloses_only_relevant_allowed_memory_and_reference_metadata(streaming) -> None:
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        private = memory(client, "project.private", PRIVATE)
        allowed = memory(client, "project.allowed", ALLOWED, sensitivity="remote_allowed")
        memory(client, "unrelated.cooking", IRRELEVANT, category="workflow", sensitivity="remote_allowed")
        response, events = chat(client, "Tell me about my project", streaming)
        result = assert_success(response, events, streaming)
        body = json.dumps(harness.calls)
        assert ALLOWED in body
        assert PRIVATE not in body and IRRELEVANT not in body
        assert private["key"] not in body
        for forbidden in (private["id"], allowed["id"], allowed["updated_at"],
                          "remote_allowed", "local_only", "updated_at", "source_conversation_id"):
            assert forbidden not in body
        context = next(m["content"] for m in harness.calls[0]["messages"] if "MEMORY_CONTEXT_V1" in m["content"])
        items = json.loads(context.split("<memory_context>")[1].split("</memory_context>")[0])["items"]
        assert items == [{"category": "project", "key": "project.allowed", "value": ALLOWED}]
        assert all("value" not in ref and "sensitivity" not in ref for ref in result["metadata"]["memory"])
        stored = client.get(f"/api/conversations/{result['conversation_id']}").json()
        assert PRIVATE not in json.dumps(stored) and ALLOWED not in json.dumps(stored)
        assert {m["value"] for m in client.get("/api/memory").json()} == {PRIVATE, ALLOWED, IRRELEVANT}
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_all_local_memory_context_is_omitted_without_placeholder(streaming) -> None:
    harness = RemoteHarness()
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)) as client:
        record = memory(client, "project.private", PRIVATE)
        response, events = chat(client, "Tell me about my project", streaming)
        assert_success(response, events, streaming)
        body = json.dumps(harness.calls)
        assert PRIVATE not in body and record["key"] not in body
        assert "MEMORY_CONTEXT_V1" not in body
        assert len(harness.calls[0]["messages"]) == 2  # system + current, no count placeholder
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_local_transformers_retains_both_memory_policies_and_commands_skip_inference(streaming) -> None:
    calls = []

    class Engine:
        def generate_detailed(self, messages, generation_config):
            calls.append(messages)
            return GenerationOutput("A safe answer", 11, 3)

    provider = LocalTransformersInferenceProvider(CONFIG.model, 42, engine_factory=lambda *_: Engine())
    generator = ProviderChatGenerator(CONFIG, provider=provider)
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=generator)) as client:
        memory(client, "project.private", PRIVATE)
        memory(client, "project.allowed", ALLOWED, sensitivity="remote_allowed")
        response, events = chat(client, "Tell me about my project", streaming)
        assert_success(response, events, streaming)
        assert PRIVATE in json.dumps(calls[0]) and ALLOWED in json.dumps(calls[0])
        command = f"Remember project {COMMAND_KEY}: {COMMAND_VALUE}"
        response, events = chat(client, command, streaming)
        assert_success(response, events, streaming)
        assert len(calls) == 1
        response, events = chat(client, "My password: LOCAL_CREDENTIAL_CANARY", streaming)
        assert_success(response, events, streaming)
        assert "LOCAL_CREDENTIAL_CANARY" in json.dumps(calls[-1])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("command", [
    f"Remember project {COMMAND_KEY}: {COMMAND_VALUE}",
    f"Update project {COMMAND_KEY}: {COMMAND_VALUE}",
    f"Forget project {COMMAND_KEY}",
    f"Remember project {COMMAND_KEY}: password: REJECTED_CREDENTIAL_CANARY",
    f"Remember ambiguous {COMMAND_VALUE}",
])
def test_current_and_historical_memory_commands_and_acks_are_projected(streaming, command) -> None:
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        memory(client, COMMAND_KEY, "PRIOR_TARGET_VALUE_CANARY", sensitivity="remote_allowed")
        response, events = chat(client, command, streaming)
        result = assert_success(response, events, streaming)
        assert harness.calls == []
        body = json.dumps(harness.calls)
        for private in (COMMAND_KEY, COMMAND_VALUE, "PRIOR_TARGET_VALUE_CANARY", "REJECTED_CREDENTIAL_CANARY"):
            assert private not in body
        assert "MEMORY_COMMAND_V1" not in body
        conversation_id = result["conversation_id"]
        # Legacy acknowledgments may echo every command field. Only the generation
        # projection changes; the full ordinary conversation API stays untouched.
        with app.state.database.session_factory() as session, session.begin():
            assistant = session.get(Message, result["message_id"])
            assistant.content = f"Acknowledged {command}"
        before = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        response, events = chat(client, "Continue this conversation", streaming, conversation_id)
        assert_success(response, events, streaming)
        later_body = json.dumps(harness.calls[-1])
        assert COMMAND_KEY not in later_body and COMMAND_VALUE not in later_body
        assert "REJECTED_CREDENTIAL_CANARY" not in later_body
        after = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert after[:len(before)] == before
        assert before[0]["content"] == command
        assert before[1]["content"] == f"Acknowledged {command}"
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_command_retry_and_calculator_followup_cannot_restore_private_context(streaming) -> None:
    tool = '<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'
    harness = RemoteHarness([tool, "The memory update noted", "Memory update noted"])
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        memory(client, "project.private", PRIVATE)
        # A long legacy command/ack exceeds raw-history budget; its safe projection
        # must happen first and remain safe on every tool and repair invocation.
        app.state.generator = lambda _: ChatGenerationResult(response="ACK_PRIVATE_CANARY " + "x" * 22000)
        seeded = client.post("/api/chat", json={"message": "Remember ambiguous " + "y" * 21000}).json()
        with app.state.database.session_factory() as session, session.begin():
            session.get(Message, seeded["message_id"]).content = "ACK_PRIVATE_CANARY " + "x" * 22000
        app.state.generator = harness.generator
        command = f"Remember project {COMMAND_KEY}: {COMMAND_VALUE}"
        response, events = chat(client, command, streaming, seeded["conversation_id"])
        assert_success(response, events, streaming)
        assert harness.calls == []
        response, events = chat(client, "Calculate 17*83. Answer in exactly 3 words.",
                                streaming, seeded["conversation_id"])
        result = assert_success(response, events, streaming)
        assert result["response"] == "Memory update noted"
        assert result["metadata"]["validator"]["retry_count"] == 1
        assert result["metadata"]["validator"]["final_validation"]["passed"] is True
        assert result["metadata"]["tools"][0]["success"] is True
        assert len(harness.calls) == 3
        for call in harness.calls:
            body = json.dumps(call)
            assert PRIVATE not in body and COMMAND_KEY not in body and COMMAND_VALUE not in body
            assert "ACK_PRIVATE_CANARY" not in body
            assert "local memory command was requested" in body
        assert harness.calls[1]["messages"][:-2] == harness.calls[0]["messages"]
        assert "1411" in harness.calls[1]["messages"][-1]["content"]
        assert "Original user request:" in harness.calls[-1]["messages"][-1]["content"]
        assert {m["value"] for m in client.get("/api/memory").json()} >= {PRIVATE}
        if streaming:
            assert [data["delta"] for name, data in events if name == "text"] == ["Memory update noted"]
    harness.provider.close()


CREDENTIALS = [
    "OPENAI_API_KEY=REMOTE_ENV_CANARY_1122", "MY_PASSWORD=PASSWORD_ENV_CANARY_3344",
    "GITHUB_ACCESS_TOKEN=GITHUB_ENV_CANARY_5566",
    '{"api_key":"QUOTED_JSON_SECRET_CANARY"}',
    "password: PASSWORD_CANARY_73612", "passcode=PASSCODE_CANARY_98217",
    "api_key: API_KEY_CANARY_87612", "access token=ACCESS_CANARY_67512",
    "refresh-token: REFRESH_CANARY_85211", "auth token: AUTH_CANARY_91627",
    "client_secret=CLIENT_SECRET_CANARY_55321", "-----BEGIN RSA PRIVATE KEY-----\nPRIVATE_KEY_CANARY",
    "eyJabcdefghi.abcdefghijk.abcdefghijk", "Authorization: Bearer BEARER_CANARY_94831",
]


@pytest.mark.parametrize("credential", CREDENTIALS)
@pytest.mark.parametrize("streaming", [False, True])
def test_current_credentials_block_before_http_with_sanitized_error_and_no_persistence(credential, streaming, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert contains_credential_like_text(credential)
    with pytest.raises(MemoryValidationError):
        validate_memory_value(credential)
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        before = durable_snapshot(app)
        response, events = chat(client, credential, streaming)
        assert_blocked(response, events, streaming)
        assert harness.calls == []
        assert durable_snapshot(app) == before
        assert credential not in response.text and credential not in caplog.text
        assert "CANARY" not in response.text and "CANARY" not in caplog.text
        assert REMOTE_TOKEN not in caplog.text
    harness.provider.close()


def test_eager_provider_stream_privacy_failure_keeps_sanitized_error_type() -> None:
    class EagerProvider:
        execution_scope = InferenceExecutionScope.REMOTE
        model_name = EXPECTED_MODEL_NAME
        provider_name = "eager-test"

        def stream(self, *args, **kwargs):
            raise RemoteDisclosureBlockedError()

    generator = ProviderChatGenerator(CONFIG, provider=EagerProvider())
    with pytest.raises(ChatPrivacyError, match=f"^{ERROR}$"):
        list(generator.stream_response([GenerationMessage("user", "Hello")], cancel_event=Event()))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("source", ["history", "legacy_memory", "staged_mutation"])
def test_retained_credentials_block_and_existing_state_remains_unchanged(streaming, source) -> None:
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:")
    with TestClient(app) as client:
        seed_prompt = "api_key: HISTORY_CREDENTIAL_CANARY" if source != "legacy_memory" else "Hello"
        seed = client.post("/api/chat", json={"message": seed_prompt}).json()
        record = memory(client, "project.legacy", "Safe initial value", sensitivity="remote_allowed")
        if source == "legacy_memory":
            with app.state.database.session_factory() as session, session.begin():
                revision = session.scalar(select(MemoryRevision).where(MemoryRevision.memory_id == record["id"]))
                revision.value = "password: LEGACY_BAD_MEMORY_CANARY"
        app.state.generator = harness.generator
        before_chat = durable_snapshot(app)
        before_memory = client.get("/api/memory").json()
        prompt = (f"Remember project {COMMAND_KEY}: {COMMAND_VALUE}" if source == "staged_mutation"
                  else "Tell me about my project")
        response, events = chat(client, prompt, streaming, seed["conversation_id"])
        if source == "staged_mutation":
            # Local commands need no disclosure: existing private history stays local.
            assert_success(response, events, streaming)
            assert harness.calls == []
            assert any(item["key"] == COMMAND_KEY for item in client.get("/api/memory").json())
            before_chat = durable_snapshot(app)
            before_memory = client.get("/api/memory").json()
            response, events = chat(client, "Tell me about my project", streaming, seed["conversation_id"])
        assert_blocked(response, events, streaming)
        assert harness.calls == []
        assert durable_snapshot(app) == before_chat
        assert client.get("/api/memory").json() == before_memory
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("location", ["message", "config_value", "config_key", "nested_config", "labeled_config", "numeric_passcode"])
def test_last_mile_scans_full_body_and_exact_token_including_json_escaping(streaming, location) -> None:
    harness = RemoteHarness()
    config = {"max_new_tokens": 8}
    content = "ordinary request"
    if location == "message":
        content += REMOTE_TOKEN
    elif location == "config_value":
        config["extra"] = REMOTE_TOKEN
    elif location == "config_key":
        config[REMOTE_TOKEN] = "unused"
    elif location == "nested_config":
        config["extra"] = [{"nested": REMOTE_TOKEN}]
    elif location == "numeric_passcode":
        config["passcode"] = 735291
    else:
        config["password"] = "CONFIG_CREDENTIAL_CANARY"
    with pytest.raises(RemoteDisclosureBlockedError, match=f"^{ERROR}$") as caught:
        if streaming:
            list(harness.provider.stream([{"role": "user", "content": content}], config, cancel_event=Event()))
        else:
            harness.provider.generate([{"role": "user", "content": content}], config)
    assert caught.value.__cause__ is None
    assert harness.calls == []
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("later_request", ["retry", "tool"])
@pytest.mark.parametrize("prefixed", [False, True])
def test_every_later_invocation_is_guarded_and_rolls_back(streaming, later_request, prefixed) -> None:
    bad_output = ("password: RETRY_CREDENTIAL_CANARY four words" if later_request == "retry" else
                  '<tool_call>{"name":"calculator","arguments":{"expression":"password: TOOL_CREDENTIAL_CANARY"}}</tool_call>')
    if prefixed:
        bad_output = bad_output.replace("password:", "OPENAI_API_KEY=")
    harness = RemoteHarness([bad_output])
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        before = durable_snapshot(app)
        prompt = "Calculate 17*83. Answer in exactly 3 words."
        response, events = chat(client, prompt, streaming)
        assert_blocked(response, events, streaming)
        assert len(harness.calls) == 1  # Only the earlier, safe request crossed the boundary.
        assert durable_snapshot(app) == before
        assert client.get("/api/memory").json() == []
    harness.provider.close()


@pytest.mark.parametrize("text", [
    "Explain API key rotation", "How does password hashing work?", "What is Authorization?",
    "Explain JWT verification", "My name is Alice", "Explain public key cryptography",
])
def test_benign_discussions_and_ordinary_identity_text_remain_allowed(text) -> None:
    assert not contains_credential_like_text(text)
    assert validate_memory_value(text) == text
    harness = RemoteHarness()
    harness.generator.generate_response([GenerationMessage("user", text)])
    assert harness.calls[0]["messages"][-1]["content"] == text
    harness.provider.close()


@pytest.mark.parametrize("scope", [None, "local", "remote", "unknown", object()])
def test_unknown_provider_scope_fails_closed_without_name_based_trust(scope) -> None:
    class Provider:
        provider_name = "local-transformers"
        execution_scope = scope
    with pytest.raises(TypeError, match="valid execution scope"):
        ProviderChatGenerator(CONFIG, provider=Provider())
    with pytest.raises(TypeError, match="valid execution scope"):
        ProviderChatGenerator(CONFIG, provider=object())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("text", BENIGN_TEXT)
def test_benign_decoded_content_reaches_real_remote_transport(text, streaming):
    harness = RemoteHarness()
    messages = [{"role": "user", "content": text}]
    if streaming:
        events = list(harness.provider.stream(messages, {"max_tokens": 20}, cancel_event=Event()))
        assert "".join(event for event in events if isinstance(event, str)) == "A safe answer"
        assert events[-1].text == "A safe answer"
    else:
        assert harness.provider.generate(messages, {"max_tokens": 20}).text == "A safe answer"
    assert len(harness.calls) == 1
    assert harness.calls[0]["messages"] == messages
    if text == "password:":
        assert b'"content":"password:"' in harness.bodies[0]
    assert harness.paths == ["/v1/generate/stream" if streaming else "/v1/generate"]
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("content,config", [
    ('{"api_key":"QUOTED_JSON_SECRET_CANARY"}', {}),
    ('{"OPENAI_API_KEY":"QUOTED_ENV_SECRET_CANARY"}', {}),
    (('MEMORY_CONTEXT_V1 <memory_context>{"items":[{"key":"openai_api_key",'
      '"value":"STRUCTURED_PAIR_CANARY"}]}</memory_context>'), {}),
    ("Safe prompt", {"OPENAI_API_KEY": "GENERATION_CONFIG_SECRET_CANARY"}),
    ("Safe prompt", {"nested": [{"OPENAI_API_KEY": "NESTED_CONFIG_CANARY"}]}),
    ("Safe prompt", {"nested": [{"key": "my_password", "value": "NESTED_PAIR_CANARY"}]}),
    ("Safe prompt", {"nested": {"OPENAI_API_KEY=KEY_CANARY": None}}),
])
def test_semantic_credentials_block_low_level_provider_before_http(content, config, streaming, caplog):
    caplog.set_level(logging.INFO)
    harness = RemoteHarness()
    with pytest.raises(RemoteDisclosureBlockedError, match=f"^{ERROR}$") as caught:
        if streaming:
            list(harness.provider.stream([{"role": "user", "content": content}], config,
                                         cancel_event=Event()))
        else:
            harness.provider.generate([{"role": "user", "content": content}], config)
    assert harness.calls == []
    assert "CANARY" not in str(caught.value) and "CANARY" not in caplog.text
    assert REMOTE_TOKEN not in caplog.text
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_seeded_remote_allowed_credential_pair_is_independently_blocked(streaming, existing, caplog):
    caplog.set_level(logging.INFO)
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:")
    with TestClient(app) as client:
        conversation_id = None
        if existing:
            conversation_id = client.post("/api/chat", json={"message": "Hello"}).json()["conversation_id"]
        with app.state.database.session_factory() as session, session.begin():
            slot = MemorySlot(owner_id="local-default", category="project", key="openai_api_key",
                              sensitivity="remote_allowed")
            session.add(slot)
            session.flush()
            session.add(MemoryRevision(memory_id=slot.id, revision=1, status="active",
                                       value="LEGACY_REMOTE_SECRET_CANARY_7788"))
        before = durable_snapshot(app)
        with app.state.database.engine.connect() as connection:
            before_memory = [connection.execute(select(table)).all()
                             for table in (MemorySlot.__table__, MemoryRevision.__table__)]
        app.state.generator = harness.generator
        response, events = chat(client, "Tell me about my project", streaming, conversation_id)
        assert_blocked(response, events, streaming)
        assert harness.calls == []
        assert durable_snapshot(app) == before
        with app.state.database.engine.connect() as connection:
            assert [connection.execute(select(table)).all()
                    for table in (MemorySlot.__table__, MemoryRevision.__table__)] == before_memory
        assert "CANARY" not in response.text and "CANARY" not in caplog.text
        assert "openai_api_key" not in response.text and "openai_api_key" not in caplog.text
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_credential_memory_command_is_not_applied_in_either_scope(remote, streaming, caplog):
    caplog.set_level(logging.INFO)
    harness = RemoteHarness()
    local_calls = []

    class Engine:
        def generate_detailed(self, messages, generation_config):
            local_calls.append(messages)
            return GenerationOutput("Memory was not stored", 11, 4)

    local_provider = LocalTransformersInferenceProvider(CONFIG.model, 42,
                                                       engine_factory=lambda *_: Engine())
    generator = harness.generator if remote else ProviderChatGenerator(CONFIG, provider=local_provider)
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=generator)
    command = "Remember project openai_api_key: MEMORY_COMMAND_PAIR_CANARY"
    with TestClient(app) as client:
        response, events = chat(client, command, streaming)
        result = assert_success(response, events, streaming)
        assert client.get("/api/memory").json() == []
        with app.state.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(MemorySlot)) == 0
            assert session.scalar(select(func.count()).select_from(MemoryRevision)) == 0
        assert harness.calls == [] and local_calls == []
        assert "No memory was changed" in result["response"]
        assert result["metadata"]["memory"] == []
        assert "CANARY" not in response.text and "CANARY" not in caplog.text
        # Ordinary local conversation text is not rewritten by storage rejection.
        stored = client.get(f"/api/conversations/{result['conversation_id']}").json()
        assert stored["messages"][0]["content"] == command
        if not remote:
            response, events = chat(client, "OPENAI_API_KEY=LOCAL_ONLY_CANARY", streaming)
            assert_success(response, events, streaming)
            assert local_calls[-1][-1]["content"] == "OPENAI_API_KEY=LOCAL_ONLY_CANARY"
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("token", ['TRANSPORT_TEST_CANARY_quote"and\\slash',
                                   'TRANSPORT_TEST_CANARY_escaped\\backslash'])
def test_escaped_transport_token_cannot_reach_http(token, streaming, caplog):
    caplog.set_level(logging.INFO)
    calls = []

    def unexpected_http(request):
        calls.append(request)
        raise AssertionError("Guard must run before transport")

    provider = RemoteInferenceProvider("https://inference.invalid", token, EXPECTED_MODEL_NAME,
                                       allowed_origins=["https://inference.invalid"],
                                       resolver=lambda _host, _port: ["8.8.8.8"],
                                       transport=httpx.MockTransport(unexpected_http))
    messages = [{"role": "user", "content": json.dumps({"note": token}, ensure_ascii=True)}]
    with pytest.raises(RemoteDisclosureBlockedError, match=f"^{ERROR}$"):
        if streaming:
            list(provider.stream(messages, {}, cancel_event=Event()))
        else:
            provider.generate(messages, {})
    assert calls == [] and token not in caplog.text
    provider.close()


def test_remote_projection_precedes_history_budget_and_preserves_orphan_rule() -> None:
    messages = [
        GenerationMessage("user", "DROP_ME", RemoteProjection(None)),
        GenerationMessage("assistant", "orphan"),
        GenerationMessage("user", "RAW_COMMAND_CANARY" * 2000, RemoteProjection("safe command")),
        GenerationMessage("assistant", "RAW_ACK_CANARY" * 2000, RemoteProjection("safe acknowledgment")),
        GenerationMessage("user", "CURRENT_REQUEST"),
    ]
    compiled = compile_model_messages(messages, runtime_system_prompt="rules", tool_instructions="tools",
                                      execution_scope=InferenceExecutionScope.REMOTE)
    assert [m["content"] for m in compiled[1:]] == ["safe command", "safe acknowledgment", "CURRENT_REQUEST"]
    assert MAX_HISTORY_MESSAGES == 20 and MAX_HISTORY_CONTEXT_CHARS == 20_000
    assert MAX_RETRIEVED_MEMORIES == 8 and MAX_MEMORY_CONTEXT_CHARS == 4_000


def test_sync_and_stream_receive_equivalent_context_and_no_environment_secrets(monkeypatch) -> None:
    monkeypatch.setenv("AMITAI_DB_KEY", "12" * 32)
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", "LOCAL_AUTH_CANARY_998127")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", REMOTE_TOKEN)
    harness = RemoteHarness()
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)) as client:
        memory(client, "project.private", PRIVATE)
        memory(client, "project.allowed", ALLOWED, sensitivity="remote_allowed")
        for streaming in (False, True):
            response, events = chat(client, "Tell me about my project", streaming)
            assert_success(response, events, streaming)
    assert harness.calls[0]["messages"] == harness.calls[1]["messages"]
    assert harness.paths == ["/v1/generate", "/v1/generate/stream"]
    for body in harness.calls:
        encoded = json.dumps(body)
        for secret in ("12" * 32, "LOCAL_AUTH_CANARY_998127", REMOTE_TOKEN):
            assert secret not in encoded
    harness.provider.close()


def test_direct_service_privacy_failure_has_no_transaction_or_raw_exception() -> None:
    harness = RemoteHarness(config=replace(CONFIG, generation={"label": "password: CONFIG_CANARY"}))
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app), app.state.database.session_factory() as session:
        with pytest.raises(ChatPrivacyError, match=f"^{ERROR}$") as caught:
            ChatService(session, generator=harness.generator).chat(
                conversation_id=None, message="Tell me about my project",
            )
        assert not session.in_transaction()
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        assert session.scalar(select(func.count()).select_from(Message)) == 0
        assert MemoryService(session).list_memories() == []
    assert harness.calls == []
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_old_history_is_dropped_before_guard_without_changing_durable_history(streaming) -> None:
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=lambda _: ChatGenerationResult(response="z" * 1100))
    with TestClient(app) as client:
        seed = client.post("/api/chat", json={"message": "password: OLD_CREDENTIAL_CANARY_82817"}).json()
        conversation_id = seed["conversation_id"]
        for index in range(13):
            message = f"history {index} " + "q" * 1100
            if index == 12:
                message += " RECENT_RETAINED_CANARY_73916"
            client.post("/api/chat", json={"message": message, "conversation_id": conversation_id})
        before = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert len(before) > MAX_HISTORY_MESSAGES
        assert sum(len(m["content"]) for m in before) > MAX_HISTORY_CONTEXT_CHARS
        app.state.generator = harness.generator
        response, events = chat(client, "CURRENT_USER_CANARY_73291", streaming, conversation_id)
        assert_success(response, events, streaming)
        compiled = harness.calls[0]["messages"]
        body = json.dumps(compiled)
        assert "OLD_CREDENTIAL_CANARY_82817" not in body
        assert "RECENT_RETAINED_CANARY_73916" in body
        assert compiled[-1]["content"] == "CURRENT_USER_CANARY_73291"
        assert len(compiled[1:-1]) <= 20
        assert sum(len(m["content"]) for m in compiled[1:-1]) <= 20000
        after = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert after[:len(before)] == before
    harness.provider.close()


@pytest.mark.parametrize("value_size", [20, 800])
def test_remote_filter_never_tops_up_retrieval_budget(value_size) -> None:
    harness = RemoteHarness()
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        memory(client, "project.allowed", ALLOWED, sensitivity="remote_allowed")
        for index in range(8):
            memory(client, f"project.local{index}", "p" * value_size)
        selected = client.post("/api/memory/search", json={"query": "project"}).json()
        assert len(selected) <= 8
        assert sum(len(json.dumps(m, ensure_ascii=False, separators=(",", ":"), sort_keys=True)) for m in selected) <= 4000
        if value_size == 20:
            assert len(selected) == 8
            assert all(m["sensitivity"] == "local_only" for m in selected)
        response, events = chat(client, "project", False)
        assert_success(response, events, False)
        context = json.dumps(harness.calls[0])
        if any(m["sensitivity"] == "remote_allowed" for m in selected):
            assert ALLOWED in context
        else:
            assert ALLOWED not in context and "MEMORY_CONTEXT_V1" not in context
        assert "project.local" not in context
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_ordinary_tool_and_retry_requests_keep_allowed_memory_but_never_restore_dropped_context(streaming, constrained) -> None:
    tool = '<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'
    outputs = [tool, "The result is 1411", "Result is 1411"] if constrained else [tool, "It is 1411"]
    harness = RemoteHarness(outputs)
    # The explicit scope, not this deliberately misleading label, determines policy.
    harness.provider.provider_name = "local-transformers"
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=lambda _: ChatGenerationResult(response="r" * 1100))
    with TestClient(app) as client:
        memory(client, "project.private", PRIVATE)
        memory(client, "project.allowed", ALLOWED, sensitivity="remote_allowed")
        seed = client.post("/api/chat", json={"message": "DROPPED_TOOL_RETRY_HISTORY_CANARY"}).json()
        conversation_id = seed["conversation_id"]
        for index in range(11):
            client.post("/api/chat", json={
                "conversation_id": conversation_id, "message": f"Recent turn {index} " + "s" * 1100,
            })
        app.state.generator = harness.generator
        prompt = "For my project, what is 17 * 83?"
        if constrained:
            prompt += " Answer in exactly 3 words."
        response, events = chat(client, prompt, streaming, conversation_id)
        result = assert_success(response, events, streaming)
        assert result["response"] == ("Result is 1411" if constrained else "It is 1411")
        assert result["metadata"]["validator"]["retry_count"] == int(constrained)
        assert result["metadata"]["tools"][0]["success"] is True
        assert len(harness.calls) == (3 if constrained else 2)
        for call in harness.calls:
            encoded = json.dumps(call)
            assert ALLOWED in encoded
            assert PRIVATE not in encoded
            assert "DROPPED_TOOL_RETRY_HISTORY_CANARY" not in encoded
        assert harness.calls[1]["messages"][:-2] == harness.calls[0]["messages"]
        assert "1411" in harness.calls[1]["messages"][-1]["content"]
        if streaming and not constrained:
            assert len([name for name, _ in events if name == "text"]) >= 2
    harness.provider.close()
