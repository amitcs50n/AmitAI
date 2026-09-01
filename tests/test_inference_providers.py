import json
import logging
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.chat_service import GenerationMessage
from evaluation.hf_backend import GenerationOutput
from runtime.config import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    EXPECTED_MODEL_NAME,
    load_runtime_config,
)
from runtime.generator import ProviderChatGenerator
from runtime.inference_app import create_inference_app
from runtime.privacy import InferenceExecutionScope
from runtime.providers import InferenceProviderError, RemoteInferenceProvider
from tests.app_factory import create_test_runtime_app as create_runtime_app


class RecordingProvider:
    execution_scope = InferenceExecutionScope.REMOTE
    provider_name = "recording"
    model_name = EXPECTED_MODEL_NAME

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.generate_calls = []
        self.stream_calls = []

    def generate(self, messages, generation_config):
        self.generate_calls.append((messages, generation_config))
        if self.failure is not None:
            raise self.failure
        return GenerationOutput("Remote answer", input_tokens=12, output_tokens=2)

    def stream(self, messages, generation_config, *, cancel_event):
        self.stream_calls.append((messages, generation_config, cancel_event))
        if self.failure is not None:
            raise self.failure
        yield "Remote"
        yield " answer"
        yield GenerationOutput("Remote answer", input_tokens=12, output_tokens=2)


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _remote_app(tmp_path: Path, provider: RecordingProvider):
    return create_runtime_app(
        _database_url(tmp_path / "local-control-plane.sqlite3"),
        mode="remote",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
        remote_endpoint="https://inference.invalid",
        remote_token="development-token",
        remote_provider_factory=lambda **_kwargs: provider,
    )


def test_remote_provider_sends_only_stateless_generation_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.update(payload)
        assert request.url.path == "/v1/generate"
        assert request.headers["Authorization"] == "Bearer service-token"
        return httpx.Response(
            200,
            json={
                "request_id": payload["request_id"],
                "model": EXPECTED_MODEL_NAME,
                "text": "Provider answer",
                "input_tokens": 7,
                "output_tokens": 2,
            },
        )

    provider = RemoteInferenceProvider(
        "https://gpu.example",
        "service-token",
        EXPECTED_MODEL_NAME,
        transport=httpx.MockTransport(handler),
    )

    output = provider.generate(
        [{"role": "user", "content": "Private generation request"}],
        {"max_new_tokens": 64, "do_sample": False},
    )

    assert output == GenerationOutput("Provider answer", 7, 2)
    assert set(captured) == {"request_id", "messages", "generation_config"}
    assert captured["messages"] == [
        {"role": "user", "content": "Private generation request"}
    ]
    serialized = json.dumps(captured)
    for forbidden in ("database_url", "conversation_id", "owner_id", "amitai.db"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://gpu.example",
        "http://127.0.0.1:9000",
        "http://localhost:9000",
        "http://[::1]:9000",
    ],
)
def test_remote_provider_accepts_https_and_loopback_http(endpoint: str) -> None:
    provider = RemoteInferenceProvider(endpoint, "development-token", EXPECTED_MODEL_NAME)
    provider.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gpu.example",
        "http://192.168.1.20:9000",
        "http://10.0.0.5:9000",
    ],
)
def test_remote_provider_rejects_non_loopback_plaintext_http(endpoint: str) -> None:
    with pytest.raises(ValueError, match="requires HTTPS"):
        RemoteInferenceProvider(endpoint, "development-token", EXPECTED_MODEL_NAME)


def test_remote_provider_streams_deltas_and_terminal_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        request_id = payload["request_id"]
        body = (
            'event: delta\ndata: {"delta":"Remote"}\n\n'
            'event: delta\ndata: {"delta":" stream"}\n\n'
            "event: final\ndata: "
            + json.dumps(
                {
                    "request_id": request_id,
                    "model": EXPECTED_MODEL_NAME,
                    "text": "Remote stream",
                    "input_tokens": 9,
                    "output_tokens": 2,
                },
                separators=(",", ":"),
            )
            + "\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = RemoteInferenceProvider(
        "https://gpu.example",
        "service-token",
        EXPECTED_MODEL_NAME,
        transport=httpx.MockTransport(handler),
    )

    items = list(
        provider.stream(
            [{"role": "user", "content": "Stream this"}],
            {"max_new_tokens": 64},
            cancel_event=threading.Event(),
        )
    )

    assert items == [
        "Remote",
        " stream",
        GenerationOutput("Remote stream", 9, 2),
    ]


def test_remote_normal_and_streaming_bodies_share_compiled_context_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_key = "ab" * 32
    local_token = "LOCAL_TOKEN_CANARY_554433"
    remote_token = "REMOTE_TOKEN_CANARY_665544"
    monkeypatch.setenv("AMITAI_DB_KEY", database_key)
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", local_token)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", remote_token)
    captured: dict[str, dict] = {}
    raw_bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw_body = request.content.decode()
        raw_bodies.append(raw_body)
        payload = json.loads(raw_body)
        captured[request.url.path] = payload
        assert request.headers["Authorization"] == f"Bearer {remote_token}"
        if request.url.path == "/v1/generate":
            return httpx.Response(
                200,
                json={
                    "request_id": payload["request_id"],
                    "model": EXPECTED_MODEL_NAME,
                    "text": "Remote answer",
                    "input_tokens": 12,
                    "output_tokens": 2,
                },
            )
        body = (
            'event: delta\ndata: {"delta":"Remote"}\n\n'
            'event: delta\ndata: {"delta":" answer"}\n\n'
            "event: final\ndata: "
            + json.dumps(
                {
                    "request_id": payload["request_id"],
                    "model": EXPECTED_MODEL_NAME,
                    "text": "Remote answer",
                    "input_tokens": 12,
                    "output_tokens": 2,
                },
                separators=(",", ":"),
            )
            + "\n\n"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = RemoteInferenceProvider(
        "https://gpu.example",
        remote_token,
        EXPECTED_MODEL_NAME,
        transport=httpx.MockTransport(handler),
    )
    generator = ProviderChatGenerator(load_runtime_config(), provider=provider)
    messages = [
        GenerationMessage(role="user", content="Earlier question"),
        GenerationMessage(role="assistant", content="Earlier answer"),
        GenerationMessage(role="user", content="Current request"),
    ]
    try:
        generator.generate_response(messages)
        list(
            generator.stream_response(
                messages,
                cancel_event=threading.Event(),
            )
        )
    finally:
        provider.close()

    normal = captured["/v1/generate"]
    streaming = captured["/v1/generate/stream"]
    assert normal["messages"] == streaming["messages"]
    assert normal["generation_config"] == streaming["generation_config"]
    assert normal["messages"][-1] == {
        "role": "user",
        "content": "Current request",
    }
    for body in raw_bodies:
        assert database_key not in body
        assert local_token not in body
        assert remote_token not in body


def test_successful_remote_inference_logs_only_operational_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "PRIVATE_PROMPT_92831"
    memory = "PRIVATE_MEMORY_55221"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": payload["request_id"],
                "model": EXPECTED_MODEL_NAME,
                "text": "Safe output",
                "input_tokens": 11,
                "output_tokens": 2,
            },
        )

    provider = RemoteInferenceProvider(
        "https://gpu.example",
        "REMOTE_API_SECRET_19281",
        EXPECTED_MODEL_NAME,
        transport=httpx.MockTransport(handler),
    )

    with caplog.at_level(logging.INFO, logger="runtime.providers"):
        provider.generate(
            [
                {"role": "system", "content": memory},
                {"role": "user", "content": prompt},
            ],
            {"max_new_tokens": 64},
        )

    assert "Inference completed" in caplog.text
    for sentinel in (prompt, memory, "REMOTE_API_SECRET_19281", "Safe output"):
        assert sentinel not in caplog.text


def test_inference_service_is_authenticated_and_has_no_application_database() -> None:
    provider = RecordingProvider()
    application = create_inference_app(
        provider=provider,
        auth_token="server-token",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
    )

    assert not hasattr(application.state, "database")
    route_paths = {route.path for route in application.routes}
    assert "/api/chat" not in route_paths
    assert "/api/memory" not in route_paths
    assert "/api/conversations" not in route_paths

    payload = {
        "request_id": "9a52c8db-221f-444c-913b-6be1f57965e0",
        "messages": [{"role": "user", "content": "Needed model input"}],
        "generation_config": {"max_new_tokens": 32},
    }
    with TestClient(application) as client:
        health = client.get("/health")
        assert client.post("/v1/generate", json=payload).status_code == 401
        response = client.post(
            "/v1/generate",
            json=payload,
            headers={"Authorization": "Bearer server-token"},
        )
        stream_response = client.post(
            "/v1/generate/stream",
            json=payload,
            headers={"Authorization": "Bearer server-token"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    assert response.json()["text"] == "Remote answer"
    assert provider.generate_calls == [
        ([{"role": "user", "content": "Needed model input"}], {"max_new_tokens": 32})
    ]
    assert stream_response.status_code == 200
    assert 'event: delta\ndata: {"delta":"Remote"}' in stream_response.text
    assert 'event: delta\ndata: {"delta":" answer"}' in stream_response.text
    assert "event: final" in stream_response.text
    assert len(provider.stream_calls) == 1


def test_remote_chat_keeps_conversation_and_memory_persistence_local(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    database_path = tmp_path / "local-control-plane.sqlite3"
    application = _remote_app(tmp_path, provider)

    with TestClient(application) as client:
        memory = client.post(
            "/api/memory",
            json={
                "category": "project",
                "key": "project.name",
                "value": "AmitAI",
                "sensitivity": "remote_allowed",
            },
        )
        response = client.post(
            "/api/chat",
            json={"message": "What is my AmitAI project name?"},
        )
        conversations = client.get("/api/conversations").json()
        memories = client.get("/api/memory").json()

    assert memory.status_code == 201
    assert response.status_code == 200
    assert database_path.exists()
    assert len(conversations) == 1
    assert memories[0]["value"] == "AmitAI"
    assert response.json()["metadata"]["memory"][0]["key"] == "project.name"
    assert "value" not in response.json()["metadata"]["memory"][0]
    assert len(provider.generate_calls) == 1
    remote_messages, remote_generation = provider.generate_calls[0]
    assert any("MEMORY_CONTEXT_V1" in item["content"] for item in remote_messages)
    assert remote_generation["max_new_tokens"] == 512
    assert not hasattr(provider, "database")
    serialized_request = json.dumps(provider.generate_calls)
    assert str(database_path) not in serialized_request


def test_remote_provider_gets_minimized_history_while_local_history_stays_complete(
    tmp_path: Path,
) -> None:
    old_canary = "OLD_PRIVATE_HISTORY_CANARY_918273"
    very_old_canary = "VERY_OLD_PRIVATE_HISTORY_CANARY_817263"
    recent_canary = "RECENT_HISTORY_CANARY_192837"
    current_request = "CURRENT_USER_REQUEST_CANARY_445566"
    provider = RecordingProvider()
    application = _remote_app(tmp_path, provider)

    with TestClient(application) as client:
        conversation_id = None
        for index in range(11):
            prefix = f"historical request {index}"
            if index == 0:
                prefix += f" {very_old_canary}"
            elif index == 1:
                prefix += f" {old_canary}"
            elif index == 10:
                prefix += f" {recent_canary}"
            response = client.post(
                "/api/chat",
                json={
                    "conversation_id": conversation_id,
                    "message": prefix + " " + ("history " * 300),
                },
            )
            assert response.status_code == 200
            conversation_id = response.json()["conversation_id"]

        final_response = client.post(
            "/api/chat",
            json={
                "conversation_id": conversation_id,
                "message": current_request,
            },
        )
        detail = client.get(f"/api/conversations/{conversation_id}").json()

    assert final_response.status_code == 200
    provider_messages = provider.generate_calls[-1][0]
    serialized_provider_context = json.dumps(provider_messages)
    assert old_canary not in serialized_provider_context
    assert very_old_canary not in serialized_provider_context
    assert recent_canary in serialized_provider_context
    assert provider_messages[-1] == {
        "role": "user",
        "content": current_request,
    }
    assert len(provider_messages[1:-1]) <= 20
    assert sum(len(message["content"]) for message in provider_messages[1:-1]) <= 20_000

    persisted_history = json.dumps(detail["messages"])
    assert old_canary in persisted_history
    assert very_old_canary in persisted_history
    assert recent_canary in persisted_history
    assert len(detail["messages"]) == 24


def test_streaming_crosses_provider_boundary_and_persists_final_local_text(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    application = _remote_app(tmp_path, provider)

    with TestClient(application) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "Give me an unconstrained response"},
        )
        conversations = client.get("/api/conversations").json()
        detail = client.get(f"/api/conversations/{conversations[0]['id']}").json()

    assert response.status_code == 200
    assert 'event: text\ndata: {"delta":"Remote"}' in response.text
    assert 'event: text\ndata: {"delta":" answer"}' in response.text
    assert "event: final" in response.text
    assert "event: done" in response.text
    assert len(provider.stream_calls) == 1
    assert [message["content"] for message in detail["messages"]] == [
        "Give me an unconstrained response",
        "Remote answer",
    ]


def test_failed_remote_inference_persists_neither_chat_nor_staged_memory(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider(failure=InferenceProviderError("Remote inference failed"))
    application = _remote_app(tmp_path, provider)

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Remember preference ui.theme = dark"},
        )
        conversations = client.get("/api/conversations").json()
        memories = client.get("/api/memory").json()

    assert response.status_code == 500
    assert response.json() == {"detail": "Assistant generation failed"}
    assert conversations == []
    assert memories == []


def test_failed_remote_stream_does_not_persist_partial_assistant_output(
    tmp_path: Path,
) -> None:
    class PartialFailureProvider(RecordingProvider):
        def stream(self, messages, generation_config, *, cancel_event):
            self.stream_calls.append((messages, generation_config, cancel_event))
            yield "Partial public text"
            raise InferenceProviderError("Remote inference failed")

    provider = PartialFailureProvider()
    application = _remote_app(tmp_path, provider)

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "Start then fail"},
        )
        conversations = client.get("/api/conversations").json()

    assert response.status_code == 200
    assert "Partial public text" in response.text
    assert "event: error" in response.text
    assert "event: final" not in response.text
    assert "event: done" not in response.text
    assert conversations == []


def test_remote_failures_do_not_log_prompts_tokens_or_response_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "PRIVATE_PROMPT_92831"
    token = "REMOTE_API_SECRET_19281"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"echoed {prompt} with {token}")

    provider = RemoteInferenceProvider(
        "https://gpu.example",
        token,
        EXPECTED_MODEL_NAME,
        transport=httpx.MockTransport(handler),
    )

    with (
        caplog.at_level(logging.INFO, logger="runtime.providers"),
        pytest.raises(InferenceProviderError, match="Remote inference failed"),
    ):
        provider.generate(
            [{"role": "user", "content": prompt}],
            {"max_new_tokens": 64},
        )

    logs = caplog.text
    assert "failure=http_500" in logs
    assert prompt not in logs
    assert token not in logs
    assert "echoed" not in logs


def test_inference_service_logs_only_sanitized_failure_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "PRIVATE_PROMPT_92831"
    token = "REMOTE_API_SECRET_19281"
    provider = RecordingProvider(failure=RuntimeError(f"failure included {prompt} {token}"))
    application = create_inference_app(
        provider=provider,
        auth_token=token,
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
    )

    with (
        caplog.at_level(logging.INFO, logger="runtime.inference_app"),
        TestClient(application) as client,
    ):
        response = client.post(
            "/v1/generate",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "request_id": "79d2a827-ac55-4ba1-a203-23aaec4f3e9d",
                "messages": [{"role": "user", "content": prompt}],
                "generation_config": {"max_new_tokens": 32},
            },
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Inference failed"}
    assert "failure=RuntimeError" in caplog.text
    assert prompt not in caplog.text
    assert token not in caplog.text
