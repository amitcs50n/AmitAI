"""CPU-only cancellation regressions. Deadlines here fail tests, never kill workers."""

import asyncio
import json
import socket
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.chat_service import ChatService, ChatStreamEvent
from backend.models import Message
from evaluation.hf_backend import GenerationOutput
from runtime.config import EXPECTED_MODEL_NAME, load_runtime_config
from runtime.generator import ProviderChatGenerator
from runtime.inference_app import create_inference_app
from runtime.providers import LocalTransformersInferenceProvider, RemoteInferenceProvider
from tests.app_factory import create_test_app

TOKEN = "STREAM_CANCEL_TEST_TOKEN_0123456789abcdef"
MESSAGES = [{"role": "user", "content": "Hello"}]


class ControlledEngine:
    """First request waits for cancellation without needing another model token."""

    def __init__(self):
        self.escape = Event()
        self.stopped = Event()
        self.signal = None
        self.calls = 0

    def generate_detailed_stream(self, messages, config, *, cancel_event):
        self.calls += 1
        if self.calls == 1:
            self.signal = cancel_event
            try:
                yield "Partial"
                while not cancel_event.wait(0.01) and not self.escape.is_set():
                    pass
            finally:
                self.stopped.set()
        else:
            yield "Second succeeds"
            yield GenerationOutput("Second succeeds", 1, 2)


def local_provider(engine):
    return LocalTransformersInferenceProvider(
        load_runtime_config().model, 42, engine_factory=lambda *_: engine,
    )


async def disconnect_request(application, *, spec="2.3", mode="disconnect", path="/v1/generate/stream", body=None):
    body = body or json.dumps({
        "request_id": str(uuid4()), "messages": MESSAGES, "generation_config": {},
    }).encode()
    disconnect = asyncio.Event()
    sent = []
    read = False

    async def receive():
        nonlocal read
        if not read:
            read = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if b"Partial" in message.get("body", b""):
            if mode == "send_error":
                raise OSError("client closed the socket")
            disconnect.set()
            if mode == "blocked_send":
                await asyncio.Event().wait()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec},
        "method": "POST", "scheme": "http", "path": path, "query_string": b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"authorization", f"Bearer {TOKEN}".encode())],
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234), "http_version": "1.1",
    }
    await application(scope, receive, send)
    return sent


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("mode", ["disconnect", "send_error", "blocked_send"])
def test_inference_disconnect_releases_generation_lock_before_second_request(spec, mode):
    engine = ControlledEngine()
    provider = local_provider(engine)
    application = create_inference_app(provider=provider, auth_token=TOKEN)

    async def scenario():
        task = asyncio.create_task(disconnect_request(application, spec=spec, mode=mode))
        try:
            done, _ = await asyncio.wait({task}, timeout=2)
            cleaned = bool(done) and engine.stopped.is_set() and not provider._generation_lock.locked()
        finally:
            engine.escape.set()  # Release the fake on old/broken code so failure cannot hang pytest.
            if engine.signal is not None:
                engine.signal.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert cleaned, "disconnect left the provider worker/lock alive"
        assert list(provider.stream(MESSAGES, {}, cancel_event=Event()))[-1].text == "Second succeeds"

    asyncio.run(scenario())


@pytest.mark.parametrize("socket_read", [False, True])
def test_remote_stop_interrupts_idle_response_without_waiting_for_more_bytes(socket_read):
    reading, closed, escaped = Event(), Event(), Event()
    calls = []
    reader, writer = socket.socketpair()

    class IdleBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'event: delta\ndata: {"delta":"Partial"}\n\n'
            reading.set()
            if socket_read:
                try:
                    reader.recv(1)  # A real blocked socket read, no bytes and no read timeout.
                except OSError:
                    raise httpx.ReadError("synthetic cancelled read") from None
                return
            while not closed.wait(0.01) and not escaped.is_set():
                pass

        def close(self):
            closed.set()
            if socket_read:
                reader.close()

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            extensions = {"network_stream": SimpleNamespace(get_extra_info=lambda _: reader)} if socket_read else {}
            return httpx.Response(200, stream=IdleBody(), extensions=extensions)
        request_id = json.loads(request.content)["request_id"]
        output = {"request_id": request_id, "model": EXPECTED_MODEL_NAME,
                  "text": "Second succeeds", "input_tokens": 1, "output_tokens": 2}
        return httpx.Response(200, text='event: delta\ndata: {"delta":"Second succeeds"}\n\n'
                              + "event: final\ndata: " + json.dumps(output) + "\n\n")

    provider = RemoteInferenceProvider(
        "https://inference.invalid", TOKEN, EXPECTED_MODEL_NAME,
        allowed_origins=["https://inference.invalid"], resolver=lambda *_: ["8.8.8.8"],
        transport=httpx.MockTransport(handle),
    )
    signal, done = Event(), Event()
    results, errors = [], []
    def consume():
        try:
            results.extend(provider.stream(MESSAGES, {}, cancel_event=signal))
        except Exception as exc:  # noqa: BLE001 - assert all consumer failures below.
            errors.append(exc)
        finally:
            done.set()

    worker = Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert reading.wait(2)
        signal.set()
        stopped = done.wait(2)
    finally:
        escaped.set()
        writer.close()
        worker.join(2)
    try:
        assert stopped, "remote read did not observe cancellation until more bytes arrived"
        assert closed.is_set() and results == ["Partial"] and errors == []
        assert list(provider.stream(MESSAGES, {}, cancel_event=Event()))[-1].text == "Second succeeds"
    finally:
        provider.close()
        reader.close()


@pytest.mark.parametrize("failure", ["create", "iterate", "close"])
def test_provider_exceptions_release_lock_and_allow_next_stream(failure):
    signal = Event()
    class Engine:
        calls = 0
        def generate_detailed_stream(self, messages, config, *, cancel_event):
            self.calls += 1
            if self.calls > 1:
                return iter(("Second succeeds", GenerationOutput("Second succeeds", 1, 2)))
            if failure == "create":
                raise ValueError("synthetic creation failure")
            class Iterator:
                def __iter__(self):
                    return self
                def __next__(self):
                    if failure == "iterate":
                        raise ValueError("synthetic iteration failure")
                    raise StopIteration
                def close(self):
                    if failure == "close":
                        raise ValueError("synthetic cleanup failure")
                    assert cancel_event.is_set()
            return Iterator()

    provider = local_provider(Engine())
    with pytest.raises(ValueError, match="synthetic"):
        list(provider.stream(MESSAGES, {}, cancel_event=signal))
    assert not provider._generation_lock.locked()
    assert list(provider.stream(MESSAGES, {}, cancel_event=Event()))[-1].text == "Second succeeds"


def test_inference_stream_worker_failure_terminates_response_and_second_succeeds():
    class Engine:
        calls = 0
        def generate_detailed_stream(self, messages, config, *, cancel_event):
            self.calls += 1
            if self.calls == 1:
                yield "Partial"
                raise ValueError("PRIVATE_WORKER_FAILURE_CANARY")
            yield "Second succeeds"
            yield GenerationOutput("Second succeeds", 1, 2)
    engine = Engine()
    provider = local_provider(engine)
    application = create_inference_app(provider=provider, auth_token=TOKEN)
    payload = {"request_id": str(uuid4()), "messages": MESSAGES, "generation_config": {}}
    with TestClient(application) as client:
        first = client.post(
            "/v1/generate/stream", json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert "event: error" in first.text and "event: final" not in first.text
        assert "PRIVATE_WORKER_FAILURE_CANARY" not in first.text
        payload["request_id"] = str(uuid4())
        second = client.post(
            "/v1/generate/stream", json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert "event: final" in second.text and "Second succeeds" in second.text
    assert engine.calls == 2 and not provider._generation_lock.locked()


def test_close_signals_delegate_before_cleanup_and_repeated_start_does_not_deadlock():
    class Engine:
        def generate_detailed_stream(self, messages, config, *, cancel_event):
            try:
                yield "Partial"
            finally:
                assert cancel_event.is_set(), "delegate close ran before its stop signal"
    provider = local_provider(Engine())
    for _ in range(20):
        signal = Event()
        stream = provider.stream(MESSAGES, {}, cancel_event=signal)
        assert next(stream) == "Partial"
        stream.close()
        assert signal.is_set() and not provider._generation_lock.locked()


def test_cancelled_request_waiting_for_lock_exits_without_starting_model():
    engine = ControlledEngine()
    provider = local_provider(engine)
    first = provider.stream(MESSAGES, {}, cancel_event=Event())
    assert next(first) == "Partial"
    signal, finished = Event(), Event()
    result = []
    def queued():
        try:
            result.extend(provider.stream(MESSAGES, {}, cancel_event=signal))
        finally:
            finished.set()
    worker = Thread(target=queued, daemon=True)
    worker.start()
    try:
        signal.set()
        assert finished.wait(2), "a cancelled queued request still waits for the generation lock"
        assert engine.calls == 1 and result == []
    finally:
        first.close()
        worker.join(2)
    assert list(provider.stream(MESSAGES, {}, cancel_event=Event()))[-1].text == "Second succeeds"


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("mode", ["disconnect", "send_error", "blocked_send"])
def test_backend_disconnect_has_no_partial_persistence_and_immediate_second_request_works(tmp_path, spec, mode):
    engine = ControlledEngine()
    provider = local_provider(engine)
    application = create_test_app(
        f"sqlite+pysqlite:///{(tmp_path / 'chat.db').as_posix()}",
        generator=ProviderChatGenerator(load_runtime_config(), provider=provider),
    )

    async def scenario():
        async with application.router.lifespan_context(application):
            try:
                first = await asyncio.wait_for(disconnect_request(
                    application, spec=spec, mode=mode, path="/api/chat/stream",
                    body=json.dumps({"message": "Hello"}).encode(),
                ), 3)
                second = await asyncio.wait_for(disconnect_request(
                    application, spec=spec, path="/api/chat/stream",
                    body=json.dumps({"message": "Second request"}).encode(),
                ), 3)
                assert engine.stopped.is_set() and not provider._generation_lock.locked()
                assert not any(b"event: final" in item.get("body", b"") for item in first)
                assert any(b"Second succeeds" in item.get("body", b"") for item in second)
                with application.state.database.session_factory() as session:
                    assert session.scalar(select(func.count()).select_from(Message)) == 2
                    assert [row.content for row in session.scalars(select(Message).order_by(Message.created_at))] == [
                        "Second request", "Second succeeds",
                    ]
            finally:
                engine.escape.set()
                if engine.signal is not None:
                    engine.signal.set()

    asyncio.run(scenario())


def test_real_http_remote_stop_then_second_request_repeatedly_succeeds(tmp_path):
    """Backend ASGI -> real HTTP socket -> inference ASGI -> one cached fake model."""
    engine = ControlledEngine()
    local = local_provider(engine)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        create_inference_app(provider=local, auth_token=TOKEN), log_level="error",
    ))
    serving = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    serving.start()
    remote = RemoteInferenceProvider(f"http://127.0.0.1:{port}", TOKEN, EXPECTED_MODEL_NAME)
    application = create_test_app(
        f"sqlite+pysqlite:///{(tmp_path / 'remote-chat.db').as_posix()}",
        generator=ProviderChatGenerator(load_runtime_config(), provider=remote),
    )

    async def scenario():
        async with application.router.lifespan_context(application):
            for cycle in range(3):
                engine.calls = 0
                engine.stopped.clear()
                await asyncio.wait_for(disconnect_request(
                    application, spec="2.4", path="/api/chat/stream",
                    body=json.dumps({"message": "Hello"}).encode(),
                ), 5)
                # Deliberately no wait for cleanup before the next HTTP generation.
                second = await asyncio.wait_for(disconnect_request(
                    application, spec="2.4", path="/api/chat/stream",
                    body=json.dumps({"message": "Second request"}).encode(),
                ), 5)
                assert any(b"Second succeeds" in item.get("body", b"") for item in second)
                assert engine.stopped.is_set() and engine.calls == 2
                assert not local._generation_lock.locked() and local._engine is engine
                assert not remote._client.is_closed
                with application.state.database.session_factory() as session:
                    assert session.scalar(select(func.count()).select_from(Message)) == (cycle + 1) * 2

    try:
        asyncio.run(scenario())
    finally:
        engine.escape.set()
        if engine.signal is not None:
            engine.signal.set()
        remote.close()
        server.should_exit = True
        serving.join(5)
        listener.close()
    assert not serving.is_alive()


def test_backend_cleanup_exception_is_sanitized_and_does_not_hold_stream_gate(tmp_path, monkeypatch, caplog):
    original = ChatService.stream_chat
    calls = 0
    class BrokenCleanup:
        def __iter__(self):
            return iter([ChatStreamEvent("text", {"delta": "Partial"})])
        def close(self):
            raise RuntimeError("CLEANUP_PRIVATE_CANARY")
    def stream(self, **kwargs):
        nonlocal calls
        calls += 1
        return BrokenCleanup() if calls == 1 else original(self, **kwargs)
    monkeypatch.setattr(ChatService, "stream_chat", stream)
    application = create_test_app(f"sqlite+pysqlite:///{(tmp_path / 'cleanup.db').as_posix()}")
    with TestClient(application) as client:
        first = client.post("/api/chat/stream", json={"message": "First"})
        assert "event: error" in first.text and "event: final" not in first.text
        assert client.get("/api/conversations").json() == []
        second = client.post("/api/chat/stream", json={"message": "Second"})
        assert "event: final" in second.text and "event: done" in second.text
        assert "CLEANUP_PRIVATE_CANARY" not in first.text + second.text + caplog.text
