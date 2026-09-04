"""Synthetic media, fake model, real protocol/privacy boundaries; never GPU/network."""

import asyncio
import base64
import builtins
import io
import json
import logging
import os
import ssl
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from backend.assets import AssetError, AssetService
from backend.chat_service import ChatPrivacyError, ChatService, GenerationMessage
from backend.models import Message, utc_now
from backend.security import LocalApiAuthMiddleware
from backend.vision_grant import RemoteVisionGrant
from evaluation.hf_backend import GenerationOutput
from runtime.config import EXPECTED_MODEL_NAME, load_runtime_config
from runtime.generator import ProviderChatGenerator
from runtime.inference_app import create_inference_app
from runtime.providers import InferenceProviderError, RemoteInferenceProvider
from runtime.vision_wire import (
    decode_vision_body,
    encode_vision_body,
)
from tests.test_assets import counts, image_bytes, upload
from tests.test_backend_streaming import _parse_sse
from tests.test_backend_vision import final_response, make_app
from tests.test_vision import VisionEngine, vision_generator

TOKEN = "remote_transport_test_token_1234567890"
ORIGIN = "https://inference.example"
PROMPT = "VISION_PROMPT_CANARY describe the shapes."


class Harness:
    def __init__(self, outputs=("Red square shown.",), **kwargs):
        self.engine = VisionEngine(outputs)
        local, self.loads = vision_generator(self.engine)
        self.server = create_inference_app(provider=local._provider, auth_token=TOKEN)
        self.client = TestClient(self.server)
        self.requests = []
        self.decoded = []
        self.grants = []
        self.dns = []
        self.provider = RemoteInferenceProvider(
            ORIGIN,
            TOKEN,
            EXPECTED_MODEL_NAME,
            allowed_origins=[ORIGIN],
            resolver=kwargs.pop("resolver", self.resolve),
            transport=httpx.MockTransport(kwargs.pop("handler", self.handle)),
            **kwargs,
        )
        self.generator = ProviderChatGenerator(load_runtime_config(), provider=self.provider)

    def resolve(self, host, port):
        self.dns.append((host, port))
        return ["8.8.8.8"]

    def handle(self, request):
        self.requests.append(request)
        if request.url.path.startswith("/v1/vision"):
            self.decoded.append(
                decode_vision_body(request.content, request.headers["content-type"])
            )
        response = self.client.post(
            request.url.path,
            content=request.content,
            headers={
                "authorization": request.headers["authorization"],
                "content-type": request.headers["content-type"],
            },
        )
        return httpx.Response(
            response.status_code, content=response.content, headers=response.headers
        )

    def run(self, streaming, prompt=PROMPT, grant=None):
        grant = grant or RemoteVisionGrant(str(uuid4()), True)
        kwargs = {"remote_grant": grant}
        messages = [GenerationMessage("user", prompt)]
        if streaming:
            return list(
                self.generator.stream_vision_response(
                    messages,
                    image_bytes(),
                    cancel_event=Event(),
                    **kwargs,
                )
            )[-1]
        return self.generator.generate_vision_response(messages, image_bytes(), **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
def test_explicit_consent_exact_wire_local_persistence_no_history_image_resend(
    tmp_path,
    monkeypatch,
    caplog,
    streaming,
):
    harness = Harness(["Red square shown.", "Further text."])
    app = make_app(tmp_path, harness.generator)
    grants = []
    original = AssetService.processing_bytes

    def processing(service, asset_id, **kwargs):
        if "remote_grant" in kwargs:
            grants.append(kwargs["remote_grant"])
        return original(service, asset_id, **kwargs)

    monkeypatch.setattr(AssetService, "processing_bytes", processing)
    monkeypatch.setenv("AMITAI_DB_KEY", "ab" * 32)
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", "LOCAL_TOKEN_CANARY_918273")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", TOKEN)
    endpoint = "/api/chat/stream" if streaming else "/api/chat"
    with TestClient(app) as client, caplog.at_level(logging.DEBUG):
        asset = upload(client, filename="PRIVATE_FILENAME_CANARY.png").json()
        canonical = client.get(f"/api/assets/{asset['id']}/content").content
        encrypted = (app.state.asset_storage.root / f"{asset['id']}.asset").read_bytes()
        asset_key = app.state.asset_storage._key_bytes()
        client.post(
            "/api/memory",
            json={
                "category": "project",
                "key": "shapes.private",
                "value": "LOCAL_MEMORY_CANARY",
            },
        )
        result = final_response(
            client.post(
                endpoint,
                json={
                    "message": PROMPT,
                    "asset_ids": [asset["id"]],
                    "allow_remote_vision": True,
                },
            ),
            streaming,
        )
        assert result["response"] == "Red square shown." and result["metadata"]["memory"] == []
        assert counts(app) == (1, 2, 1)
        assert len(harness.requests) == 1 and len(grants) == 1
        with pytest.raises(PermissionError):
            grants[0].require(asset["id"])
        request = harness.requests[0]
        metadata, png = harness.decoded[0]
        assert png == canonical and request.content.count(canonical) == 1
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.url.path == ("/v1/vision/stream" if streaming else "/v1/vision")
        assert PROMPT in str(metadata.messages)
        for marker in (
            asset["id"],
            "PRIVATE_FILENAME_CANARY",
            "DATABASE_PATH_CANARY",
            "ASSET_PATH_CANARY",
            "LOCAL_MEMORY_CANARY",
            "LOCAL_TOKEN_CANARY_918273",
            TOKEN,
            "ab" * 32,
            asset["sha256"],
            "filename=",
            "processing_scope",
            "persistence_mode",
            "AEK",
        ):
            assert marker.encode() not in request.content
        assert encrypted not in request.content
        for encoded_key in (asset_key, asset_key.hex().encode(), base64.b64encode(asset_key)):
            assert encoded_key not in request.content
        assert client.get(f"/api/assets/{asset['id']}").json()["processing_scope"] == "local_only"
        history = client.get(f"/api/conversations/{result['conversation_id']}").json()
        assert "allow_remote_vision" not in json.dumps(history)
        assert history["messages"][1]["content"] == result["response"]
        # Upload scope and previous consent never authorize another request.
        denied = client.post(
            endpoint,
            json={
                "conversation_id": result["conversation_id"],
                "message": "Again",
                "asset_ids": [asset["id"]],
            },
        )
        assert "Remote vision disclosure is not enabled" in denied.text
        assert len(harness.requests) == 1
        final_response(
            client.post(
                endpoint,
                json={
                    "conversation_id": result["conversation_id"],
                    "message": "Explain more",
                },
            ),
            streaming,
        )
        assert len(grants) == 1 and len(harness.engine.images) == 1 and len(harness.loads) == 1
        assert harness.requests[-1].headers["content-type"] == "application/json"
        assert counts(app) == (1, 4, 1)
    assert not any(
        s in caplog.text for s in (PROMPT, TOKEN, "LOCAL_MEMORY_CANARY", "PRIVATE_FILENAME_CANARY")
    )


def test_grant_mismatch_revocation_and_remote_boolean_cannot_decrypt(tmp_path, monkeypatch):
    harness = Harness()
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        monkeypatch.setattr(app.state.asset_storage, "read", lambda *_: pytest.fail("decryption"))
        with app.state.database.session_factory() as session:
            service = AssetService(session, app.state.asset_storage)
            grant = RemoteVisionGrant(str(uuid4()), True)
            for kwargs in ({"remote": True}, {"remote_grant": grant}):
                with pytest.raises(AssetError):
                    service.processing_bytes(asset["id"], **kwargs)
            grant = RemoteVisionGrant(asset["id"], True)
            grant.revoke()
            with pytest.raises(AssetError):
                service.processing_bytes(asset["id"], remote_grant=grant)
    with pytest.raises(PermissionError):
        RemoteVisionGrant(str(uuid4()), False)


@pytest.mark.parametrize("consent", [False, None, "true", 1, {}])
def test_strict_consent_defaults_fail_closed(tmp_path, consent):
    harness = Harness()
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        response = client.post(
            "/api/chat",
            json={
                "message": "Describe",
                "asset_ids": [asset["id"]],
                "allow_remote_vision": consent,
            },
        )
        assert response.status_code in {403, 422}
        assert not harness.requests and counts(app) == (0, 0, 1)


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_vision_tools_repairs_projection_and_grant_lifetime(
    tmp_path, monkeypatch, streaming
):
    tool = '<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'
    harness = Harness([tool, "The product is 1411.", "Product is 1411."])
    app = make_app(tmp_path, harness.generator)
    original = AssetService.processing_bytes
    grants = []

    def processing(service, asset_id, **kwargs):
        grants.append(kwargs["remote_grant"])
        return original(service, asset_id, **kwargs)

    monkeypatch.setattr(AssetService, "processing_bytes", processing)
    with TestClient(app) as client:
        asset = upload(client).json()
        with app.state.database.session_factory() as session:
            harness.engine.before_generate = lambda: assert_no_transaction(session)
            service = ChatService(session, harness.generator, asset_storage=app.state.asset_storage)
            kwargs = {
                "conversation_id": None,
                "message": "What is 17 * 83? Answer in exactly 3 words.",
                "asset_ids": (asset["id"],),
                "allow_remote_vision": True,
            }
            if streaming:
                events = list(service.stream_chat(**kwargs))
                assert [e.event for e in events] == ["start", "text", "final", "done"]
                assert events[1].data["delta"] == "Product is 1411."
            else:
                assert service.chat(**kwargs).response == "Product is 1411."
            assert counts(app) == (1, 2, 1) and len(harness.requests) == 3
            assert len(grants) == 1
            with pytest.raises(PermissionError):
                grants[0].require()
            assert all(png == harness.decoded[0][1] for _, png in harness.decoded)
            assert "tool_result" in str(harness.decoded[1][0].messages)


def assert_no_transaction(session):
    assert not session.in_transaction()


@pytest.mark.parametrize("streaming", [False, True])
def test_vision_text_guard_and_memory_command_projection(streaming):
    harness = Harness()
    with pytest.raises(ChatPrivacyError):
        harness.run(streaming, "My password: PRIVATE_PASSWORD_CANARY_123")
    assert not harness.requests and not harness.dns


@pytest.mark.parametrize("streaming", [False, True])
def test_image_memory_command_values_never_cross(tmp_path, streaming):
    harness = Harness()
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        final_response(
            client.post(
                "/api/chat/stream" if streaming else "/api/chat",
                json={
                    "message": "Remember project shapes.private = MEMORY_COMMAND_CANARY",
                    "asset_ids": [asset["id"]],
                    "allow_remote_vision": True,
                },
            ),
            streaming,
        )
        assert b"MEMORY_COMMAND_CANARY" not in harness.requests[0].content
        assert b"MEMORY_COMMAND_V1" not in harness.requests[0].content


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [300, 301, 302, 303, 307, 308, 401, 500])
def test_vision_redirects_and_errors_never_resend_or_persist(tmp_path, status, streaming):
    calls = []

    def fail(request):
        calls.append(request)
        return httpx.Response(
            status, headers={"Location": "https://attacker.example"}, text="PRIVATE_ERROR_BODY"
        )

    harness = Harness(handler=fail)
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "message": PROMPT,
                "asset_ids": [asset["id"]],
                "allow_remote_vision": True,
            },
        )
        assert "PRIVATE_ERROR_BODY" not in response.text
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == [
                "start",
                "error",
            ]
        else:
            assert response.status_code == 500
        assert len(calls) == 1 and counts(app) == (0, 0, 1)


def test_remote_vision_uses_existing_tls_client_and_dns_policy(monkeypatch):
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return httpx.Client(**kwargs)

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.setenv(key, "PRIVATE_ENV_CANARY")
    harness = Harness(client_factory=factory, resolver=lambda *_: ["8.8.8.8", "127.0.0.1"])
    assert len(captured) == 1
    assert captured[0]["trust_env"] is False and captured[0]["follow_redirects"] is False
    tls = captured[0]["verify"]
    assert tls.check_hostname and tls.verify_mode == ssl.CERT_REQUIRED
    assert tls.minimum_version >= ssl.TLSVersion.TLSv1_2
    with pytest.raises(Exception, match="Assistant generation failed"):
        harness.run(False)
    assert not harness.requests


def wire(png=None):
    return encode_vision_body(
        json.dumps(
            {
                "version": 1,
                "request_id": str(uuid4()),
                "messages": [{"role": "user", "content": PROMPT}],
                "generation_config": load_runtime_config().generation,
            }
        ).encode(),
        image_bytes() if png is None else png,
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_server_dispatches_only_the_selected_vision_method(monkeypatch, streaming):
    harness = Harness()
    provider = harness.server.state.inference_provider
    calls = []
    closed = Event()

    def generate(request, config):
        assert not streaming, "JSON/stream dispatch was crossed"
        assert request.image.getpixel((0, 0)) is not None
        calls.append(config)
        return GenerationOutput("Red square shown.", 1029, 3)

    def stream(request, config, *, cancel_event):
        assert streaming, "non-streaming endpoint called stream_vision"
        assert not cancel_event.is_set()
        assert request.image.getpixel((0, 0)) is not None
        calls.append(config)
        try:
            yield "Red "
            yield "square shown."
            yield GenerationOutput("Red square shown.", 1029, 3)
        finally:
            closed.set()

    def forbidden(*args, **kwargs):
        pytest.fail("Endpoint called the other generation method")

    monkeypatch.setattr(provider, "generate_vision", forbidden if streaming else generate)
    monkeypatch.setattr(provider, "stream_vision", stream if streaming else forbidden)
    body, content_type = wire()
    metadata, _ = decode_vision_body(body, content_type)
    response = harness.client.post(
        "/v1/vision/stream" if streaming else "/v1/vision",
        content=body,
        headers={"content-type": content_type, "authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    if streaming:
        events = list(_parse_sse(response.text.splitlines()))
        assert [e["event"] for e in events] == ["delta", "delta", "final"]
        result = events[-1]["data"]
        assert "".join(e["data"]["delta"] for e in events[:-1]) == result["text"]
        assert closed.wait(2)
    else:
        result = response.json()
    assert result == {
        "request_id": str(metadata.request_id), "model": provider.model_name,
        "text": "Red square shown.", "input_tokens": 1029, "output_tokens": 3,
    }
    assert calls == [metadata.generation_config.model_dump(exclude_none=True)]
    assert not harness.loads


@pytest.mark.parametrize("mode", ["nonstream", "before_delta", "after_delta"])
def test_worker_failure_safe_through_http_client_and_local_persistence(
    tmp_path, monkeypatch, caplog, mode
):
    harness = Harness()
    provider = harness.server.state.inference_provider
    attempts = []
    failure = ValueError("VISION_STREAM_CANARY " + TOKEN)

    def generate(*args, **kwargs):
        assert mode == "nonstream", "unexpected fallback"
        attempts.append("nonstream")
        raise failure

    def stream(*args, **kwargs):
        assert mode != "nonstream", "unexpected fallback"
        attempts.append("stream")
        if mode == "after_delta":
            yield "Partial"
        raise failure

    monkeypatch.setattr(provider, "generate_vision", generate)
    monkeypatch.setattr(provider, "stream_vision", stream)
    # Inspect the raw server boundary as well as its client-facing translation.
    body, content_type = wire()
    path = "/v1/vision" if mode == "nonstream" else "/v1/vision/stream"
    with caplog.at_level(logging.DEBUG):
        response = harness.client.post(
            path, content=body,
            headers={"content-type": content_type, "authorization": f"Bearer {TOKEN}"},
        )
        if mode == "nonstream":
            assert response.status_code == 502
            assert response.json() == {"detail": "Inference failed"}
        else:
            events = list(_parse_sse(response.text.splitlines()))
            assert [e["event"] for e in events] == (
                ["delta", "error"] if mode == "after_delta" else ["error"]
            )
            assert events[-1]["data"] == {"detail": "Inference failed"}
        app = make_app(tmp_path, harness.generator)
        with TestClient(app) as client:
            asset = upload(client).json()
            chat = client.post(
                "/api/chat" if mode == "nonstream" else "/api/chat/stream",
                json={"message": "Describe", "asset_ids": [asset["id"]],
                      "allow_remote_vision": True},
            )
            if mode == "nonstream":
                assert chat.status_code == 500
            else:
                assert [e["event"] for e in _parse_sse(chat.text.splitlines())] == (
                    ["start", "text", "error"] if mode == "after_delta" else ["start", "error"]
                )
            assert counts(app) == (0, 0, 1)
        assert [r.url.path for r in harness.requests] == [path]
    assert attempts == (["nonstream"] if mode == "nonstream" else ["stream"]) * 2
    assert not any(s in caplog.text + response.text + chat.text for s in (
        "VISION_STREAM_CANARY", TOKEN, "Traceback (most recent call last)"
    ))


@pytest.mark.parametrize("path", ["/v1/vision", "/v1/vision/stream"])
def test_server_auth_before_body_and_ram_only_validated_media(monkeypatch, path, caplog):
    harness = Harness()
    info = PngImagePlugin.PngInfo()
    info.add_text("private", "EXIF_GPS_CANARY")
    body, media_type = wire(image_bytes(pnginfo=info))
    assert b"EXIF_GPS_CANARY" not in body
    # An untrusted caller can skip the client normalizer: the server strips it again.
    _, canonical = decode_vision_body(body, media_type)
    body = body.replace(canonical, image_bytes(pnginfo=info))
    assert b"EXIF_GPS_CANARY" in body
    assert (
        harness.client.post(
            path, content=b"malformed", headers={"content-type": media_type}
        ).status_code
        == 401
    )
    assert not harness.loads

    # PIL and routes are imported before instrumentation; inference has no file access.
    def no_file(*args, **kwargs):
        pytest.fail("Remote vision attempted filesystem access")

    with monkeypatch.context() as scoped, caplog.at_level(logging.DEBUG):
        for name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile", "mkstemp"):
            scoped.setattr(tempfile, name, no_file)
        scoped.setattr(builtins, "open", no_file)
        scoped.setattr(io, "open", no_file)
        scoped.setattr(os, "open", no_file)
        response = harness.client.post(
            path,
            content=body,
            headers={
                "content-type": media_type,
                "authorization": f"Bearer {TOKEN}",
            },
        )
        assert response.status_code == 200
    assert len(harness.engine.images) == 1 and len(harness.loads) == 1
    assert not any(value in caplog.text for value in (PROMPT, TOKEN, "EXIF_GPS_CANARY"))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "extra",
        "filename",
        "missing",
        "encoded",
        "wrong-mime",
        "json",
        "truncated",
        "jpeg",
        "apng",
        "extra-metadata",
        "large-dimension",
        "epilogue",
        "query",
        "duplicate-json",
        "empty-image",
        "oversized-image",
        "duplicate-image",
        "missing-image",
        "too-many-pixels",
        "transfer-encoding",
        "malformed-json",
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_server_rejects_invalid_media_before_generation(mutation, streaming):
    harness = Harness()
    body, media_type = wire()
    _, png = decode_vision_body(body, media_type)
    headers = {"content-type": media_type, "authorization": f"Bearer {TOKEN}"}
    path = "/v1/vision/stream" if streaming else "/v1/vision"
    if mutation == "duplicate":
        delimiter = b"--" + media_type.split("boundary=")[1].encode()
        segments = body.split(delimiter)
        body = delimiter.join([segments[0], segments[1], segments[1], segments[-1]])
    elif mutation == "extra":
        delimiter = b"--" + media_type.split("boundary=")[1].encode()
        body = body.replace(
            delimiter + b"--", delimiter + body.split(delimiter)[1] + delimiter + b"--"
        )
    elif mutation == "filename":
        body = body.replace(b'name="image"', b'name="image"; filename="PRIVATE_PATH.png"')
    elif mutation == "missing":
        body = body.replace(b'name="image"', b'name="other"')
    elif mutation == "encoded":
        headers["content-encoding"] = "gzip"
    elif mutation == "wrong-mime":
        body = body.replace(b"image/png", b"image/jpeg")
    elif mutation == "json":
        headers["content-type"] = "application/json"
    elif mutation == "truncated":
        body = body.replace(png, png[:-12])
    elif mutation == "jpeg":
        body = body.replace(png, image_bytes("JPEG"))
    elif mutation == "apng":
        body = body.replace(
            png, image_bytes(save_all=True, append_images=[Image.new("RGB", (12, 8), "blue")])
        )
    elif mutation == "extra-metadata":
        body = body.replace(b'"version": 1', b'"asset_id": "PRIVATE_ID", "version": 1')
    elif mutation == "large-dimension":
        body = body.replace(png, image_bytes(size=(8193, 1)))
    elif mutation == "epilogue":
        body += b"PRIVATE_TRAILING_DATA"
    elif mutation == "query":
        path += "?token=PRIVATE_QUERY"
    elif mutation == "duplicate-json":
        body = body.replace(b'"version": 1', b'"version": 1, "version": 1')
    elif mutation == "empty-image":
        body = body.replace(png, b"")
    elif mutation == "oversized-image":
        body = body.replace(png, b"x" * (20 * 1024 * 1024 + 1))
    elif mutation == "duplicate-image":
        delimiter = b"--" + media_type.split("boundary=")[1].encode()
        segments = body.split(delimiter)
        body = delimiter.join([segments[0], segments[2], segments[2], segments[-1]])
    elif mutation == "missing-image":
        delimiter = b"--" + media_type.split("boundary=")[1].encode()
        segments = body.split(delimiter)
        body = delimiter + segments[1] + delimiter + segments[-1]
    elif mutation == "too-many-pixels":
        body = body.replace(png, image_bytes(size=(6000, 4001)))
    elif mutation == "transfer-encoding":
        body = body.replace(
            b"Content-Type: image/png",
            b"Content-Transfer-Encoding: base64\r\nContent-Type: image/png",
        )
    elif mutation == "malformed-json":
        body = body.replace(b'"version": 1', b'"version": invalid')
    response = harness.client.post(path, content=body, headers=headers)
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "Invalid vision request"}
    assert not harness.loads


def test_server_bounds_actual_reads_without_content_length(monkeypatch):
    harness = Harness()
    body, media_type = wire()
    import runtime.vision_api as api

    monkeypatch.setattr(api, "MAX_VISION_BODY_BYTES", len(body) - 1)
    response = harness.client.post(
        "/v1/vision",
        content=iter([body[:20], body[20:]]),
        headers={
            "content-type": media_type,
            "authorization": f"Bearer {TOKEN}",
        },
    )
    assert response.status_code == 422 and not harness.loads


def test_remote_stream_causality_and_cancellation_no_terminal(tmp_path, monkeypatch):
    first_emitted = Event()
    closed = Event()

    class Bytes(httpx.SyncByteStream):
        def __iter__(self):
            yield b'event: delta\ndata: {"delta":"Python"}\n\n'
            assert first_emitted.is_set(), "first delta was buffered"
            yield b": keep-alive\n\n"

        def close(self):
            closed.set()

    harness = Harness(handler=lambda _: httpx.Response(200, stream=Bytes()))
    app = make_app(tmp_path, harness.generator)
    grants = []
    original = AssetService.processing_bytes

    def processing(service, asset_id, **kwargs):
        grants.append(kwargs["remote_grant"])
        return original(service, asset_id, **kwargs)

    monkeypatch.setattr(AssetService, "processing_bytes", processing)
    with TestClient(app) as client:
        asset = upload(client).json()
        with app.state.database.session_factory() as session:
            signal = Event()
            service = ChatService(session, harness.generator, asset_storage=app.state.asset_storage)
            stream = service.stream_chat(
                conversation_id=None,
                message=PROMPT,
                asset_ids=(asset["id"],),
                allow_remote_vision=True,
                cancel_event=signal,
            )
            assert next(stream).event == "start"
            assert next(stream).data == {"delta": "Python"}
            first_emitted.set()
            signal.set()
            assert list(stream) == [] and closed.is_set()
            assert not session.in_transaction() and counts(app) == (0, 0, 1)
            with pytest.raises(PermissionError):
                grants[0].require()


def test_capabilities_minimal_mock_local_remote_and_authenticated(tmp_path):
    harness = Harness()
    local, _ = vision_generator(VisionEngine())
    for index, (generator, expected, mode) in enumerate(
        [
            (None, {"enabled": False, "scope": None}, "mock"),
            (local, {"enabled": True, "scope": "local"}, "local"),
            (harness.generator, {"enabled": True, "scope": "remote"}, "remote"),
            (SimpleNamespace(supports_vision=False, vision_scope="local"), {"enabled": False, "scope": "local"}, "local"),
            (SimpleNamespace(supports_vision=False, vision_scope="remote"), {"enabled": False, "scope": "remote"}, "remote"),
            (SimpleNamespace(), {"enabled": False, "scope": None}, "unknown"),
        ]
    ):
        directory = tmp_path / str(index)
        directory.mkdir()
        app = make_app(directory, generator)
        with TestClient(app) as client:
            response = client.get("/api/capabilities")
            assert response.json() == {"vision": expected, "inference": {"mode": mode}}
            assert response.headers["cache-control"] == "no-store"
        protected = TestClient(LocalApiAuthMiddleware(app, token="cd" * 32))
        assert protected.get("/api/capabilities").status_code == 401
        assert (
            protected.get(
                "/api/capabilities", headers={"authorization": "Bearer " + "cd" * 32}
            ).status_code
            == 200
        )
    assert not harness.loads and not harness.requests


@pytest.mark.parametrize("streaming", [False, True])
def test_minimized_remote_vision_history_retries_do_not_restore_private_context(
    tmp_path, streaming
):
    harness = Harness(["This has four words", "Red square shown."])
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        conversation = client.post("/api/conversations", json={"title": "PRIVATE_TITLE"}).json()
        with app.state.database.session_factory.begin() as session:
            now = utc_now()
            for index in range(30):
                session.add(
                    Message(
                        id=str(uuid4()),
                        conversation_id=conversation["id"],
                        role="user" if index % 2 == 0 else "assistant",
                        content=("OLD_HISTORY_CANARY " if index < 20 else "RECENT_HISTORY_CANARY ")
                        + "x" * 2000,
                        created_at=now + timedelta(microseconds=index),
                    )
                )
        before = client.get(f"/api/conversations/{conversation['id']}").json()["messages"]
        asset = upload(client).json()
        final_response(
            client.post(
                "/api/chat/stream" if streaming else "/api/chat",
                json={
                    "conversation_id": conversation["id"],
                    "message": "Describe. Answer in exactly 3 words.",
                    "asset_ids": [asset["id"]],
                    "allow_remote_vision": True,
                },
            ),
            streaming,
        )
        assert len(harness.requests) == 2
        for request in harness.requests:
            assert b"OLD_HISTORY_CANARY" not in request.content
            assert b"RECENT_HISTORY_CANARY" in request.content
            assert b"PRIVATE_TITLE" not in request.content
            assert conversation["id"].encode() not in request.content
        after = client.get(f"/api/conversations/{conversation['id']}").json()["messages"]
        assert after[:30] == before


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_failed_validation_existing_conversation_unchanged(tmp_path, streaming):
    harness = Harness(["This has four words"] * 3)
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        conversation = client.post("/api/conversations", json={"title": "Existing"}).json()
        before = client.get(f"/api/conversations/{conversation['id']}").json()
        asset = upload(client).json()
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "conversation_id": conversation["id"],
                "message": "Answer in exactly 3 words.",
                "asset_ids": [asset["id"]],
                "allow_remote_vision": True,
            },
        )
        assert "This has four words" not in response.text
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == [
                "start",
                "error",
            ]
        else:
            assert response.status_code == 500
        assert client.get(f"/api/conversations/{conversation['id']}").json() == before
        assert len(harness.requests) == 2 and counts(app) == (1, 0, 1)


def test_remote_partial_failure_has_no_persistence(tmp_path):
    class Partial(httpx.SyncByteStream):
        def __iter__(self):
            yield b'event: delta\ndata: {"delta":"Partial"}\n\n'
            raise httpx.ReadError("PRIVATE_NETWORK_ERROR")

    harness = Harness(handler=lambda _: httpx.Response(200, stream=Partial()))
    app = make_app(tmp_path, harness.generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "Describe",
                "asset_ids": [asset["id"]],
                "allow_remote_vision": True,
            },
        )
        assert [e["event"] for e in _parse_sse(response.text.splitlines())] == [
            "start",
            "text",
            "error",
        ]
        assert "PRIVATE_NETWORK_ERROR" not in response.text and counts(app) == (0, 0, 1)


def test_remote_server_disconnect_signals_model_before_terminal_and_closes_image():
    stopped = Event()

    class ControlledEngine(VisionEngine):
        def generate_detailed_stream(self, messages, config, *, cancel_event):
            self._output(messages, config)
            yield "Python"
            assert cancel_event.wait(5), "ASGI disconnect did not cancel model generation"
            stopped.set()

    engine = ControlledEngine()
    local, _ = vision_generator(engine)
    server = create_inference_app(provider=local._provider, auth_token=TOKEN)
    body, content_type = wire()

    async def scenario():
        disconnected = asyncio.Event()
        sent = []
        first_read = True

        async def receive():
            nonlocal first_read
            if first_read:
                first_read = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if b"Python" in message.get("body", b""):
                disconnected.set()

        await server(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "method": "POST",
                "scheme": "http",
                "path": "/v1/vision/stream",
                "query_string": b"",
                "headers": [
                    (b"content-type", content_type.encode()),
                    (b"authorization", f"Bearer {TOKEN}".encode()),
                ],
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 1234),
                "http_version": "1.1",
            },
            receive,
            send,
        )
        assert await asyncio.to_thread(stopped.wait, 5)
        assert not any(b"event: final" in m.get("body", b"") for m in sent)

    asyncio.run(scenario())


def test_nonstream_server_disconnect_discards_result_but_keeps_image_until_return(monkeypatch):
    import runtime.vision_api as api

    harness = Harness()
    started, release, returned, image_closed = Event(), Event(), Event(), Event()
    decode = api.decoded_vision_image

    @contextmanager
    def tracked_decode(png):
        with decode(png) as image:
            yield image
        if started.is_set():
            image_closed.set()

    def generate(request, config):
        started.set()
        assert release.wait(5), "disconnected request did not return promptly"
        # A disconnect must not close the borrowed PIL input underneath the call.
        assert request.image.getpixel((0, 0)) is not None
        returned.set()
        return GenerationOutput("DISCARDED_RESULT_CANARY", 10, 2)

    monkeypatch.setattr(api, "decoded_vision_image", tracked_decode)
    monkeypatch.setattr(harness.server.state.inference_provider, "generate_vision", generate)
    body, content_type = wire()

    async def scenario():
        sent = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            assert started.is_set()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        try:
            await asyncio.wait_for(harness.server(
                {
                    "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "method": "POST", "scheme": "http", "path": "/v1/vision",
                    "query_string": b"", "http_version": "1.1",
                    "headers": [(b"content-type", content_type.encode()),
                                (b"authorization", f"Bearer {TOKEN}".encode())],
                    "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
                }, receive, send,
            ), timeout=3)
            assert started.is_set() and not returned.is_set() and not image_closed.is_set()
            assert not any(b"DISCARDED_RESULT_CANARY" in m.get("body", b"") for m in sent)
        finally:
            release.set()
        assert await asyncio.to_thread(image_closed.wait, 2)
        assert returned.is_set()

    asyncio.run(scenario())


def test_direct_remote_vision_invalid_or_revoked_grant_never_transports():
    harness = Harness()
    grant = RemoteVisionGrant(str(uuid4()), True)
    grant.revoke()
    with pytest.raises(PermissionError):
        harness.provider.generate_vision([], {}, image_bytes(), remote_grant=grant)
    with pytest.raises(InferenceProviderError):
        harness.provider.generate_vision(
            [], {}, None, remote_grant=RemoteVisionGrant(str(uuid4()), True)
        )
    assert not harness.requests


@pytest.mark.parametrize(
    "endpoint,allowed,token",
    [
        (ORIGIN, [], TOKEN),
        ("http://inference.example", ["http://inference.example"], TOKEN),
        ("https://8.8.8.8", ["https://8.8.8.8"], TOKEN),
        (ORIGIN, [ORIGIN], "short"),
        (ORIGIN, [ORIGIN], " " + TOKEN),
    ],
)
def test_invalid_vision_transport_configuration_fails_before_client(endpoint, allowed, token):
    with pytest.raises(ValueError):
        RemoteInferenceProvider(
            endpoint,
            token,
            EXPECTED_MODEL_NAME,
            allowed_origins=allowed,
            client_factory=lambda **_: pytest.fail("Invalid vision transport reached client"),
        )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "endpoint,answers",
    [
        (ORIGIN, ["192.168.1.1"]),
        (ORIGIN, ["8.8.8.8", "10.0.0.1"]),
        ("http://localhost:8000", ["8.8.8.8"]),
    ],
)
def test_vision_dns_preflight_blocks_private_mixed_and_loopback_rebinding(
    endpoint, answers, streaming
):
    provider = RemoteInferenceProvider(
        endpoint,
        TOKEN,
        EXPECTED_MODEL_NAME,
        allowed_origins=[endpoint],
        resolver=lambda *_: answers,
        transport=httpx.MockTransport(
            lambda _: pytest.fail("DNS policy allowed vision disclosure")
        ),
    )
    grant = RemoteVisionGrant(str(uuid4()), True)
    args = ([{"role": "user", "content": PROMPT}], load_runtime_config().generation, image_bytes())
    with pytest.raises(InferenceProviderError, match="Remote inference failed"):
        if streaming:
            list(provider.stream_vision(*args, remote_grant=grant, cancel_event=Event()))
        else:
            provider.generate_vision(*args, remote_grant=grant)


@pytest.mark.parametrize("streaming", [False, True])
def test_vision_tls_errors_are_sanitized_in_results_and_logs(caplog, streaming):
    def fail(_):
        raise httpx.ConnectError("PRIVATE_TLS_CANARY " + TOKEN + " PRIVATE_IMAGE_BYTE_CANARY")

    harness = Harness(handler=fail)
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(Exception, match="Assistant generation failed") as caught,
    ):
        harness.run(streaming)
    assert not any(
        value in str(caught.value) + caplog.text
        for value in (
            "PRIVATE_TLS_CANARY",
            "PRIVATE_IMAGE_BYTE_CANARY",
            TOKEN,
            PROMPT,
        )
    )
