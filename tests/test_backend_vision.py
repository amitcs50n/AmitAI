"""Real encrypted local assets with mocked native inference, never real weights."""

import json
import logging
from datetime import timedelta
from threading import Event
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.assets import AssetService
from backend.chat_service import ChatService, GenerationMessage, RemoteVisionDisclosureError
from backend.models import MemorySlot, UploadedAsset, utc_now
from runtime.config import EXPECTED_MODEL_NAME, load_runtime_config
from runtime.generator import ProviderChatGenerator, TransformersChatGenerator
from runtime.providers import RemoteInferenceProvider
from tests.app_factory import create_test_app
from tests.test_assets import counts, upload
from tests.test_backend_streaming import _parse_sse
from tests.test_vision import VisionEngine, assert_closed, vision_generator


def make_app(tmp_path, generator):
    return create_test_app(
        f"sqlite:///{tmp_path / 'DATABASE_PATH_CANARY.db'}",
        generator=generator,
        asset_directory=tmp_path / "ASSET_PATH_CANARY",
    )


def final_response(response, streaming):
    assert response.status_code == 200
    if not streaming:
        return response.json()
    events = _parse_sse(response.text.splitlines())
    assert events[0]["event"] == "start"
    assert [e["event"] for e in events[-2:]] == ["final", "done"]
    final = events[-2]["data"]
    assert "".join(e["data"]["delta"] for e in events if e["event"] == "text") == final["response"]
    return final


@pytest.mark.parametrize("streaming", [False, True])
def test_encrypted_bytes_vision_persistence_and_text_followup_without_image_reload(
    tmp_path,
    monkeypatch,
    streaming,
    caplog,
):
    engine = VisionEngine(["Red square shown.", "Further explanation."])
    generator, loads = vision_generator(engine)
    app = make_app(tmp_path, generator)
    reads = []
    original = AssetService.processing_bytes

    def read(service, asset_id, **kwargs):
        png = original(service, asset_id, **kwargs)
        reads.append(png)
        return png

    monkeypatch.setattr(AssetService, "processing_bytes", read)
    endpoint = "/api/chat/stream" if streaming else "/api/chat"
    with TestClient(app) as client, caplog.at_level(logging.DEBUG):
        asset = upload(client, filename=r"C:\PRIVATE_ORIGINAL_PATH\photo.png").json()
        path = app.state.asset_storage.root / f"{asset['id']}.asset"
        encrypted_before = path.read_bytes()
        assert not encrypted_before.startswith(b"\x89PNG")
        result = final_response(
            client.post(
                endpoint,
                json={
                    "message": "VISION_PROMPT_CANARY What is shown?",
                    "asset_ids": [asset["id"]],
                },
            ),
            streaming,
        )
        assert result["response"] == "Red square shown."
        assert result["metadata"]["model"] == EXPECTED_MODEL_NAME
        assert result["metadata"]["input_tokens"] == 100
        assert result["metadata"]["memory"] == []
        assert len(reads) == 1 and reads[0].startswith(b"\x89PNG\r\n\x1a\n")
        assert reads[0] == client.get(f"/api/assets/{asset['id']}/content").content
        assert path.read_bytes() == encrypted_before
        assert list(app.state.asset_storage.root.iterdir()) == [path]
        assert counts(app) == (1, 2, 1)
        history = client.get(f"/api/conversations/{result['conversation_id']}").json()
        assert history["messages"][0]["content"] == "VISION_PROMPT_CANARY What is shown?"
        assert history["messages"][0]["assets"][0]["id"] == asset["id"]
        assert history["messages"][1]["content"] == "Red square shown."
        assert all(
            marker not in json.dumps(history) for marker in ["base64", "image_pad", "pixel_values"]
        )
        assert len(engine.images) == 1
        # The explicit preview above also reads bytes. Isolate the followup request.
        reads.clear()
        final_response(
            client.post(
                endpoint,
                json={
                    "conversation_id": result["conversation_id"],
                    "message": "Explain more.",
                },
            ),
            streaming,
        )
        assert not reads and len(engine.images) == 1 and len(loads) == 1
        assert counts(app) == (1, 4, 1)
        assert all(isinstance(m["content"], str) for m in engine.calls[-1][0])
        assert "Red square shown." in str(engine.calls[-1][0])
        for messages, _ in engine.calls:
            assert not any(
                canary in str(messages)
                for canary in (
                    "PRIVATE_ORIGINAL_PATH",
                    "DATABASE_PATH_CANARY",
                    "ASSET_PATH_CANARY",
                    asset["id"],
                    ".asset",
                )
            )
        with app.state.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(MemorySlot)) == 0
    assert "VISION_PROMPT_CANARY" not in caplog.text


@pytest.mark.parametrize("streaming", [False, True])
def test_vision_memory_commands_are_read_only_and_memory_context_is_not_retrieved(
    tmp_path, streaming
):
    engine = VisionEngine(["Red square shown."])
    generator, _ = vision_generator(engine)
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        stored = client.post(
            "/api/memory",
            json={
                "category": "project",
                "key": "vision.canary",
                "value": "PRIVATE_MEMORY_CANARY",
            },
        )
        assert stored.status_code == 201
        asset = upload(client).json()
        result = final_response(
            client.post(
                "/api/chat/stream" if streaming else "/api/chat",
                json={
                    "message": "Remember project vision.caption = VISION_CAPTION_CANARY",
                    "asset_ids": [asset["id"]],
                },
            ),
            streaming,
        )
        assert result["metadata"]["memory"] == []
        assert "PRIVATE_MEMORY_CANARY" not in str(engine.calls)
        assert "Memory commands are not applied on image turns" in str(engine.calls)
        with app.state.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(MemorySlot)) == 1


@pytest.mark.parametrize("kind", ["multiple", "missing", "expired", "deleted"])
@pytest.mark.parametrize("streaming", [False, True])
def test_invalid_asset_turns_never_infer_or_create_conversation(tmp_path, kind, streaming):
    engine = VisionEngine()
    generator, loads = vision_generator(engine)
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        ids = [asset["id"]]
        if kind == "multiple":
            ids.append(upload(client).json()["id"])
        elif kind == "missing":
            ids = [str(uuid4())]
        elif kind == "deleted":
            assert client.delete(f"/api/assets/{asset['id']}").status_code == 204
        else:
            with app.state.database.session_factory.begin() as session:
                session.get(UploadedAsset, asset["id"]).created_at = utc_now() - timedelta(days=2)
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "message": "Describe",
                "asset_ids": ids,
            },
        )
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == ["error"]
        else:
            assert response.status_code in {404, 422}
        if kind == "multiple":
            assert "Vision currently supports one image per message." in response.text
            assert counts(app) == (0, 0, 2)
        assert counts(app)[:2] == (0, 0) and not loads and not engine.calls


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_vision_fails_before_decrypt_and_http(tmp_path, monkeypatch, streaming):
    http_calls = []
    provider = RemoteInferenceProvider(
        "https://inference.example",
        "test_token_material_0123456789abcdefgh",
        EXPECTED_MODEL_NAME,
        allowed_origins=["https://inference.example"],
        resolver=lambda *_: ["8.8.8.8"],
        transport=httpx.MockTransport(lambda req: http_calls.append(req) or httpx.Response(500)),
    )
    generator = ProviderChatGenerator(load_runtime_config(), provider=provider)
    assert generator.supports_vision  # Capability does not imply disclosure consent.
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        monkeypatch.setattr(
            AssetService, "processing_bytes", lambda *_: pytest.fail("remote decrypt")
        )
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "message": "Describe",
                "asset_ids": [asset["id"]],
            },
        )
        assert "Remote vision disclosure is not enabled" in response.text
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == ["error"]
        else:
            assert response.status_code == 403
        assert not http_calls and counts(app) == (0, 0, 1)
        assert client.get(f"/api/assets/{asset['id']}").json()["persistence_mode"] == "temporary"
        with pytest.raises(RemoteVisionDisclosureError):
            generator.generate_vision_response([GenerationMessage("user", "Hi")], b"not PNG")
    provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_validation_failure_does_not_persist_or_change_existing_history(
    tmp_path, streaming, existing
):
    engine = VisionEngine(
        ["Text answer.", *(["This has four words"] * 3)]
        if existing
        else ["This has four words"] * 3
    )
    generator, _ = vision_generator(engine)
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        conversation_id = None
        if existing:
            conversation_id = client.post("/api/chat", json={"message": "Hi"}).json()[
                "conversation_id"
            ]
            before = client.get(f"/api/conversations/{conversation_id}").json()
        asset = upload(client).json()
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "conversation_id": conversation_id,
                "message": "Answer in exactly 3 words.",
                "asset_ids": [asset["id"]],
            },
        )
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == [
                "start",
                "error",
            ]
        else:
            assert response.status_code == 500
        assert "This has four words" not in response.text
        assert counts(app) == ((1, 2, 1) if existing else (0, 0, 1))
        assert client.get(f"/api/assets/{asset['id']}").json()["persistence_mode"] == "temporary"
        if existing:
            assert client.get(f"/api/conversations/{conversation_id}").json() == before


def test_cancellation_and_partial_failure_outside_transactions_do_not_promote_asset(tmp_path):
    engine = VisionEngine(["Red square shown.", "Red square shown."])
    generator, _ = vision_generator(engine)
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        with app.state.database.session_factory() as session:
            service = ChatService(
                session, generator=generator, asset_storage=app.state.asset_storage
            )

            def outside_transaction():
                assert not session.in_transaction()

            engine.before_generate = outside_transaction
            signal = Event()
            stream = service.stream_chat(
                conversation_id=None,
                message="Describe",
                asset_ids=(asset["id"],),
                cancel_event=signal,
            )
            assert next(stream).event == "start"
            assert next(stream).event == "text"
            signal.set()
            assert list(stream) == []
            assert not session.in_transaction() and counts(app) == (0, 0, 1)
            assert_closed(engine.images[0])

        def fail_after_delta():
            raise RuntimeError("PRIVATE_CUDA_TENSOR_PATH_CANARY")

        engine.before_generate = lambda: None
        engine.after_first = fail_after_delta
        response = client.post(
            "/api/chat/stream", json={"message": "Describe", "asset_ids": [asset["id"]]}
        )
        events = _parse_sse(response.text.splitlines())
        assert [e["event"] for e in events] == ["start", "text", "error"]
        assert "PRIVATE_CUDA_TENSOR_PATH_CANARY" not in response.text
        assert counts(app) == (0, 0, 1)
        assert_closed(engine.images[-1])
        assert client.get(f"/api/assets/{asset['id']}").json()["persistence_mode"] == "temporary"


@pytest.mark.parametrize("streaming", [False, True])
def test_advertised_native_vision_load_failure_never_falls_back_to_stub(tmp_path, streaming):
    def incompatible_factory(*_args):
        raise RuntimeError("PRIVATE_CACHE_PATH")

    incompatible_factory.supports_vision = True
    generator = TransformersChatGenerator(
        load_runtime_config(), engine_factory=incompatible_factory
    )
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        response = client.post(
            "/api/chat/stream" if streaming else "/api/chat",
            json={
                "message": "Describe",
                "asset_ids": [asset["id"]],
            },
        )
        assert "Assistant generation failed" in response.text
        assert (
            "PRIVATE_CACHE_PATH" not in response.text and "media-not-enabled" not in response.text
        )
        assert counts(app) == (0, 0, 1)
        if streaming:
            assert [e["event"] for e in _parse_sse(response.text.splitlines())] == [
                "start",
                "error",
            ]
        else:
            assert response.status_code == 500


@pytest.mark.parametrize("streaming", [False, True])
def test_successful_vision_generation_and_retry_always_outside_sql_transaction(tmp_path, streaming):
    engine = VisionEngine(["Invalid answer with four words", "Red square shown."])
    generator, _ = vision_generator(engine)
    app = make_app(tmp_path, generator)
    with TestClient(app) as client:
        asset = upload(client).json()
        with app.state.database.session_factory() as session:
            service = ChatService(
                session, generator=generator, asset_storage=app.state.asset_storage
            )

            def outside_transaction():
                assert not session.in_transaction()

            engine.before_generate = outside_transaction
            kwargs = {
                "conversation_id": None,
                "message": "Answer in exactly 3 words.",
                "asset_ids": (asset["id"],),
            }
            if streaming:
                events = list(service.stream_chat(**kwargs))
                assert [event.event for event in events] == ["start", "text", "final", "done"]
                assert events[1].data["delta"] == "Red square shown."
            else:
                assert service.chat(**kwargs).response == "Red square shown."
            assert len(engine.calls) == 2 and not session.in_transaction()
            assert counts(app) == (1, 2, 1)
