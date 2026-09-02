import json
import logging
import os
from datetime import timedelta
from io import BytesIO
from threading import Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
from sqlalchemy import event, func, select

from backend.app import create_app
from backend.asset_routes import MAX_UPLOAD_BODY_BYTES, parse_upload
from backend.asset_storage import MAX_ASSET_BYTES, AssetStorageError
from backend.assets import (
    MAX_IMAGE_DIMENSION,
    TEMPORARY_TTL,
    VISION_NOT_ENABLED,
    AssetError,
    AssetService,
    normalize_image,
)
from backend.chat_service import ChatService
from backend.models import Conversation, Message, UploadedAsset, utc_now
from tests.app_factory import create_test_app


def image_bytes(format="PNG", size=(12, 8), **kwargs):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=format, **kwargs)
    return output.getvalue()


@pytest.fixture
def app(tmp_path):
    def never_infer(_messages):
        pytest.fail("Attachment requests must never reach a text inference provider")

    return create_test_app(
        f"sqlite:///{tmp_path / 'assets.db'}",
        asset_directory=tmp_path / "private-assets",
        generator=never_infer,
    )


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def upload(client, content=None, mime="image/png", filename="sample.png", **fields):
    return client.post(
        "/api/assets",
        files={
            "file": (filename, image_bytes() if content is None else content, mime),
        },
        data=fields,
    )


def counts(app):
    with app.state.database.session_factory() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (Conversation, Message, UploadedAsset)
        )


@pytest.mark.parametrize(
    ("format", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_valid_upload_metadata_canonical_preview_and_delete(client, app, format, mime):
    response = upload(client, image_bytes(format), mime)
    assert response.status_code == 201
    asset = response.json()
    assert asset["id"] != asset["original_filename"]
    assert asset["content_type"] == "image/png"
    assert (asset["width"], asset["height"]) == (12, 8)
    assert asset["processing_scope"] == "local_only"
    assert asset["persistence_mode"] == "temporary"
    assert counts(app) == (0, 0, 1)
    assert client.get(f"/api/assets/{asset['id']}").json() == asset
    preview = client.get(f"/api/assets/{asset['id']}/content")
    assert preview.headers["content-type"] == "image/png"
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert len(preview.content) == asset["byte_size"]
    assert [path.name for path in app.state.asset_storage.root.iterdir()] == [f"{asset['id']}.asset"]
    with Image.open(BytesIO(preview.content)) as decoded:
        assert decoded.format == "PNG"
        assert not decoded.info
    assert client.delete(f"/api/assets/{asset['id']}").status_code == 204
    assert not list(app.state.asset_storage.root.iterdir())
    assert client.get(f"/api/assets/{asset['id']}/content").status_code == 404


@pytest.mark.parametrize(
    "name",
    [
        "../../secret.png",
        r"..\..\secret.png",
        r"C:\Users\Amit\Desktop\img.png",
        "x" * 600 + ".png",
        "\u202eevil\u200b.png",
        "$(whoami);<script>.png",
    ],
)
def test_filename_is_only_sanitized_leaf_display_metadata(client, app, name):
    response = upload(client, filename=name)
    assert response.status_code == 201
    asset = response.json()
    safe = asset["original_filename"]
    assert len(safe) <= 120
    assert all(
        character.isascii() and (character.isalnum() or character in " ._-") for character in safe
    )
    assert all(
        fragment not in json.dumps(asset)
        for fragment in ["C:\\Users", "Desktop", "../", "<script>"]
    )
    assert next(app.state.asset_storage.root.iterdir()).name == f"{asset['id']}.asset"


def test_strip_exif_orientation_and_png_text_metadata(client, caplog):
    exif = Image.Exif()
    exif[274] = 6
    exif[271] = "PRIVATE_DEVICE_CANARY"
    exif[315] = "PRIVATE_LOCATION_CANARY"
    source = image_bytes("JPEG", exif=exif)
    with caplog.at_level(logging.DEBUG):
        result = upload(client, source, "image/jpeg")
    assert result.status_code == 201
    asset = result.json()
    assert (asset["width"], asset["height"]) == (8, 12)
    content = client.get(f"/api/assets/{asset['id']}/content").content
    assert b"PRIVATE_" not in content
    with Image.open(BytesIO(content)) as normalized:
        assert not normalized.getexif()
        assert not normalized.info
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Location", "PRIVATE_LOCATION_CANARY")
    png = upload(client, image_bytes(pnginfo=metadata)).json()
    assert b"PRIVATE_" not in client.get(f"/api/assets/{png['id']}/content").content
    assert "PRIVATE_" not in caplog.text


@pytest.mark.parametrize(
    ("content", "mime"),
    [
        (b"", "image/png"),
        (b"not an image", "image/png"),
        (b"<svg></svg>", "image/svg+xml"),
        (image_bytes(), "image/jpeg"),
        (image_bytes(), "application/octet-stream"),
        (image_bytes()[:35], "image/png"),
        (image_bytes("JPEG")[:-10], "image/jpeg"),
    ],
)
def test_invalid_bytes_and_types_do_not_persist(client, app, content, mime):
    result = upload(client, content, mime)
    assert result.status_code in {415, 422}
    assert counts(app) == (0, 0, 0)
    assert not app.state.asset_storage.root.exists()


def test_dimension_and_pixel_limits_before_decode(client, app, monkeypatch):
    result = upload(client, image_bytes(size=(MAX_IMAGE_DIMENSION + 1, 1)))
    assert result.status_code == 422
    monkeypatch.setattr("backend.assets.MAX_IMAGE_PIXELS", 50)
    assert upload(client).status_code == 422
    assert counts(app) == (0, 0, 0)


@pytest.mark.parametrize(
    ("format", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
@pytest.mark.parametrize("missing", [1, 2, 4, 8])
def test_truncated_trailer_is_not_an_accepted_image(client, format, mime, missing):
    assert upload(client, image_bytes(format)[:-missing], mime).status_code == 422


def test_bytes_limit_without_content_length(client, app):
    response = client.post(
        "/api/assets",
        content=iter([b"x" * MAX_UPLOAD_BODY_BYTES, b"x"]),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert response.status_code == 413
    assert counts(app) == (0, 0, 0)
    with pytest.raises(AssetError):
        normalize_image(b"x" * (MAX_ASSET_BYTES + 1), "image/png")


@pytest.mark.parametrize(
    "extra",
    [
        [("unexpected", (None, "private"))],
        [("file", ("second.png", image_bytes(), "image/png"))],
        [("persistence_mode", (None, "temporary")), ("persistence_mode", (None, "temporary"))],
        [("processing_scope", (None, "remote_allowed"))],
        [("path", (None, r"C:\secret.png"))],
    ],
)
def test_strict_multipart_fields(client, extra):
    result = client.post(
        "/api/assets", files=[("file", ("image.png", image_bytes(), "image/png")), *extra]
    )
    assert result.status_code == 422
    assert "private" not in result.text and "secret.png" not in result.text


@pytest.mark.parametrize(
    "body", [b"", b"RAW_PRIVATE_MULTIPART", b"--x\r\ninvalid\r\n--x--", b"--x--"]
)
def test_malformed_multipart_is_sanitized(client, caplog, body):
    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/api/assets", content=body, headers={"Content-Type": "multipart/form-data; boundary=x"}
        )
    assert response.status_code == 422
    assert "RAW_PRIVATE_MULTIPART" not in response.text + caplog.text


def test_incomplete_multipart_and_invalid_metadata(client):
    with pytest.raises(AssetError):
        parse_upload(
            b'--x\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n\r\nraw',
            "multipart/form-data; boundary=x",
        )
    assert upload(client, persistence_mode="conversation").status_code == 422
    assert (
        upload(client, persistence_mode="temporary", conversation_id=str(uuid4())).status_code
        == 422
    )
    assert (
        upload(client, persistence_mode="conversation", conversation_id=str(uuid4())).status_code
        == 404
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_attachment_chat_local_acknowledgment_history_and_delete(client, app, streaming):
    asset = upload(client).json()
    path = "/api/chat/stream" if streaming else "/api/chat"
    response = client.post(
        path, json={"message": "Describe this image", "asset_ids": [asset["id"]]}
    )
    assert response.status_code == 200
    if streaming:
        lines = response.text.splitlines()
        assert [line for line in lines if line.startswith("event:")] == [
            "event: start",
            "event: text",
            "event: final",
            "event: done",
        ]
        result = json.loads(lines[lines.index("event: final") + 1][6:])
    else:
        result = response.json()
    assert result["response"] == VISION_NOT_ENABLED
    assert result["metadata"]["model"] == "media-not-enabled"
    conversation_id = result["conversation_id"]
    history = client.get(f"/api/conversations/{conversation_id}").json()
    stored = history["messages"][0]["assets"][0]
    assert stored["id"] == asset["id"]
    assert stored["persistence_mode"] == "conversation"
    assert stored["conversation_id"] == conversation_id
    assert history["messages"][1]["assets"] == []
    assert counts(app) == (1, 2, 1)
    assert client.delete(f"/api/assets/{asset['id']}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").json()["messages"][0]["assets"] == []
    assert not list(app.state.asset_storage.root.iterdir())


def test_explicit_conversation_mode_and_conversation_delete(client, app):
    conversation_id = client.post("/api/conversations").json()["id"]
    asset = upload(client, persistence_mode="conversation", conversation_id=conversation_id).json()
    response = client.post(
        "/api/chat",
        json={
            "message": "Attached",
            "conversation_id": conversation_id,
            "asset_ids": [asset["id"]],
        },
    )
    assert response.status_code == 200
    other = client.post(
        "/api/chat", json={"message": "Wrong conversation", "asset_ids": [asset["id"]]}
    )
    assert other.status_code == 409
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert counts(app) == (0, 0, 0)
    assert not list(app.state.asset_storage.root.iterdir())


def test_invalid_attachment_references_and_no_ambient_paths(client, app):
    asset = upload(client).json()
    for ids in [
        [str(uuid4())],
        [r"C:\Users\Amit\Desktop\image.png"],
        ["../secret.png"],
        [asset["id"]] * 2,
        [str(uuid4()) for _ in range(5)],
    ]:
        response = client.post("/api/chat", json={"message": "No authority", "asset_ids": ids})
        assert response.status_code in {404, 422}
        assert "Desktop" not in response.text
    assert counts(app) == (0, 0, 1)
    assert client.delete(f"/api/assets/{asset['id']}").status_code == 204
    response = client.post(
        "/api/chat/stream", json={"message": "Deleted", "asset_ids": [asset["id"]]}
    )
    assert "event: error" in response.text
    assert "event: final" not in response.text
    for path in [
        "/api/assets/import-path",
        "/api/assets/import",
        "/api/assets/path",
        "/api/assets/%2e%2e%2fsecret/content",
    ]:
        assert client.post(path, json={"path": r"C:\private.png"}).status_code in {404, 405}
        assert client.get(path).status_code in {404, 422}


def test_temporary_cleanup_orphans_restart_and_remote_fail_closed(client, app):
    asset = upload(client).json()
    with app.state.database.session_factory() as session:
        service = AssetService(session, app.state.asset_storage)
        with pytest.raises(AssetError, match="Remote image processing"):
            service.processing_bytes(asset["id"], remote=True)
        session.rollback()
        assert service.cleanup(now=utc_now() + TEMPORARY_TTL + timedelta(seconds=1)) == 1
    assert not list(app.state.asset_storage.root.iterdir())
    assert counts(app) == (0, 0, 0)
    orphan = str(uuid4())
    app.state.asset_storage.write(orphan, image_bytes())
    path = app.state.asset_storage.root / f"{orphan}.asset"
    os.utime(path, (0, 0))
    with app.state.database.session_factory() as session:
        assert AssetService(session, app.state.asset_storage).cleanup() == 1
    assert not path.exists()


def test_cancel_or_failed_commit_never_promotes_or_persists(client, app, monkeypatch):
    asset = upload(client).json()
    with app.state.database.session_factory() as session:
        service = ChatService(session, asset_storage=app.state.asset_storage)
        cancel = Event()
        stream = service.stream_chat(
            conversation_id=None, message="Attached", asset_ids=(asset["id"],), cancel_event=cancel
        )
        assert next(stream).event == "start"
        assert not session.in_transaction()
        assert next(stream).event == "text"
        cancel.set()
        assert list(stream) == []
        assert not session.in_transaction()
        assert counts(app) == (0, 0, 1)

        def fail(*args, **kwargs):
            raise RuntimeError("forced commit failure")

        monkeypatch.setattr(service.messages, "add_metadata", fail)
        with pytest.raises(RuntimeError):
            service.chat(conversation_id=None, message="Attached", asset_ids=(asset["id"],))
        assert counts(app) == (0, 0, 1)
    assert client.get(f"/api/assets/{asset['id']}").json()["persistence_mode"] == "temporary"


def test_failed_upload_cleans_file_and_bad_storage_returns_safe_error(client, app, monkeypatch):
    original = app.state.asset_storage.write

    def fail(asset_id, content):
        raise AssetStorageError("Local asset storage is unavailable")

    monkeypatch.setattr(app.state.asset_storage, "write", fail)
    assert upload(client).status_code == 503
    assert counts(app) == (0, 0, 0)
    monkeypatch.setattr(app.state.asset_storage, "write", original)


def test_all_asset_routes_require_auth(tmp_path):
    application = create_app(
        f"sqlite:///{tmp_path / 'auth.db'}",
        encrypted_storage=False,
        local_api_token="a" * 64,
        asset_directory=tmp_path / "assets",
    )
    with TestClient(application) as client:
        for method, path in [
            ("POST", "/api/assets"),
            ("GET", f"/api/assets/{uuid4()}"),
            ("GET", f"/api/assets/{uuid4()}/content"),
            ("DELETE", f"/api/assets/{uuid4()}"),
        ]:
            assert client.request(method, path).status_code == 401
        assert not application.state.asset_storage.root.exists()


def test_metadata_and_bytes_survive_restart(tmp_path):
    options = {"asset_directory": tmp_path / "assets"}
    url = f"sqlite:///{tmp_path / 'restart.db'}"
    with TestClient(create_test_app(url, **options)) as client:
        asset = upload(client).json()
    with TestClient(create_test_app(url, **options)) as client:
        assert client.get(f"/api/assets/{asset['id']}").json() == asset
        assert client.get(f"/api/assets/{asset['id']}/content").status_code == 200


def test_upload_commit_failure_removes_complete_file(app, client):
    with app.state.database.session_factory() as session:
        service = AssetService(session, app.state.asset_storage)

        def reject_commit(_session):
            raise RuntimeError("simulated commit failure")

        event.listen(session, "before_commit", reject_commit)
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            service.create(normalize_image(image_bytes(), "image/png"), filename="sample.png")
    assert counts(app) == (0, 0, 0)
    assert not list(app.state.asset_storage.root.iterdir())


def test_duplicate_uploads_are_independent_and_four_can_attach(client, app):
    assets = [upload(client).json() for _ in range(4)]
    ids = [asset["id"] for asset in assets]
    assert len(set(ids)) == 4
    assert len({asset["sha256"] for asset in assets}) == 1
    result = client.post("/api/chat", json={"message": "Keep these", "asset_ids": ids})
    assert result.status_code == 200
    assert client.delete(f"/api/assets/{ids[0]}").status_code == 204
    assert client.get(f"/api/assets/{ids[1]}/content").status_code == 200
    history = client.get(f"/api/conversations/{result.json()['conversation_id']}").json()
    assert len(history["messages"][0]["assets"]) == 3


def test_failed_physical_delete_is_inaccessible_and_cleanup_retries(client, app, monkeypatch):
    asset = upload(client).json()
    original_delete = app.state.asset_storage.delete

    def disk_failure(_asset_id):
        raise AssetStorageError("Local asset storage is unavailable")

    monkeypatch.setattr(app.state.asset_storage, "delete", disk_failure)
    assert client.delete(f"/api/assets/{asset['id']}").status_code == 503
    assert client.get(f"/api/assets/{asset['id']}/content").status_code == 404
    monkeypatch.setattr(app.state.asset_storage, "delete", original_delete)
    with app.state.database.session_factory() as session:
        service = AssetService(session, app.state.asset_storage)
        assert service.cleanup(now=utc_now() + TEMPORARY_TTL + timedelta(seconds=1)) == 1
    assert not list(app.state.asset_storage.root.iterdir())


def test_storage_has_no_arbitrary_path_authority(client, app):
    for untrusted in ["../outside", r"C:\private.png", "a/b", "", "sample.png"]:
        with pytest.raises(AssetStorageError):
            app.state.asset_storage.read(untrusted)
        with pytest.raises(AssetStorageError):
            app.state.asset_storage.delete(untrusted)
