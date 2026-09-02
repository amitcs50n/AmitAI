import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import PngImagePlugin
from sqlalchemy import select

from backend.app import create_app
from backend.asset_crypto import (
    ASSET_HEADER,
    ASSET_OVERHEAD,
    MAX_ASSET_BYTES,
    MAX_ASSET_CIPHERTEXT_BYTES,
    PNG_SIGNATURE,
    AssetCryptoError,
    decrypt_asset,
    encrypt_asset,
)
from backend.asset_keys import AssetKeyError
from backend.asset_storage import AssetStorage, AssetStorageError, _assert_private
from backend.assets import normalize_image
from backend.models import AssetEncryptionState
from backend.secure_memory import SecretHandle
from runtime.key_store import KeyStore, KeyStorePolicy, UnlockError
from tests.app_factory import create_test_app
from tests.test_assets import image_bytes, upload

TEST_KEY = b"AEK_CANARY_918273".ljust(32, b"!")
PLAINTEXT = normalize_image(image_bytes(), "image/png").content


@pytest.fixture
def encrypted_app(tmp_path):
    # Explicit plaintext SQLite test mode; never a production protection claim.
    app = create_test_app(
        f"sqlite:///{tmp_path / 'app.db'}",
        asset_directory=tmp_path / "assets",
    )
    with TestClient(app) as client:
        yield app, client


def test_aead_roundtrip_random_nonces_and_exact_format():
    first_id, second_id = str(uuid4()), str(uuid4())
    first = encrypt_asset(first_id, PLAINTEXT, TEST_KEY)
    repeat = encrypt_asset(first_id, PLAINTEXT, TEST_KEY)
    other = encrypt_asset(second_id, PLAINTEXT, TEST_KEY)
    assert ASSET_HEADER == b"AMITASST\x01"
    assert ASSET_OVERHEAD == 37
    assert MAX_ASSET_CIPHERTEXT_BYTES == 20 * 1024 * 1024 + 37
    assert len(first) == len(PLAINTEXT) + ASSET_OVERHEAD
    assert len({first, repeat, other}) == 3
    assert len({item[9:21] for item in (first, repeat, other)}) == 3
    for identifier, ciphertext in [(first_id, first), (first_id, repeat), (second_id, other)]:
        assert ciphertext.startswith(ASSET_HEADER)
        assert not ciphertext.startswith(PNG_SIGNATURE)
        assert PLAINTEXT not in ciphertext
        assert decrypt_asset(identifier, ciphertext, TEST_KEY) == PLAINTEXT


@pytest.mark.parametrize(
    "mutation",
    [
        "magic",
        "version",
        "nonce",
        "first",
        "middle",
        "tag",
        "append",
        "garbage",
        "oversized",
        *range(38),
        -1,
    ],
)
def test_strict_parser_and_aead_tampering_are_sanitized(mutation):
    identifier = str(uuid4())
    envelope = encrypt_asset(identifier, PLAINTEXT, TEST_KEY)
    if isinstance(mutation, int):
        damaged = envelope[:mutation]
    elif mutation == "append":
        damaged = envelope + b"x"
    elif mutation == "garbage":
        damaged = b"garbage" * 8
    elif mutation == "oversized":
        damaged = ASSET_HEADER + b"x" * MAX_ASSET_CIPHERTEXT_BYTES
    else:
        offsets = {
            "magic": 0,
            "version": 8,
            "nonce": 9,
            "first": 21,
            "middle": len(envelope) // 2,
            "tag": len(envelope) - 1,
        }
        mutable = bytearray(envelope)
        mutable[offsets[mutation]] ^= 1
        damaged = bytes(mutable)
    with pytest.raises(AssetCryptoError, match="^Stored image is unavailable$"):
        decrypt_asset(identifier, damaged, TEST_KEY)


@pytest.mark.parametrize("wrong_key", [b"", b"x" * 16, b"x" * 31, b"x" * 32, b"x" * 33])
def test_wrong_key_and_wrong_asset_identity_fail(wrong_key):
    identifier = str(uuid4())
    ciphertext = encrypt_asset(identifier, PLAINTEXT, TEST_KEY)
    with pytest.raises(AssetCryptoError):
        decrypt_asset(identifier, ciphertext, wrong_key)
    with pytest.raises(AssetCryptoError):
        decrypt_asset(str(uuid4()), ciphertext, TEST_KEY)


def test_plaintext_and_ciphertext_limits():
    identifier = str(uuid4())
    large = b"p" * MAX_ASSET_BYTES
    ciphertext = encrypt_asset(identifier, large, TEST_KEY)
    assert len(ciphertext) == MAX_ASSET_CIPHERTEXT_BYTES
    assert decrypt_asset(identifier, ciphertext, TEST_KEY) == large
    for invalid in (b"", large + b"p"):
        with pytest.raises(AssetCryptoError):
            encrypt_asset(identifier, invalid, TEST_KEY)
    for invalid_key in (b"x" * 16, b"x" * 31, b"x" * 33):
        with pytest.raises(AssetCryptoError):
            encrypt_asset(identifier, PLAINTEXT, invalid_key)


def test_actual_files_encrypted_preview_unchanged_and_swap_fails_aead(encrypted_app):
    app, client = encrypted_app
    assets = [upload(client).json(), upload(client).json()]
    storage = app.state.asset_storage
    paths = [storage.root / f"{asset['id']}.asset" for asset in assets]
    ciphertext = [path.read_bytes() for path in paths]
    assert ciphertext[0] != ciphertext[1]
    assert assets[0]["sha256"] == assets[1]["sha256"]  # Hash checks CANNOT detect this swap.
    for asset, path in zip(assets, paths, strict=True):
        assert path.read_bytes().startswith(ASSET_HEADER)
        assert PLAINTEXT not in path.read_bytes()
        _assert_private(storage.root, directory=True)
        _assert_private(path, directory=False)
        response = client.get(f"/api/assets/{asset['id']}/content")
        assert response.content == PLAINTEXT
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
    paths[0].write_bytes(ciphertext[1])
    paths[1].write_bytes(ciphertext[0])
    for asset in assets:
        with pytest.raises(AssetStorageError):
            storage.read(asset["id"])  # Direct storage read, before AssetService's hash check.
        response = client.get(f"/api/assets/{asset['id']}/content")
        assert response.status_code == 503
        assert response.json() == {"detail": "Local asset storage is unavailable"}
        # Known generated files can be deleted even though authentication fails.
        assert client.delete(f"/api/assets/{asset['id']}").status_code == 204
    assert list(storage.root.iterdir()) == []


def test_disk_failure_only_writes_ciphertext_temps_and_rolls_back(encrypted_app, monkeypatch):
    app, client = encrypted_app
    storage = app.state.asset_storage
    observed = []

    def failed_replace(source, _destination):
        data = Path(source).read_bytes()
        observed.append(data)
        assert data.startswith(ASSET_HEADER)
        assert PLAINTEXT not in data
        assert not data.startswith(PNG_SIGNATURE)
        _assert_private(Path(source), directory=False)
        raise OSError("PRIVATE_PATH_CANARY")

    monkeypatch.setattr("backend.asset_storage.os.replace", failed_replace)
    response = upload(client)
    assert response.status_code == 503
    assert "PRIVATE_PATH_CANARY" not in response.text
    assert len(observed) == 1
    assert list(storage.root.iterdir()) == []
    with app.state.database.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT count(*) FROM uploaded_assets").scalar() == 0


def test_read_is_bounded_even_if_file_is_oversized(encrypted_app, monkeypatch):
    app, _client = encrypted_app
    path = app.state.asset_storage.root / f"{uuid4()}.asset"
    reads = []

    class Oversized(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return b"x" * size

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Oversized())
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.read(path.stem)
    assert reads == [MAX_ASSET_CIPHERTEXT_BYTES + 1]


def test_private_key_never_enters_logs_reprs_metadata_or_export(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(
        "backend.asset_keys.secrets", SimpleNamespace(token_bytes=lambda n: TEST_KEY)
    )
    app = create_test_app(
        f"sqlite:///{tmp_path / 'app.db'}",
        asset_directory=tmp_path / "PRIVATE_PATH_CANARY",
    )
    app.state.database.engine.echo = "debug"  # Even opt-in SQL logging must not expose key rows.
    with caplog.at_level(logging.DEBUG), TestClient(app) as client:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("location", "PRIVATE_IMAGE_METADATA_CANARY")
        asset = upload(client, image_bytes(pnginfo=metadata)).json()
        response = client.get(f"/api/assets/{asset['id']}/content")
        chat = client.post("/api/chat", json={"message": "Attached", "asset_ids": [asset["id"]]})
        exported = client.get(f"/api/conversations/{chat.json()['conversation_id']}").text
        public = (
            response.text + exported + repr(app.state._state) + repr(app.state.asset_storage._key)
        )
        assert "key_material" not in exported
        assert "data:image" not in exported
        assert "ciphertext" not in exported
        assert response.content == PLAINTEXT
        for value in (
            TEST_KEY.decode(),
            TEST_KEY.hex(),
            base64.b64encode(TEST_KEY).decode(),
            "PRIVATE_IMAGE_METADATA_CANARY",
            "PRIVATE_PATH_CANARY",
        ):
            assert value not in public + caplog.text
        handle = app.state.asset_storage._key
        assert handle.locked
    assert handle.closed
    assert bytes(handle._buffer) == b"\x00" * 32


def test_missing_or_wrong_key_never_read_as_plaintext(encrypted_app):
    app, client = encrypted_app
    asset = upload(client).json()
    storage = app.state.asset_storage
    storage.close()
    with pytest.raises(AssetStorageError):
        storage.read(asset["id"])
    storage.bind_key(SecretHandle(b"w" * 32))
    with pytest.raises(AssetStorageError):
        storage.read(asset["id"])


def test_key_creation_commit_failure_does_not_write_images(tmp_path, monkeypatch):
    app = create_test_app(f"sqlite:///{tmp_path / 'app.db'}", asset_directory=tmp_path / "assets")
    app.state.database.create_schema()
    with app.state.database.engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TRIGGER reject_aek BEFORE INSERT ON asset_encryption_state "
            "BEGIN SELECT RAISE(ABORT, 'PRIVATE_DB_FAILURE_CANARY'); END"
        )
    with (
        pytest.raises(AssetKeyError, match="^Asset encryption key unavailable$"),
        TestClient(app),
    ):
        pytest.fail("Must not start")
    assert not app.state.asset_storage.root.exists()
    with app.state.database.engine.connect() as connection:
        assert (
            connection.exec_driver_sql("SELECT count(*) FROM asset_encryption_state").scalar() == 0
        )
    app.state.database.engine.dispose()


def test_sqlcipher_passphrase_change_and_database_rotation_preserve_aek_and_ciphertext(tmp_path):
    passphrase, replacement = "asset original passphrase", "asset replacement passphrase"
    store = KeyStore(tmp_path / "secrets" / "database-key.json", policy=KeyStorePolicy.for_tests())
    database_path = tmp_path / "app.db"
    store.initialize(passphrase)
    original_database_key = None
    options = {"asset_directory": tmp_path / "assets", "enforce_local_auth": False}

    def application(handle):
        return create_app(f"sqlite:///{database_path}", database_key=handle, **options)

    with store.unlock(passphrase) as handle:
        original_database_key = handle.copy_bytes()
        app = application(handle)
        with TestClient(app) as client:
            asset = upload(client).json()
            chat = client.post("/api/chat", json={"message": "Keep", "asset_ids": [asset["id"]]})
            conversation_id = chat.json()["conversation_id"]
            with app.state.database.session_factory() as session:
                aek = session.scalar(select(AssetEncryptionState.key_material))
            assert len(aek) == 32 and aek != original_database_key
    path = options["asset_directory"] / f"{asset['id']}.asset"
    ciphertext = path.read_bytes()
    before_db = database_path.read_bytes()
    assert aek not in before_db and aek.hex().encode() not in before_db
    assert not before_db.startswith(b"SQLite format 3")
    store.change_passphrase(passphrase, replacement)
    assert database_path.read_bytes() == before_db
    assert path.read_bytes() == ciphertext
    with pytest.raises(UnlockError):
        store.unlock(passphrase)

    def verify():
        with store.unlock(replacement, database_path=database_path) as handle:
            app = application(handle)
            with TestClient(app) as client:
                assert client.get(f"/api/assets/{asset['id']}/content").content == PLAINTEXT
                history = client.get(f"/api/conversations/{conversation_id}").json()
                assert history["messages"][0]["assets"][0]["id"] == asset["id"]
                with app.state.database.session_factory() as session:
                    assert session.scalar(select(AssetEncryptionState.key_material)) == aek
        assert path.read_bytes() == ciphertext

    verify()
    store.rotate_database_key(database_path, replacement)
    with store.unlock(replacement) as handle:
        assert handle.copy_bytes() != original_database_key
    assert aek not in database_path.read_bytes()
    verify()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode checks; Windows ACLs checked above")
def test_posix_asset_modes_and_insecure_mode_rejected(encrypted_app):
    app, client = encrypted_app
    asset = upload(client).json()
    root = app.state.asset_storage.root
    path = root / f"{asset['id']}.asset"
    assert root.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    path.chmod(0o644)
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.read(asset["id"])


def test_owned_paths_reject_symlink_or_reparse(encrypted_app, monkeypatch):
    app, client = encrypted_app
    asset = upload(client).json()
    path = app.state.asset_storage.root / f"{asset['id']}.asset"
    original = Path.lstat

    def reparse(candidate, *args, **kwargs):
        if candidate == path:
            details = original(candidate, *args, **kwargs)
            return SimpleNamespace(st_mode=details.st_mode, st_file_attributes=0x400)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.read(asset["id"])
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.delete(asset["id"])


def test_asset_storage_without_initialized_key_fails_closed(tmp_path):
    with pytest.raises(AssetStorageError):
        AssetStorage(tmp_path / "assets").write(str(uuid4()), PLAINTEXT)


def test_plaintext_byte_canary_never_written_even_in_temp(encrypted_app, monkeypatch):
    app, _client = encrypted_app
    storage = app.state.asset_storage
    # Deliberate crypto-layer payload: compressed pixel bytes need not contain literal text.
    payload = PNG_SIGNATURE + b"PRIVATE_IMAGE_CANARY_817263" * 16
    original_replace = os.replace
    inspected = []

    def inspect(source, target):
        ciphertext = Path(source).read_bytes()
        assert ciphertext.startswith(ASSET_HEADER)
        assert b"PRIVATE_IMAGE_CANARY" not in ciphertext
        assert PNG_SIGNATURE not in ciphertext
        inspected.append(source)
        original_replace(source, target)

    monkeypatch.setattr("backend.asset_storage.os.replace", inspect)
    identifier = str(uuid4())
    storage.write(identifier, payload)
    ciphertext = (storage.root / f"{identifier}.asset").read_bytes()
    assert b"PRIVATE_IMAGE_CANARY" not in ciphertext
    assert payload not in ciphertext
    assert storage.read(identifier) == payload
    assert len(inspected) == 1


@pytest.mark.parametrize("failure", ["tamper", "wrong_key", "missing_key"])
def test_read_failure_never_logs_crypto_material_or_paths(
    encrypted_app,
    monkeypatch,
    caplog,
    failure,
):
    app, client = encrypted_app
    asset = upload(client).json()
    storage = app.state.asset_storage
    original_key = storage._key.copy_bytes()
    path = storage.root / f"{asset['id']}.asset"
    ciphertext = path.read_bytes()
    if failure == "tamper":
        path.write_bytes(ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]))
    else:
        storage.close()
        if failure == "wrong_key":
            storage.bind_key(SecretHandle(TEST_KEY))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AssetStorageError) as caught:
            storage.read(asset["id"])
        response = client.get(f"/api/assets/{asset['id']}/content")
    assert response.status_code == 503
    exposed = str(caught.value) + response.text + caplog.text
    for value in (
        TEST_KEY.decode(),
        TEST_KEY.hex(),
        original_key.hex(),
        str(path),
        base64.b64encode(original_key).decode(),
        ciphertext[9:21].hex(),
        "InvalidTag",
        "PRIVATE_IMAGE_CANARY",
        "PRIVATE_IMAGE_METADATA_CANARY",
    ):
        assert value not in exposed


def test_symlink_file_rejected_without_following(encrypted_app, monkeypatch):
    app, client = encrypted_app
    asset = upload(client).json()
    path = app.state.asset_storage.root / f"{asset['id']}.asset"
    original = Path.lstat

    def symlink(candidate, *args, **kwargs):
        if candidate == path:
            return SimpleNamespace(st_mode=0o120777, st_file_attributes=0)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", symlink)
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.read(asset["id"])
    with pytest.raises(AssetStorageError):
        app.state.asset_storage.delete(asset["id"])
