import hashlib
import logging
import os
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app import create_app
from backend.asset_crypto import ASSET_HEADER, ASSET_MAGIC, PNG_SIGNATURE, encrypt_asset
from backend.asset_keys import AssetKeyError, load_asset_key
from backend.asset_migration import migrate_assets
from backend.asset_storage import AssetStorage, AssetStorageError
from backend.assets import normalize_image
from backend.database import Database
from backend.models import (
    AssetEncryptionState,
    Conversation,
    Message,
    UploadedAsset,
    message_assets,
)
from backend.schemas import AssetRead
from runtime.paths import atomic_write_private
from tests.test_assets import image_bytes

PLAINTEXT = normalize_image(image_bytes(), "image/png").content
DB_KEY = "a1" * 32


class SimulatedCrash(BaseException):
    pass


def legacy_state(tmp_path, *, count=1, encrypted=False):
    """Real previous-version metadata + owner-only plaintext PNG files (no key row)."""
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    options = {
        "database_url": url,
        "encrypted_storage": encrypted,
        "database_key": DB_KEY if encrypted else None,
        "enforce_local_auth": False,
        "asset_directory": tmp_path / "assets",
    }
    database = Database.from_url(url, encrypted=encrypted, encryption_key=options["database_key"])
    database.create_schema()
    snapshots = []
    with database.session_factory.begin() as session:
        conversation = Conversation(title="Preserved legacy image conversation")
        session.add(conversation)
        session.flush()
        message = Message(
            conversation_id=conversation.id, content="Existing attachment", role="user"
        )
        session.add(message)
        session.flush()
        for index in range(count):
            asset = UploadedAsset(
                id=str(UUID(int=index + 1)),
                original_filename="legacy.png",
                kind="image",
                content_type="image/png",
                byte_size=len(PLAINTEXT),
                width=12,
                height=8,
                sha256=hashlib.sha256(PLAINTEXT).hexdigest(),
                conversation_id=conversation.id,
                persistence_mode="conversation",
                processing_scope="local_only",
            )
            session.add(asset)
            session.flush()
            session.execute(
                message_assets.insert().values(message_id=message.id, asset_id=asset.id)
            )
            snapshots.append(AssetRead.model_validate(asset).model_dump(mode="json"))
            atomic_write_private(options["asset_directory"] / f"{asset.id}.png", PLAINTEXT)
        conversation_id = conversation.id
    database.engine.dispose()
    return options, snapshots, conversation_id


def open_database(options):
    return Database.from_url(
        options["database_url"],
        encrypted=options["encrypted_storage"],
        encryption_key=options["database_key"],
    )


def key_value(options):
    database = open_database(options)
    try:
        with database.session_factory() as session:
            return session.scalar(select(AssetEncryptionState.key_material))
    finally:
        database.engine.dispose()


def prepare_migration(options):
    database = open_database(options)
    storage = AssetStorage(options["asset_directory"])
    storage.bind_key(load_asset_key(database.engine, storage))
    return database, storage


@pytest.mark.parametrize("encrypted", [False, True])
def test_legacy_migration_preserves_identity_metadata_linkage_and_is_idempotent(
    tmp_path, encrypted
):
    options, snapshots, conversation_id = legacy_state(tmp_path, encrypted=encrypted)
    identifier = snapshots[0]["id"]
    root = options["asset_directory"]
    assert key_value(options) is None
    with TestClient(create_app(**options)) as client:
        assert client.get(f"/api/assets/{identifier}").json() == snapshots[0]
        assert client.get(f"/api/assets/{identifier}/content").content == PLAINTEXT
        history = client.get(f"/api/conversations/{conversation_id}").json()
        assert history["messages"][0]["assets"] == snapshots
        assert history["messages"][0]["content"] == "Existing attachment"
    assert not (root / f"{identifier}.png").exists()
    ciphertext = (root / f"{identifier}.asset").read_bytes()
    assert ciphertext.startswith(ASSET_HEADER) and PLAINTEXT not in ciphertext
    aek = key_value(options)
    assert len(aek) == 32
    with TestClient(create_app(**options)) as client:
        assert client.get(f"/api/assets/{identifier}/content").content == PLAINTEXT
    assert (root / f"{identifier}.asset").read_bytes() == ciphertext
    assert key_value(options) == aek
    assert sorted(path.name for path in root.iterdir()) == [f"{identifier}.asset"]


@pytest.mark.parametrize(
    "phase", ["after_temp_fsync", "after_legacy_replace", "after_legacy_rename"]
)
def test_crash_recovery_uses_durable_same_key_and_never_copies_plaintext(tmp_path, phase):
    options, snapshots, _ = legacy_state(tmp_path, count=2, encrypted=True)
    database, storage = prepare_migration(options)
    aek = key_value(options)  # A separate connection sees the committed AEK BEFORE any migration.
    assert aek is not None
    identifier = snapshots[0]["id"]
    root = options["asset_directory"]
    observed = []

    def crash(current):
        with database.engine.connect() as connection:
            # File operations don't own a SQL transaction (another writer can BEGIN).
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.rollback()
        if current == "after_temp_fsync":
            temps = list(root.glob(".*.asset.tmp-*"))
            assert len(temps) == 1
            assert temps[0].read_bytes().startswith(ASSET_HEADER)
            assert PLAINTEXT not in temps[0].read_bytes()
            # No destination .asset exists while the original remains plaintext.
            assert not (root / f"{identifier}.asset").exists()
        if current == phase:
            observed.append(current)
            raise SimulatedCrash

    try:
        with pytest.raises(SimulatedCrash):
            migrate_assets(database.session_factory, storage, phase_hook=crash)
    finally:
        storage.close()
        database.engine.dispose()
    assert observed == [phase]
    legacy = root / f"{identifier}.png"
    if phase == "after_temp_fsync":
        assert legacy.read_bytes() == PLAINTEXT
        assert len(list(root.glob(".*.asset.tmp-*"))) == 1
    elif phase == "after_legacy_replace":
        assert legacy.read_bytes().startswith(ASSET_HEADER)
        assert not (root / f"{identifier}.asset").exists()
    else:
        assert not legacy.exists()
    with TestClient(create_app(**options)) as client:
        for snapshot in snapshots:
            assert client.get(f"/api/assets/{snapshot['id']}/content").content == PLAINTEXT
    assert key_value(options) == aek
    assert len(list(root.iterdir())) == 2
    assert all(path.suffix == ".asset" for path in root.iterdir())


def test_mixed_encrypted_legacy_missing_and_interrupted_names(tmp_path):
    options, snapshots, conversation_id = legacy_state(tmp_path, count=4)
    root = options["asset_directory"]
    database, storage = prepare_migration(options)
    key = key_value(options)
    first, second, missing, interrupted = [snapshot["id"] for snapshot in snapshots]
    storage.migrate_asset(first, lambda _data: None)
    first_before = (root / f"{first}.asset").read_bytes()
    (root / f"{missing}.png").unlink()
    interrupted_bytes = encrypt_asset(interrupted, PLAINTEXT, key)
    atomic_write_private(root / f"{interrupted}.png", interrupted_bytes)
    storage.close()
    database.engine.dispose()
    with TestClient(create_app(**options)) as client:
        for identifier in (first, second, interrupted):
            assert client.get(f"/api/assets/{identifier}/content").content == PLAINTEXT
        assert client.get(f"/api/assets/{missing}").status_code == 200
        assert client.get(f"/api/assets/{missing}/content").status_code == 503
        history = client.get(f"/api/conversations/{conversation_id}").json()
        assert len(history["messages"][0]["assets"]) == 4
    assert (root / f"{first}.asset").read_bytes() == first_before
    assert (root / f"{interrupted}.asset").read_bytes() == interrupted_bytes
    assert not (root / f"{missing}.asset").exists()
    assert key_value(options) == key


@pytest.mark.parametrize(
    "damage", ["unknown", "size", "hash", "dimensions", "truncated", "version"]
)
def test_invalid_legacy_stops_startup_without_serving_or_destroying_source(
    tmp_path, damage, caplog
):
    options, snapshots, _ = legacy_state(tmp_path)
    identifier = snapshots[0]["id"]
    path = options["asset_directory"] / f"{identifier}.png"
    database, storage = prepare_migration(options)
    key = key_value(options)
    if damage == "unknown":
        path.write_bytes(b"PRIVATE_UNKNOWN_BYTES_CANARY")
    elif damage == "truncated":
        path.write_bytes(PLAINTEXT[:-1])
    elif damage == "version":
        path.write_bytes(ASSET_MAGIC + b"\x02" + b"x" * 100)
    else:
        with database.session_factory.begin() as session:
            asset = session.get(UploadedAsset, identifier)
            if damage == "size":
                asset.byte_size += 1
            elif damage == "hash":
                asset.sha256 = "0" * 64
            else:
                asset.width += 1
    storage.close()
    database.engine.dispose()
    before = path.read_bytes()
    app = create_app(**options)
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(AssetStorageError, match="^Local asset storage is unavailable$") as caught,
        TestClient(app),
    ):
        pytest.fail("Migration failure must stop startup")
    assert app.state.asset_storage._key is None
    exposed = str(caught.value) + caplog.text
    for value in ("PRIVATE_UNKNOWN_BYTES_CANARY", key.hex(), str(path), "InvalidTag"):
        assert value not in exposed
    assert path.read_bytes() == before
    assert key_value(options) == key
    assert not (options["asset_directory"] / f"{identifier}.asset").exists()


@pytest.mark.parametrize("representation", ["asset", "png", "temp", "orphan"])
def test_encrypted_files_without_key_fail_closed_including_orphans_and_temps(
    tmp_path, representation
):
    options, snapshots, _ = legacy_state(tmp_path)
    identifier = snapshots[0]["id"]
    root = options["asset_directory"]
    database, storage = prepare_migration(options)
    ciphertext = encrypt_asset(identifier, PLAINTEXT, key_value(options))
    (root / f"{identifier}.png").unlink()
    filename = f"{identifier}.png" if representation == "png" else f"{identifier}.asset"
    if representation == "temp":
        filename = f".{identifier}.asset.tmp-0123456789abcdef"
    path = root / filename
    atomic_write_private(path, ciphertext)
    with database.engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM asset_encryption_state")
        if representation == "orphan":
            connection.exec_driver_sql("DELETE FROM uploaded_assets")
    storage.close()
    database.engine.dispose()
    with (
        pytest.raises(AssetKeyError, match="^Asset encryption key unavailable$"),
        TestClient(create_app(**options)),
    ):
        pytest.fail("Must not serve")
    assert path.read_bytes() == ciphertext
    assert key_value(options) is None


@pytest.mark.parametrize("damage", ["length", "version", "wrong_key", "extra_row"])
def test_corrupt_key_never_regenerated_and_ciphertext_retained(tmp_path, damage):
    options, snapshots, _ = legacy_state(tmp_path)
    with TestClient(create_app(**options)):
        pass
    path = options["asset_directory"] / f"{snapshots[0]['id']}.asset"
    before = path.read_bytes()
    database = open_database(options)
    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA ignore_check_constraints=ON")
        if damage == "version":
            connection.exec_driver_sql("UPDATE asset_encryption_state SET format_version=2")
        elif damage == "extra_row":
            connection.exec_driver_sql(
                "INSERT INTO asset_encryption_state SELECT 2, format_version, key_material, "
                "created_at FROM asset_encryption_state"
            )
        else:
            connection.exec_driver_sql(
                "UPDATE asset_encryption_state SET key_material=?",
                (b"k" * (31 if damage == "length" else 32),),
            )
    database.engine.dispose()
    before_key = key_value(options)
    with (
        pytest.raises(AssetStorageError if damage == "wrong_key" else AssetKeyError),
        TestClient(create_app(**options)),
    ):
        pytest.fail("Must not serve")
    assert path.read_bytes() == before
    assert key_value(options) == before_key


@pytest.mark.parametrize("failure", ["write", "replace", "permissions"])
def test_io_and_permissions_failure_are_sanitized_and_preserve_plaintext_for_retry(
    tmp_path,
    monkeypatch,
    failure,
):
    options, snapshots, _ = legacy_state(tmp_path)
    path = options["asset_directory"] / f"{snapshots[0]['id']}.png"
    app = create_app(**options)
    with monkeypatch.context() as patch:

        def fail(*_args, **_kwargs):
            raise OSError("PRIVATE_FILE_PATH_CANARY")

        if failure == "write":
            patch.setattr(AssetStorage, "_atomic_ciphertext", fail)
        elif failure == "replace":
            patch.setattr("backend.asset_storage.os.replace", fail)
        else:
            patch.setattr("backend.asset_storage._assert_private", fail)
        with pytest.raises((AssetStorageError, AssetKeyError)) as caught, TestClient(app):
            pytest.fail("Must not serve")
        assert "PRIVATE_FILE_PATH_CANARY" not in str(caught.value)
    assert path.read_bytes() == PLAINTEXT
    assert not list(options["asset_directory"].glob("*.asset"))
    assert not list(options["asset_directory"].glob(".*.tmp-*"))
    with TestClient(create_app(**options)) as client:
        assert client.get(f"/api/assets/{snapshots[0]['id']}/content").content == PLAINTEXT


def test_startup_temp_cleanup_and_orphans_stay_flat_generated_and_ciphertext_only(tmp_path):
    options, snapshots, _ = legacy_state(tmp_path)
    root = options["asset_directory"]
    orphan = str(uuid4())
    atomic_write_private(root / f"{orphan}.png", PLAINTEXT)
    # Previous-version abandoned plaintext upload temp, never a committed file.
    atomic_write_private(root / f".{uuid4()}.png.tmp-0123456789abcdef", PLAINTEXT)
    unrelated = root / "operator-notes.txt"
    unrelated.write_text("keep this file", encoding="utf-8")
    nested = root / "unrelated-folder"
    nested.mkdir()
    nested_file = nested / f"{uuid4()}.png"
    nested_file.write_bytes(PLAINTEXT)
    with TestClient(create_app(**options)):
        pass
    assert not list(root.glob("*.png"))
    assert not list(root.glob(".*.tmp-*"))
    assert (root / f"{orphan}.asset").read_bytes().startswith(ASSET_HEADER)
    assert unrelated.read_text() == "keep this file"
    assert nested_file.read_bytes() == PLAINTEXT
    # Hourly/startup orphan TTL still applies to complete files; don't sweep fresh writes.
    old_orphan = root / f"{orphan}.asset"
    os.utime(old_orphan, (0, 0))
    with TestClient(create_app(**options)):
        pass
    assert not old_orphan.exists()
    assert (root / f"{snapshots[0]['id']}.asset").exists()
    assert unrelated.exists() and nested_file.exists()


def test_conflicting_representations_fail_without_removing_either(tmp_path):
    options, snapshots, _ = legacy_state(tmp_path)
    with TestClient(create_app(**options)):
        pass
    identifier = snapshots[0]["id"]
    root = options["asset_directory"]
    encrypted = (root / f"{identifier}.asset").read_bytes()
    atomic_write_private(root / f"{identifier}.png", PLAINTEXT)
    with pytest.raises(AssetStorageError), TestClient(create_app(**options)):
        pytest.fail("Must not serve conflicting state")
    assert (root / f"{identifier}.png").read_bytes() == PLAINTEXT
    assert (root / f"{identifier}.asset").read_bytes() == encrypted


def test_key_and_ciphertext_survive_database_and_asset_directory_relocation(tmp_path):
    options, snapshots, _ = legacy_state(tmp_path, encrypted=True)
    with TestClient(create_app(**options)):
        pass
    old_root = options["asset_directory"]
    identifier = snapshots[0]["id"]
    ciphertext = (old_root / f"{identifier}.asset").read_bytes()
    new_root = tmp_path / "moved-assets"
    old_root.rename(new_root)
    new_db = tmp_path / "moved.db"
    (tmp_path / "legacy.db").rename(new_db)
    options.update(database_url=f"sqlite:///{new_db}", asset_directory=new_root)
    with TestClient(create_app(**options)) as client:
        assert client.get(f"/api/assets/{identifier}/content").content == PLAINTEXT
    assert (new_root / f"{identifier}.asset").read_bytes() == ciphertext
    assert not ciphertext.startswith(PNG_SIGNATURE)
