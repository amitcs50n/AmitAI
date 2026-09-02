"""Real SQLCipher + envelope + synthetic encrypted-image backup/recovery tests."""

import base64
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.asset_crypto import ASSET_HEADER, encrypt_asset
from backend.asset_storage import database_asset_namespace
from backend.database import SQLITE_HEADER, _fsync_file, _load_sqlcipher_driver
from runtime import backup
from runtime.backup_archive import DATABASE, ENVELOPE, FIXED_DATE, MANIFEST, validated_archive
from runtime.backup_files import new_private_file
from runtime.key_store import KeyStore, KeyStoreError, KeyStorePolicy
from runtime.paths import assert_owner_only, atomic_write_private, restore_journal_path
from tests.test_assets import upload

PASSPHRASE = "BACKUP_PASSPHRASE_CANARY_427381"
KEY = b"BACKUP_RAW_DB_KEY_CANARY".ljust(32, b"!")
MESSAGE = "BACKUP_PRIVATE_MESSAGE_128937"
MEMORY = "BACKUP_PRIVATE_MEMORY_647382"
LOCAL_TOKEN = "BACKUP_LOCAL_TOKEN_CANARY_91827364"
REMOTE_TOKEN = "BACKUP_REMOTE_TOKEN_CANARY_73642812"
POLICY = KeyStorePolicy.for_tests()


@pytest.mark.parametrize("ending", [b"\x1a", b"\x1a\x1a", b"\r\n", b"\x00"])
def test_snapshot_fsync_preserves_binary_trailing_ctrl_z(tmp_path, ending):
    path = tmp_path / "ciphertext.bin"
    original = b"synthetic ciphertext with arbitrary binary ending" + ending
    path.write_bytes(original)
    _fsync_file(path)
    assert path.read_bytes() == original


class SimulatedCrash(BaseException):
    pass


def prompt(_label):
    return PASSPHRASE


@pytest.fixture
def installation(tmp_path, monkeypatch):
    asset_base = tmp_path / "local-data" / "assets"
    directory = lambda namespace: asset_base / namespace
    monkeypatch.setattr("backend.app.default_asset_directory", directory)
    monkeypatch.setattr(backup, "default_asset_directory", directory)
    database = tmp_path / "old-machine" / "amitai.db"
    database.parent.mkdir()
    store = KeyStore(database.parent / "secrets" / "database-key.json", policy=POLICY)
    store.initialize(PASSPHRASE, database_key=KEY)
    with store.unlock(PASSPHRASE) as handle:
        app = create_app(
            f"sqlite:///{database.as_posix()}", database_key=handle, enforce_local_auth=False
        )
        with TestClient(app) as client:
            chat = client.post("/api/chat", json={"message": MESSAGE}).json()
            conversation_id = chat["conversation_id"]
            memory = client.post(
                "/api/memory",
                json={
                    "category": "project",
                    "key": "backup.canary",
                    "value": MEMORY,
                },
            ).json()
            images = [
                upload(
                    client,
                    filename="PRIVATE_CLIENT_FILENAME.png",
                    persistence_mode="conversation",
                    conversation_id=conversation_id,
                ).json()
                for _ in range(2)
            ]
            attached = client.post(
                "/api/chat",
                json={
                    "conversation_id": conversation_id,
                    "message": "Attached locally",
                    "asset_ids": [images[0]["id"]],
                },
            )
            assert attached.status_code == 200
            history = client.get(f"/api/conversations/{conversation_id}").json()
            normalized = client.get(f"/api/assets/{images[0]['id']}/content").content
            assets = app.state.asset_storage.root
            with app.state.database.engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA user_version=17")
            raw = app.state.database.engine.raw_connection()
            try:
                aek = raw.execute("SELECT key_material FROM asset_encryption_state").fetchone()[0]
            finally:
                raw.close()
    output = tmp_path / "separate-drive" / "aevon.amitai-backup"
    target = tmp_path / "new-machine" / "different-location" / "restored.db"
    target_store = KeyStore(tmp_path / "new-private" / "database-key.json", policy=POLICY)
    return SimpleNamespace(
        database=database,
        store=store,
        assets=assets,
        images=images,
        history=history,
        memory=memory,
        normalized=normalized,
        aek=aek,
        output=output,
        target=target,
        target_store=target_store,
        target_assets=directory(database_asset_namespace(target)),
    )


def create(state, **kwargs):
    return backup.create_backup(state.output, state.database, state.store, prompt=prompt, **kwargs)


def restore(state, **kwargs):
    return backup.restore_backup(
        state.output, state.target, state.target_store, prompt=prompt, **kwargs
    )


def payloads(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(path, values):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, content in values.items():
            info = zipfile.ZipInfo(name, FIXED_DATE)
            info.create_system, info.external_attr = 0, 0x20
            archive.writestr(info, content)


def refresh_hash(values, name):
    manifest = json.loads(values[MANIFEST])
    item = (
        manifest["database"]
        if name == DATABASE
        else manifest["key_envelope"]
        if name == ENVELOPE
        else next(a for a in manifest["assets"] if a["id"] == name[7:-6])
    )
    item.update(size=len(values[name]), sha256=hashlib.sha256(values[name]).hexdigest())
    values[MANIFEST] = json.dumps(manifest).encode()


def assert_not_installed(state):
    assert not state.target.exists()
    assert not state.target_store.key_file.exists()
    assert not state.target_assets.exists() or not list(state.target_assets.iterdir())
    assert not restore_journal_path(state.target_store.key_file).exists()


def assert_restored(state):
    assert state.target_assets != state.assets
    expected = hashlib.sha256(str(state.target.resolve()).encode()).hexdigest()[:24]
    assert state.target_assets.name == expected
    with state.target_store.unlock(PASSPHRASE) as key:
        assert key.copy_bytes() == KEY
        app = create_app(
            f"sqlite:///{state.target.as_posix()}",
            database_key=key,
            local_api_token=LOCAL_TOKEN,
            enforce_local_auth=True,
        )
        with TestClient(app) as client:
            assert client.get("/api/conversations").status_code == 401
            client.headers["Authorization"] = f"Bearer {LOCAL_TOKEN}"
            assert client.get(f"/api/conversations/{state.history['id']}").json() == state.history
            assert {**state.memory, "operation": "current"} in client.get("/api/memory").json()
            for image in state.images:
                assert client.get(f"/api/assets/{image['id']}").json() == image
                assert client.get(f"/api/assets/{image['id']}/content").content == state.normalized
            assert app.state.asset_storage.root == state.target_assets
            with app.state.database.engine.connect() as connection:
                assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 17
    with sqlite3.connect(state.target) as plain, pytest.raises(sqlite3.DatabaseError):
        plain.execute("SELECT * FROM sqlite_master").fetchall()


def test_complete_encrypted_roundtrip_without_original_installation(
    installation, monkeypatch, caplog
):
    state = installation
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", LOCAL_TOKEN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("AMITAI_DB_KEY", KEY.hex())
    monkeypatch.setenv("AMITAI_UNLOCK_PASSPHRASE", "ENV_PASSPHRASE_MUST_NOT_BE_USED")
    new_private_file(state.store.key_file.parent / "local-api-token", LOCAL_TOKEN.encode())
    orphan = str(uuid4())
    new_private_file(
        state.assets / f"{orphan}.asset", encrypt_asset(orphan, state.normalized, state.aek)
    )
    original_key = state.store.key_file.read_bytes()
    with caplog.at_level(logging.DEBUG):
        backup_id = create(state)
        values = payloads(state.output)
        assert set(values) == {
            MANIFEST,
            DATABASE,
            ENVELOPE,
            *(f"assets/{a['id']}.asset" for a in state.images),
        }
        assert values[ENVELOPE] == original_key
        assert not values[DATABASE].startswith(SQLITE_HEADER)
        for image in state.images:
            name = f"assets/{image['id']}.asset"
            assert values[name] == (state.assets / Path(name).name).read_bytes()
            assert values[name].startswith(ASSET_HEADER)
        with zipfile.ZipFile(state.output) as archive:
            assert not archive.comment
            for info in archive.infolist():
                assert info.compress_type == zipfile.ZIP_STORED and info.date_time == FIXED_DATE
                assert info.create_system == 0 and info.external_attr == 0x20
                assert not info.extra and not info.comment
        assert_owner_only(state.output, directory=False)
        raw = state.output.read_bytes()
        for secret in (
            PASSPHRASE.encode(),
            KEY,
            KEY.hex().encode(),
            state.aek,
            state.normalized,
            MESSAGE.encode(),
            MEMORY.encode(),
            LOCAL_TOKEN.encode(),
            REMOTE_TOKEN.encode(),
            str(state.database).encode(),
            str(state.assets).encode(),
            str(state.store.key_file).encode(),
            b"PRIVATE_CLIENT_FILENAME",
        ):
            assert secret not in raw
        # A separate machine cannot fall back to any old DB, envelope or asset.
        state.database.rename(state.database.with_suffix(".unavailable"))
        state.store.key_file.rename(state.store.key_file.with_suffix(".unavailable"))
        state.assets.rename(state.assets.with_name(state.assets.name + "-unavailable"))
        assert restore(state) == backup_id
        assert_restored(state)
    for value in (PASSPHRASE, KEY.hex(), MESSAGE, MEMORY, LOCAL_TOKEN, REMOTE_TOKEN):
        # The application may log SQL queries if tests explicitly enable DEBUG;
        # backup code itself uses raw DBAPI for key and verification row reads.
        assert value not in caplog.text


@pytest.mark.parametrize("part", [DATABASE, ENVELOPE, "asset"])
@pytest.mark.parametrize("rehash", [False, True])
def test_ciphertext_tampering_fails_even_with_recomputed_manifest(installation, part, rehash):
    state = installation
    create(state)
    values = payloads(state.output)
    name = f"assets/{state.images[0]['id']}.asset" if part == "asset" else part
    if name == ENVELOPE:
        envelope = json.loads(values[name])
        changed = bytearray(base64.b64decode(envelope["wrap"]["ciphertext"]))
        changed[0] ^= 1
        envelope["wrap"]["ciphertext"] = base64.b64encode(changed).decode()
        values[name] = json.dumps(envelope).encode()
    else:
        changed = bytearray(values[name])
        changed[len(changed) // 2] ^= 1
        values[name] = bytes(changed)
    if rehash:
        refresh_hash(values, name)
    rewrite(state.output, values)
    with pytest.raises(backup.BackupError):
        restore(state)
    assert_not_installed(state)


@pytest.mark.parametrize("remove_manifest_entry", [False, True])
def test_missing_archive_asset_cannot_be_hidden_by_manifest(installation, remove_manifest_entry):
    state = installation
    create(state)
    values = payloads(state.output)
    del values[f"assets/{state.images[0]['id']}.asset"]
    if remove_manifest_entry:
        manifest = json.loads(values[MANIFEST])
        manifest["assets"] = [a for a in manifest["assets"] if a["id"] != state.images[0]["id"]]
        manifest["asset_count"] -= 1
        values[MANIFEST] = json.dumps(manifest).encode()
    rewrite(state.output, values)
    with pytest.raises(backup.BackupError):
        restore(state)
    assert_not_installed(state)


def test_wrong_passphrase_no_valid_installation_and_no_private_error(installation, caplog):
    state = installation
    create(state)
    with pytest.raises(backup.BackupError) as error:
        backup.restore_backup(
            state.output,
            state.target,
            state.target_store,
            prompt=lambda _: "WRONG_PRIVATE_PASSPHRASE",
        )
    assert "WRONG_PRIVATE_PASSPHRASE" not in str(error.value) + caplog.text
    assert_not_installed(state)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "legacy", "missing-key"])
def test_create_requires_every_db_asset_and_existing_aek(installation, damage):
    state = installation
    path = state.assets / f"{state.images[0]['id']}.asset"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"PRIVATE_CORRUPT_ASSET")
    elif damage == "legacy":
        path.rename(path.with_suffix(".png"))
    else:
        connection = cipher_connection(state.database)
        connection.execute("DELETE FROM asset_encryption_state")
        connection.commit()
        connection.close()
    original = state.database.read_bytes(), state.store.key_file.read_bytes()
    with pytest.raises(backup.BackupError):
        create(state)
    assert not state.output.exists()
    assert not list(state.output.parent.glob(".amitai-backup-*"))
    assert (state.database.read_bytes(), state.store.key_file.read_bytes()) == original


def test_pending_rotation_refused_before_prompt_or_snapshot(installation):
    state = installation
    atomic_write_private(state.store.rotation_file, b"PRIVATE_ROTATION_STATE")
    with pytest.raises(backup.BackupError, match="rotation"):
        backup.create_backup(
            state.output,
            state.database,
            state.store,
            prompt=lambda _: pytest.fail("Prompted during pending rotation"),
        )
    assert not state.output.exists()


@pytest.mark.parametrize("existing", ["database", "key", "assets", "sidecar"])
def test_restore_never_overwrites_existing_installation(installation, existing):
    state = installation
    create(state)
    path = {
        "database": state.target,
        "key": state.target_store.key_file,
        "assets": state.target_assets / f"{uuid4()}.asset",
        "sidecar": Path(f"{state.target}-wal"),
    }[existing]
    new_private_file(path, b"EXISTING_TARGET_CANARY")
    with pytest.raises(backup.BackupError):
        backup.restore_backup(
            state.output,
            state.target,
            state.target_store,
            prompt=lambda _: pytest.fail("Prompted before target refusal"),
        )
    assert path.read_bytes() == b"EXISTING_TARGET_CANARY"
    assert not restore_journal_path(state.target_store.key_file).exists()


def test_existing_backup_not_overwritten_and_failure_before_commit_is_clean(installation):
    state = installation
    create(state)
    before = state.output.read_bytes()
    with pytest.raises(backup.BackupError, match="already exists"):
        create(state)
    assert state.output.read_bytes() == before

    def fail(phase):
        if phase == "restore_verified":
            raise OSError("PRIVATE_FAILURE " + PASSPHRASE)

    with pytest.raises(backup.BackupError) as error:
        restore(state, phase_hook=fail)
    assert PASSPHRASE not in str(error.value)
    assert_not_installed(state)
    assert not list(state.target_store.key_file.parent.glob(".amitai-backup-*"))


@pytest.mark.parametrize(
    "phase", ["restore_marked", "assets_installed", "database_installed", "key_installed"]
)
def test_interrupted_restore_key_last_blocks_unlock_and_resumes(installation, phase):
    state = installation
    create(state)

    def crash(current):
        if current == phase:
            raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        restore(state, phase_hook=crash)
    marker = restore_journal_path(state.target_store.key_file)
    assert set(json.loads(marker.read_bytes())) == {"backup_id", "phase"}
    assert state.target_store.key_file.exists() is (phase == "key_installed")
    with pytest.raises(KeyStoreError, match="Restore recovery required"):
        state.target_store.unlock(PASSPHRASE)
    with pytest.raises(backup.BackupError, match="Interrupted restore"):
        restore(state)
    restore(state, resume=True)
    assert not marker.exists()
    assert_restored(state)


def test_resume_refuses_mismatched_destination_without_deleting_it(installation):
    state = installation
    create(state)

    def crash(phase):
        if phase == "database_installed":
            raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        restore(state, phase_hook=crash)
    state.target.write_bytes(b"CHANGED_DESTINATION_CANARY")
    with pytest.raises(backup.BackupError, match="do not match"):
        restore(state, resume=True)
    assert state.target.read_bytes() == b"CHANGED_DESTINATION_CANARY"
    assert not state.target_store.key_file.exists()


def cipher_connection(path):
    connection = _load_sqlcipher_driver().connect(str(path), timeout=0)
    connection.execute(f"PRAGMA key = \"x'{KEY.hex()}'\"")
    return connection


@pytest.mark.parametrize("_repeat", range(10))
def test_snapshot_captures_committed_wal_and_excludes_later_writes(installation, tmp_path, _repeat):
    state = installation
    connection = cipher_connection(state.database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE snapshot_test (value TEXT)")
        connection.execute("INSERT INTO snapshot_test VALUES ('COMMITTED_WAL_CANARY')")
        connection.commit()
        assert Path(f"{state.database}-wal").stat().st_size > 0

        def observe(phase):
            if phase == "snapshot_exported":
                with pytest.raises(_load_sqlcipher_driver().OperationalError, match="locked"):
                    connection.execute("BEGIN IMMEDIATE")
            if phase == "snapshot_verified":
                [stage] = state.output.parent.glob(".amitai-backup-*")
                snapshot_hash = hashlib.sha256((stage / DATABASE).read_bytes()).digest()
                connection.execute("INSERT INTO snapshot_test VALUES ('LATER_WRITE_CANARY')")
                connection.commit()
                assert hashlib.sha256((stage / DATABASE).read_bytes()).digest() == snapshot_hash

        create(state, phase_hook=observe)
        snapshot = tmp_path / "snapshot-for-test.db"
        snapshot.write_bytes(payloads(state.output)[DATABASE])
        verify = cipher_connection(snapshot)
        try:
            assert verify.execute("SELECT * FROM snapshot_test").fetchall() == [
                ("COMMITTED_WAL_CANARY",)
            ]
            assert verify.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert verify.execute("PRAGMA cipher_integrity_check").fetchall() == []
            assert verify.execute("PRAGMA user_version").fetchone() == (17,)
        finally:
            verify.close()
        assert connection.execute("SELECT count(*) FROM snapshot_test").fetchone() == (2,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        connection.close()


def test_key_envelope_change_during_backup_fails_without_publishing(installation):
    state = installation

    def change(phase):
        if phase == "snapshot_verified":
            state.store.change_passphrase(PASSPHRASE, "Replacement unlock passphrase")

    with pytest.raises(backup.BackupError, match="Key envelope changed"):
        create(state, phase_hook=change)
    assert not state.output.exists()


def test_cli_defaults_explicit_output_getpass_and_no_secret_arguments(
    installation, monkeypatch, capsys
):
    state = installation
    monkeypatch.setattr(backup, "default_key_file", lambda: state.store.key_file)
    monkeypatch.setattr(
        backup, "KeyStore", lambda path, **kwargs: KeyStore(path, **{"policy": POLICY, **kwargs})
    )
    monkeypatch.chdir(state.database.parent)
    prompts = []
    backup.main(
        ["create", "--output", str(state.output)],
        prompt=lambda label: prompts.append(label) or PASSPHRASE,
    )
    assert len(prompts) == 1
    assert capsys.readouterr().out == "Encrypted local backup created.\n"
    for args in (
        ["create"],
        ["restore"],
        ["create", "--output", str(state.output), "--passphrase", PASSPHRASE],
        ["restore", "--input", str(state.output), "--force"],
    ):
        with pytest.raises(SystemExit):
            backup._parser().parse_args(args)
    assert PASSPHRASE not in capsys.readouterr().err
    with validated_archive(state.output):
        pass


def test_cli_restore_uses_destination_overrides_and_hidden_prompt(
    installation, monkeypatch, capsys
):
    state = installation
    create(state)
    monkeypatch.setattr(
        backup, "KeyStore", lambda path, **kwargs: KeyStore(path, **{"policy": POLICY, **kwargs})
    )
    prompts = []
    backup.main(
        [
            "restore",
            "--input",
            str(state.output),
            "--database-file",
            str(state.target),
            "--key-file",
            str(state.target_store.key_file),
        ],
        prompt=lambda label: prompts.append(label) or PASSPHRASE,
    )
    assert len(prompts) == 1
    assert capsys.readouterr().out == "Encrypted local backup restored.\n"
    assert_restored(state)


def test_zero_assets_roundtrip_keeps_existing_asset_key(installation):
    state = installation
    connection = cipher_connection(state.database)
    connection.execute("DELETE FROM message_assets")
    connection.execute("DELETE FROM uploaded_assets")
    connection.commit()
    connection.close()
    create(state)
    assert json.loads(payloads(state.output)[MANIFEST])["asset_count"] == 0
    restore(state)
    verify = cipher_connection(state.target)
    try:
        assert verify.execute("SELECT key_material FROM asset_encryption_state").fetchone() == (
            state.aek,
        )
        assert verify.execute("SELECT count(*) FROM uploaded_assets").fetchone() == (0,)
    finally:
        verify.close()
    assert not list(state.target_assets.iterdir())


def test_restore_checks_partial_asset_install_and_resumes(installation, monkeypatch):
    state = installation
    create(state)
    original = backup._install
    installed = []

    def crash(source, destination, entry, *, resume):
        original(source, destination, entry, resume=resume)
        installed.append(destination)
        raise SimulatedCrash

    with monkeypatch.context() as patch:
        patch.setattr(backup, "_install", crash)
        with pytest.raises(SimulatedCrash):
            restore(state)
    assert len(installed) == 1 and installed[0].suffix == ".asset"
    assert not state.target.exists() and not state.target_store.key_file.exists()
    restore(state, resume=True)
    assert_restored(state)


def test_real_process_exit_leaves_detectable_resumable_restore(installation):
    state = installation
    create(state)
    script = """
import os
import sys
from pathlib import Path
from runtime import backup
from runtime.key_store import KeyStore, KeyStorePolicy
from tests.test_backup import PASSPHRASE

backup.default_asset_directory = lambda _: Path(sys.argv[4])
def crash(phase):
    if phase == "database_installed":
        os._exit(71)
backup.restore_backup(Path(sys.argv[1]), Path(sys.argv[2]),
    KeyStore(Path(sys.argv[3]), policy=KeyStorePolicy.for_tests()),
    prompt=lambda _: PASSPHRASE, phase_hook=crash)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(state.output),
            str(state.target),
            str(state.target_store.key_file),
            str(state.target_assets),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 71, result.stderr
    assert not result.stdout and PASSPHRASE not in result.stderr
    assert state.target.exists() and not state.target_store.key_file.exists()
    assert list(state.target_store.key_file.parent.glob(".amitai-backup-*"))
    with pytest.raises(KeyStoreError, match="Restore recovery required"):
        state.target_store.unlock(PASSPHRASE)
    restore(state, resume=True)
    assert_restored(state)


@pytest.mark.parametrize(
    "operation", ["initialize", "import", "change-passphrase", "rotate", "recover"]
)
def test_pending_restore_blocks_key_mutations(installation, operation):
    state = installation
    new_private_file(restore_journal_path(state.store.key_file), b'{"phase":"installing"}')
    before = state.database.read_bytes(), state.store.key_file.read_bytes()
    calls = {
        "initialize": lambda: state.store.initialize(PASSPHRASE),
        "import": lambda: state.store.import_existing(state.database, KEY.hex(), PASSPHRASE),
        "change-passphrase": lambda: state.store.change_passphrase(
            PASSPHRASE, "Replacement passphrase"
        ),
        "rotate": lambda: state.store.rotate_database_key(state.database, PASSPHRASE),
        "recover": lambda: state.store.recover_rotation(state.database, PASSPHRASE),
    }
    with pytest.raises(KeyStoreError, match="Restore recovery required"):
        calls[operation]()
    assert (state.database.read_bytes(), state.store.key_file.read_bytes()) == before


def test_generated_staging_contains_no_plaintext_or_secret_exports(installation):
    state = installation
    observed = []

    def inspect(phase):
        if phase != "archive_verified":
            return
        for directory in state.output.parent.glob(".amitai-backup-*"):
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                observed.append(path.name)
                data = path.read_bytes()
                for private in (
                    state.normalized,
                    state.aek,
                    KEY,
                    PASSPHRASE.encode(),
                    MESSAGE.encode(),
                    MEMORY.encode(),
                    LOCAL_TOKEN.encode(),
                ):
                    assert private not in data

    create(state, phase_hook=inspect)
    assert {DATABASE, ENVELOPE, "archive.bin"}.issubset(observed)
    assert not list(state.output.parent.glob(".amitai-backup-*"))


def test_overlapping_destination_paths_fail_before_creating_directories(installation):
    state = installation
    create(state)
    # A DB file cannot also be the parent directory of the key file.
    store = KeyStore(state.target / "secrets" / "key.json", policy=POLICY)
    with pytest.raises(backup.BackupError, match="must not overlap"):
        backup.restore_backup(state.output, state.target, store, prompt=prompt)
    assert not state.target.exists()


def test_new_target_appearing_before_publication_is_not_overwritten(installation):
    state = installation
    create(state)

    def race(phase):
        if phase == "restore_verified":
            state.target.write_bytes(b"CONCURRENT_INSTALLATION_CANARY")

    with pytest.raises(backup.BackupError, match="already exists"):
        restore(state, phase_hook=race)
    assert state.target.read_bytes() == b"CONCURRENT_INSTALLATION_CANARY"
    assert not state.target_store.key_file.exists()
    assert not restore_journal_path(state.target_store.key_file).exists()
