import base64
import json
from argparse import Namespace
from pathlib import Path

import pytest

from backend.database import Database
from runtime.key_store import (
    ENVELOPE_PURPOSE,
    ENVELOPE_VERSION,
    KDF_NAME,
    ROTATION_RECOVERY_REQUIRED_MESSAGE,
    WRAP_ALGORITHM,
    Argon2Parameters,
    KeyStore,
    KeyStoreError,
    KeyStorePolicy,
    UnlockError,
    file_sha256,
)
from runtime.keyctl import _parser, command_init
from runtime.keyctl import main as keyctl_main
from runtime.paths import (
    PrivatePathError,
    assert_owner_only,
    atomic_write_private,
    default_runtime_token_file,
    validate_key_file_path,
)
from runtime.secure_memory import DatabaseKeyHandle, SecureMemoryError

PASSPHRASE = "correct horse battery staple"
NEW_PASSPHRASE = "new correct horse battery staple"
WRONG_PASSPHRASE = "wrong horse battery staple"
DATABASE_KEY = b"DATABASE_KEY_CANARY_1234567890!!"
TEST_POLICY = KeyStorePolicy.for_tests()


def _store(tmp_path: Path) -> KeyStore:
    return KeyStore(tmp_path / "private" / "database-key.json", policy=TEST_POLICY)


def _document(store: KeyStore) -> dict:
    return json.loads(store.key_file.read_text(encoding="utf-8"))


def _write_document(store: KeyStore, document: dict) -> None:
    atomic_write_private(
        store.key_file,
        (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode(),
    )


def test_production_policy_is_strong_and_test_policy_is_explicit() -> None:
    production = KeyStorePolicy()

    assert production.creation == Argon2Parameters(
        memory_cost_kib=65_536,
        time_cost=3,
        parallelism=1,
        hash_len=32,
    )
    assert TEST_POLICY.creation.memory_cost_kib == 64
    assert TEST_POLICY.creation.time_cost == 1


def test_initialize_creates_strict_authenticated_envelope_without_raw_key(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_key=DATABASE_KEY)

    document = _document(store)
    assert set(document) == {"version", "purpose", "kdf", "wrap"}
    assert document["version"] == ENVELOPE_VERSION
    assert document["purpose"] == ENVELOPE_PURPOSE
    assert document["kdf"]["name"] == KDF_NAME
    assert document["wrap"]["algorithm"] == WRAP_ALGORITHM
    raw = store.key_file.read_bytes()
    assert DATABASE_KEY not in raw
    assert DATABASE_KEY.hex().encode() not in raw
    assert not list(store.key_file.parent.glob(f".{store.key_file.name}.tmp-*"))

    with store.unlock(PASSPHRASE) as handle:
        assert handle.copy_bytes() == DATABASE_KEY


def test_wrong_passphrase_is_sanitized_and_does_not_mutate_files(
    tmp_path: Path,
) -> None:
    database = tmp_path / "existing.sqlite3"
    database.write_bytes(b"SQLite format 3\x00" + b"database sentinel")
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_path=database, database_key=DATABASE_KEY)
    envelope_before = file_sha256(store.key_file)
    database_before = file_sha256(database)

    with pytest.raises(UnlockError) as failure:
        store.unlock(WRONG_PASSPHRASE, database_path=database)

    assert str(failure.value) == "Unlock failed"
    assert PASSPHRASE not in str(failure.value)
    assert WRONG_PASSPHRASE not in str(failure.value)
    assert file_sha256(store.key_file) == envelope_before
    assert file_sha256(database) == database_before
    assert not list(store.key_file.parent.glob("*.tmp-*"))
    with store.unlock(PASSPHRASE) as handle:
        assert handle.copy_bytes() == DATABASE_KEY


def test_envelope_tampering_and_parser_abuse_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_key=DATABASE_KEY)
    original = _document(store)

    def changed_base64(value: str) -> str:
        decoded = bytearray(base64.b64decode(value))
        decoded[0] ^= 1
        return base64.b64encode(decoded).decode()

    mutations = (
        lambda item: item["wrap"].__setitem__(
            "ciphertext", changed_base64(item["wrap"]["ciphertext"])
        ),
        lambda item: item["wrap"].__setitem__(
            "nonce", changed_base64(item["wrap"]["nonce"])
        ),
        lambda item: item["kdf"].__setitem__(
            "salt", changed_base64(item["kdf"]["salt"])
        ),
        lambda item: item["kdf"].__setitem__("name", "pbkdf2"),
        lambda item: item["wrap"].__setitem__("algorithm", "aes-128-gcm"),
        lambda item: item.__setitem__("version", 999),
        lambda item: item["kdf"].__setitem__("memory_cost_kib", 2_000_000),
        lambda item: item["kdf"].__setitem__("time_cost", 1_000),
        lambda item: item["wrap"].__setitem__("ciphertext", "AA=="),
        lambda item: item.__setitem__("unexpected", "x" * 35_000),
    )

    for mutate in mutations:
        candidate = json.loads(json.dumps(original))
        mutate(candidate)
        _write_document(store, candidate)
        with pytest.raises(UnlockError, match="^Unlock failed$"):
            store.unlock(PASSPHRASE)

    atomic_write_private(store.key_file, b'{"version":1,"version":1}\n')
    with pytest.raises(UnlockError, match="^Unlock failed$"):
        store.unlock(PASSPHRASE)


def test_change_passphrase_rewraps_same_database_key(tmp_path: Path) -> None:
    database = tmp_path / "existing.sqlite3"
    database.write_bytes(b"SQLite format 3\x00" + b"unchanged database")
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_path=database, database_key=DATABASE_KEY)
    database_before = database.read_bytes()
    envelope_before = store.key_file.read_bytes()

    store.change_passphrase(PASSPHRASE, NEW_PASSPHRASE)

    assert database.read_bytes() == database_before
    assert store.key_file.read_bytes() != envelope_before
    with pytest.raises(UnlockError, match="^Unlock failed$"):
        store.unlock(PASSPHRASE)
    with store.unlock(NEW_PASSPHRASE) as handle:
        assert handle.copy_bytes() == DATABASE_KEY


def test_init_refuses_encrypted_existing_database_and_accepts_plaintext(
    tmp_path: Path,
) -> None:
    encrypted = tmp_path / "encrypted.sqlite3"
    encrypted.write_bytes(b"not a plaintext SQLite header")
    refused = _store(tmp_path / "refused")

    with pytest.raises(KeyStoreError, match="use import-existing"):
        refused.initialize(PASSPHRASE, database_path=encrypted)
    assert not refused.key_file.exists()

    plaintext = tmp_path / "plaintext.sqlite3"
    plaintext.write_bytes(b"SQLite format 3\x00" + b"plaintext")
    accepted = _store(tmp_path / "accepted")
    accepted.initialize(PASSPHRASE, database_path=plaintext)
    assert accepted.key_file.exists()
    assert plaintext.read_bytes().startswith(b"SQLite format 3\x00")


def test_import_existing_proves_key_and_does_not_modify_encrypted_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-encrypted.sqlite3"
    database = Database.from_url(
        f"sqlite+pysqlite:///{database_path.as_posix()}",
        encryption_key=DATABASE_KEY.hex(),
    )
    database.create_schema()
    database.engine.dispose()
    database_before = file_sha256(database_path)

    wrong_store = _store(tmp_path / "wrong")
    wrong_key = b"W" * 32
    with pytest.raises(KeyStoreError, match="could not open") as failure:
        wrong_store.import_existing(
            database_path,
            wrong_key.hex(),
            PASSPHRASE,
        )
    assert wrong_key.hex() not in str(failure.value)
    assert not wrong_store.key_file.exists()
    assert file_sha256(database_path) == database_before

    store = _store(tmp_path / "correct")
    store.import_existing(database_path, DATABASE_KEY.hex(), PASSPHRASE)

    assert file_sha256(database_path) == database_before
    with store.unlock(PASSPHRASE) as handle:
        assert handle.copy_bytes() == DATABASE_KEY
    assert DATABASE_KEY not in store.key_file.read_bytes()
    assert DATABASE_KEY.hex().encode() not in store.key_file.read_bytes()


def test_key_file_location_rejects_repository_and_windows_onedrive(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    with pytest.raises(PrivatePathError, match="outside"):
        validate_key_file_path(
            repository / "secret.json",
            repository_root=repository,
            platform="linux",
        )

    one_drive = tmp_path / "OneDrive"
    with pytest.raises(PrivatePathError, match="OneDrive"):
        validate_key_file_path(
            one_drive / "secret.json",
            repository_root=repository,
            platform="win32",
            environ={"OneDrive": str(one_drive)},
        )

    with pytest.raises(PrivatePathError, match="must be absolute"):
        default_runtime_token_file(
            environ={"AMITAI_LOCAL_API_TOKEN_FILE": "relative/token"},
            platform="linux",
        )


def test_private_envelope_permissions_are_owner_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_key=DATABASE_KEY)

    assert_owner_only(store.key_file.parent, directory=True)
    assert_owner_only(store.key_file, directory=False)


def test_database_key_handle_hides_zeroizes_and_rejects_use_after_close() -> None:
    handle = DatabaseKeyHandle(DATABASE_KEY)
    underlying_buffer = handle._buffer

    assert handle.locked is True
    assert DATABASE_KEY.hex() not in repr(handle)
    assert DATABASE_KEY.hex() not in str(handle)
    handle.close()

    assert bytes(underlying_buffer) == b"\x00" * 32
    assert handle.closed is True
    with pytest.raises(SecureMemoryError, match="closed"):
        handle.copy_bytes()
    with pytest.raises(SecureMemoryError, match="closed"), handle.temporary_hex():
        pass


def test_keyctl_has_no_secret_arguments_and_prompts_for_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--passphrase" not in option_strings
    assert "--database-key" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--passphrase", "not-accepted"])

    prompts: list[str] = []
    answers = iter((PASSPHRASE, PASSPHRASE))
    arguments = Namespace(
        key_file=tmp_path / "private" / "database-key.json",
        database_file=tmp_path / "amitai.db",
    )
    monkeypatch.setattr(
        "runtime.keyctl.KeyStore",
        lambda path: KeyStore(path, policy=TEST_POLICY),
    )

    def prompt(label: str) -> str:
        prompts.append(label)
        return next(answers)

    command_init(arguments, prompt=prompt)

    assert prompts == [
        "New unlock passphrase: ",
        "Confirm new unlock passphrase: ",
    ]
    assert PASSPHRASE not in "".join(prompts)


def test_keyctl_change_passphrase_refuses_recovery_before_prompt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_key=DATABASE_KEY)
    atomic_write_private(store.rotation_file, b"pending authenticated recovery state\n")
    prompts: list[str] = []

    def prompt(label: str) -> str:
        prompts.append(label)
        raise AssertionError("change-passphrase must refuse before prompting")

    with pytest.raises(SystemExit, match=f"^{ROTATION_RECOVERY_REQUIRED_MESSAGE}$"):
        keyctl_main(
            ["change-passphrase", "--key-file", str(store.key_file)],
            prompt=prompt,
        )

    assert prompts == []
    assert store.rotation_file.exists()
