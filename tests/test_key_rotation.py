import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.database as database_module
from backend.app import create_app
from backend.database import SQLITE_HEADER, database_key_opens
from runtime.key_store import (
    ROTATION_RECOVERY_REQUIRED_MESSAGE,
    KeyRotationError,
    KeyStore,
    KeyStorePolicy,
    UnlockError,
    file_sha256,
)
from runtime.paths import rotation_candidate_path

PASSPHRASE = "rotation passphrase for tests"
NEW_PASSPHRASE = "new rotation passphrase for tests"
OLD_KEY = b"OLD_DB_KEY_CANARY_12345678901234"
NEW_KEY = b"NEW_DB_KEY_CANARY_98765432109876"
MESSAGE_CANARY = "ROTATION_MESSAGE_CANARY_918273"
MEMORY_CANARY = "ROTATION_MEMORY_CANARY_817263"
TITLE_CANARY = "Rotation preserved title"
TEST_POLICY = KeyStorePolicy.for_tests()


class SimulatedCrash(BaseException):
    pass


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _store(tmp_path: Path) -> KeyStore:
    return KeyStore(tmp_path / "private" / "database-key.json", policy=TEST_POLICY)


def _initialize_database(tmp_path: Path) -> tuple[KeyStore, Path, str, str]:
    database_path = tmp_path / "amitai.db"
    store = _store(tmp_path)
    store.initialize(PASSPHRASE, database_key=OLD_KEY)
    with store.unlock(PASSPHRASE) as handle:
        application = create_app(
            _database_url(database_path),
            database_key=handle,
            enforce_local_auth=False,
        )
        with TestClient(application) as client:
            chat = client.post("/api/chat", json={"message": MESSAGE_CANARY})
            assert chat.status_code == 200
            conversation_id = chat.json()["conversation_id"]
            renamed = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"title": TITLE_CANARY},
            )
            assert renamed.status_code == 200
            memory = client.post(
                "/api/memory",
                json={
                    "category": "project",
                    "key": "rotation.canary",
                    "value": MEMORY_CANARY,
                },
            )
            assert memory.status_code == 201
            memory_id = memory.json()["id"]
            with application.state.database.engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA user_version=7")
    return store, database_path, conversation_id, memory_id


def _assert_preserved(
    store: KeyStore,
    database_path: Path,
    conversation_id: str,
    memory_id: str,
    *,
    passphrase: str = PASSPHRASE,
) -> None:
    with store.unlock(passphrase, database_path=database_path) as handle:
        application = create_app(
            _database_url(database_path),
            database_key=handle,
            enforce_local_auth=False,
        )
        with TestClient(application) as client:
            conversation = client.get(f"/api/conversations/{conversation_id}")
            memories = client.get("/api/memory")
            assert conversation.status_code == 200
            assert conversation.json()["title"] == TITLE_CANARY
            assert conversation.json()["messages"][0]["content"] == MESSAGE_CANARY
            assert any(
                item["id"] == memory_id and item["value"] == MEMORY_CANARY
                for item in memories.json()
            )
            with application.state.database.engine.connect() as connection:
                assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 7


def _assert_no_raw_keys(path: Path) -> None:
    raw = path.read_bytes()
    for key in (OLD_KEY, NEW_KEY):
        assert key not in raw
        assert key.hex().encode() not in raw


def _temporary_key_store_artifacts(store: KeyStore) -> list[Path]:
    return list(store.key_file.parent.glob(".*.tmp-*"))


def test_database_key_rotation_preserves_data_and_rejects_old_key(
    tmp_path: Path,
) -> None:
    store, database_path, conversation_id, memory_id = _initialize_database(tmp_path)
    database_before = file_sha256(database_path)

    store.rotate_database_key(
        database_path,
        PASSPHRASE,
        new_database_key=NEW_KEY,
    )

    assert file_sha256(database_path) != database_before
    assert database_path.read_bytes()[: len(SQLITE_HEADER)] != SQLITE_HEADER
    assert MESSAGE_CANARY.encode() not in database_path.read_bytes()
    assert database_key_opens(database_path, NEW_KEY.hex()) is True
    assert database_key_opens(database_path, OLD_KEY.hex()) is False
    assert not store.rotation_file.exists()
    assert not rotation_candidate_path(database_path).exists()
    _assert_no_raw_keys(store.key_file)
    _assert_preserved(store, database_path, conversation_id, memory_id)


@pytest.mark.parametrize(
    "phase",
    (
        "after_journal_written",
        "after_candidate_created",
        "after_candidate_verified",
        "after_database_replaced",
        "before_envelope_finalization",
        "after_envelope_finalized",
    ),
)
def test_rotation_recovers_deterministically_from_every_crash_phase(
    tmp_path: Path,
    phase: str,
) -> None:
    store, database_path, conversation_id, memory_id = _initialize_database(tmp_path)

    def crash_hook(current_phase: str) -> None:
        if current_phase == phase:
            raise SimulatedCrash(phase)

    with pytest.raises(SimulatedCrash, match=phase):
        store.rotate_database_key(
            database_path,
            PASSPHRASE,
            new_database_key=NEW_KEY,
            phase_hook=crash_hook,
        )

    assert store.rotation_file.exists()
    assert database_path.read_bytes()[: len(SQLITE_HEADER)] != SQLITE_HEADER
    _assert_no_raw_keys(store.rotation_file)
    candidate = rotation_candidate_path(database_path)
    if candidate.exists():
        candidate_raw = candidate.read_bytes()
        assert candidate_raw[: len(SQLITE_HEADER)] != SQLITE_HEADER
        assert MESSAGE_CANARY.encode() not in candidate_raw
        for key in (OLD_KEY, NEW_KEY):
            assert key not in candidate_raw
            assert key.hex().encode() not in candidate_raw
    opens_before_recovery = (
        database_key_opens(database_path, OLD_KEY.hex()),
        database_key_opens(database_path, NEW_KEY.hex()),
    )
    assert opens_before_recovery in {(True, False), (False, True)}

    store.recover_rotation(database_path, PASSPHRASE)

    assert not store.rotation_file.exists()
    assert not rotation_candidate_path(database_path).exists()
    with store.unlock(PASSPHRASE) as recovered:
        assert recovered.copy_bytes() in {OLD_KEY, NEW_KEY}
    _assert_no_raw_keys(store.key_file)
    _assert_preserved(store, database_path, conversation_id, memory_id)


def test_passphrase_change_refuses_pending_post_replacement_rotation_then_recovers(
    tmp_path: Path,
) -> None:
    store, database_path, conversation_id, memory_id = _initialize_database(tmp_path)
    stable_envelope_before_rotation = file_sha256(store.key_file)

    def crash_after_replacement(phase: str) -> None:
        if phase == "after_database_replaced":
            raise SimulatedCrash(phase)

    with pytest.raises(SimulatedCrash, match="after_database_replaced"):
        store.rotate_database_key(
            database_path,
            PASSPHRASE,
            new_database_key=NEW_KEY,
            phase_hook=crash_after_replacement,
        )

    assert store.rotation_recovery_required is True
    assert file_sha256(store.key_file) == stable_envelope_before_rotation
    assert database_path.read_bytes()[: len(SQLITE_HEADER)] != SQLITE_HEADER
    assert (
        database_key_opens(database_path, OLD_KEY.hex()),
        database_key_opens(database_path, NEW_KEY.hex()),
    ) == (False, True)
    database_before_refusal = file_sha256(database_path)
    envelope_before_refusal = file_sha256(store.key_file)
    journal_before_refusal = file_sha256(store.rotation_file)

    with pytest.raises(
        KeyRotationError,
        match=f"^{ROTATION_RECOVERY_REQUIRED_MESSAGE}$",
    ):
        store.change_passphrase(PASSPHRASE, NEW_PASSPHRASE)

    assert file_sha256(database_path) == database_before_refusal
    assert file_sha256(store.key_file) == envelope_before_refusal
    assert file_sha256(store.rotation_file) == journal_before_refusal
    assert _temporary_key_store_artifacts(store) == []

    store.recover_rotation(database_path, PASSPHRASE)

    assert store.rotation_recovery_required is False
    assert not store.rotation_file.exists()
    assert file_sha256(database_path) == database_before_refusal
    assert database_key_opens(database_path, OLD_KEY.hex()) is False
    assert database_key_opens(database_path, NEW_KEY.hex()) is True
    with store.unlock(PASSPHRASE) as recovered:
        assert recovered.copy_bytes() == NEW_KEY
    _assert_preserved(store, database_path, conversation_id, memory_id)

    database_before_passphrase_change = file_sha256(database_path)
    store.change_passphrase(PASSPHRASE, NEW_PASSPHRASE)

    assert file_sha256(database_path) == database_before_passphrase_change
    with pytest.raises(UnlockError, match="^Unlock failed$"):
        store.unlock(PASSPHRASE)
    with store.unlock(NEW_PASSPHRASE) as unlocked:
        assert unlocked.copy_bytes() == NEW_KEY
    _assert_preserved(
        store,
        database_path,
        conversation_id,
        memory_id,
        passphrase=NEW_PASSPHRASE,
    )


def test_passphrase_change_refuses_pending_pre_replacement_rotation(
    tmp_path: Path,
) -> None:
    store, database_path, _conversation_id, _memory_id = _initialize_database(tmp_path)

    def crash_after_journal(phase: str) -> None:
        if phase == "after_journal_written":
            raise SimulatedCrash(phase)

    with pytest.raises(SimulatedCrash, match="after_journal_written"):
        store.rotate_database_key(
            database_path,
            PASSPHRASE,
            new_database_key=NEW_KEY,
            phase_hook=crash_after_journal,
        )

    assert store.rotation_recovery_required is True
    assert (
        database_key_opens(database_path, OLD_KEY.hex()),
        database_key_opens(database_path, NEW_KEY.hex()),
    ) == (True, False)
    database_before_refusal = file_sha256(database_path)
    envelope_before_refusal = file_sha256(store.key_file)
    journal_before_refusal = file_sha256(store.rotation_file)

    with pytest.raises(
        KeyRotationError,
        match=f"^{ROTATION_RECOVERY_REQUIRED_MESSAGE}$",
    ):
        store.change_passphrase(PASSPHRASE, NEW_PASSPHRASE)

    assert file_sha256(database_path) == database_before_refusal
    assert file_sha256(store.key_file) == envelope_before_refusal
    assert file_sha256(store.rotation_file) == journal_before_refusal
    assert _temporary_key_store_artifacts(store) == []


def test_busy_database_aborts_rotation_without_source_mutation(tmp_path: Path) -> None:
    store, database_path, conversation_id, memory_id = _initialize_database(tmp_path)
    source_before = file_sha256(database_path)
    driver = importlib.import_module("sqlcipher3.dbapi2")
    connection = driver.connect(str(database_path), isolation_level=None)
    cursor = connection.cursor()
    try:
        cursor.execute(f'PRAGMA key = "x\'{OLD_KEY.hex()}\'"')
        cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
        cursor.execute("BEGIN EXCLUSIVE")

        with pytest.raises(
            KeyRotationError,
            match="^Database is busy; stop AmitAI before key rotation$",
        ):
            store.rotate_database_key(
                database_path,
                PASSPHRASE,
                new_database_key=NEW_KEY,
            )
    finally:
        connection.rollback()
        cursor.close()
        connection.close()

    assert file_sha256(database_path) == source_before
    assert database_key_opens(database_path, OLD_KEY.hex()) is True
    assert database_key_opens(database_path, NEW_KEY.hex()) is False
    assert not store.rotation_file.exists()
    assert not rotation_candidate_path(database_path).exists()
    _assert_preserved(store, database_path, conversation_id, memory_id)


def test_rotation_aborts_if_source_changes_after_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, database_path, _conversation_id, _memory_id = _initialize_database(tmp_path)
    verify_candidate = database_module._verify_encrypted_candidate
    changed = False

    def verify_then_change(*args, **kwargs) -> None:
        nonlocal changed
        verify_candidate(*args, **kwargs)
        if changed:
            return
        changed = True
        driver = importlib.import_module("sqlcipher3.dbapi2")
        connection = driver.connect(str(database_path), isolation_level=None)
        try:
            cursor = connection.cursor()
            cursor.execute(f'PRAGMA key = "x\'{OLD_KEY.hex()}\'"')
            cursor.execute("UPDATE conversations SET title='late source write'")
        finally:
            connection.close()

    monkeypatch.setattr(
        "backend.database._verify_encrypted_candidate",
        verify_then_change,
    )

    with pytest.raises(KeyRotationError, match="^Database key rotation failed$"):
        store.rotate_database_key(
            database_path,
            PASSPHRASE,
            new_database_key=NEW_KEY,
        )

    assert changed is True
    assert database_key_opens(database_path, OLD_KEY.hex()) is True
    assert database_key_opens(database_path, NEW_KEY.hex()) is False
    store.recover_rotation(database_path, PASSPHRASE)
    assert not store.rotation_file.exists()


def test_rotation_recovery_fails_closed_and_retains_journal_if_neither_key_opens(
    tmp_path: Path,
) -> None:
    store, database_path, _conversation_id, _memory_id = _initialize_database(tmp_path)

    def crash_after_journal(phase: str) -> None:
        if phase == "after_journal_written":
            raise SimulatedCrash(phase)

    with pytest.raises(SimulatedCrash):
        store.rotate_database_key(
            database_path,
            PASSPHRASE,
            new_database_key=NEW_KEY,
            phase_hook=crash_after_journal,
        )
    envelope_before = file_sha256(store.key_file)
    journal_before = file_sha256(store.rotation_file)
    database_path.write_bytes(b"corrupted encrypted database")
    database_before = file_sha256(database_path)

    with pytest.raises(
        KeyRotationError,
        match="^Database key rotation recovery failed$",
    ):
        store.recover_rotation(database_path, PASSPHRASE)

    assert store.rotation_file.exists()
    assert file_sha256(store.rotation_file) == journal_before
    assert file_sha256(store.key_file) == envelope_before
    assert file_sha256(database_path) == database_before
    _assert_no_raw_keys(store.rotation_file)
