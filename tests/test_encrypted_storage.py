import hashlib
import importlib.metadata
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app import create_app
from backend.database import (
    MIGRATION_TEMP_MARKER,
    SQLITE_HEADER,
    Database,
    EncryptedStorageError,
    EncryptedStorageUnavailableError,
    PlaintextDatabaseError,
)
from backend.models import Conversation, MemoryRevision, MemorySlot, Message
from runtime.app import create_configured_app, create_runtime_app
from runtime.serve import load_local_server_config
from tests.app_factory import create_test_app

KEY_A = "11" * 32
KEY_B = "22" * 32
LOCAL_TOKEN = "LOCAL_API_SECRET_91233_secure_test_padding"
MESSAGE_CANARY = "UNIQUE_PLAINTEXT_CANARY_MESSAGE_9283174"
MEMORY_CANARY = "UNIQUE_PLAINTEXT_CANARY_MEMORY_5522184"
TITLE_CANARY = "UNIQUE_PLAINTEXT_CANARY_TITLE_1938217"
PRIVATE_MESSAGE = "PRIVATE_DB_MESSAGE_192837"
PRIVATE_MEMORY = "PRIVATE_DB_MEMORY_817263"
MALFORMED_KEY = "DB_SECRET_KEY_CANARY_918273"
WAL_CANARY = "PLAINTEXT_WAL_CANARY_6619042"


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _artifacts(path: Path) -> list[Path]:
    return [
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    ]


def _migration_candidates(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}{MIGRATION_TEMP_MARKER}*"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_encrypted_app(path: Path, key: str = KEY_A):
    return create_app(
        _database_url(path),
        database_key=key,
        enforce_local_auth=False,
    )


def _seed_application(application) -> tuple[str, str]:
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
                "key": "encryption.canary",
                "value": MEMORY_CANARY,
            },
        )
        assert memory.status_code == 201
        return conversation_id, memory.json()["id"]


def _assert_no_plaintext_canaries(path: Path) -> None:
    canaries = (MESSAGE_CANARY, MEMORY_CANARY, TITLE_CANARY)
    for artifact in _artifacts(path):
        if not artifact.exists():
            continue
        raw = artifact.read_bytes()
        assert SQLITE_HEADER not in raw[: len(SQLITE_HEADER)]
        for canary in canaries:
            assert canary.encode() not in raw


def _assert_plain_sqlite_cannot_read(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT name FROM sqlite_master").fetchall()
    finally:
        connection.close()


def _seed_plaintext_database(path: Path) -> tuple[str, str]:
    application = create_test_app(_database_url(path))
    conversation_id, memory_id = _seed_application(application)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version=7")
        connection.commit()
    finally:
        connection.close()
    return conversation_id, memory_id


def test_real_sqlcipher_encrypts_raw_database_and_reopens_through_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "encrypted.sqlite3"
    first_application = _create_encrypted_app(path)
    conversation_id, memory_id = _seed_application(first_application)

    assert first_application.state.database.encrypted is True
    assert importlib.metadata.version("sqlcipher3") == "0.6.2"
    assert first_application.state.database.cipher_version.startswith("4.")
    assert path.read_bytes()[: len(SQLITE_HEADER)] != SQLITE_HEADER
    _assert_no_plaintext_canaries(path)
    _assert_plain_sqlite_cannot_read(path)

    reopened_application = _create_encrypted_app(path)
    with TestClient(reopened_application) as client:
        conversation = client.get(f"/api/conversations/{conversation_id}")
        memories = client.get("/api/memory")

        assert conversation.status_code == 200
        assert conversation.json()["title"] == TITLE_CANARY
        assert [item["role"] for item in conversation.json()["messages"]] == [
            "user",
            "assistant",
        ]
        assert conversation.json()["messages"][0]["content"] == MESSAGE_CANARY
        assert any(
            item["id"] == memory_id and item["value"] == MEMORY_CANARY
            for item in memories.json()
        )

        with reopened_application.state.database.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "delete"


def test_sqlcipher_rollback_journal_never_contains_plaintext_canary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "encrypted-journal.sqlite3"
    application = _create_encrypted_app(path)
    _seed_application(application)
    database = Database.from_url(_database_url(path), encryption_key=KEY_A)
    raw_connection = database.engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "UPDATE messages SET content=? WHERE content=?",
                ("temporary replacement", MESSAGE_CANARY),
            )
            journal = Path(f"{path}-journal")
            assert journal.exists()
            journal_raw = journal.read_bytes()
            assert MESSAGE_CANARY.encode() not in journal_raw
            assert MEMORY_CANARY.encode() not in journal_raw
            assert TITLE_CANARY.encode() not in journal_raw
            raw_connection.rollback()
        finally:
            cursor.close()
    finally:
        raw_connection.close()
        database.engine.dispose()

    assert not Path(f"{path}-journal").exists()


def test_runtime_factory_uses_encrypted_storage_by_default(tmp_path: Path) -> None:
    path = tmp_path / "runtime-encrypted.sqlite3"
    application = create_runtime_app(
        _database_url(path),
        database_key=KEY_A,
        mode="mock",
        enforce_local_auth=False,
    )

    with TestClient(application) as client:
        assert client.get("/api/health").json() == {"status": "ok"}

    assert application.state.database.encrypted is True
    assert path.read_bytes()[: len(SQLITE_HEADER)] != SQLITE_HEADER


def test_wrong_key_fails_without_mutating_or_emptying_database(tmp_path: Path) -> None:
    path = tmp_path / "wrong-key.sqlite3"
    application = _create_encrypted_app(path)
    _seed_application(application)
    before = _sha256(path)

    with pytest.raises(EncryptedStorageError) as failure:
        Database.from_url(_database_url(path), encryption_key=KEY_B)

    assert str(failure.value) == "Encrypted database could not be opened"
    assert KEY_A not in str(failure.value)
    assert KEY_B not in str(failure.value)
    assert _sha256(path) == before
    _assert_plain_sqlite_cannot_read(path)

    reopened = Database.from_url(_database_url(path), encryption_key=KEY_A)
    try:
        with reopened.session_factory() as session:
            assert session.scalar(select(Conversation.title)) == TITLE_CANARY
    finally:
        reopened.engine.dispose()


def test_missing_malformed_key_and_missing_driver_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "closed.sqlite3"

    with pytest.raises(EncryptedStorageError) as missing:
        Database.from_url(_database_url(path))
    assert "64 hexadecimal characters" in str(missing.value)
    assert not path.exists()

    with pytest.raises(EncryptedStorageError) as malformed:
        Database.from_url(_database_url(path), encryption_key=MALFORMED_KEY)
    assert MALFORMED_KEY not in str(malformed.value)
    assert not path.exists()

    monkeypatch.setattr(
        "backend.database.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("private import failure")),
    )
    with pytest.raises(EncryptedStorageUnavailableError) as unavailable:
        Database.from_url(_database_url(path), encryption_key=KEY_A)
    assert "install AmitAI encrypted-storage dependencies" in str(unavailable.value)
    assert KEY_A not in str(unavailable.value)
    assert not path.exists()


def test_ordinary_sqlite_driver_cannot_masquerade_as_encrypted_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "not-sqlcipher.sqlite3"
    monkeypatch.setattr("backend.database._load_sqlcipher_driver", lambda: sqlite3)

    with pytest.raises(EncryptedStorageUnavailableError) as failure:
        Database.from_url(_database_url(path), encryption_key=KEY_A)

    assert str(failure.value) == (
        "Encrypted storage support is unavailable or is not SQLCipher"
    )
    assert not path.exists()


def test_canonical_startup_requires_database_key_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", LOCAL_TOKEN)
    monkeypatch.delenv("AMITAI_DB_KEY", raising=False)

    with pytest.raises(EncryptedStorageError):
        load_local_server_config({"AMITAI_LOCAL_API_TOKEN": LOCAL_TOKEN})
    with pytest.raises(EncryptedStorageError):
        create_configured_app()

    monkeypatch.setenv("AMITAI_DB_KEY", MALFORMED_KEY)
    with pytest.raises(EncryptedStorageError) as failure:
        create_configured_app()
    assert MALFORMED_KEY not in str(failure.value)


def test_plaintext_database_requires_explicit_migration_without_modification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plaintext-refused.sqlite3"
    _seed_plaintext_database(path)
    before = path.read_bytes()

    with pytest.raises(PlaintextDatabaseError) as failure:
        Database.from_url(_database_url(path), encryption_key=KEY_A)

    assert "AMITAI_ENCRYPT_EXISTING_DB=1" in str(failure.value)
    assert path.read_bytes() == before
    assert path.read_bytes().startswith(SQLITE_HEADER)
    assert MESSAGE_CANARY.encode() in path.read_bytes()
    assert _migration_candidates(path) == []


def test_plaintext_migration_exports_verifies_and_atomically_replaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plaintext-migrated.sqlite3"
    conversation_id, memory_id = _seed_plaintext_database(path)

    migrated = Database.from_url(
        _database_url(path),
        encryption_key=KEY_A,
        migrate_plaintext=True,
    )
    try:
        assert migrated.cipher_version.startswith("4.")
        with migrated.session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            memory = session.get(MemorySlot, memory_id)
            messages = list(
                session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at, Message.id)
                )
            )
            revisions = list(
                session.scalars(
                    select(MemoryRevision).where(MemoryRevision.memory_id == memory_id)
                )
            )
            assert conversation is not None
            assert conversation.title == TITLE_CANARY
            assert next(item.content for item in messages) == MESSAGE_CANARY
            assert memory is not None
            assert [item.value for item in revisions] == [MEMORY_CANARY]
        with migrated.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 7
            assert connection.exec_driver_sql("PRAGMA integrity_check").scalar() == "ok"
    finally:
        migrated.engine.dispose()

    _assert_no_plaintext_canaries(path)
    _assert_plain_sqlite_cannot_read(path)
    assert _migration_candidates(path) == []
    assert all(not sidecar.exists() for sidecar in _artifacts(path)[1:])


def test_plaintext_migration_checkpoints_unclosed_wal_before_export(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plaintext-wal.sqlite3"
    conversation_id, _ = _seed_plaintext_database(path)
    script = (
        "import os, sqlite3, sys; "
        "connection = sqlite3.connect(sys.argv[1]); "
        "connection.execute('PRAGMA journal_mode=WAL'); "
        "connection.execute('PRAGMA wal_autocheckpoint=0'); "
        "connection.execute('UPDATE conversations SET title=? WHERE id=?', "
        "(sys.argv[2], sys.argv[3])); "
        "connection.commit(); "
        "os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(path), WAL_CANARY, conversation_id],
        check=True,
    )
    wal = Path(f"{path}-wal")
    shm = Path(f"{path}-shm")
    assert wal.exists()
    assert shm.exists()
    assert WAL_CANARY.encode() in wal.read_bytes()

    migrated = Database.from_url(
        _database_url(path),
        encryption_key=KEY_A,
        migrate_plaintext=True,
    )
    try:
        with migrated.session_factory() as session:
            assert session.get(Conversation, conversation_id).title == WAL_CANARY
    finally:
        migrated.engine.dispose()

    assert WAL_CANARY.encode() not in path.read_bytes()
    assert not wal.exists()
    assert not shm.exists()
    assert not Path(f"{path}-journal").exists()


def test_busy_plaintext_database_refuses_migration_and_preserves_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "busy.sqlite3"
    _seed_plaintext_database(path)
    before = path.read_bytes()
    locker = sqlite3.connect(path, timeout=0, isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(EncryptedStorageError) as failure:
            Database.from_url(
                _database_url(path),
                encryption_key=KEY_A,
                migrate_plaintext=True,
            )
        assert str(failure.value) == (
            "Plaintext database is busy; stop AmitAI before migration"
        )
    finally:
        locker.rollback()
        locker.close()

    assert path.read_bytes() == before
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1
    finally:
        connection.close()
    assert _migration_candidates(path) == []


def test_migration_swap_failure_keeps_plaintext_source_and_removes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "migration-failure.sqlite3"
    _seed_plaintext_database(path)
    before = path.read_bytes()

    def fail_replace(_source: Path, _candidate: Path) -> None:
        raise OSError(f"private failure {PRIVATE_MESSAGE} {PRIVATE_MEMORY} {KEY_A}")

    monkeypatch.setattr("backend.database._replace_database", fail_replace)
    with caplog.at_level(logging.INFO), pytest.raises(EncryptedStorageError) as failure:
        Database.from_url(
            _database_url(path),
            encryption_key=KEY_A,
            migrate_plaintext=True,
        )

    assert str(failure.value) == "Plaintext database migration failed"
    assert path.read_bytes() == before
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT title FROM conversations").fetchone()[0] == TITLE_CANARY
    finally:
        connection.close()
    assert _migration_candidates(path) == []
    for sentinel in (PRIVATE_MESSAGE, PRIVATE_MEMORY, KEY_A):
        assert sentinel not in caplog.text
        assert sentinel not in str(failure.value)


def test_database_key_never_enters_engine_url_or_public_database_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no-key-metadata.sqlite3"
    database = Database.from_url(_database_url(path), encryption_key=KEY_A)
    try:
        assert KEY_A not in str(database.engine.url)
        assert KEY_A not in repr(database)
        assert set(database.__dataclass_fields__) == {
            "engine",
            "session_factory",
            "encrypted",
            "cipher_version",
        }
    finally:
        database.engine.dispose()
