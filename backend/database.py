"""Central SQLAlchemy construction for encrypted and explicit test storage."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import secrets
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DATABASE_URL = "sqlite+pysqlite:///./amitai.db"
SQLITE_HEADER = b"SQLite format 3\x00"
DATABASE_KEY_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
MIGRATION_TEMP_MARKER = ".amitai-migrating-"
REQUIRED_APPLICATION_TABLES = frozenset(
    {
        "conversations",
        "messages",
        "message_metadata",
        "memory_slots",
        "memory_revisions",
    }
)
REPRESENTATIVE_TABLES = (
    "conversations",
    "messages",
    "memory_slots",
    "memory_revisions",
)


class EncryptedStorageError(RuntimeError):
    """Sanitized failure opening, creating, or migrating encrypted storage."""


class PlaintextDatabaseError(EncryptedStorageError):
    """An existing plaintext database requires explicit migration permission."""


class EncryptedStorageUnavailableError(EncryptedStorageError):
    """The native SQLCipher dependency is unavailable or not a real codec."""


class Base(DeclarativeBase):
    """Base class for backend ORM models."""


class DatabaseKeySource(Protocol):
    """Provide a temporary hex key without exposing a permanent string."""

    def temporary_hex(self) -> AbstractContextManager[str]: ...


DatabaseKeyInput = str | DatabaseKeySource


def validate_database_key(value: str | None) -> str:
    """Return a normalized raw SQLCipher key without ever echoing invalid input."""

    if not isinstance(value, str) or DATABASE_KEY_PATTERN.fullmatch(value) is None:
        raise EncryptedStorageError(
            "Database key must contain exactly 64 hexadecimal characters"
        )
    return value.lower()


@contextmanager
def _temporary_database_key(value: DatabaseKeyInput | None) -> Iterator[str]:
    if isinstance(value, str) or value is None:
        yield validate_database_key(value)
        return
    try:
        with value.temporary_hex() as temporary:
            yield validate_database_key(temporary)
    except EncryptedStorageError:
        raise
    except Exception:  # noqa: BLE001 - protocol implementations are untrusted
        raise EncryptedStorageError("Database key is unavailable") from None


def _load_sqlcipher_driver() -> ModuleType:
    try:
        return importlib.import_module("sqlcipher3.dbapi2")
    except (ImportError, OSError):
        raise EncryptedStorageUnavailableError(
            "Encrypted storage support is not installed; install AmitAI "
            "encrypted-storage dependencies"
        ) from None


def _verify_sqlcipher_driver(driver: ModuleType) -> None:
    connection = None
    try:
        connection = driver.connect(":memory:")
        cursor = connection.cursor()
        try:
            _cipher_version(cursor)
        finally:
            cursor.close()
    except EncryptedStorageError:
        raise
    except (AttributeError, TypeError, driver.Error):
        raise EncryptedStorageUnavailableError(
            "Encrypted storage support is unavailable or is not SQLCipher"
        ) from None
    finally:
        if connection is not None:
            connection.close()


def _database_path(url: URL) -> Path | None:
    if url.get_backend_name() != "sqlite":
        return None
    if url.database in {None, "", ":memory:"}:
        return None
    return Path(url.database).expanduser().resolve()


def _looks_like_plaintext_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as database_file:
            return database_file.read(len(SQLITE_HEADER)) == SQLITE_HEADER
    except OSError:
        raise EncryptedStorageError("Encrypted database could not be opened") from None


def _key_pragma(key: str) -> str:
    return f'PRAGMA key = "x\'{key}\'"'


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _qualified(schema: str, name: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(name)}"


def _sidecar_paths(path: Path) -> tuple[Path, Path, Path]:
    return (
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    )


def _migration_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}{MIGRATION_TEMP_MARKER}{secrets.token_hex(8)}")


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remove_candidate(candidate: Path) -> None:
    for artifact in (candidate, *_sidecar_paths(candidate)):
        try:
            _remove_file(artifact)
        except OSError:
            pass


def _user_tables(cursor: Any, schema: str) -> tuple[str, ...]:
    statement = (
        f"SELECT name FROM {_qualified(schema, 'sqlite_master')} "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return tuple(row[0] for row in cursor.execute(statement).fetchall())


def _schema_records(cursor: Any, schema: str) -> tuple[tuple[Any, ...], ...]:
    statement = (
        f"SELECT type, name, tbl_name, sql FROM {_qualified(schema, 'sqlite_master')} "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    )
    return tuple(cursor.execute(statement).fetchall())


def _table_counts(cursor: Any, schema: str, tables: tuple[str, ...]) -> dict[str, int]:
    return {
        table: int(
            cursor.execute(f"SELECT count(*) FROM {_qualified(schema, table)}").fetchone()[0]
        )
        for table in tables
    }


def _representative_rows(
    cursor: Any,
    schema: str,
    tables: tuple[str, ...],
) -> dict[str, tuple[Any, ...] | None]:
    available = set(tables)
    return {
        table: cursor.execute(
            f"SELECT * FROM {_qualified(schema, table)} LIMIT 1"
        ).fetchone()
        for table in REPRESENTATIVE_TABLES
        if table in available
    }


def _integrity_is_ok(cursor: Any, schema: str = "main") -> bool:
    rows = cursor.execute(f"PRAGMA {_quote_identifier(schema)}.integrity_check").fetchall()
    return rows == [("ok",)]


def _cipher_integrity_is_ok(cursor: Any) -> bool:
    return cursor.execute("PRAGMA cipher_integrity_check").fetchall() == []


def _cipher_version(cursor: Any) -> str:
    row = cursor.execute("PRAGMA cipher_version").fetchone()
    if row is None or not isinstance(row[0], str) or not row[0].strip():
        raise EncryptedStorageUnavailableError(
            "Encrypted storage support is unavailable or is not SQLCipher"
        )
    return row[0].strip()


def _is_database_busy(error: Exception) -> bool:
    return getattr(error, "sqlite_errorcode", None) in {5, 6}


def _artifact_fingerprint(path: Path) -> tuple[tuple[str, int, str], ...]:
    fingerprints: list[tuple[str, int, str]] = []
    for artifact in (path, *_sidecar_paths(path)):
        if not artifact.exists():
            continue
        digest = hashlib.sha256()
        with artifact.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints.append((artifact.name, artifact.stat().st_size, digest.hexdigest()))
    return tuple(fingerprints)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _verify_encrypted_candidate(
    candidate: Path,
    key: DatabaseKeyInput,
    driver: ModuleType,
    *,
    expected_tables: tuple[str, ...],
    expected_schema: tuple[tuple[Any, ...], ...],
    expected_counts: dict[str, int],
    expected_rows: dict[str, tuple[Any, ...] | None],
    expected_user_version: int,
) -> None:
    connection = None
    try:
        connection = driver.connect(str(candidate), timeout=0)
        cursor = connection.cursor()
        try:
            with _temporary_database_key(key) as temporary_key:
                cursor.execute(_key_pragma(temporary_key))
            _cipher_version(cursor)
            cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
            candidate_tables = _user_tables(cursor, "main")
            if not REQUIRED_APPLICATION_TABLES.issubset(candidate_tables):
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if candidate_tables != expected_tables:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _schema_records(cursor, "main") != expected_schema:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _table_counts(cursor, "main", candidate_tables) != expected_counts:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _representative_rows(cursor, "main", candidate_tables) != expected_rows:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            user_version = int(cursor.execute("PRAGMA main.user_version").fetchone()[0])
            if user_version != expected_user_version:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if not _integrity_is_ok(cursor) or not _cipher_integrity_is_ok(cursor):
                raise EncryptedStorageError("Encrypted database migration verification failed")
        finally:
            cursor.close()
    except EncryptedStorageError:
        raise
    except (driver.Error, OSError, TypeError, ValueError):
        raise EncryptedStorageError("Encrypted database migration verification failed") from None
    finally:
        if connection is not None:
            connection.close()


def _replace_database(source: Path, candidate: Path) -> None:
    os.replace(candidate, source)


def _migrate_plaintext_database(
    path: Path,
    key: DatabaseKeyInput,
    driver: ModuleType,
) -> None:
    candidate = _migration_temp_path(path)
    connection = None
    attached = False
    transaction_open = False
    expected_tables: tuple[str, ...] = ()
    expected_schema: tuple[tuple[Any, ...], ...] = ()
    expected_counts: dict[str, int] = {}
    expected_rows: dict[str, tuple[Any, ...] | None] = {}
    locked_source_fingerprint: tuple[tuple[str, int, str], ...] = ()
    try:
        connection = driver.connect(str(path), timeout=0, isolation_level=None)
        cursor = connection.cursor()
        try:
            _cipher_version(cursor)
            cursor.execute("PRAGMA busy_timeout=0")
            cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
            checkpoint = cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise EncryptedStorageError(
                    "Plaintext database is busy; stop AmitAI before migration"
                )
            journal_mode = cursor.execute("PRAGMA journal_mode=DELETE").fetchone()
            if journal_mode is None or str(journal_mode[0]).casefold() != "delete":
                raise EncryptedStorageError(
                    "Plaintext database is busy; stop AmitAI before migration"
                )

            escaped_candidate = str(candidate).replace("'", "''")
            with _temporary_database_key(key) as temporary_key:
                cursor.execute(
                    f"ATTACH DATABASE '{escaped_candidate}' "
                    f"AS encrypted KEY \"x'{temporary_key}'\""
                )
            attached = True
            cursor.execute("BEGIN EXCLUSIVE")
            transaction_open = True

            expected_tables = _user_tables(cursor, "main")
            if not REQUIRED_APPLICATION_TABLES.issubset(expected_tables):
                raise EncryptedStorageError("Plaintext database schema is not supported")
            expected_schema = _schema_records(cursor, "main")
            expected_counts = _table_counts(cursor, "main", expected_tables)
            expected_rows = _representative_rows(cursor, "main", expected_tables)
            user_version = int(cursor.execute("PRAGMA main.user_version").fetchone()[0])

            cursor.execute("SELECT sqlcipher_export('encrypted')").fetchone()
            cursor.execute(f"PRAGMA encrypted.user_version={user_version}")
            if _user_tables(cursor, "encrypted") != expected_tables:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _schema_records(cursor, "encrypted") != expected_schema:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _table_counts(cursor, "encrypted", expected_tables) != expected_counts:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if _representative_rows(cursor, "encrypted", expected_tables) != expected_rows:
                raise EncryptedStorageError("Encrypted database migration verification failed")
            if not _integrity_is_ok(cursor, "encrypted"):
                raise EncryptedStorageError("Encrypted database migration verification failed")

            # Capture the exact source artifacts represented by the export while
            # BEGIN EXCLUSIVE still prevents another SQLite writer from changing
            # them. The WAL has already been checkpointed and journal mode has
            # been normalized, so any later sidecar change is meaningful.
            locked_source_fingerprint = _artifact_fingerprint(path)

            cursor.execute("COMMIT")
            transaction_open = False
            cursor.execute("DETACH DATABASE encrypted")
            attached = False
        finally:
            cursor.close()
            connection.close()
            connection = None

        _verify_encrypted_candidate(
            candidate,
            key,
            driver,
            expected_tables=expected_tables,
            expected_schema=expected_schema,
            expected_counts=expected_counts,
            expected_rows=expected_rows,
            expected_user_version=user_version,
        )
        _fsync_file(candidate)
        if _artifact_fingerprint(path) != locked_source_fingerprint:
            raise EncryptedStorageError(
                "Plaintext database changed during migration; stop AmitAI and retry"
            )
        _replace_database(path, candidate)
        _fsync_directory(path.parent)
        for sidecar in _sidecar_paths(path):
            _remove_file(sidecar)
    except EncryptedStorageError:
        if transaction_open and connection is not None:
            try:
                connection.rollback()
            except driver.Error:
                transaction_open = False
        if attached and connection is not None:
            try:
                connection.execute("DETACH DATABASE encrypted")
            except driver.Error:
                attached = False
        if connection is not None:
            connection.close()
        _remove_candidate(candidate)
        raise
    except (driver.Error, OSError, TypeError, ValueError) as exc:
        if connection is not None:
            try:
                if transaction_open:
                    connection.rollback()
            finally:
                connection.close()
        _remove_candidate(candidate)
        if _is_database_busy(exc):
            raise EncryptedStorageError(
                "Plaintext database is busy; stop AmitAI before migration"
            ) from None
        raise EncryptedStorageError("Plaintext database migration failed") from None


def database_key_opens(path: Path, key: DatabaseKeyInput) -> bool:
    """Return whether one key opens an existing SQLCipher database."""

    if not path.exists() or _looks_like_plaintext_sqlite(path):
        return False
    driver = _load_sqlcipher_driver()
    _verify_sqlcipher_driver(driver)
    connection = None
    try:
        connection = driver.connect(str(path), timeout=0)
        cursor = connection.cursor()
        try:
            with _temporary_database_key(key) as temporary_key:
                cursor.execute(_key_pragma(temporary_key))
            _cipher_version(cursor)
            cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
            return _integrity_is_ok(cursor) and _cipher_integrity_is_ok(cursor)
        finally:
            cursor.close()
    except (EncryptedStorageError, driver.Error, OSError, TypeError, ValueError):
        return False
    finally:
        if connection is not None:
            connection.close()


def rotate_encrypted_database(
    path: Path,
    old_key: DatabaseKeyInput,
    new_key: DatabaseKeyInput,
    *,
    candidate: Path,
    phase_hook: Callable[[str], None] | None = None,
) -> None:
    """Offline encrypted export with locked-snapshot consistency verification."""

    hook = phase_hook or (lambda _phase: None)
    driver = _load_sqlcipher_driver()
    _verify_sqlcipher_driver(driver)
    _remove_candidate(candidate)
    connection = None
    attached = False
    transaction_open = False
    replaced = False
    expected_tables: tuple[str, ...] = ()
    expected_schema: tuple[tuple[Any, ...], ...] = ()
    expected_counts: dict[str, int] = {}
    expected_rows: dict[str, tuple[Any, ...] | None] = {}
    expected_user_version = 0
    locked_source_fingerprint: tuple[tuple[str, int, str], ...] = ()
    try:
        connection = driver.connect(str(path), timeout=0, isolation_level=None)
        cursor = connection.cursor()
        try:
            with _temporary_database_key(old_key) as temporary_old_key:
                cursor.execute(_key_pragma(temporary_old_key))
            _cipher_version(cursor)
            cursor.execute("PRAGMA busy_timeout=0")
            cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
            checkpoint = cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise EncryptedStorageError(
                    "Database is busy; stop AmitAI before key rotation"
                )
            journal_mode = cursor.execute("PRAGMA journal_mode=DELETE").fetchone()
            if journal_mode is None or str(journal_mode[0]).casefold() != "delete":
                raise EncryptedStorageError(
                    "Database is busy; stop AmitAI before key rotation"
                )

            escaped_candidate = str(candidate).replace("'", "''")
            with _temporary_database_key(new_key) as temporary_new_key:
                cursor.execute(
                    f"ATTACH DATABASE '{escaped_candidate}' "
                    f"AS rotated KEY \"x'{temporary_new_key}'\""
                )
            attached = True
            cursor.execute("BEGIN EXCLUSIVE")
            transaction_open = True

            expected_tables = _user_tables(cursor, "main")
            if not REQUIRED_APPLICATION_TABLES.issubset(expected_tables):
                raise EncryptedStorageError("Encrypted database schema is not supported")
            expected_schema = _schema_records(cursor, "main")
            expected_counts = _table_counts(cursor, "main", expected_tables)
            expected_rows = _representative_rows(cursor, "main", expected_tables)
            expected_user_version = int(
                cursor.execute("PRAGMA main.user_version").fetchone()[0]
            )

            cursor.execute("SELECT sqlcipher_export('rotated')").fetchone()
            cursor.execute(f"PRAGMA rotated.user_version={expected_user_version}")
            hook("after_candidate_created")
            if _user_tables(cursor, "rotated") != expected_tables:
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            if _schema_records(cursor, "rotated") != expected_schema:
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            if _table_counts(cursor, "rotated", expected_tables) != expected_counts:
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            if _representative_rows(cursor, "rotated", expected_tables) != expected_rows:
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            rotated_user_version = int(
                cursor.execute("PRAGMA rotated.user_version").fetchone()[0]
            )
            if rotated_user_version != expected_user_version:
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            if not _integrity_is_ok(cursor, "rotated"):
                raise EncryptedStorageError("Encrypted database rotation verification failed")
            hook("after_candidate_verified")
            locked_source_fingerprint = _artifact_fingerprint(path)

            cursor.execute("COMMIT")
            transaction_open = False
            cursor.execute("DETACH DATABASE rotated")
            attached = False
        finally:
            cursor.close()
            connection.close()
            connection = None

        _verify_encrypted_candidate(
            candidate,
            new_key,
            driver,
            expected_tables=expected_tables,
            expected_schema=expected_schema,
            expected_counts=expected_counts,
            expected_rows=expected_rows,
            expected_user_version=expected_user_version,
        )
        _fsync_file(candidate)
        if _artifact_fingerprint(path) != locked_source_fingerprint:
            raise EncryptedStorageError(
                "Encrypted database changed during key rotation; stop AmitAI and retry"
            )
        _replace_database(path, candidate)
        replaced = True
        _fsync_directory(path.parent)
        for sidecar in _sidecar_paths(path):
            _remove_file(sidecar)
        hook("after_database_replaced")
        if not database_key_opens(path, new_key) or database_key_opens(path, old_key):
            raise EncryptedStorageError("Encrypted database rotation verification failed")
    except EncryptedStorageError:
        if transaction_open and connection is not None:
            try:
                connection.rollback()
            except driver.Error:
                pass
        if attached and connection is not None:
            try:
                connection.execute("DETACH DATABASE rotated")
            except driver.Error:
                pass
        if connection is not None:
            connection.close()
        if not replaced:
            _remove_candidate(candidate)
        raise
    except (driver.Error, OSError, TypeError, ValueError) as exc:
        if connection is not None:
            try:
                if transaction_open:
                    connection.rollback()
            finally:
                connection.close()
        if not replaced:
            _remove_candidate(candidate)
        if _is_database_busy(exc):
            raise EncryptedStorageError(
                "Database is busy; stop AmitAI before key rotation"
            ) from None
        raise EncryptedStorageError("Database key rotation failed") from None


def _sqlite_engine_options(url: URL) -> dict[str, object]:
    options: dict[str, object] = {
        "pool_pre_ping": True,
        "connect_args": {"check_same_thread": False},
    }
    if url.database in {None, "", ":memory:"}:
        options["poolclass"] = StaticPool
    return options


def _register_plaintext_connection_settings(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _configure_plaintext_connection(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def _register_encrypted_connection_settings(
    engine: Engine,
    key: DatabaseKeyInput,
    driver: ModuleType,
) -> None:
    @event.listens_for(engine, "connect")
    def _configure_encrypted_connection(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            with _temporary_database_key(key) as temporary_key:
                cursor.execute(_key_pragma(temporary_key))
            _cipher_version(cursor)
            cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
            cursor.execute("PRAGMA foreign_keys=ON")
        except EncryptedStorageError:
            raise
        except driver.Error:
            raise EncryptedStorageError("Encrypted database could not be opened") from None
        finally:
            cursor.close()


@dataclass(frozen=True)
class Database:
    """Own an engine and session factory without exposing encryption secrets."""

    engine: Engine
    session_factory: sessionmaker[Session]
    encrypted: bool
    cipher_version: str | None = None

    @classmethod
    def from_url(
        cls,
        database_url: str = DEFAULT_DATABASE_URL,
        *,
        encrypted: bool = True,
        encryption_key: DatabaseKeyInput | None = None,
        migrate_plaintext: bool = False,
    ) -> Database:
        url = make_url(database_url)
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        database_path = _database_path(url)
        database_existed = database_path is not None and database_path.exists()

        if encrypted:
            if url.get_backend_name() != "sqlite":
                raise EncryptedStorageError("Encrypted storage requires a SQLite database URL")
            with _temporary_database_key(encryption_key):
                pass
            driver = _load_sqlcipher_driver()
            _verify_sqlcipher_driver(driver)
            if (
                database_path is not None
                and database_path.exists()
                and _looks_like_plaintext_sqlite(database_path)
            ):
                if not migrate_plaintext:
                    raise PlaintextDatabaseError(
                        "Existing plaintext AmitAI database detected; set "
                        "AMITAI_ENCRYPT_EXISTING_DB=1 for one intentional migration launch"
                    )
                assert encryption_key is not None
                _migrate_plaintext_database(database_path, encryption_key, driver)
            engine_options = _sqlite_engine_options(url)
            engine_options["module"] = driver
        elif url.get_backend_name() == "sqlite":
            engine_options = _sqlite_engine_options(url)

        engine = create_engine(database_url, **engine_options)
        if encrypted:
            assert encryption_key is not None
            _register_encrypted_connection_settings(engine, encryption_key, driver)
        elif url.get_backend_name() == "sqlite":
            _register_plaintext_connection_settings(engine)

        cipher_version: str | None = None
        try:
            with engine.connect() as connection:
                if encrypted:
                    cipher_version = str(
                        connection.exec_driver_sql("PRAGMA cipher_version").scalar()
                    )
                    journal_mode = connection.exec_driver_sql(
                        "PRAGMA journal_mode=DELETE"
                    ).scalar()
                    if str(journal_mode).casefold() != "delete":
                        raise EncryptedStorageError("Encrypted database could not be opened")
        except EncryptedStorageError:
            engine.dispose()
            if encrypted and database_path is not None and not database_existed:
                _remove_candidate(database_path)
            raise
        except (SQLAlchemyError, OSError):
            engine.dispose()
            if encrypted:
                if database_path is not None and not database_existed:
                    _remove_candidate(database_path)
                raise EncryptedStorageError("Encrypted database could not be opened") from None
            raise

        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        return cls(
            engine=engine,
            session_factory=factory,
            encrypted=encrypted,
            cipher_version=cipher_version,
        )

    def create_schema(self) -> None:
        # Importing registers all mapped classes on Base.metadata.
        from . import models as _models  # noqa: F401

        Base.metadata.create_all(self.engine)
