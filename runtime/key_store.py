"""Passphrase-wrapped SQLCipher database keys and rotation recovery state."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argon2.exceptions import HashingError
from argon2.low_level import ARGON2_VERSION, Type, hash_secret_raw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.database import (
    DATABASE_KEY_PATTERN,
    SQLITE_HEADER,
    EncryptedStorageError,
    database_key_opens,
    rotate_encrypted_database,
)

from .paths import (
    PrivatePathError,
    atomic_write_private,
    read_private,
    remove_private,
    rotation_candidate_path,
    rotation_journal_path,
    validate_key_file_path,
)
from .secure_memory import DatabaseKeyHandle, SecureMemoryError

ENVELOPE_VERSION = 1
ENVELOPE_PURPOSE = "amitai-sqlcipher-database-key"
ROTATION_PURPOSE = "amitai-sqlcipher-database-key-rotation"
KDF_NAME = "argon2id"
WRAP_ALGORITHM = "aes-256-gcm"
MIN_PASSPHRASE_CHARS = 12
SALT_BYTES = 16
NONCE_BYTES = 12
DATABASE_KEY_BYTES = 32
GCM_TAG_BYTES = 16
MAX_JSON_TEXT_CHARS = 32 * 1024
ROTATION_STATES = frozenset({"prepared", "database_replaced", "finalized"})


class KeyStoreError(RuntimeError):
    """A sanitized key-store operation failure."""


class UnlockError(KeyStoreError):
    pass


class KeyRotationError(KeyStoreError):
    pass


@dataclass(frozen=True)
class Argon2Parameters:
    memory_cost_kib: int = 65_536
    time_cost: int = 3
    parallelism: int = 1
    hash_len: int = 32


@dataclass(frozen=True)
class Argon2Bounds:
    minimum_memory_cost_kib: int = 65_536
    maximum_memory_cost_kib: int = 1_048_576
    minimum_time_cost: int = 3
    maximum_time_cost: int = 10
    minimum_parallelism: int = 1
    maximum_parallelism: int = 8


@dataclass(frozen=True)
class KeyStorePolicy:
    creation: Argon2Parameters = Argon2Parameters()
    bounds: Argon2Bounds = Argon2Bounds()

    @classmethod
    def for_tests(cls) -> KeyStorePolicy:
        return cls(
            creation=Argon2Parameters(
                memory_cost_kib=64,
                time_cost=1,
                parallelism=1,
                hash_len=32,
            ),
            bounds=Argon2Bounds(
                minimum_memory_cost_kib=8,
                maximum_memory_cost_kib=1_024,
                minimum_time_cost=1,
                maximum_time_cost=2,
                minimum_parallelism=1,
                maximum_parallelism=2,
            ),
        )


PRODUCTION_KEY_STORE_POLICY = KeyStorePolicy()


@dataclass(frozen=True)
class KeyStoreStatus:
    envelope_exists: bool
    envelope_version: int | None
    kdf_name: str | None
    memory_cost_kib: int | None
    time_cost: int | None
    parallelism: int | None
    key_file: str
    rotation_recovery_required: bool


def _zero(value: bytearray) -> None:
    for index in range(len(value)):
        value[index] = 0


def _require_exact_keys(value: object, expected: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        raise KeyStoreError("Key store is invalid")
    return value


def _require_int(value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KeyStoreError("Key store is invalid")
    if not minimum <= value <= maximum:
        raise KeyStoreError("Key store is invalid")
    return value


def _decode_base64(value: object, *, expected_bytes: int) -> bytes:
    if not isinstance(value, str) or len(value) > (expected_bytes * 2) + 16:
        raise KeyStoreError("Key store is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise KeyStoreError("Key store is invalid") from None
    if len(decoded) != expected_bytes or base64.b64encode(decoded).decode("ascii") != value:
        raise KeyStoreError("Key store is invalid")
    return decoded


def _encode_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _json_without_duplicates(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_JSON_TEXT_CHARS:
        raise KeyStoreError("Key store is invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise KeyStoreError("Key store is invalid")
            result[key] = value
        return result

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise KeyStoreError("Key store is invalid") from None
    if not isinstance(document, dict):
        raise KeyStoreError("Key store is invalid")
    return document


def _serialized(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_passphrase(passphrase: str) -> None:
    if not isinstance(passphrase, str) or len(passphrase) < MIN_PASSPHRASE_CHARS:
        raise KeyStoreError(
            f"Unlock passphrase must contain at least {MIN_PASSPHRASE_CHARS} characters"
        )


def _has_plaintext_header(path: Path) -> bool:
    try:
        with path.open("rb") as database_file:
            return database_file.read(len(SQLITE_HEADER)) == SQLITE_HEADER
    except OSError:
        raise KeyStoreError("Database could not be inspected") from None


def _kdf_document(parameters: Argon2Parameters, salt: bytes) -> dict[str, Any]:
    return {
        "name": KDF_NAME,
        "salt": _encode_base64(salt),
        "memory_cost_kib": parameters.memory_cost_kib,
        "time_cost": parameters.time_cost,
        "parallelism": parameters.parallelism,
        "hash_len": parameters.hash_len,
    }


def _parse_kdf(
    value: object,
    *,
    bounds: Argon2Bounds,
) -> tuple[dict[str, Any], Argon2Parameters, bytes]:
    document = _require_exact_keys(
        value,
        frozenset(
            {
                "name",
                "salt",
                "memory_cost_kib",
                "time_cost",
                "parallelism",
                "hash_len",
            }
        ),
    )
    if document["name"] != KDF_NAME:
        raise KeyStoreError("Key store is invalid")
    parameters = Argon2Parameters(
        memory_cost_kib=_require_int(
            document["memory_cost_kib"],
            minimum=bounds.minimum_memory_cost_kib,
            maximum=bounds.maximum_memory_cost_kib,
        ),
        time_cost=_require_int(
            document["time_cost"],
            minimum=bounds.minimum_time_cost,
            maximum=bounds.maximum_time_cost,
        ),
        parallelism=_require_int(
            document["parallelism"],
            minimum=bounds.minimum_parallelism,
            maximum=bounds.maximum_parallelism,
        ),
        hash_len=_require_int(document["hash_len"], minimum=32, maximum=32),
    )
    salt = _decode_base64(document["salt"], expected_bytes=SALT_BYTES)
    return document, parameters, salt


def _derive_wrapping_key(
    passphrase: str,
    *,
    parameters: Argon2Parameters,
    salt: bytes,
) -> bytearray:
    try:
        derived = hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=parameters.time_cost,
            memory_cost=parameters.memory_cost_kib,
            parallelism=parameters.parallelism,
            hash_len=parameters.hash_len,
            type=Type.ID,
            version=ARGON2_VERSION,
        )
    except (HashingError, MemoryError, ValueError):
        raise KeyStoreError("Unlock failed") from None
    return bytearray(derived)


def _stable_aad(kdf: dict[str, Any]) -> bytes:
    return _serialized(
        {
            "version": ENVELOPE_VERSION,
            "purpose": ENVELOPE_PURPOSE,
            "kdf": kdf,
        }
    ).rstrip(b"\n")


def _rotation_aad(state: str, kdf: dict[str, Any], label: str) -> bytes:
    return _serialized(
        {
            "version": ENVELOPE_VERSION,
            "purpose": ROTATION_PURPOSE,
            "state": state,
            "kdf": kdf,
            "label": label,
        }
    ).rstrip(b"\n")


def _wrap(secret: bytes, wrapping_key: bytearray, *, aad: bytes) -> dict[str, str]:
    nonce = secrets.token_bytes(NONCE_BYTES)
    ciphertext = AESGCM(bytes(wrapping_key)).encrypt(nonce, secret, aad)
    return {
        "algorithm": WRAP_ALGORITHM,
        "nonce": _encode_base64(nonce),
        "ciphertext": _encode_base64(ciphertext),
    }


def _unwrap(
    value: object,
    wrapping_key: bytearray,
    *,
    aad: bytes,
) -> bytes:
    document = _require_exact_keys(
        value,
        frozenset({"algorithm", "nonce", "ciphertext"}),
    )
    if document["algorithm"] != WRAP_ALGORITHM:
        raise KeyStoreError("Key store is invalid")
    nonce = _decode_base64(document["nonce"], expected_bytes=NONCE_BYTES)
    ciphertext = _decode_base64(
        document["ciphertext"],
        expected_bytes=DATABASE_KEY_BYTES + GCM_TAG_BYTES,
    )
    try:
        plaintext = AESGCM(bytes(wrapping_key)).decrypt(nonce, ciphertext, aad)
    except (InvalidTag, ValueError):
        raise UnlockError("Unlock failed") from None
    if len(plaintext) != DATABASE_KEY_BYTES:
        raise UnlockError("Unlock failed")
    return plaintext


class KeyStore:
    def __init__(
        self,
        key_file: Path,
        *,
        policy: KeyStorePolicy = PRODUCTION_KEY_STORE_POLICY,
        repository_root: Path | None = None,
        validate_location: bool = True,
    ) -> None:
        if validate_location:
            options = {}
            if repository_root is not None:
                options["repository_root"] = repository_root
            self.key_file = validate_key_file_path(key_file, **options)
        else:
            self.key_file = key_file.expanduser().resolve(strict=False)
        self.rotation_file = rotation_journal_path(self.key_file)
        self.policy = policy

    def _read_envelope(self) -> tuple[dict[str, Any], dict[str, Any], Argon2Parameters, bytes]:
        document = _require_exact_keys(
            _json_without_duplicates(read_private(self.key_file)),
            frozenset({"version", "purpose", "kdf", "wrap"}),
        )
        if document["version"] != ENVELOPE_VERSION or document["purpose"] != ENVELOPE_PURPOSE:
            raise KeyStoreError("Key store is invalid")
        kdf, parameters, salt = _parse_kdf(
            document["kdf"],
            bounds=self.policy.bounds,
        )
        return document, kdf, parameters, salt

    def _build_envelope_with_kek(
        self,
        database_key: bytes,
        wrapping_key: bytearray,
        *,
        kdf: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "version": ENVELOPE_VERSION,
            "purpose": ENVELOPE_PURPOSE,
            "kdf": kdf,
            "wrap": _wrap(database_key, wrapping_key, aad=_stable_aad(kdf)),
        }

    def _write_new_envelope(self, database_key: bytes, passphrase: str) -> None:
        parameters = self.policy.creation
        salt = secrets.token_bytes(SALT_BYTES)
        kdf = _kdf_document(parameters, salt)
        wrapping_key = _derive_wrapping_key(
            passphrase,
            parameters=parameters,
            salt=salt,
        )
        try:
            document = self._build_envelope_with_kek(
                database_key,
                wrapping_key,
                kdf=kdf,
            )
            atomic_write_private(self.key_file, _serialized(document))
        finally:
            _zero(wrapping_key)

    def _unlock_material(
        self,
        passphrase: str,
    ) -> tuple[bytes, bytearray, dict[str, Any]]:
        document, kdf, parameters, salt = self._read_envelope()
        wrapping_key = _derive_wrapping_key(
            passphrase,
            parameters=parameters,
            salt=salt,
        )
        try:
            database_key = _unwrap(
                document["wrap"],
                wrapping_key,
                aad=_stable_aad(kdf),
            )
        except Exception:
            _zero(wrapping_key)
            raise
        return database_key, wrapping_key, kdf

    def initialize(
        self,
        passphrase: str,
        *,
        database_path: Path | None = None,
        database_key: bytes | None = None,
    ) -> None:
        _validate_passphrase(passphrase)
        if self.key_file.exists() or self.rotation_file.exists():
            raise KeyStoreError("Key store is already initialized")
        if (
            database_path is not None
            and database_path.exists()
            and not _has_plaintext_header(database_path)
        ):
            raise KeyStoreError("Encrypted database exists; use import-existing")
        generated = secrets.token_bytes(DATABASE_KEY_BYTES) if database_key is None else database_key
        if len(generated) != DATABASE_KEY_BYTES:
            raise KeyStoreError("Database key is invalid")
        self._write_new_envelope(generated, passphrase)

    def import_existing(
        self,
        database_path: Path,
        database_key_hex: str,
        passphrase: str,
    ) -> None:
        _validate_passphrase(passphrase)
        if self.key_file.exists() or self.rotation_file.exists():
            raise KeyStoreError("Key store is already initialized")
        if not database_path.exists() or _has_plaintext_header(database_path):
            raise KeyStoreError("Existing encrypted database is required")
        if DATABASE_KEY_PATTERN.fullmatch(database_key_hex) is None:
            raise KeyStoreError("Existing database key is invalid")
        key_bytes = bytes.fromhex(database_key_hex)
        with DatabaseKeyHandle(key_bytes) as handle:
            if not database_key_opens(database_path, handle):
                raise KeyStoreError("Existing database key could not open the database")
        self._write_new_envelope(key_bytes, passphrase)

    def unlock(
        self,
        passphrase: str,
        *,
        database_path: Path | None = None,
    ) -> DatabaseKeyHandle:
        try:
            if self.rotation_file.exists():
                if database_path is None:
                    raise UnlockError("Unlock failed")
                self.recover_rotation(database_path, passphrase)
            database_key, wrapping_key, _ = self._unlock_material(passphrase)
            try:
                return DatabaseKeyHandle(database_key)
            finally:
                _zero(wrapping_key)
        except (KeyStoreError, PrivatePathError, SecureMemoryError):
            raise UnlockError("Unlock failed") from None

    def change_passphrase(self, old_passphrase: str, new_passphrase: str) -> None:
        _validate_passphrase(new_passphrase)
        try:
            database_key, wrapping_key, _ = self._unlock_material(old_passphrase)
        except (KeyStoreError, PrivatePathError):
            raise UnlockError("Unlock failed") from None
        try:
            self._write_new_envelope(database_key, new_passphrase)
        finally:
            _zero(wrapping_key)

    def _rotation_document(
        self,
        *,
        state: str,
        old_key: bytes,
        new_key: bytes,
        wrapping_key: bytearray,
        kdf: dict[str, Any],
    ) -> dict[str, Any]:
        if state not in ROTATION_STATES:
            raise KeyRotationError("Database key rotation failed")
        return {
            "version": ENVELOPE_VERSION,
            "purpose": ROTATION_PURPOSE,
            "state": state,
            "kdf": kdf,
            "keys": {
                "old": _wrap(
                    old_key,
                    wrapping_key,
                    aad=_rotation_aad(state, kdf, "old"),
                ),
                "new": _wrap(
                    new_key,
                    wrapping_key,
                    aad=_rotation_aad(state, kdf, "new"),
                ),
            },
        }

    def _write_rotation(
        self,
        *,
        state: str,
        old_key: bytes,
        new_key: bytes,
        wrapping_key: bytearray,
        kdf: dict[str, Any],
    ) -> None:
        document = self._rotation_document(
            state=state,
            old_key=old_key,
            new_key=new_key,
            wrapping_key=wrapping_key,
            kdf=kdf,
        )
        atomic_write_private(self.rotation_file, _serialized(document))

    def _read_rotation(
        self,
        passphrase: str,
    ) -> tuple[str, bytes, bytes, bytearray, dict[str, Any]]:
        document = _require_exact_keys(
            _json_without_duplicates(read_private(self.rotation_file)),
            frozenset({"version", "purpose", "state", "kdf", "keys"}),
        )
        if (
            document["version"] != ENVELOPE_VERSION
            or document["purpose"] != ROTATION_PURPOSE
            or document["state"] not in ROTATION_STATES
        ):
            raise KeyStoreError("Key store is invalid")
        state = document["state"]
        kdf, parameters, salt = _parse_kdf(
            document["kdf"],
            bounds=self.policy.bounds,
        )
        keys = _require_exact_keys(document["keys"], frozenset({"old", "new"}))
        wrapping_key = _derive_wrapping_key(
            passphrase,
            parameters=parameters,
            salt=salt,
        )
        try:
            old_key = _unwrap(
                keys["old"],
                wrapping_key,
                aad=_rotation_aad(state, kdf, "old"),
            )
            new_key = _unwrap(
                keys["new"],
                wrapping_key,
                aad=_rotation_aad(state, kdf, "new"),
            )
        except Exception:
            _zero(wrapping_key)
            raise
        return state, old_key, new_key, wrapping_key, kdf

    def rotate_database_key(
        self,
        database_path: Path,
        passphrase: str,
        *,
        phase_hook: Callable[[str], None] | None = None,
        new_database_key: bytes | None = None,
    ) -> None:
        hook = phase_hook or (lambda _phase: None)
        if self.rotation_file.exists():
            self.recover_rotation(database_path, passphrase)
        try:
            old_key, wrapping_key, kdf = self._unlock_material(passphrase)
        except (KeyStoreError, PrivatePathError):
            raise UnlockError("Unlock failed") from None
        new_key = (
            secrets.token_bytes(DATABASE_KEY_BYTES)
            if new_database_key is None
            else new_database_key
        )
        if len(new_key) != DATABASE_KEY_BYTES:
            _zero(wrapping_key)
            raise KeyRotationError("Database key rotation failed")
        candidate = rotation_candidate_path(database_path)
        try:
            self._write_rotation(
                state="prepared",
                old_key=old_key,
                new_key=new_key,
                wrapping_key=wrapping_key,
                kdf=kdf,
            )
            hook("after_journal_written")
            with DatabaseKeyHandle(old_key) as old_handle, DatabaseKeyHandle(
                new_key
            ) as new_handle:
                rotate_encrypted_database(
                    database_path,
                    old_handle,
                    new_handle,
                    candidate=candidate,
                    phase_hook=hook,
                )
            self._write_rotation(
                state="database_replaced",
                old_key=old_key,
                new_key=new_key,
                wrapping_key=wrapping_key,
                kdf=kdf,
            )
            hook("before_envelope_finalization")
            envelope = self._build_envelope_with_kek(
                new_key,
                wrapping_key,
                kdf=kdf,
            )
            atomic_write_private(self.key_file, _serialized(envelope))
            self._write_rotation(
                state="finalized",
                old_key=old_key,
                new_key=new_key,
                wrapping_key=wrapping_key,
                kdf=kdf,
            )
            hook("after_envelope_finalized")
            remove_private(self.rotation_file)
        except EncryptedStorageError as exc:
            if str(exc) == "Database is busy; stop AmitAI before key rotation":
                remove_private(self.rotation_file)
                raise KeyRotationError(str(exc)) from None
            raise KeyRotationError("Database key rotation failed") from None
        except (KeyStoreError, PrivatePathError, SecureMemoryError):
            raise
        except Exception:  # noqa: BLE001 - sanitize phase-hook and platform failures
            raise KeyRotationError("Database key rotation failed") from None
        finally:
            _zero(wrapping_key)

    def recover_rotation(self, database_path: Path, passphrase: str) -> None:
        try:
            _, old_key, new_key, wrapping_key, kdf = self._read_rotation(passphrase)
        except (KeyStoreError, PrivatePathError):
            raise UnlockError("Unlock failed") from None
        candidate = rotation_candidate_path(database_path)
        try:
            with DatabaseKeyHandle(old_key) as old_handle, DatabaseKeyHandle(
                new_key
            ) as new_handle:
                old_opens = database_key_opens(database_path, old_handle)
                new_opens = database_key_opens(database_path, new_handle)
            if old_opens == new_opens:
                raise KeyRotationError("Database key rotation recovery failed")
            selected = new_key if new_opens else old_key
            envelope = self._build_envelope_with_kek(
                selected,
                wrapping_key,
                kdf=kdf,
            )
            atomic_write_private(self.key_file, _serialized(envelope))
            for artifact in (
                candidate,
                Path(f"{candidate}-wal"),
                Path(f"{candidate}-shm"),
                Path(f"{candidate}-journal"),
            ):
                try:
                    artifact.unlink()
                except FileNotFoundError:
                    pass
            remove_private(self.rotation_file)
        finally:
            _zero(wrapping_key)

    def status(self) -> KeyStoreStatus:
        if not self.key_file.exists():
            return KeyStoreStatus(
                envelope_exists=False,
                envelope_version=None,
                kdf_name=None,
                memory_cost_kib=None,
                time_cost=None,
                parallelism=None,
                key_file=str(self.key_file),
                rotation_recovery_required=self.rotation_file.exists(),
            )
        document, _, parameters, _ = self._read_envelope()
        return KeyStoreStatus(
            envelope_exists=True,
            envelope_version=document["version"],
            kdf_name=KDF_NAME,
            memory_cost_kib=parameters.memory_cost_kib,
            time_cost=parameters.time_cost,
            parallelism=parameters.parallelism,
            key_file=str(self.key_file),
            rotation_recovery_required=self.rotation_file.exists(),
        )


def file_sha256(path: Path) -> str:
    """Return a non-secret fingerprint for tests and operator verification."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
