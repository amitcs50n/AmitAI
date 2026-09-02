"""Explicit encrypted local backup/restore. No network, model, or new key hierarchy."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from backend.asset_crypto import decrypt_asset
from backend.asset_keys import load_asset_key
from backend.asset_migration import validate_normalized_png
from backend.asset_storage import AssetStorage, database_asset_namespace, default_asset_directory
from backend.database import (
    Database,
    DatabaseKeyInput,
    database_key_opens,
    export_encrypted_snapshot,
)

from .backup_archive import (
    DATABASE,
    ENVELOPE,
    MAX_ARCHIVE_BYTES,
    MAX_ASSETS,
    MAX_DATABASE_BYTES,
    Entry,
    Manifest,
    canonical_uuid,
    extract_verified,
    strict_json,
    validated_archive,
    write_archive,
)
from .backup_files import (
    checked_path,
    file_fingerprint,
    install_new,
    journal_lock,
    new_private_file,
    private_stage,
    regular_file,
    stage_copy,
)
from .key_store import KeyStore
from .paths import (
    _fsync_directory,
    assert_owner_only,
    default_key_file,
    ensure_private_directory,
    read_private,
    restore_journal_path,
    rotation_candidate_path,
)
from .process_hardening import apply_process_hardening
from .secure_memory import DatabaseKeyHandle

SecretPrompt = Callable[[str], str]
PhaseHook = Callable[[str], None]
RECOVERY_MESSAGE = (
    "Interrupted restore detected; rerun runtime.backup restore --resume "
    "with the same input and destinations"
)


class BackupError(RuntimeError):
    """Only fixed, public-safe diagnostics may leave this module."""


def _hook(callback: PhaseHook | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def _asset_root(database: Path) -> Path:
    return checked_path(default_asset_directory(database_asset_namespace(database)))


def _distinct_targets(*paths: Path) -> None:
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise BackupError("Backup paths must not overlap")


def _require_stable_key(store: KeyStore, database: Path, envelope: bytes | None = None) -> None:
    store.require_no_restore()
    if (
        checked_path(store.rotation_file).exists()
        or checked_path(rotation_candidate_path(database)).exists()
    ):
        raise BackupError("Resolve pending database-key rotation before backup or restore")
    if envelope is not None and read_private(store.key_file) != envelope:
        raise BackupError("Key envelope changed during backup; stop key management and retry")


def _unlock(store: KeyStore, prompt: SecretPrompt) -> DatabaseKeyHandle:
    passphrase = prompt("Aevon unlock passphrase: ")
    try:
        # No database_path: do not implicitly recover a rotation in backup code.
        return store.unlock(passphrase)
    finally:
        passphrase = ""


def _assets(
    snapshot: Path,
    key: DatabaseKeyInput,
    storage: AssetStorage,
    *,
    destination: Path | None = None,
) -> dict[str, Entry]:
    if not database_key_opens(snapshot, key):
        raise BackupError("Encrypted backup validation failed")
    database = Database.from_url(f"sqlite+pysqlite:///{snapshot.as_posix()}", encryption_key=key)
    try:
        # No ORM result logging and no schema creation/migration in a backup.
        with (
            closing(database.engine.raw_connection()) as connection,
            closing(connection.cursor()) as cursor,
        ):
            cursor.execute(
                "SELECT id, byte_size, sha256, width, height FROM uploaded_assets ORDER BY id"
            )
            rows = cursor.fetchmany(MAX_ASSETS + 1)
        if len(rows) > MAX_ASSETS:
            raise BackupError("Backup exceeds V1 asset limit")
        # Backup/restore MUST NOT create a replacement AEK, even for an empty DB.
        with load_asset_key(database.engine, storage, create_if_missing=False) as asset_key:
            entries = {}
            total_size = 0
            for identifier, size, digest, width, height in rows:
                canonical_uuid(identifier)
                encrypted = storage.read_ciphertext(identifier)
                total_size += len(encrypted)
                if total_size > MAX_ARCHIVE_BYTES:
                    raise BackupError("Backup exceeds V1 size limit")
                plaintext = b""
                try:
                    plaintext = decrypt_asset(identifier, encrypted, asset_key.copy_bytes())
                    validate_normalized_png(plaintext, expected=(size, digest, width, height))
                finally:
                    plaintext = b""  # immutable library copies cannot be guaranteed zeroized
                name = f"assets/{identifier}.asset"
                entries[name] = Entry(len(encrypted), hashlib.sha256(encrypted).hexdigest())
                if destination is not None:
                    new_private_file(destination / name, encrypted)
            return entries
    finally:
        database.engine.dispose()


def create_backup(
    output: Path,
    database_file: Path,
    store: KeyStore,
    *,
    prompt: SecretPrompt = getpass.getpass,
    phase_hook: PhaseHook | None = None,
) -> str:
    try:
        output = checked_path(output, absolute=True)
        database = checked_path(database_file)
        checked_path(store.key_file)
        if output.exists():
            raise BackupError("Backup output already exists; choose a new filename")
        if output.suffix != ".amitai-backup" or output in {database, store.key_file}:
            raise BackupError("Choose an absolute .amitai-backup output filename")
        _distinct_targets(output, database, store.key_file, _asset_root(database))
        regular_file(database, MAX_DATABASE_BYTES)
        _require_stable_key(store, database)
        envelope = read_private(store.key_file)
        with _unlock(store, prompt) as key, private_stage(output.parent) as stage:
            _require_stable_key(store, database, envelope)
            new_private_file(stage / ENVELOPE, envelope)
            new_private_file(stage / DATABASE)
            export_encrypted_snapshot(database, key, stage / DATABASE, phase_hook=phase_hook)
            regular_file(stage / DATABASE, MAX_DATABASE_BYTES)
            _hook(phase_hook, "snapshot_verified")
            entries = _assets(
                stage / DATABASE, key, AssetStorage(_asset_root(database)), destination=stage
            )
            entries[DATABASE] = Entry(*file_fingerprint(stage / DATABASE, MAX_DATABASE_BYTES))
            entries[ENVELOPE] = Entry(len(envelope), hashlib.sha256(envelope).hexdigest())
            manifest = Manifest(str(uuid4()), entries)
            candidate = stage / "archive.bin"
            new_private_file(candidate)
            write_archive(candidate, stage, manifest)
            with validated_archive(candidate) as (_, verified):
                if verified != manifest:
                    raise BackupError("Backup verification failed")
            _hook(phase_hook, "archive_verified")
            _require_stable_key(store, database, envelope)
            install_new(candidate, output)
            return manifest.backup_id
    except BackupError:
        raise
    except Exception:  # noqa: BLE001 - filesystem, SQL, key and ZIP details are private
        raise BackupError("Backup creation failed") from None


def _sidecars(database: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal"))


def _targets_empty(database: Path, store: KeyStore, assets: Path) -> None:
    for path in (
        database,
        store.key_file,
        store.rotation_file,
        rotation_candidate_path(database),
        *_sidecars(database),
    ):
        if checked_path(path).exists():
            raise BackupError("Restore target already exists; choose empty destinations")
    if assets.exists():
        assert_owner_only(assets, directory=True)
        if any(assets.iterdir()):
            raise BackupError("Restore asset destination is not empty")


def _matches(path: Path, entry: Entry) -> bool:
    assert_owner_only(path, directory=False)
    return file_fingerprint(path, entry.size) == entry.fingerprint


def _resume_targets(database: Path, store: KeyStore, assets: Path, manifest: Manifest) -> None:
    for path in (store.rotation_file, rotation_candidate_path(database), *_sidecars(database)):
        if checked_path(path).exists():
            raise BackupError("Restore recovery targets do not match the backup")
    for path, name in ((database, DATABASE), (store.key_file, ENVELOPE)):
        if checked_path(path).exists() and not _matches(path, manifest.entries[name]):
            raise BackupError("Restore recovery targets do not match the backup")
    if assets.exists():
        assert_owner_only(assets, directory=True)
        for path in assets.iterdir():
            name = f"assets/{path.name}"
            if name not in manifest.entries or not _matches(path, manifest.entries[name]):
                raise BackupError("Restore recovery assets do not match the backup")


def _install(source: Path, target: Path, entry: Entry, *, resume: bool) -> None:
    if checked_path(target).exists():
        if resume and _matches(target, entry):
            return
        raise BackupError("Restore destination changed; nothing will be overwritten")
    install_new(source, target)


def restore_backup(
    source: Path,
    database_file: Path,
    store: KeyStore,
    *,
    resume: bool = False,
    prompt: SecretPrompt = getpass.getpass,
    phase_hook: PhaseHook | None = None,
) -> str:
    try:
        source = checked_path(source)
        database = checked_path(database_file)
        checked_path(store.key_file)
        assets = _asset_root(database)
        marker = checked_path(restore_journal_path(store.key_file))
        _distinct_targets(source, database, store.key_file, marker, assets)
        if marker.exists() != resume:
            raise BackupError(
                RECOVERY_MESSAGE if marker.exists() else "No interrupted restore to resume"
            )
        if not resume:
            _targets_empty(database, store, assets)
        # Entire container validation precedes extraction, prompting and final writes.
        with validated_archive(source) as (archive, manifest):
            ensure_private_directory(store.key_file.parent)
            with private_stage(store.key_file.parent) as stage:
                extract_verified(archive, manifest, stage)
                staged_store = KeyStore(
                    stage / ENVELOPE, policy=store.policy, validate_location=False
                )
                with _unlock(staged_store, prompt) as key:
                    entries = _assets(stage / DATABASE, key, AssetStorage(stage / "assets"))
                if entries != {
                    name: item
                    for name, item in manifest.entries.items()
                    if name.startswith("assets/")
                }:
                    raise BackupError("Backup assets do not match the database snapshot")
                # Prepare copies on each target filesystem; no cross-volume renames
                # or unverified final copies. DB and key are installed without overwrite.
                with (
                    private_stage(database.parent) as db_stage,
                    private_stage(assets.parent) as asset_stage,
                ):
                    for name, entry in manifest.entries.items():
                        if name == ENVELOPE:
                            continue
                        destination = (
                            db_stage / DATABASE
                            if name == DATABASE
                            else asset_stage / Path(name).name
                        )
                        with (stage / name).open("rb") as member:
                            if stage_copy(member, destination, entry.size) != entry.fingerprint:
                                raise BackupError("Restore staging verification failed")
                    _hook(phase_hook, "restore_verified")
                    marker_content = json.dumps(
                        {"backup_id": manifest.backup_id, "phase": "installing"},
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                    if not resume:
                        _targets_empty(database, store, assets)
                        new_private_file(stage / "restore-state", marker_content)
                        install_new(stage / "restore-state", marker)
                    with journal_lock(marker) as locked:
                        document = strict_json(locked.read(1025))
                        if document != {"backup_id": manifest.backup_id, "phase": "installing"}:
                            raise BackupError("Restore recovery does not match this backup")
                        _resume_targets(database, store, assets, manifest)
                        _hook(phase_hook, "restore_marked")
                        ensure_private_directory(assets)
                        _fsync_directory(assets.parent)
                        for name, entry in manifest.entries.items():
                            if name.startswith("assets/"):
                                _install(
                                    asset_stage / Path(name).name,
                                    assets / Path(name).name,
                                    entry,
                                    resume=resume,
                                )
                        _fsync_directory(assets)
                        _hook(phase_hook, "assets_installed")
                        _install(
                            db_stage / DATABASE, database, manifest.entries[DATABASE], resume=resume
                        )
                        _hook(phase_hook, "database_installed")
                        _install(
                            stage / ENVELOPE,
                            store.key_file,
                            manifest.entries[ENVELOPE],
                            resume=resume,
                        )
                        _hook(phase_hook, "key_installed")
                    # Key publication is last. A leftover marker blocks normal
                    # unlock/mutations until a fully revalidated --resume finishes.
                    if read_private(marker) != marker_content:
                        raise BackupError("Restore recovery state changed")
                    marker.unlink()
                    _fsync_directory(marker.parent)
                    return manifest.backup_id
    except BackupError:
        raise
    except Exception:  # noqa: BLE001 - no archive, database, key, or path diagnostics
        raise BackupError(
            "Restore failed; if interrupted, retry the same command with --resume"
        ) from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default message echoes invalid arguments, including secrets
        # mistakenly supplied on the command line. Do not repeat any argv values.
        self.exit(2, "Invalid backup command; use --help for supported arguments.\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Encrypted local AmitAI backup and recovery", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "restore"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument(
            "--output" if name == "create" else "--input", type=Path, required=True
        )
        command.add_argument("--database-file", type=Path, default=Path("amitai.db"))
        command.add_argument("--key-file", type=Path, default=None)
        if name == "restore":
            command.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, prompt: SecretPrompt = getpass.getpass) -> None:
    arguments = _parser().parse_args(argv)
    try:
        apply_process_hardening()
        store = KeyStore(checked_path(arguments.key_file or default_key_file()))
        if arguments.command == "create":
            create_backup(arguments.output, arguments.database_file, store, prompt=prompt)
            print("Encrypted local backup created.")
        else:
            restore_backup(
                arguments.input,
                arguments.database_file,
                store,
                prompt=prompt,
                resume=arguments.resume,
            )
            print("Encrypted local backup restored.")
    except BackupError as exc:
        raise SystemExit(str(exc)) from None
    except Exception:  # noqa: BLE001 - never print native/platform exceptions or secret input
        raise SystemExit("Local backup operation failed") from None


if __name__ == "__main__":
    main()
