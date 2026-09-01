"""Interactive management for AmitAI's wrapped local database key."""

from __future__ import annotations

import argparse
import getpass
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from backend.database import EncryptedStorageError

from .key_store import (
    MIN_PASSPHRASE_CHARS,
    ROTATION_RECOVERY_REQUIRED_MESSAGE,
    KeyRotationError,
    KeyStore,
    KeyStoreError,
)
from .paths import PrivatePathError, default_key_file
from .secure_memory import SecureMemoryError

SecretPrompt = Callable[[str], str]


def _confirmed_passphrase(prompt: SecretPrompt, *, label: str = "New") -> str:
    first = prompt(f"{label} unlock passphrase: ")
    second = prompt(f"Confirm {label.lower()} unlock passphrase: ")
    if first != second:
        raise KeyStoreError("Unlock passphrases did not match")
    if len(first) < MIN_PASSPHRASE_CHARS:
        raise KeyStoreError(
            f"Unlock passphrase must contain at least {MIN_PASSPHRASE_CHARS} characters"
        )
    return first


def _store(arguments: argparse.Namespace) -> KeyStore:
    return KeyStore(arguments.key_file or default_key_file())


def _database(arguments: argparse.Namespace) -> Path:
    return arguments.database_file.expanduser().resolve()


def command_init(
    arguments: argparse.Namespace,
    *,
    prompt: SecretPrompt = getpass.getpass,
) -> None:
    passphrase = _confirmed_passphrase(prompt)
    try:
        _store(arguments).initialize(passphrase, database_path=_database(arguments))
    finally:
        passphrase = ""


def command_import_existing(
    arguments: argparse.Namespace,
    *,
    prompt: SecretPrompt = getpass.getpass,
) -> None:
    database_key = prompt("Current 64-hex SQLCipher database key: ")
    passphrase = ""
    try:
        passphrase = _confirmed_passphrase(prompt)
        _store(arguments).import_existing(
            _database(arguments),
            database_key,
            passphrase,
        )
    finally:
        database_key = ""
        passphrase = ""


def command_change_passphrase(
    arguments: argparse.Namespace,
    *,
    prompt: SecretPrompt = getpass.getpass,
) -> None:
    store = _store(arguments)
    if store.rotation_recovery_required:
        raise KeyRotationError(ROTATION_RECOVERY_REQUIRED_MESSAGE)
    old_passphrase = prompt("Current unlock passphrase: ")
    new_passphrase = ""
    try:
        new_passphrase = _confirmed_passphrase(prompt)
        store.change_passphrase(old_passphrase, new_passphrase)
    finally:
        old_passphrase = ""
        new_passphrase = ""


def command_rotate_database_key(
    arguments: argparse.Namespace,
    *,
    prompt: SecretPrompt = getpass.getpass,
) -> None:
    passphrase = prompt("Current unlock passphrase: ")
    try:
        _store(arguments).rotate_database_key(_database(arguments), passphrase)
    finally:
        passphrase = ""


def command_status(arguments: argparse.Namespace) -> None:
    status = _store(arguments).status()
    print(json.dumps(status.__dict__, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage AmitAI secure local keys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_paths(command: str, *, database: bool) -> argparse.ArgumentParser:
        selected = subparsers.add_parser(command)
        selected.add_argument("--key-file", type=Path, default=None)
        if database:
            selected.add_argument(
                "--database-file",
                type=Path,
                default=Path("amitai.db"),
            )
        return selected

    add_paths("init", database=True).set_defaults(handler=command_init)
    add_paths("import-existing", database=True).set_defaults(
        handler=command_import_existing
    )
    add_paths("change-passphrase", database=False).set_defaults(
        handler=command_change_passphrase
    )
    add_paths("rotate-db-key", database=True).set_defaults(
        handler=command_rotate_database_key
    )
    add_paths("status", database=False).set_defaults(handler=command_status)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prompt: SecretPrompt = getpass.getpass,
) -> None:
    arguments = _parser().parse_args(argv)
    handler = arguments.handler
    try:
        if handler is command_status:
            handler(arguments)
        else:
            handler(arguments, prompt=prompt)
    except (
        EncryptedStorageError,
        KeyStoreError,
        PrivatePathError,
        SecureMemoryError,
    ) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
