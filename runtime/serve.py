"""Canonical interactive and loopback-first AmitAI control-plane launcher."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.database import EncryptedStorageError
from backend.security import environment_flag

from .app import create_runtime_app
from .key_store import KeyStore, KeyStoreError, UnlockError
from .paths import (
    PrivatePathError,
    atomic_write_private,
    default_key_file,
    default_runtime_token_file,
    remove_private,
)
from .process_hardening import ProcessHardeningError, apply_process_hardening
from .secure_memory import SecureMemoryError

LEGACY_SECRET_ENVIRONMENT = (
    "AMITAI_DB_KEY",
    "AMITAI_LOCAL_API_TOKEN",
    "AMITAI_UNLOCK_PASSPHRASE",
)


@dataclass(frozen=True)
class LocalServerConfig:
    host: str
    port: int
    encrypt_existing_database: bool = False
    enable_dev_docs: bool = False


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def reject_legacy_secret_environment(environ: Mapping[str, str]) -> None:
    if any(name in environ for name in LEGACY_SECRET_ENVIRONMENT):
        raise ValueError(
            "Legacy secret environment variables are not supported by secure startup"
        )


def load_local_server_config(
    environ: Mapping[str, str] | None = None,
) -> LocalServerConfig:
    values = os.environ if environ is None else environ
    reject_legacy_secret_environment(values)
    host = values.get("AMITAI_HOST", "127.0.0.1").strip()
    if not host:
        raise ValueError("AMITAI_HOST must not be empty")
    allow_lan = environment_flag("AMITAI_ALLOW_LAN", environ=values)
    if not _is_loopback_host(host) and not allow_lan:
        raise ValueError(
            "Refusing non-loopback control-plane host; set AMITAI_ALLOW_LAN=1 "
            "only for an intentional LAN deployment"
        )

    raw_port = values.get("AMITAI_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("AMITAI_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("AMITAI_PORT must be between 1 and 65535")

    return LocalServerConfig(
        host=host,
        port=port,
        encrypt_existing_database=environment_flag(
            "AMITAI_ENCRYPT_EXISTING_DB",
            environ=values,
        ),
        enable_dev_docs=environment_flag(
            "AMITAI_ENABLE_DEV_DOCS",
            environ=values,
        ),
    )


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.expanduser().resolve().as_posix()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the secure AmitAI control plane")
    parser.add_argument("--key-file", type=Path, default=None)
    parser.add_argument("--database-file", type=Path, default=Path("amitai.db"))
    return parser


def run_secure_server(
    *,
    config: LocalServerConfig,
    key_file: Path,
    database_file: Path,
    token_file: Path,
    passphrase_prompt: Callable[[str], str] = getpass.getpass,
    uvicorn_run: Callable[..., None] | None = None,
) -> None:
    apply_process_hardening()
    store = KeyStore(key_file)
    passphrase = passphrase_prompt("AmitAI unlock passphrase: ")
    try:
        key_handle = store.unlock(passphrase, database_path=database_file)
    finally:
        passphrase = ""
    local_token = secrets.token_hex(32)
    application = None
    try:
        atomic_write_private(token_file, (local_token + "\n").encode("ascii"))
        application = create_runtime_app(
            _database_url(database_file),
            database_key=key_handle,
            encrypted_storage=True,
            encrypt_existing_database=config.encrypt_existing_database,
            local_api_token=local_token,
            enforce_local_auth=True,
            enable_dev_docs=config.enable_dev_docs,
        )
        if uvicorn_run is None:
            import uvicorn

            selected_run = uvicorn.run
        else:
            selected_run = uvicorn_run
        selected_run(
            application,
            host=config.host,
            port=config.port,
            workers=1,
            reload=False,
            access_log=False,
        )
    finally:
        try:
            if application is not None:
                application.state.database.engine.dispose()
        finally:
            try:
                remove_private(token_file)
            finally:
                key_handle.close()
                local_token = ""


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    config = load_local_server_config()
    key_file = arguments.key_file or default_key_file()
    token_file = default_runtime_token_file()
    try:
        run_secure_server(
            config=config,
            key_file=key_file,
            database_file=arguments.database_file.expanduser().resolve(),
            token_file=token_file,
        )
    except UnlockError:
        raise SystemExit("Unlock failed") from None
    except (
        EncryptedStorageError,
        KeyStoreError,
        PrivatePathError,
        ProcessHardeningError,
        SecureMemoryError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
