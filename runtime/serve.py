"""Canonical loopback-first launcher for the local AmitAI control plane."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass

from backend.database import validate_database_key
from backend.security import environment_flag, validate_local_api_token


@dataclass(frozen=True)
class LocalServerConfig:
    host: str
    port: int


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def load_local_server_config(
    environ: Mapping[str, str] | None = None,
) -> LocalServerConfig:
    values = os.environ if environ is None else environ
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

    if validate_local_api_token(values.get("AMITAI_LOCAL_API_TOKEN")) is None:
        raise ValueError("AMITAI_LOCAL_API_TOKEN must be configured")
    validate_database_key(values.get("AMITAI_DB_KEY"))
    return LocalServerConfig(host=host, port=port)


def main() -> None:
    import uvicorn

    config = load_local_server_config()
    uvicorn.run(
        "runtime.app:app",
        host=config.host,
        port=config.port,
        workers=1,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
