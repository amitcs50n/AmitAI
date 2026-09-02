"""Local control-plane entrypoint with replaceable inference providers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from fastapi import FastAPI

from backend.app import LazyConfiguredApplication
from backend.app import create_app as create_backend_app
from backend.chat_service import ResponseGenerator
from backend.database import DEFAULT_DATABASE_URL, DatabaseKeyInput

from .config import DEFAULT_RUNTIME_CONFIG_PATH, RuntimeConfig, load_production_runtime_config
from .generator import ProviderChatGenerator, TransformersChatGenerator
from .providers import InferenceProvider, RemoteInferenceProvider

RuntimeGeneratorFactory = Callable[[RuntimeConfig], ResponseGenerator]
RemoteProviderFactory = Callable[..., InferenceProvider]


def select_response_generator(
    *,
    mode: str | None = None,
    config_path: str | Path | None = None,
    generator_factory: RuntimeGeneratorFactory = TransformersChatGenerator,
    remote_endpoint: str | None = None,
    remote_token: str | None = None,
    remote_allowed_origins: str | Sequence[str] | None = None,
    remote_provider_factory: RemoteProviderFactory = RemoteInferenceProvider,
) -> ResponseGenerator | None:
    selected_mode = (
        mode
        or os.getenv("AMITAI_INFERENCE_PROVIDER")
        or os.getenv("AMITAI_GENERATOR", "mock")
    ).strip().lower()
    if selected_mode == "mock":
        return None
    if selected_mode not in {"transformers", "remote"}:
        raise ValueError(
            "Unsupported inference provider. Use 'mock', 'transformers', or 'remote'."
        )

    selected_config = config_path or os.getenv(
        "AMITAI_RUNTIME_CONFIG",
        str(DEFAULT_RUNTIME_CONFIG_PATH),
    )
    config = load_production_runtime_config(selected_config)
    if selected_mode == "transformers":
        return generator_factory(config)

    endpoint = remote_endpoint if remote_endpoint is not None else os.getenv("AMITAI_REMOTE_INFERENCE_URL")
    token = remote_token if remote_token is not None else os.getenv("AMITAI_REMOTE_INFERENCE_TOKEN")
    if not endpoint or not token:
        raise ValueError(
            "Remote inference requires AMITAI_REMOTE_INFERENCE_URL and "
            "AMITAI_REMOTE_INFERENCE_TOKEN"
        )
    provider = remote_provider_factory(
        endpoint=endpoint,
        token=token,
        model_name=str(config.model["name"]),
        allowed_origins=(
            remote_allowed_origins if remote_allowed_origins is not None
            else os.getenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS")
        ),
    )
    return ProviderChatGenerator(config, provider=provider)


def create_runtime_app(
    database_url: str = DEFAULT_DATABASE_URL,
    *,
    database_key: DatabaseKeyInput | None = None,
    encrypted_storage: bool = True,
    encrypt_existing_database: bool = False,
    mode: str | None = None,
    config_path: str | Path | None = None,
    generator_factory: RuntimeGeneratorFactory = TransformersChatGenerator,
    remote_endpoint: str | None = None,
    remote_token: str | None = None,
    remote_allowed_origins: str | Sequence[str] | None = None,
    remote_provider_factory: RemoteProviderFactory = RemoteInferenceProvider,
    local_api_token: str | None = None,
    enforce_local_auth: bool = True,
    enable_dev_docs: bool = False,
) -> FastAPI:
    generator = select_response_generator(
        mode=mode,
        config_path=config_path,
        generator_factory=generator_factory,
        remote_endpoint=remote_endpoint,
        remote_token=remote_token,
        remote_allowed_origins=remote_allowed_origins,
        remote_provider_factory=remote_provider_factory,
    )
    return create_backend_app(
        database_url,
        database_key=database_key,
        encrypted_storage=encrypted_storage,
        encrypt_existing_database=encrypt_existing_database,
        generator=generator,
        local_api_token=local_api_token,
        enforce_local_auth=enforce_local_auth,
        enable_dev_docs=enable_dev_docs,
    )


def create_configured_app() -> FastAPI:
    """Fail closed when bypassing the interactive secure launcher."""

    raise RuntimeError(
        "Direct runtime ASGI startup is unsupported; use python -m runtime.serve"
    )


app = LazyConfiguredApplication(create_configured_app)
