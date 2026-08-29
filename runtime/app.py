"""Runtime-aware FastAPI entrypoint for mock or real model generation."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from backend.app import create_app as create_backend_app
from backend.chat_service import ResponseGenerator
from backend.database import DEFAULT_DATABASE_URL

from .config import DEFAULT_RUNTIME_CONFIG_PATH, RuntimeConfig, load_runtime_config
from .generator import TransformersChatGenerator

RuntimeGeneratorFactory = Callable[[RuntimeConfig], ResponseGenerator]


def select_response_generator(
    *,
    mode: str | None = None,
    config_path: str | Path | None = None,
    generator_factory: RuntimeGeneratorFactory = TransformersChatGenerator,
) -> ResponseGenerator | None:
    selected_mode = (mode or os.getenv("AMITAI_GENERATOR", "mock")).strip().lower()
    if selected_mode == "mock":
        return None
    if selected_mode != "transformers":
        raise ValueError("Unsupported AMITAI_GENERATOR value. Use 'mock' or 'transformers'.")

    selected_config = config_path or os.getenv(
        "AMITAI_RUNTIME_CONFIG",
        str(DEFAULT_RUNTIME_CONFIG_PATH),
    )
    return generator_factory(load_runtime_config(selected_config))


def create_runtime_app(
    database_url: str = DEFAULT_DATABASE_URL,
    *,
    mode: str | None = None,
    config_path: str | Path | None = None,
    generator_factory: RuntimeGeneratorFactory = TransformersChatGenerator,
) -> FastAPI:
    generator = select_response_generator(
        mode=mode,
        config_path=config_path,
        generator_factory=generator_factory,
    )
    return create_backend_app(database_url, generator=generator)


app = create_runtime_app()
