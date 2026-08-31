"""Explicit unauthenticated application factories for deterministic legacy tests."""

from typing import Any

from fastapi import FastAPI

from backend.app import create_app
from runtime.app import create_runtime_app


def create_test_app(*args: Any, **kwargs: Any) -> FastAPI:
    kwargs["enforce_local_auth"] = False
    return create_app(*args, **kwargs)


def create_test_runtime_app(*args: Any, **kwargs: Any) -> FastAPI:
    kwargs["enforce_local_auth"] = False
    return create_runtime_app(*args, **kwargs)
