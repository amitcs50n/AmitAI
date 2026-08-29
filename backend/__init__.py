"""Persistent HTTP backend for AmitAI."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Expose the app factory without creating the default database on package import."""

    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
