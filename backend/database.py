"""Database setup shared by the API and repository layers."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DATABASE_URL = "sqlite:///./amitai.db"


class Base(DeclarativeBase):
    """Base class for backend ORM models."""


@dataclass(frozen=True)
class Database:
    """Own an engine and its session factory without leaking SQLite details."""

    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def from_url(cls, database_url: str = DEFAULT_DATABASE_URL) -> "Database":
        url = make_url(database_url)
        engine_options: dict[str, object] = {"pool_pre_ping": True}

        if url.get_backend_name() == "sqlite":
            engine_options["connect_args"] = {"check_same_thread": False}
            if url.database in {None, "", ":memory:"}:
                engine_options["poolclass"] = StaticPool

        engine = create_engine(database_url, **engine_options)

        if url.get_backend_name() == "sqlite":

            @event.listens_for(engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
                try:
                    cursor.execute("PRAGMA foreign_keys=ON")
                finally:
                    cursor.close()

        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        return cls(engine=engine, session_factory=factory)

    def create_schema(self) -> None:
        # Importing registers all mapped classes on Base.metadata.
        from . import models as _models  # noqa: F401

        Base.metadata.create_all(self.engine)
