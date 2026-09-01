from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from backend.app import create_app
from backend.database import Base, Database
from backend.memory import MemoryConflictError, MemoryService
from backend.models import MemoryRevision, MemorySlot
from tests.app_factory import create_test_app


def test_memory_policy_defaults_explicit_updates_and_chat_preservation() -> None:
    app = create_test_app("sqlite+pysqlite:///:memory:")
    with TestClient(app) as client:
        created = client.post("/api/memory", json={
            "category": "preference", "key": "ui.theme", "value": "dark",
        }).json()
        assert created["sensitivity"] == "local_only"
        route = f"/api/memory/{created['id']}"
        allowed = client.patch(route, json={"sensitivity": "remote_allowed"})
        assert allowed.status_code == 200
        assert allowed.json()["value"] == "dark"
        assert allowed.json()["sensitivity"] == "remote_allowed"
        changed = client.patch(route, json={"value": "light"})
        assert changed.json()["sensitivity"] == "remote_allowed"
        for command in ("Remember preference ui.theme: blue", "Update preference ui.theme: green"):
            assert client.post("/api/chat", json={"message": command}).status_code == 200
            assert client.get("/api/memory").json()[0]["sensitivity"] == "remote_allowed"
        both = client.patch(route, json={"value": "amber", "sensitivity": "local_only"})
        assert both.json()["sensitivity"] == "local_only"
        assert both.json()["value"] == "amber"
        client.patch(route, json={"sensitivity": "remote_allowed"})
        assert client.delete(route).status_code == 204
        assert client.patch(route, json={"sensitivity": "local_only"}).status_code == 409
        assert client.get("/api/memory", params={"status": "deleted"}).json()[0]["value"] is None
        assert client.post("/api/chat", json={
            "message": "Remember preference ui.theme: new explicit value",
        }).status_code == 200
        record = client.get("/api/memory").json()[0]
        assert record["id"] == created["id"]
        assert record["sensitivity"] == "local_only"
        assert record["value"] == "new explicit value"
        assert client.post("/api/chat", json={
            "message": "Remember profile display.name: Alice",
        }).status_code == 200
        assert next(record for record in client.get("/api/memory").json()
                    if record["key"] == "display.name")["sensitivity"] == "local_only"
        with app.state.database.session_factory() as session:
            old_revisions = list(session.scalars(select(MemoryRevision).where(
                MemoryRevision.memory_id == created["id"], MemoryRevision.status != "active",
            )))
            assert old_revisions and all(item.value is None for item in old_revisions)


@pytest.mark.parametrize("invalid", ["public", "private", "remote", "local", "sensitive", "", None, 1])
def test_invalid_memory_sensitivity_is_rejected_on_create_and_patch(invalid) -> None:
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:")) as client:
        body = {"category": "project", "key": "project.name", "value": "Aevon"}
        assert client.post("/api/memory", json={**body, "sensitivity": invalid}).status_code == 422
        memory = client.post("/api/memory", json=body).json()
        assert client.patch(f"/api/memory/{memory['id']}", json={
            "sensitivity": invalid,
        }).status_code == 422
        assert client.get("/api/memory").json()[0]["sensitivity"] == "local_only"


@pytest.mark.parametrize("body", [{}, {"value": None}, {"value": "okay", "sensitivity": None}])
def test_empty_or_explicit_null_patch_fails(body) -> None:
    with TestClient(create_test_app("sqlite+pysqlite:///:memory:")) as client:
        assert client.patch("/api/memory/missing", json=body).status_code == 422


def test_policy_change_participates_in_revision_conflict_checks(tmp_path: Path) -> None:
    db = Database.from_url(f"sqlite+pysqlite:///{(tmp_path / 'memory.db').as_posix()}", encrypted=False)
    db.create_schema()
    with db.session_factory() as session, session.begin():
        service = MemoryService(session)
        memory = service.apply(service.stage_create(category="project", key="name", value="old"))
    with db.session_factory() as first, db.session_factory() as second:
        with first.begin():
            stale = MemoryService(first).stage_update(memory["id"], value="stale edit")
        with second.begin():
            service = MemoryService(second)
            service.apply(service.stage_update(memory["id"], sensitivity="remote_allowed"))
        with pytest.raises(MemoryConflictError), first.begin():
            MemoryService(first).apply(stale)
    with db.session_factory() as session:
        record = MemoryService(session).list_memories()[0]
        assert record["value"] == "old"
        assert record["sensitivity"] == "remote_allowed"
    db.engine.dispose()


# The real pre-sensitivity table definition, not a new-schema table with a column removed.
LEGACY_MEMORY_SLOTS = """
CREATE TABLE memory_slots (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    owner_id VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL,
    key VARCHAR(128) NOT NULL,
    status VARCHAR(16) NOT NULL CHECK (status IN ('active', 'deleted')),
    current_revision INTEGER NOT NULL CHECK (current_revision >= 1),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    deleted_at DATETIME,
    UNIQUE (owner_id, category, key)
)
"""


@pytest.mark.parametrize("encrypted", [False, True])
def test_pre_sensitivity_schema_upgrade_is_safe_and_idempotent(tmp_path: Path, encrypted: bool) -> None:
    path = tmp_path / "legacy.db"
    url = f"sqlite+pysqlite:///{path.as_posix()}"
    key = "31" * 32 if encrypted else None
    database = Database.from_url(url, encrypted=encrypted, encryption_key=key)
    timestamp = "2026-08-30 10:00:00.000000"
    with database.engine.begin() as connection:
        connection.exec_driver_sql(LEGACY_MEMORY_SLOTS)
        connection.exec_driver_sql(
            "CREATE INDEX ix_memory_slots_owner_status_category_updated "
            "ON memory_slots (owner_id, status, category, updated_at)"
        )
        for table in Base.metadata.sorted_tables:
            if table.name != "memory_slots":
                table.create(connection)
        connection.exec_driver_sql(
            "INSERT INTO conversations (id, title, created_at, updated_at, archived) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-conversation", "Local history", timestamp, timestamp, 0),
        )
        connection.exec_driver_sql(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-message", "legacy-conversation", "user", "Unchanged local text", timestamp),
        )
        connection.exec_driver_sql(
            "INSERT INTO memory_slots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy-id", "local-default", "project", "legacy.name", "active", 2,
             timestamp, timestamp, None),
        )
        connection.exec_driver_sql(
            "INSERT INTO memory_slots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("deleted-id", "local-default", "project", "deleted.name", "deleted", 1,
             timestamp, timestamp, timestamp),
        )
        for revision_id, slot_id, revision, value, status in (
            ("rev1", "legacy-id", 1, "LEGACY_OLD_VALUE", "stale"),
            ("rev2", "legacy-id", 2, "LEGACY_CURRENT_VALUE", "active"),
            ("rev3", "deleted-id", 1, None, "deleted"),
        ):
            connection.exec_driver_sql(
                "INSERT INTO memory_revisions "
                "(id, memory_id, revision, value, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (revision_id, slot_id, revision, value, status, timestamp),
            )
        connection.exec_driver_sql(
            "UPDATE memory_revisions SET source_conversation_id='legacy-conversation', "
            "source_message_id='legacy-message' WHERE memory_id='legacy-id'"
        )
        before = connection.exec_driver_sql("SELECT * FROM memory_revisions ORDER BY id").all()
        assert "sensitivity" not in {c["name"] for c in inspect(connection).get_columns("memory_slots")}
    database.engine.dispose()

    for _ in range(2):
        # Lifespan must upgrade before even the first memory read, including SQLCipher.
        app = create_app(url, encrypted_storage=encrypted, database_key=key, enforce_local_auth=False)
        with TestClient(app) as client:
            active = client.get("/api/memory").json()
            assert active[0]["sensitivity"] == "local_only"
            assert active[0]["value"] == "LEGACY_CURRENT_VALUE"
            assert active[0]["source"] == {
                "conversation_id": "legacy-conversation", "message_id": "legacy-message",
            }
            history = client.get("/api/conversations/legacy-conversation").json()
            assert history["messages"][0]["content"] == "Unchanged local text"
            deleted = client.get("/api/memory", params={"status": "deleted"}).json()
            assert deleted[0]["sensitivity"] == "local_only"
            assert deleted[0]["value"] is None
            with app.state.database.engine.connect() as connection:
                assert connection.exec_driver_sql("SELECT * FROM memory_revisions ORDER BY id").all() == before
                assert connection.exec_driver_sql(
                    "SELECT current_revision, created_at FROM memory_slots WHERE id='legacy-id'"
                ).one() == (2, timestamp)
            with pytest.raises(IntegrityError), app.state.database.engine.begin() as connection:
                connection.exec_driver_sql("UPDATE memory_slots SET sensitivity='public'")
    if encrypted:
        assert not path.read_bytes().startswith(b"SQLite format 3")
        assert b"LEGACY_CURRENT_VALUE" not in path.read_bytes()
        assert b"LEGACY_OLD_VALUE" not in path.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.db"]


def test_new_schema_rejects_invalid_sensitivity_structurally() -> None:
    database = Database.from_url("sqlite+pysqlite:///:memory:", encrypted=False)
    database.create_schema()
    with database.session_factory() as session, pytest.raises(IntegrityError), session.begin():
        session.add(MemorySlot(owner_id="local-default", category="project", key="name", sensitivity="public"))
        session.flush()
    database.engine.dispose()
