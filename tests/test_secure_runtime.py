from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.database import Database, EncryptedStorageError
from runtime.key_store import KeyStore, KeyStorePolicy, UnlockError
from runtime.paths import assert_owner_only
from runtime.process_hardening import apply_process_hardening
from runtime.secure_memory import DatabaseKeyHandle
from runtime.serve import LocalServerConfig, load_local_server_config, run_secure_server

PASSPHRASE = "secure runtime passphrase"
DATABASE_KEY = b"RUNTIME_DB_KEY_CANARY_1234567890"
TEST_POLICY = KeyStorePolicy.for_tests()


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _initialized_store(tmp_path: Path) -> tuple[KeyStore, Path, Path, Path]:
    key_file = tmp_path / "secrets" / "database-key.json"
    token_file = tmp_path / "runtime" / "local-api-token"
    database_file = tmp_path / "amitai.db"
    store = KeyStore(key_file, policy=TEST_POLICY)
    store.initialize(PASSPHRASE, database_key=DATABASE_KEY)
    return store, key_file, token_file, database_file


def test_canonical_configuration_rejects_every_legacy_secret_environment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canaries = {
        "AMITAI_DB_KEY": "DB_ENV_CANARY_918273",
        "AMITAI_LOCAL_API_TOKEN": "LOCAL_ENV_CANARY_817263",
        "AMITAI_UNLOCK_PASSPHRASE": "PASS_ENV_CANARY_716253",
    }

    with pytest.raises(ValueError) as failure:
        load_local_server_config(canaries)

    assert str(failure.value) == (
        "Legacy secret environment variables are not supported by secure startup"
    )
    for canary in canaries.values():
        assert canary not in str(failure.value)
        assert canary not in caplog.text

    clean = load_local_server_config({})
    assert clean.host == "127.0.0.1"
    assert clean.port == 8000


def test_launcher_uses_app_object_ephemeral_token_and_cleans_up_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, key_file, token_file, database_file = _initialized_store(tmp_path)
    monkeypatch.setattr(
        "runtime.serve.KeyStore",
        lambda _path: KeyStore(key_file, policy=TEST_POLICY),
    )
    monkeypatch.setenv("AMITAI_GENERATOR", "mock")
    tokens: list[str] = []
    uvicorn_calls: list[dict] = []

    def fake_uvicorn_run(application, **kwargs) -> None:
        assert not isinstance(application, str)
        token = token_file.read_text(encoding="ascii").removesuffix("\n")
        assert len(token) == 64
        assert token == token.lower()
        assert_owner_only(token_file.parent, directory=True)
        assert_owner_only(token_file, directory=False)
        tokens.append(token)
        uvicorn_calls.append(kwargs)

        with TestClient(application) as client:
            assert client.get("/api/health").json() == {"status": "ok"}
            assert client.get("/api/conversations").status_code == 401
            assert client.get(
                "/api/conversations",
                headers={"Authorization": f"Bearer {token}"},
            ).status_code == 200
            if len(tokens) > 1:
                assert client.get(
                    "/api/conversations",
                    headers={"Authorization": f"Bearer {tokens[0]}"},
                ).status_code == 401
            state_text = repr(application.state._state)
            assert token not in state_text
            assert DATABASE_KEY.hex() not in state_text
            assert DATABASE_KEY.hex() not in repr(application.state.database)
            assert DATABASE_KEY.hex() not in str(application.state.database.engine.url)

    config = LocalServerConfig(host="127.0.0.1", port=8000)
    for _ in range(2):
        run_secure_server(
            config=config,
            key_file=key_file,
            database_file=database_file,
            token_file=token_file,
            passphrase_prompt=lambda _label: PASSPHRASE,
            uvicorn_run=fake_uvicorn_run,
        )
        assert not token_file.exists()

    assert len(set(tokens)) == 2
    assert uvicorn_calls == [
        {
            "host": "127.0.0.1",
            "port": 8000,
            "workers": 1,
            "reload": False,
            "access_log": False,
        },
        {
            "host": "127.0.0.1",
            "port": 8000,
            "workers": 1,
            "reload": False,
            "access_log": False,
        },
    ]
    assert store.key_file.exists()


def test_unlock_failure_creates_no_token_or_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _store, key_file, token_file, database_file = _initialized_store(tmp_path)
    monkeypatch.setattr(
        "runtime.serve.KeyStore",
        lambda _path: KeyStore(key_file, policy=TEST_POLICY),
    )
    uvicorn_called = False

    def should_not_run(*_args, **_kwargs) -> None:
        nonlocal uvicorn_called
        uvicorn_called = True

    with pytest.raises(UnlockError, match="^Unlock failed$"):
        run_secure_server(
            config=LocalServerConfig(host="127.0.0.1", port=8000),
            key_file=key_file,
            database_file=database_file,
            token_file=token_file,
            passphrase_prompt=lambda _label: "wrong secure passphrase",
            uvicorn_run=should_not_run,
        )

    assert uvicorn_called is False
    assert not token_file.exists()
    assert PASSPHRASE not in caplog.text


def test_server_failure_removes_token_and_zeroizes_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, key_file, token_file, database_file = _initialized_store(tmp_path)
    handles: list[DatabaseKeyHandle] = []

    class CapturingStore:
        def unlock(self, passphrase: str, *, database_path: Path) -> DatabaseKeyHandle:
            handle = store.unlock(passphrase, database_path=database_path)
            handles.append(handle)
            return handle

    monkeypatch.setattr("runtime.serve.KeyStore", lambda _path: CapturingStore())
    monkeypatch.setenv("AMITAI_GENERATOR", "mock")

    with pytest.raises(RuntimeError, match="server failed"):
        run_secure_server(
            config=LocalServerConfig(host="127.0.0.1", port=8000),
            key_file=key_file,
            database_file=database_file,
            token_file=token_file,
            passphrase_prompt=lambda _label: PASSPHRASE,
            uvicorn_run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("server failed")
            ),
        )

    assert not token_file.exists()
    assert len(handles) == 1
    assert handles[0].closed is True


def test_database_handle_keys_every_new_sqlalchemy_connection(tmp_path: Path) -> None:
    path = tmp_path / "pooled.sqlite3"
    handle = DatabaseKeyHandle(DATABASE_KEY)

    class CountingKeySource:
        def __init__(self) -> None:
            self.uses = 0

        @contextmanager
        def temporary_hex(self):
            self.uses += 1
            with handle.temporary_hex() as value:
                yield value

    source = CountingKeySource()
    database = Database.from_url(_database_url(path), encryption_key=source)
    try:
        database.create_schema()
        for _ in range(2):
            database.engine.dispose()
            with database.engine.connect() as connection:
                assert connection.exec_driver_sql("SELECT count(*) FROM sqlite_master").scalar()
        assert source.uses >= 3
        assert DATABASE_KEY.hex() not in repr(database)
        assert DATABASE_KEY.hex() not in repr(database.engine)
        assert DATABASE_KEY.hex() not in str(database.engine.url)
    finally:
        database.engine.dispose()
        handle.close()

    database.engine.dispose()
    with (
        pytest.raises(EncryptedStorageError, match="Database key is unavailable"),
        database.engine.connect(),
    ):
        pass


def test_process_hardening_reports_only_controls_actually_applied() -> None:
    state = apply_process_hardening()

    if state.core_dumps_disabled:
        import resource

        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    else:
        assert state.process_dumpable_disabled is False
