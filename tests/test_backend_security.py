import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.chat_service import ChatGenerationDelta
from runtime.app import create_runtime_app
from runtime.serve import LocalServerConfig, load_local_server_config, main

LOCAL_TOKEN = "LOCAL_API_SECRET_91233_secure_test_padding"
DATABASE_KEY = "a1" * 32
AUTHORIZATION = {"Authorization": f"Bearer {LOCAL_TOKEN}"}


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _secured_app(tmp_path: Path, **kwargs):
    return create_app(
        _database_url(tmp_path / "secured.sqlite3"),
        encrypted_storage=False,
        local_api_token=LOCAL_TOKEN,
        **kwargs,
    )


def test_private_local_routes_require_the_configured_bearer_token(tmp_path: Path) -> None:
    application = _secured_app(tmp_path)

    with TestClient(application) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        missing = client.get("/api/conversations")
        incorrect = client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer incorrect-token-value-with-padding"},
        )
        valid = client.get("/api/conversations", headers=AUTHORIZATION)
        memory = client.get("/api/memory", headers=AUTHORIZATION)
        chat = client.post(
            "/api/chat",
            headers=AUTHORIZATION,
            json={"message": "Authenticated request"},
        )

    assert missing.status_code == 401
    assert missing.json() == {"detail": "Unauthorized"}
    assert missing.headers["www-authenticate"] == "Bearer"
    assert incorrect.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == []
    assert memory.status_code == 200
    assert chat.status_code == 200


def test_local_auth_uses_constant_time_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons = []

    def compare(candidate: str, expected: str) -> bool:
        comparisons.append((candidate, expected))
        return False

    monkeypatch.setattr("backend.security.secrets.compare_digest", compare)
    application = _secured_app(tmp_path)

    with TestClient(application) as client:
        response = client.get(
            "/api/memory",
            headers={"Authorization": "Bearer wrong-but-compared"},
        )

    assert response.status_code == 401
    assert comparisons == [("wrong-but-compared", LOCAL_TOKEN)]


def test_missing_server_token_fails_closed(tmp_path: Path) -> None:
    application = create_app(
        _database_url(tmp_path / "locked.sqlite3"),
        encrypted_storage=False,
        local_api_token=None,
    )

    with TestClient(application) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/conversations").status_code == 401


def test_runtime_factory_is_authenticated_by_default(tmp_path: Path) -> None:
    application = create_runtime_app(
        _database_url(tmp_path / "runtime-secured.sqlite3"),
        encrypted_storage=False,
        mode="mock",
        local_api_token=LOCAL_TOKEN,
    )

    with TestClient(application) as client:
        assert client.get("/api/conversations").status_code == 401
        assert client.get("/api/conversations", headers=AUTHORIZATION).status_code == 200


def test_docs_and_openapi_are_disabled_by_default_and_explicitly_enabled_for_dev(
    tmp_path: Path,
) -> None:
    default_app = create_app(
        _database_url(tmp_path / "no-docs.sqlite3"),
        encrypted_storage=False,
    )
    dev_app = create_app(
        _database_url(tmp_path / "dev-docs.sqlite3"),
        encrypted_storage=False,
        enable_dev_docs=True,
    )

    with TestClient(default_app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
    with TestClient(dev_app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_control_plane_has_no_wildcard_cors(tmp_path: Path) -> None:
    application = _secured_app(tmp_path)
    assert all(
        middleware.cls.__name__ != "CORSMiddleware"
        for middleware in application.user_middleware
    )

    with TestClient(application) as client:
        response = client.get(
            "/api/conversations",
            headers={**AUTHORIZATION, "Origin": "https://untrusted.example"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_canonical_launcher_defaults_to_loopback_and_requires_a_strong_token() -> None:
    config = load_local_server_config(
        {
            "AMITAI_LOCAL_API_TOKEN": LOCAL_TOKEN,
            "AMITAI_DB_KEY": DATABASE_KEY,
        }
    )

    assert config.host == "127.0.0.1"
    assert config.port == 8000

    with pytest.raises(ValueError, match="AMITAI_LOCAL_API_TOKEN"):
        load_local_server_config({})


def test_canonical_launcher_rejects_lan_binding_without_explicit_opt_in() -> None:
    values = {
        "AMITAI_LOCAL_API_TOKEN": LOCAL_TOKEN,
        "AMITAI_DB_KEY": DATABASE_KEY,
        "AMITAI_HOST": "0.0.0.0",
    }

    with pytest.raises(ValueError, match="Refusing non-loopback"):
        load_local_server_config(values)

    allowed = load_local_server_config({**values, "AMITAI_ALLOW_LAN": "1"})
    assert allowed.host == "0.0.0.0"


def test_canonical_launcher_disables_unsanitized_access_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "runtime.serve.load_local_server_config",
        lambda: LocalServerConfig(host="127.0.0.1", port=8000),
    )
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))

    main()

    assert calls == [
        (
            ("runtime.app:app",),
            {
                "host": "127.0.0.1",
                "port": 8000,
                "workers": 1,
                "reload": False,
                "access_log": False,
            },
        )
    ]


def test_sensitive_local_values_never_enter_operational_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_prompt = "PRIVATE_PROMPT_92831"
    private_memory = "PRIVATE_MEMORY_55221"
    failed_auth = "LOCAL_API_SECRET_91233_wrong_attempt"

    class SensitiveFailureGenerator:
        def stream_response(self, _messages, *, cancel_event):
            del cancel_event
            yield ChatGenerationDelta(delta="Partial output")
            raise RuntimeError(
                f"provider failure {private_prompt} {private_memory} {LOCAL_TOKEN}"
            )

    application = _secured_app(tmp_path, generator=SensitiveFailureGenerator())

    with caplog.at_level(logging.INFO), TestClient(application) as client:
        successful_memory = client.post(
            "/api/memory",
            headers=AUTHORIZATION,
            json={
                "category": "project",
                "key": "privacy.sentinel",
                "value": private_memory,
            },
        )
        rejected = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {failed_auth}"},
            json={"message": private_prompt},
        )
        failed_stream = client.post(
            "/api/chat/stream",
            headers=AUTHORIZATION,
            json={"message": private_prompt},
        )
        conversations = client.get("/api/conversations", headers=AUTHORIZATION)

    assert successful_memory.status_code == 201
    assert rejected.status_code == 401
    assert "event: error" in failed_stream.text
    assert "event: final" not in failed_stream.text
    assert conversations.json() == []
    assert "Streaming assistant generation failed" in caplog.text
    for sentinel in (private_prompt, private_memory, LOCAL_TOKEN, failed_auth):
        assert sentinel not in caplog.text
