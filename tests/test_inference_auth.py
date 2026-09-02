import re
import secrets

import httpx
import pytest
from fastapi.testclient import TestClient

from runtime import inference_token
from runtime.inference_app import create_inference_app
from runtime.inference_auth import MIN_INFERENCE_TOKEN_CHARS, validate_inference_token
from runtime.providers import RemoteInferenceProvider
from tests.test_inference_providers import RecordingProvider

TOKEN = "AUTH_TOKEN_TEST_CANARY_01234567890123456789"
INVALID_TOKENS = [
    "", "test", "secret", "abc123", "development-token", "x" * 31,
    " " + TOKEN, TOKEN + " ", TOKEN + "\n", TOKEN + "\r\n", TOKEN + "\x00", TOKEN + "\x7f",
    TOKEN + "\t", TOKEN + " internal space", TOKEN + "秘密", TOKEN + "\u0085",
]


@pytest.mark.parametrize("token", INVALID_TOKENS)
def test_client_server_and_shared_validator_reject_the_same_invalid_token(token, caplog):
    for construct in (
        lambda: validate_inference_token(token),
        lambda: RemoteInferenceProvider("http://127.0.0.1:9000", token, "model"),
        lambda: create_inference_app(provider=RecordingProvider(), auth_token=token),
    ):
        with pytest.raises(ValueError, match="Inference token must contain at least 32") as error:
            construct()
        if token:
            assert token not in str(error.value) and token not in caplog.text
        assert "CANARY" not in str(error.value) and "CANARY" not in caplog.text


@pytest.mark.parametrize("token", ["x" * 32, TOKEN, 'HEADER_SAFE_TOKEN_CANARY_0123456789_"\\'])
def test_client_server_accept_identical_header_safe_token_without_normalizing(token):
    assert MIN_INFERENCE_TOKEN_CHARS == 32
    assert validate_inference_token(token) == token
    app = create_inference_app(provider=RecordingProvider(), auth_token=token)
    with TestClient(app) as client:
        response = client.post("/v1/generate", headers={"Authorization": f"Bearer {token}"}, json={
            "request_id": "9a52c8db-221f-444c-913b-6be1f57965e0",
            "messages": [{"role": "user", "content": "safe request"}], "generation_config": {},
        })
        assert response.status_code == 200
    remote = RemoteInferenceProvider("http://127.0.0.1:9000", token, "model",
                                     transport=httpx.MockTransport(lambda _: None))
    remote.close()


@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
@pytest.mark.parametrize("configured", [False, True])
def test_server_missing_wrong_and_correct_auth_stays_generic(path, configured, monkeypatch):
    monkeypatch.delenv("AMITAI_INFERENCE_AUTH_TOKEN", raising=False)
    provider = RecordingProvider()
    app = create_inference_app(provider=provider, auth_token=TOKEN if configured else None)
    payload = {"request_id": "9a52c8db-221f-444c-913b-6be1f57965e0",
               "messages": [{"role": "user", "content": "safe request"}], "generation_config": {}}
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN},
                        {"Authorization": "Bearer " + TOKEN + " "},
                        {"Authorization": b"Bearer \xff"}):
            response = client.post(path, json=payload, headers=headers)
            assert response.status_code == (401 if configured else 503)
            assert response.json() == {"detail": "Unauthorized" if configured
                                       else "Inference authentication is not configured"}
        assert provider.generate_calls == provider.stream_calls == []
        response = client.post(path, json=payload, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == (200 if configured else 503)


def test_server_uses_and_validates_session_token_from_environment(monkeypatch):
    monkeypatch.setenv("AMITAI_INFERENCE_AUTH_TOKEN", "secret")
    with pytest.raises(ValueError, match="Inference token must contain at least 32"):
        create_inference_app(provider=RecordingProvider())
    monkeypatch.setenv("AMITAI_INFERENCE_AUTH_TOKEN", TOKEN)
    app = create_inference_app(provider=RecordingProvider())
    with TestClient(app) as client:
        response = client.post("/v1/generate", headers={"Authorization": f"Bearer {TOKEN}"}, json={
            "request_id": "9a52c8db-221f-444c-913b-6be1f57965e0",
            "messages": [{"role": "user", "content": "safe request"}], "generation_config": {},
        })
        assert response.status_code == 200


def test_token_utility_uses_32_random_bytes_and_only_prints_one_token(monkeypatch, capsys, caplog):
    calls = []

    def generate(size):
        calls.append(size)
        return TOKEN

    monkeypatch.setattr(inference_token.secrets, "token_urlsafe", generate)
    inference_token.main([])
    assert calls == [32]
    captured = capsys.readouterr()
    assert captured.out == TOKEN + "\n" and captured.err == ""
    assert TOKEN not in caplog.text
    with pytest.raises(SystemExit) as error:
        inference_token.main(["ARGUMENT_SECRET_CANARY"])
    assert "ARGUMENT_SECRET_CANARY" not in str(error.value)
    assert capsys.readouterr().out == "" and calls == [32]


def test_generated_session_tokens_meet_shared_validation():
    token = secrets.token_urlsafe(32)
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
    assert validate_inference_token(token) == token
