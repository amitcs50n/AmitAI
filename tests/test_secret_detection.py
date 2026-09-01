import json

import pytest

from backend.secret_detection import (
    contains_credential_like_data,
    contains_credential_like_pair,
    contains_credential_like_text,
    is_credential_like_label,
)
from runtime.privacy import RemoteDisclosureBlockedError, guarded_request_body

CREDENTIAL_LABELS = [
    "password", "passcode", "api_key", "access_token", "refresh_token", "auth_token",
    "client_secret", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
    "MY_PASSWORD", "DATABASE_PASSWORD", "SERVICE_PASSCODE", "GITHUB_ACCESS_TOKEN",
    "OAUTH_REFRESH_TOKEN", "BACKEND_AUTH_TOKEN", "OAUTH_CLIENT_SECRET", "api-key",
    "API KEY", "api.key", "ApiKey", "clientsecret", "service.api-key",
    "ＯＰＥＮＡＩ＿ＡＰＩ＿ＫＥＹ", "ＭＹ＿ＰＡＳＳＷＯＲＤ",
]
BENIGN_LABELS = [
    "api_key_rotation_policy", "password_hash", "password_hash_algorithm",
    "access_token_documentation", "refresh_token_strategy", "client_secret_explanation",
    "maximum_tokens", "max_tokens", "token_budget", "token_count", "input_tokens",
    "output_tokens", "password_length", "GITHUB_TOKEN", "token", "monkey", "notpassword",
    "api/key", "api@key", "api:key", "password!", "profile.name", "ui.theme",
]
BENIGN_TEXT = [
    "password:", "api_key:", "OPENAI_API_KEY=", '"password": ""', '"api_key": ""',
    'What does "password:" mean?', 'Explain the field "api_key:"',
    "Why is OPENAI_API_KEY empty?", "Explain API key rotation", "How does password hashing work?",
    "What is Authorization?", "Authorization: Bearer", "Authorization: Bearer   ",
    "Explain JWT verification", "My name is Alice", "Explain public key cryptography",
    '{"content":"password:"}', '{"password":null}', '{"api_key":""}',
    "'api_key': ''", 'password: ""', 'OPENAI_API_KEY = ""',
    "password:\nExplain the next field", "Ｓｅｒｖｉｃｅ＿ｐａｓｓｗｏｒｄ：",
    "What does the Authorization header do?", "Explain JWT structure without a real token.",
    "What does the label OPENAI_API_KEY mean?", "Why is my API key environment variable empty?",
    "a8fd82d3c0abcdef123456", "sk-unlabeled-example", "04a630e2-b2d2-4bfe-85e0-21f329d2bd09",
]


@pytest.mark.parametrize("label", CREDENTIAL_LABELS)
def test_normalized_credential_family_and_nonempty_pair(label):
    assert is_credential_like_label(label)
    assert contains_credential_like_pair(label, "x")
    assert not contains_credential_like_pair(label, "")
    assert not contains_credential_like_pair(label, " \t\n")
    for separator in (":", " = "):
        for quote in ("", '"', "'"):
            assert contains_credential_like_text(f'{quote}{label}{quote}{separator}"CANARY"')


@pytest.mark.parametrize("label", BENIGN_LABELS)
def test_no_substring_or_generic_token_label_matching(label):
    assert not is_credential_like_label(label)
    assert not contains_credential_like_pair(label, "ordinary value")
    assert not contains_credential_like_text(json.dumps({label: "ordinary value"}))
    assert not contains_credential_like_text(f"{label}=ordinary")


@pytest.mark.parametrize("text", BENIGN_TEXT)
def test_missing_values_and_security_discussion_are_not_credentials(text):
    assert not contains_credential_like_text(text)
    assert not contains_credential_like_data({"messages": [{"role": "user", "content": text}]})
    body = guarded_request_body({"messages": [{"role": "user", "content": text}]},
                                transport_token="unrelated-transport-token")
    assert json.loads(body)["messages"][0]["content"] == text


@pytest.mark.parametrize("text", [
    "password: hunter2", "password=hunter2", 'OPENAI_API_KEY = "sk-secret"',
    '{"api_key":"SECRET"}', "{'api_key': 'SECRET'}", '"api_key":"SECRET"',
    '{"password":"SECRET","password":""}',
    "Authorization: Bearer SECRET", '"Authorization": "Bearer SECRET"',
    "Send Authorization: Bearer SECRET with the request",
    "-----BEGIN PRIVATE KEY-----", "-----BEGIN RSA PRIVATE KEY-----",
    "eyJabcdefghi.abcdefghijk.abcdefghijk", "ｐａｓｓｗｏｒｄ： ｓｅｃｒｅｔ",
    ('MEMORY_CONTEXT_V1 <memory_context>{"items":[{"key":"openai_api_key",'
     '"value":"sk-secret","category":"project"}]}</memory_context>'),
    '{"value":"SECRET","category":"project","key":"my_password"}',
    '{"key":"openai_api_key","key":"benign","value":"SECRET"}',
    r'{"\u0061pi_key":"SECRET"}',
    'An actual quoted assignment: "password: SECRET"',
])
def test_credential_text_json_records_and_existing_standalone_forms(text):
    assert contains_credential_like_text(text)


def test_authorization_requires_a_supported_scheme_and_actual_value():
    assert contains_credential_like_pair("Authorization", "Bearer abc")
    assert not contains_credential_like_pair("Authorization", "header documentation")
    assert not contains_credential_like_pair("Authorization", "Bearer")
    assert not contains_credential_like_pair("Authorization", "Basic abc")


@pytest.mark.parametrize("token", ['quote"and\\slash', "unicode秘密é", "Ｆｕｌｌｗｉｄｔｈ"])
@pytest.mark.parametrize("embedded_json", [False, True])
def test_exact_transport_token_is_checked_in_decoded_content(token, embedded_json):
    content = json.dumps({"note": token}, ensure_ascii=True) if embedded_json else token
    with pytest.raises(RemoteDisclosureBlockedError):
        guarded_request_body({"messages": [{"role": "user", "content": content}]},
                             transport_token=token)


def test_guard_checks_owned_snapshot_and_serializes_only_once(monkeypatch):
    from runtime import privacy

    payload = {"messages": [{"role": "user", "content": "password:"}],
               "generation_config": {"max_tokens": 10}}
    original_check = privacy.contains_credential_like_data
    original_dumps = json.dumps
    checks = []
    serializations = []

    def check(snapshot, **kwargs):
        assert snapshot is not payload
        assert snapshot["messages"] is not payload["messages"]
        result = original_check(snapshot, **kwargs)
        checks.append(snapshot)
        payload["messages"][0]["content"] = "OPENAI_API_KEY=UNSAFE_LATE_MUTATION"
        return result

    def dumps(snapshot, **kwargs):
        assert snapshot is checks[0]
        serializations.append(snapshot)
        return original_dumps(snapshot, **kwargs)

    monkeypatch.setattr(privacy, "contains_credential_like_data", check)
    monkeypatch.setattr(privacy.json, "dumps", dumps)
    body = privacy.guarded_request_body(payload, transport_token="transport-only")
    assert len(checks) == len(serializations) == 1
    assert b'"content":"password:"' in body
    assert b"UNSAFE_LATE_MUTATION" not in body


def test_long_ordinary_prose_and_deeply_nested_data():
    # Exercise iterative semantic traversal without a recursion-dependent policy.
    assert not contains_credential_like_text("ordinary text " * 10_000)
    nested = {"OPENAI_API_KEY": "SECRET"}
    for _ in range(1500):
        nested = {"nested": [nested]}
    assert contains_credential_like_data(nested)


def test_cyclic_config_cannot_hang_the_semantic_walk():
    payload = {"generation_config": {"max_tokens": 5}}
    payload["generation_config"]["cycle"] = payload
    with pytest.raises(ValueError, match="Circular reference"):
        guarded_request_body(payload, transport_token="transport-only")
