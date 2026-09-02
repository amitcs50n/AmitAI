import json
import logging
import socket
import ssl
from dataclasses import FrozenInstanceError
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from runtime.app import select_response_generator
from runtime.config import EXPECTED_MODEL_NAME
from runtime.privacy import RemoteDisclosureBlockedError
from runtime.providers import InferenceProviderError, RemoteInferenceProvider
from runtime.remote_transport import (
    RemoteTransportPolicy,
    create_remote_ssl_context,
    resolve_addresses,
)
from tests.app_factory import create_test_app
from tests.test_remote_privacy import RemoteHarness, chat, durable_snapshot

ORIGIN = "https://pod-123.proxy.runpod.net"
TOKEN = "TRANSPORT_AUTH_CANARY_01234567890123456789"
PROMPT = "PRIVATE_TRANSPORT_PROMPT_CANARY"
MEMORY = "PRIVATE_TRANSPORT_MEMORY_CANARY"
PUBLIC = ["8.8.8.8", "2606:4700:4700::1111"]


class Harness:
    def __init__(self, *, endpoint=ORIGIN, allowed_origins=(ORIGIN,),
                 answers=PUBLIC, response_status=200, client_factory=httpx.Client):
        self.http_calls = []
        self.dns_calls = []
        self.answers = answers
        self.response_status = response_status
        self.provider = RemoteInferenceProvider(
            endpoint, TOKEN, EXPECTED_MODEL_NAME, allowed_origins=allowed_origins,
            resolver=self.resolve, transport=httpx.MockTransport(self.handle),
            client_factory=client_factory,
        )

    def resolve(self, host, port):
        self.dns_calls.append((host, port))
        if isinstance(self.answers, Exception):
            raise self.answers
        return self.answers

    def handle(self, request):
        self.http_calls.append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if self.response_status != 200:
            return httpx.Response(self.response_status, headers={
                "Location": "https://attacker.example/v1/generate?PRIVATE_LOCATION_CANARY",
            }, text="PRIVATE_RESPONSE_CANARY")
        payload = json.loads(request.content)
        final = {"request_id": payload["request_id"], "model": EXPECTED_MODEL_NAME,
                 "text": "Safe reply", "input_tokens": 3, "output_tokens": 2}
        if request.url.path.endswith("/stream"):
            body = ('event: delta\ndata: {"delta":"Safe "}\n\n'
                    'event: delta\ndata: {"delta":"reply"}\n\n'
                    f'event: final\ndata: {json.dumps(final)}\n\n')
            return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=final)

    def run(self, streaming):
        messages = [{"role": "system", "content": MEMORY}, {"role": "user", "content": PROMPT}]
        if streaming:
            events = list(self.provider.stream(messages, {}, cancel_event=Event()))
            assert "".join(event for event in events if isinstance(event, str)) == events[-1].text
            return events[-1]
        return self.provider.generate(messages, {})


@pytest.mark.parametrize("endpoint,canonical", [
    (ORIGIN, ORIGIN),
    ("HTTPS://POD-123.PROXY.RUNPOD.NET.:443/", ORIGIN),
    (ORIGIN + ":8443", ORIGIN + ":8443"),
    ("https://xn--bcher-kva.example", "https://xn--bcher-kva.example"),
])
def test_exact_origin_canonicalization(endpoint, canonical):
    policy = RemoteTransportPolicy.from_config(endpoint, canonical)
    assert policy.origin.url == canonical
    assert not policy.origin.loopback
    with pytest.raises(FrozenInstanceError):
        policy.origin.hostname = "attacker.example"
    with pytest.raises(FrozenInstanceError):
        policy.allowed_origins = frozenset()


@pytest.mark.parametrize("endpoint,allowlist", [
    (ORIGIN, None), (ORIGIN, ""), (ORIGIN, []),
    ("https://evil.proxy.runpod.net", ORIGIN),
    ("https://pod-123.proxy.runpod.net.evil.example", ORIGIN),
    ("https://child.pod-123.proxy.runpod.net", ORIGIN),
    (ORIGIN + ":8443", ORIGIN), (ORIGIN, ORIGIN + ":8443"),
    (ORIGIN, "http://pod-123.proxy.runpod.net"),
    ("http://pod-123.proxy.runpod.net", ORIGIN),
    (ORIGIN, "*"), (ORIGIN, "https://*"), (ORIGIN, "*.runpod.net"),
    (ORIGIN, "https://*.proxy.runpod.net"), (ORIGIN, "runpod.net"),
    (ORIGIN, ORIGIN + ","), (ORIGIN, ORIGIN + ",https://*"),
])
def test_unapproved_origin_fails_before_dns_or_client_creation(endpoint, allowlist):
    calls = []
    with pytest.raises(ValueError):
        RemoteInferenceProvider(endpoint, TOKEN, EXPECTED_MODEL_NAME, allowed_origins=allowlist,
                                resolver=lambda *args: calls.append(args),
                                client_factory=lambda **kwargs: calls.append(kwargs))
    assert calls == []


@pytest.mark.parametrize("endpoint", [
    ORIGIN + "/foo", ORIGIN + "/v1", ORIGIN + "//", ORIGIN + "/?x=1", ORIGIN + "?",
    ORIGIN + "#fragment", ORIGIN + "#", "https://user@pod-123.proxy.runpod.net",
    "https://user:pass@pod-123.proxy.runpod.net", "https://203.0.113.10", "https://8.8.8.8",
    "https://[2606:4700:4700::1111]", "http://192.168.1.10", "https://127.1",
    "https://2130706433", "https://0x7f000001", "https://0177.0.0.1",
    "http://localhost.attacker.example", "http://foo-localhost.example",
    "https://bad_host.example", "https://-bad.example", "https://bad-.example",
    "https://bad..example", "https://host.example..", "https://b\u00fccher.example",
    "https://host.example:", "https://host.example:0", "https://host.example:65536",
    "https://host.example:bad", "https://[::1]attacker", "https://[::1]:80:90",
    "https://host.example\\@evil.example", "https://%31%32%37.0.0.1",
    "https://[::1%25lo0]", " " + ORIGIN, ORIGIN + "\n", "https://\thost.example",
])
def test_ambiguous_endpoint_forms_are_not_normalized_into_trusted_origins(endpoint):
    with pytest.raises(ValueError):
        RemoteTransportPolicy.from_config(endpoint, [endpoint])


def test_comma_separated_exact_allowlist_and_non_default_port():
    policy = RemoteTransportPolicy.from_config(
        ORIGIN + ":8443", f" https://other.example, {ORIGIN}:8443/ ",
    )
    assert policy.origin.port == 8443
    assert len(policy.allowed_origins) == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("endpoint,answers", [
    ("http://127.0.0.1:9000", ["127.0.0.1"]),
    ("http://localhost:9000", ["127.0.0.1", "::1"]),
    ("http://LOCALHOST.:9000/", ["::1"]),
    ("http://[::1]:9000", ["::1"]),
    ("https://localhost", ["127.0.0.1", "::1"]),
])
def test_loopback_development_does_not_require_public_allowlist(endpoint, answers, streaming):
    harness = Harness(endpoint=endpoint, allowed_origins=None, answers=answers)
    assert harness.run(streaming).text == "Safe reply"
    assert len(harness.dns_calls) == len(harness.http_calls) == 1
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("bad", [
    "127.0.0.1", "::1", "10.0.0.1", "172.16.0.1", "192.168.1.10", "169.254.1.1",
    "fe80::1", "0.0.0.0", "::", "224.0.0.1", "ff02::1", "240.0.0.1", "100.64.0.1",
    "203.0.113.10", "2001:db8::1", "fc00::1", "fec0::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
])
def test_public_private_special_and_mixed_dns_answers_fail_before_http(bad, streaming, caplog):
    caplog.set_level(logging.INFO)
    for answers in ([bad], [PUBLIC[0], bad]):
        harness = Harness(answers=answers)
        with pytest.raises(InferenceProviderError, match="^Remote inference failed$") as error:
            harness.run(streaming)
        assert harness.http_calls == [] and len(harness.dns_calls) == 1
        assert error.value.__cause__ is None
        harness.provider.close()
    assert "failure=dns_policy" in caplog.text
    for private in (bad, TOKEN, PROMPT, MEMORY):
        assert private not in caplog.text and private not in str(error.value)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("answers", [
    [], None, "8.8.8.8", [None], [8], ["not an IP"], ["fe80::1%LOCAL_SCOPE_CANARY"],
    OSError("PRIVATE_DNS_FAILURE_CANARY"), RuntimeError("PRIVATE_RESOLVER_CANARY"),
])
def test_dns_failure_or_malformed_answers_never_fall_back_to_http(answers, streaming, caplog):
    caplog.set_level(logging.INFO)
    harness = Harness(answers=answers)
    with pytest.raises(InferenceProviderError, match="^Remote inference failed$"):
        harness.run(streaming)
    assert harness.http_calls == []
    assert "CANARY" not in caplog.text
    harness.provider.close()


@pytest.mark.parametrize("answers", [["127.0.0.1", "192.168.1.20"], PUBLIC])
def test_poisoned_localhost_resolution_is_not_loopback(answers):
    harness = Harness(endpoint="http://localhost:9000", answers=answers, allowed_origins=None)
    with pytest.raises(InferenceProviderError, match="^Remote inference failed$"):
        harness.run(False)
    assert harness.http_calls == []
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_dns_is_rechecked_and_later_rebinding_is_blocked(streaming):
    harness = Harness()
    assert harness.run(streaming).text == "Safe reply"
    harness.answers = ["127.0.0.1"]
    with pytest.raises(InferenceProviderError, match="^Remote inference failed$"):
        harness.run(streaming)
    assert harness.dns_calls == [("pod-123.proxy.runpod.net", 443)] * 2
    assert len(harness.http_calls) == 1
    harness.provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [300, 301, 302, 303, 307, 308])
def test_redirects_never_resend_headers_or_body(streaming, status, caplog):
    caplog.set_level(logging.INFO)
    harness = Harness(response_status=status)
    with pytest.raises(InferenceProviderError, match="^Remote inference failed$") as error:
        harness.run(streaming)
    assert len(harness.http_calls) == 1
    request = harness.http_calls[0]
    assert str(request.url) == ORIGIN + ("/v1/generate/stream" if streaming else "/v1/generate")
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    for private in (TOKEN, PROMPT, MEMORY, "PRIVATE_RESPONSE_CANARY", "PRIVATE_LOCATION_CANARY"):
        assert private not in str(error.value) and private not in caplog.text
    assert "attacker.example" not in caplog.text
    assert "failure=redirect" in caplog.text
    harness.provider.close()


def test_explicit_tls_context_ignores_environment_ca_overrides(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "UNTRUSTED_CA_FILE_CANARY")
    monkeypatch.setenv("SSL_CERT_DIR", "UNTRUSTED_CA_DIR_CANARY")
    context = create_remote_ssl_context()
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert context.cert_store_stats()["x509_ca"] > 0


@pytest.mark.parametrize("streaming", [False, True])
def test_client_construction_has_explicit_tls_and_ignores_environment_proxies(monkeypatch, streaming):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9999")
    monkeypatch.setenv("SSL_CERT_FILE", "UNTRUSTED_CA_FILE_CANARY")
    monkeypatch.setenv("SSL_CERT_DIR", "UNTRUSTED_CA_DIR_CANARY")
    captured = []

    def client_factory(**kwargs):
        captured.append(kwargs)
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        context = kwargs["verify"]
        assert isinstance(context, ssl.SSLContext) and context.check_hostname
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
        assert "proxy" not in kwargs
        return httpx.Client(**kwargs)

    # Capture the production security options at the public constructor boundary.
    harness = Harness(client_factory=client_factory)
    assert harness.run(streaming).text == "Safe reply"
    assert len(captured) == len(harness.http_calls) == 1
    harness.provider.close()


def test_system_resolver_requests_a_and_aaaa_and_validates_results(monkeypatch):
    calls = []

    def getaddrinfo(host, port, **kwargs):
        calls.append((host, port, kwargs))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC[0], port)),
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC[1], port, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    assert resolve_addresses("pod-123.proxy.runpod.net", 443) == PUBLIC
    assert calls == [("pod-123.proxy.runpod.net", 443, {
        "family": socket.AF_UNSPEC, "type": socket.SOCK_STREAM, "proto": socket.IPPROTO_TCP,
    })]


@pytest.mark.parametrize("records", [
    [], [(-1, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))],
    [(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP, "", ("8.8.8.8", 443))],
    [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 443))],
    [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", "443"))],
    [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 80))],
    [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 443))],
    [(socket.AF_INET,)],
])
def test_malformed_system_resolver_records_fail_before_http(monkeypatch, records):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: records)
    calls = []
    provider = RemoteInferenceProvider(ORIGIN, TOKEN, EXPECTED_MODEL_NAME,
                                       allowed_origins=[ORIGIN],
                                       transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(InferenceProviderError, match="^Remote inference failed$"):
        provider.generate([{"role": "user", "content": PROMPT}], {})
    assert calls == []
    provider.close()


def test_provider_destination_is_read_only_and_paths_are_fixed():
    harness = Harness()
    with pytest.raises(AttributeError):
        harness.provider.endpoint = "https://attacker.example"
    harness.run(False)
    harness.run(True)
    assert [str(request.url) for request in harness.http_calls] == [
        ORIGIN + "/v1/generate", ORIGIN + "/v1/generate/stream",
    ]
    harness.provider.close()


def test_transport_checks_do_not_bypass_existing_body_disclosure_guard():
    harness = Harness()
    with pytest.raises(RemoteDisclosureBlockedError):
        harness.provider.generate([{"role": "user", "content": TOKEN}], {})
    assert harness.http_calls == harness.dns_calls == []
    harness.provider.close()


def test_cancellation_during_dns_preflight_sends_no_http():
    cancel_event = Event()
    calls = []

    def resolver(_host, _port):
        cancel_event.set()
        return PUBLIC

    provider = RemoteInferenceProvider(ORIGIN, TOKEN, EXPECTED_MODEL_NAME,
                                       allowed_origins=[ORIGIN], resolver=resolver,
                                       transport=httpx.MockTransport(lambda request: calls.append(request)))
    assert list(provider.stream([{"role": "user", "content": PROMPT}], {},
                                cancel_event=cancel_event)) == []
    assert calls == []
    provider.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("override", [
    {"request_id": "MISMATCHED_REQUEST_CANARY"}, {"model": "MISMATCHED_MODEL_CANARY"},
    {"text": ""}, {"text": None}, {"input_tokens": -1}, {"output_tokens": True},
])
def test_transport_changes_preserve_response_identity_and_metadata_checks(streaming, override):
    calls = []

    def handle(request):
        calls.append(request)
        payload = json.loads(request.content)
        final = {"request_id": payload["request_id"], "model": EXPECTED_MODEL_NAME,
                 "text": "Safe reply", "input_tokens": 3, "output_tokens": 2, **override}
        if streaming:
            return httpx.Response(200, text=(
                'event: delta\ndata: {"delta":"Safe reply"}\n\n'
                f'event: final\ndata: {json.dumps(final)}\n\n'
            ))
        return httpx.Response(200, json=final)

    provider = RemoteInferenceProvider(ORIGIN, TOKEN, EXPECTED_MODEL_NAME,
                                       allowed_origins=[ORIGIN], resolver=lambda *_: PUBLIC,
                                       transport=httpx.MockTransport(handle))
    with pytest.raises(InferenceProviderError) as error:
        if streaming:
            list(provider.stream([{"role": "user", "content": PROMPT}], {}, cancel_event=Event()))
        else:
            provider.generate([{"role": "user", "content": PROMPT}], {})
    assert len(calls) == 1
    assert "CANARY" not in str(error.value)
    provider.close()


def test_environment_requires_public_origin_and_forwards_exact_configuration(monkeypatch):
    monkeypatch.setenv("AMITAI_INFERENCE_PROVIDER", "remote")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_URL", ORIGIN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", TOKEN)
    monkeypatch.delenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(ValueError, match="AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS"):
        select_response_generator()
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", ORIGIN)
    generator = select_response_generator()
    assert generator._provider.endpoint == ORIGIN
    generator._provider.close()


def test_empty_explicit_remote_configuration_does_not_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_URL", ORIGIN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", TOKEN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", ORIGIN)
    for overrides in ({"remote_token": ""}, {"remote_endpoint": ""}, {"remote_allowed_origins": ""}):
        with pytest.raises(ValueError):
            select_response_generator(mode="remote", **overrides)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("followup", ["initial", "retry", "tool"])
def test_dns_failure_rolls_back_chat_and_staged_memory_on_every_invocation(streaming, followup):
    dns_calls = []

    def resolver(_host, _port):
        dns_calls.append(1)
        return PUBLIC if len(dns_calls) == 1 and followup != "initial" else ["127.0.0.1"]

    output = ('<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'
              if followup == "tool" else "An invalid four words")
    harness = RemoteHarness([output], resolver=resolver)
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=harness.generator)
    with TestClient(app) as client:
        before = durable_snapshot(app)
        prompt = "Remember project transport.note: safe preference. Answer in exactly 3 words."
        response, events = chat(client, prompt, streaming)
        if streaming:
            assert response.status_code == 200
            assert [name for name, _ in events] == ["start", "error"]
            assert events[-1][1] == {"detail": "Assistant generation failed"}
        else:
            assert response.status_code == 500
            assert response.json() == {"detail": "Assistant generation failed"}
        assert len(dns_calls) == (1 if followup == "initial" else 2)
        assert len(harness.calls) == (0 if followup == "initial" else 1)
        assert durable_snapshot(app) == before
        assert client.get("/api/memory").json() == []
    harness.provider.close()
