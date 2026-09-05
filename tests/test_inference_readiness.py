"""CPU-only readiness/preload contracts; all model construction uses fake engines."""

import asyncio
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from evaluation.hf_backend import GenerationOutput
from runtime.config import load_runtime_config
from runtime.inference_app import create_inference_app
from runtime.media import VisionGenerationRequest
from runtime.providers import LocalTransformersInferenceProvider

TOKEN = "READINESS_TEST_TOKEN_0123456789abcdef"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MESSAGES = [{"role": "user", "content": "Synthetic test"}]
CANARY = "hf_SECRET_CANARY /workspace/hf/private C:\\secret\\model prompt credential"


class FakeEngine:
    supports_vision = True

    def __init__(self):
        self.calls = []

    def generate_detailed(self, messages, config):
        self.calls.append(messages)
        return GenerationOutput("Synthetic answer", 2, 2)

    def generate_detailed_stream(self, messages, config, *, cancel_event):
        if not cancel_event.is_set():
            output = self.generate_detailed(messages, config)
            yield output.text
            yield output


class Factory:
    supports_vision = True

    def __init__(self, *, blocked=False, fail=False):
        self.entered, self.release = Event(), Event()
        if not blocked:
            self.release.set()
        self.fail = fail
        self.loads = 0
        self.engine = FakeEngine()

    def __call__(self, model, seed):
        assert model == load_runtime_config().model and seed == 3407
        self.loads += 1
        self.entered.set()
        assert self.release.wait(5), "test did not release fake loader"
        if self.fail:
            raise RuntimeError(CANARY)
        return self.engine


def provider_for(factory):
    return LocalTransformersInferenceProvider(
        load_runtime_config().model, 3407, engine_factory=factory
    )


def wait_ready(client, expected):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get("/ready", headers=HEADERS)
        assert response.headers["cache-control"] == "no-store"
        if response.json() == {"state": expected}:
            assert response.status_code == (200 if expected == "ready" else 503)
            return
        time.sleep(0.01)
    pytest.fail(f"readiness did not become {expected}")


@pytest.mark.parametrize("operation", ["preload", "text", "text_stream", "vision", "vision_stream"])
def test_first_model_use_transitions_and_all_paths_reuse_cached_engine(operation):
    factory = Factory(blocked=True)
    provider = provider_for(factory)
    assert provider.initialization_state == "unloaded" and factory.loads == 0
    with Image.new("RGB", (2, 2)) as image, ThreadPoolExecutor(max_workers=1) as pool:
        request = VisionGenerationRequest(MESSAGES, image)
        operations = {
            "preload": provider.preload,
            "text": lambda: provider.generate(MESSAGES, {}),
            "text_stream": lambda: list(provider.stream(MESSAGES, {}, cancel_event=Event())),
            "vision": lambda: provider.generate_vision(request, {}),
            "vision_stream": lambda: list(
                provider.stream_vision(request, {}, cancel_event=Event())
            ),
        }
        pending = pool.submit(operations[operation])
        try:
            assert factory.entered.wait(3)
            assert provider.initialization_state == "loading"
        finally:
            factory.release.set()
        pending.result(timeout=3)
        assert provider.initialization_state == "ready"
        for use in operations.values():
            use()
        assert factory.loads == 1
        assert any(isinstance(call[-1]["content"], list) for call in factory.engine.calls)
        assert any(isinstance(call[-1]["content"], str) for call in factory.engine.calls)


def test_concurrent_preload_text_and_vision_initialize_exactly_once():
    factory = Factory(blocked=True)
    provider = provider_for(factory)
    barrier = Barrier(8)
    with Image.new("RGB", (2, 2)) as image, ThreadPoolExecutor(max_workers=8) as pool:

        def run(index):
            barrier.wait(timeout=3)
            if index % 3 == 0:
                provider.preload()
            elif index % 3 == 1:
                provider.generate(MESSAGES, {})
            else:
                provider.generate_vision(VisionGenerationRequest(MESSAGES, image), {})

        pending = [pool.submit(run, index) for index in range(8)]
        try:
            assert factory.entered.wait(3)
            assert provider.initialization_state == "loading" and factory.loads == 1
        finally:
            factory.release.set()
        for result in pending:
            result.result(timeout=3)
    assert provider.initialization_state == "ready" and factory.loads == 1


def test_liveness_readiness_and_preload_are_responsive_during_load():
    factory = Factory(blocked=True)
    provider = provider_for(factory)
    app = create_inference_app(provider=provider, auth_token=TOKEN)
    with TestClient(app) as client:
        assert factory.loads == 0  # app construction and lifespan remain lazy
        assert client.get("/health").json() == {"status": "ok"}
        wait_ready(client, "unloaded")
        assert factory.loads == 0
        try:
            accepted = client.post("/preload", headers=HEADERS)
            assert accepted.status_code == 202 and accepted.json() == {"status": "accepted"}
            assert accepted.headers["cache-control"] == "no-store"
            assert factory.entered.wait(3)
            wait_ready(client, "loading")
            assert client.get("/health").status_code == 200
            # The HTTP request has completed while the shared loader is still blocked.
            for _ in range(4):
                assert client.post("/preload", headers=HEADERS).status_code == 202
            assert factory.loads == 1 and not factory.engine.calls
        finally:
            factory.release.set()
        wait_ready(client, "ready")
        already_ready = client.post("/preload", headers=HEADERS)
        assert already_ready.status_code == 200 and already_ready.json() == {"state": "ready"}
        assert factory.loads == 1 and not factory.engine.calls


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("method,path", [("GET", "/ready"), ("POST", "/preload")])
def test_readiness_and_preload_share_inference_auth_without_loading(
    configured, method, path, monkeypatch
):
    monkeypatch.delenv("AMITAI_INFERENCE_AUTH_TOKEN", raising=False)
    factory = Factory()
    app = create_inference_app(
        provider=provider_for(factory), auth_token=TOKEN if configured else None
    )
    with TestClient(app) as client:
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": TOKEN},
            {"Authorization": "Bearer " + TOKEN + " "},
        ):
            response = client.request(method, path, headers=headers)
            assert response.status_code == (401 if configured else 503)
            assert response.json() == {
                "detail": "Unauthorized"
                if configured
                else "Inference authentication is not configured"
            }
        if not configured:
            assert client.request(method, path, headers=HEADERS).status_code == 503
        assert client.get("/health").status_code == 200
        assert factory.loads == 0


@pytest.mark.parametrize("initial_use", ["preload", "generate"])
def test_failed_initialization_is_safe_and_explicit_preload_retry_recovers(initial_use, caplog):
    factory = Factory(fail=True)
    provider = provider_for(factory)
    app = create_inference_app(provider=provider, auth_token=TOKEN)
    with TestClient(app) as client:
        if initial_use == "preload":
            assert client.post("/preload", headers=HEADERS).status_code == 202
        else:
            response = client.post(
                "/v1/generate",
                headers=HEADERS,
                json={
                    "request_id": "9a52c8db-221f-444c-913b-6be1f57965e0",
                    "messages": MESSAGES,
                    "generation_config": {},
                },
            )
            assert response.status_code == 502 and response.json() == {"detail": "Inference failed"}
        wait_ready(client, "failed")
        assert factory.loads == 1
        for _ in range(3):
            wait_ready(client, "failed")  # inspection never retries
            assert client.get("/health").json() == {"status": "ok"}
        assert factory.loads == 1 and CANARY not in caplog.text
        factory.fail = False
        assert client.post("/preload", headers=HEADERS).status_code == 202
        wait_ready(client, "ready")
        assert factory.loads == 2
        assert provider.generate(MESSAGES, {}).text == "Synthetic answer"
        assert factory.loads == 2 and CANARY not in caplog.text


def test_shutdown_waits_for_preload_without_blocking_event_loop():
    factory = Factory(blocked=True)
    provider = provider_for(factory)
    app = create_inference_app(provider=provider, auth_token=TOKEN)

    async def scenario():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        shutdown = None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (await client.post("/preload", headers=HEADERS)).status_code == 202
            try:
                assert await asyncio.to_thread(factory.entered.wait, 3)
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                done, _ = await asyncio.wait({shutdown}, timeout=0.05)
                assert not done, "shutdown abandoned a running model loader"
                assert (await client.get("/health")).status_code == 200
                assert (await client.post("/preload", headers=HEADERS)).status_code == 503
            finally:
                factory.release.set()
                if shutdown is None:
                    await lifespan.__aexit__(None, None, None)
                else:
                    await shutdown

    asyncio.run(scenario())
    assert provider.initialization_state == "ready" and factory.loads == 1


def test_custom_provider_is_not_assumed_ready():
    from tests.test_inference_providers import RecordingProvider

    provider = RecordingProvider()
    with TestClient(create_inference_app(provider=provider, auth_token=TOKEN)) as client:
        for method, path in (("GET", "/ready"), ("POST", "/preload")):
            response = client.request(method, path, headers=HEADERS)
            assert response.status_code == 503
            assert response.json() == {"detail": "Model readiness is unavailable"}
    assert not provider.generate_calls and not provider.stream_calls


def test_import_startup_and_probes_cannot_import_model_libraries():
    script = """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
            raise AssertionError("CPU-only import guard")
sys.meta_path.insert(0, NoModels())
from fastapi.testclient import TestClient
from runtime.inference_app import create_inference_app
with TestClient(create_inference_app(auth_token="READINESS_TEST_TOKEN_0123456789abcdef")) as client:
    assert client.get("/health").status_code == 200
    response = client.get("/ready", headers={"Authorization": "Bearer READINESS_TEST_TOKEN_0123456789abcdef"})
    assert response.status_code == 503 and response.json() == {"state": "unloaded"}
assert not any(name.split(".")[0] in {"torch", "transformers", "huggingface_hub"} for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        timeout=15,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
