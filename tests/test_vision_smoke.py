"""Exercise both opt-in smoke launchers with fake inference only."""

from types import SimpleNamespace

import pytest

from evaluation.hf_backend import GenerationOutput
from scripts import remote_vision_smoke, vision_smoke
from tests.test_remote_vision import ORIGIN, TOKEN, Harness
from tests.test_vision_model import Tensor


@pytest.fixture
def native_smoke(monkeypatch):
    engines = []
    state = SimpleNamespace(
        engines=engines,
        events=["Synthetic ", "answer", GenerationOutput("Synthetic answer", 10, 2)],
        closed=False,
    )

    class Engine:
        def __init__(self, *_):
            engines.append(self)
            self.calls = []
            self.dependency_versions = {"torch": "fake", "transformers": "fake"}
            cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
            for name in (
                "memory_allocated",
                "memory_reserved",
                "max_memory_allocated",
                "max_memory_reserved",
            ):
                setattr(cuda, name, lambda _: 2**30)
            self.torch = SimpleNamespace(cuda=cuda)

        def _prepare_generation(self, messages, config):
            if isinstance(messages[-1]["content"], list):
                return {"image_grid_thw": Tensor((1, 3), data=[[1, 48, 64]])}, {}, 10
            return {}, {}, 5

        def generate_detailed(self, messages, config):
            self.calls.append(messages)
            self._prepare_generation(messages, config)
            return GenerationOutput("Synthetic answer", 10, 2)

        def generate_detailed_stream(self, messages, config, *, cancel_event):
            self.calls.append(messages)
            self._prepare_generation(messages, config)
            assert not cancel_event.is_set()
            assert messages[-1]["content"][0]["image"].getpixel((0, 0)) is not None
            try:
                for item in state.events:
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                state.closed = True

    monkeypatch.setattr(vision_smoke, "NativeQwenGenerator", Engine)
    return state


def test_native_smoke_metrics_and_same_model_followup(native_smoke, capsys):
    assert vision_smoke.main() == 0
    output = capsys.readouterr().out
    for marker in (
        "Model load:",
        "Processed image dimensions: 1024 x 768",
        "Vision output tokens: 2",
        "max_reserved=1.00",
        "Same-model text followup:",
        "NATIVE VISION NONSTREAM: PASS",
        "NATIVE VISION STREAM: PASS",
        "NATIVE TEXT: PASS",
        "Stream reconstruction: PASS",
    ):
        assert marker in output
    assert len(native_smoke.engines) == 1 and len(native_smoke.engines[0].calls) == 3
    assert native_smoke.closed
    assert native_smoke.engines[0].calls[0] == native_smoke.engines[0].calls[1]


def test_native_smoke_exposes_synthetic_worker_traceback_only(native_smoke, capsys):
    native_smoke.events = ["Partial ", ValueError("VISION_STREAM_CANARY")]
    assert vision_smoke.main() == 1
    output = capsys.readouterr()
    assert "NATIVE VISION NONSTREAM: PASS" in output.out
    assert "Partial " in output.out and "NATIVE VISION STREAM: FAIL" in output.out
    assert "Traceback (most recent call last)" in output.err
    assert "generate_detailed_stream" in output.err
    assert "ValueError: VISION_STREAM_CANARY" in output.err
    assert "NATIVE TEXT: PASS" not in output.out
    assert native_smoke.closed


@pytest.mark.parametrize("events", [
    [], ["text"], [""], [123], ["text", GenerationOutput("different", 10, 1)],
    ["text", GenerationOutput("text", 10, 1), GenerationOutput("text", 10, 1)],
    ["text", GenerationOutput("text", 10, 1), "late"],
])
def test_native_smoke_rejects_broken_stream_contract(native_smoke, capsys, events):
    native_smoke.events = events
    assert vision_smoke.main() == 1
    assert "NATIVE VISION STREAM: FAIL" in capsys.readouterr().out
    assert native_smoke.closed


@pytest.mark.parametrize("failed_stage", [None, 0, 1, 2])
def test_remote_smoke_stages_paths_and_sanitized_failure(monkeypatch, capsys, failed_stage):
    outputs = ["Red square shown.", "Red square shown.", "Paris is capital."]
    if failed_stage is not None:
        outputs[failed_stage] = ValueError(f"SMOKE_PRIVATE_CANARY {TOKEN} {ORIGIN}")
    harness = Harness(outputs)

    def provider(endpoint, token, model, **kwargs):
        assert endpoint == ORIGIN and token == TOKEN
        assert kwargs["allowed_origins"] == ORIGIN
        assert model == harness.provider.model_name
        return harness.provider

    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_URL", ORIGIN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", TOKEN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setattr(remote_vision_smoke, "RemoteInferenceProvider", provider)
    assert remote_vision_smoke.main() == (0 if failed_stage is None else 1)
    stages_run = 3 if failed_stage is None else failed_stage + 1
    assert [r.url.path for r in harness.requests] == [
        "/v1/vision",
        "/v1/vision/stream",
        "/v1/generate",
    ][:stages_run]
    assert len(harness.loads) == 1
    captured = capsys.readouterr()
    for index, stage in enumerate(["REMOTE VISION NONSTREAM", "REMOTE VISION STREAM", "REMOTE TEXT"]):
        if index < stages_run:
            assert f"{stage}: {'FAIL' if index == failed_stage else 'PASS'}" in captured.out
        else:
            assert stage not in captured.out
    assert not any(s in captured.out + captured.err for s in (
        TOKEN, ORIGIN, "SMOKE_PRIVATE_CANARY", "Red square shown.", "Paris is capital."
    ))
