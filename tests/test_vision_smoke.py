"""Exercise both opt-in smoke launchers with fake inference only."""

from types import SimpleNamespace

from evaluation.hf_backend import GenerationOutput
from scripts import remote_vision_smoke, vision_smoke
from tests.test_remote_vision import ORIGIN, TOKEN, Harness


def test_native_smoke_metrics_and_same_model_followup(monkeypatch, capsys):
    engines = []

    class Engine:
        def __init__(self, *_):
            engines.append(self)
            self.calls = []
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
                return {"image_grid_thw": [SimpleNamespace(tolist=lambda: [1, 48, 64])]}, {}, 10
            return {}, {}, 5

        def generate_detailed(self, messages, config):
            self.calls.append(messages)
            self._prepare_generation(messages, config)
            return GenerationOutput("Synthetic answer", 10, 2)

    monkeypatch.setattr(vision_smoke, "NativeQwenGenerator", Engine)
    assert vision_smoke.main() == 0
    output = capsys.readouterr().out
    for marker in (
        "Model load:",
        "Processed image dimensions: 1024 x 768",
        "Vision output tokens: 2",
        "max_reserved=1.00",
        "Same-model text followup:",
    ):
        assert marker in output
    assert len(engines) == 1 and len(engines[0].calls) == 2


def test_remote_smoke_uses_configured_transport_and_synthetic_image(monkeypatch):
    harness = Harness(["Red square shown.", "Red square shown.", "Paris is capital."])

    def provider(endpoint, token, model, **kwargs):
        assert endpoint == ORIGIN and token == TOKEN
        assert kwargs["allowed_origins"] == ORIGIN
        assert model == harness.provider.model_name
        return harness.provider

    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_URL", ORIGIN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", TOKEN)
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setattr(remote_vision_smoke, "RemoteInferenceProvider", provider)
    assert remote_vision_smoke.main() == 0
    assert [r.url.path for r in harness.requests] == [
        "/v1/vision",
        "/v1/vision/stream",
        "/v1/generate",
    ]
    assert len(harness.loads) == 1
