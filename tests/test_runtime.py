import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.chat_service import (
    ChatGenerationDelta,
    ChatGenerationError,
    ChatGenerationResult,
    GenerationMessage,
)
from evaluation.hf_backend import GenerationOutput, TransformersGenerator
from runtime.app import create_runtime_app, select_response_generator
from runtime.config import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    RuntimeConfig,
    load_runtime_config,
)
from runtime.generator import TransformersChatGenerator


def _runtime_config(system_prompt: str = "Tested runtime prompt") -> RuntimeConfig:
    return RuntimeConfig(
        source_path=Path("test-runtime.yaml"),
        runtime_system_prompt=system_prompt,
        model={
            "name": EXPECTED_MODEL_NAME,
            "revision": EXPECTED_MODEL_REVISION,
            "dtype": "bfloat16",
            "load_in_4bit": False,
            "device_map": "auto",
            "trust_remote_code": False,
        },
        generation={
            "max_new_tokens": 512,
            "enable_thinking": False,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "seed": 3407,
        },
        mechanical_constraints_enabled=True,
    )


class SequenceEngine:
    def __init__(self, outputs: list[GenerationOutput]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def generate_detailed(self, messages, generation_config):
        self.calls.append((messages, generation_config))
        return next(self.outputs)


class StreamingSequenceEngine:
    def __init__(self, outputs: list[list[str | GenerationOutput]]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.cancel_events: list[threading.Event] = []

    def generate_detailed_stream(
        self,
        messages,
        generation_config,
        *,
        cancel_event,
    ):
        self.calls.append((messages, generation_config))
        self.cancel_events.append(cancel_event)
        yield from next(self.outputs)


def _generator_with_engine(
    outputs: list[GenerationOutput],
    *,
    system_prompt: str = "Tested runtime prompt",
) -> tuple[TransformersChatGenerator, SequenceEngine, list[tuple[dict, int]]]:
    engine = SequenceEngine(outputs)
    factory_calls: list[tuple[dict, int]] = []

    def factory(model_config, seed):
        factory_calls.append((model_config, seed))
        return engine

    return (
        TransformersChatGenerator(
            _runtime_config(system_prompt),
            engine_factory=factory,
        ),
        engine,
        factory_calls,
    )


def test_runtime_config_loads_the_frozen_model_prompt_and_generation_settings() -> None:
    document = yaml.safe_load(DEFAULT_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))[
        "baseline_eval"
    ]

    config = load_runtime_config()

    assert config.runtime_system_prompt == document["runtime_system_prompt"]
    assert config.model == document["model"]
    assert config.generation == document["generation"]
    assert config.model["name"] == EXPECTED_MODEL_NAME
    assert config.model["revision"] == EXPECTED_MODEL_REVISION
    assert config.model["dtype"] == "bfloat16"
    assert config.mechanical_constraints_enabled is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda document: document.pop("baseline_eval"), "baseline_eval"),
        (
            lambda document: document["baseline_eval"].pop("runtime_system_prompt"),
            "runtime_system_prompt",
        ),
        (
            lambda document: document["baseline_eval"]["model"].pop("name"),
            "Runtime model must remain",
        ),
        (
            lambda document: document["baseline_eval"].pop("generation"),
            "generation must be an object",
        ),
        (
            lambda document: document["baseline_eval"]["model"].update({"load_in_4bit": True}),
            "BF16 loading",
        ),
        (
            lambda document: document["baseline_eval"]["model"].update({"revision": "latest"}),
            "revision must remain pinned",
        ),
        (
            lambda document: document["baseline_eval"]["model"].update({"dtype": "float16"}),
            "dtype must remain bfloat16",
        ),
        (
            lambda document: document["baseline_eval"]["generation"].update(
                {"enable_thinking": True}
            ),
            "disable thinking",
        ),
    ],
)
def test_runtime_config_rejects_invalid_or_drifted_settings(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    document = yaml.safe_load(DEFAULT_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(document)
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        load_runtime_config(path)


def test_original_call_prepends_system_prompt_and_preserves_full_history() -> None:
    generator, engine, factory_calls = _generator_with_engine(
        [GenerationOutput("Second answer", input_tokens=20, output_tokens=3)]
    )

    result = generator.generate_response(
        [
            GenerationMessage(role="user", content="First question"),
            GenerationMessage(role="assistant", content="First answer"),
            GenerationMessage(role="user", content="Second question"),
        ]
    )

    assert engine.calls[0][0] == [
        {"role": "system", "content": "Tested runtime prompt"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    assert factory_calls == [(_runtime_config().model, 3407)]
    assert result.response == "Second answer"
    assert result.model == EXPECTED_MODEL_NAME
    assert isinstance(result.latency_ms, int) and result.latency_ms >= 0
    assert result.input_tokens == 20
    assert result.output_tokens == 3
    assert result.validator == {
        "retry_attempted": False,
        "retry_passed": None,
        "retry_count": 0,
        "parsed_constraints": [],
        "final_validation": {
            "passed": True,
            "checks": [],
            "failures": [],
            "normalized_response": None,
        },
    }
    assert result.tools == []
    assert result.memory == []


def test_constraints_are_parsed_only_from_the_current_user_message() -> None:
    generator, engine, _ = _generator_with_engine(
        [GenerationOutput("This historical limit no longer applies", 20, 6)]
    )

    result = generator.generate_response(
        [
            GenerationMessage(role="user", content="Answer in exactly 3 words."),
            GenerationMessage(role="assistant", content="One two three"),
            GenerationMessage(role="user", content="Now explain normally."),
        ]
    )

    assert len(engine.calls) == 1
    assert result.response == "This historical limit no longer applies"
    assert result.validator["parsed_constraints"] == []
    assert result.validator["retry_count"] == 0


def test_runtime_preserves_system_and_tool_roles_from_persisted_history() -> None:
    generator, engine, _ = _generator_with_engine([GenerationOutput("Role-aware answer", 20, 3)])

    generator.generate_response(
        [
            GenerationMessage(role="system", content="Historical system note"),
            GenerationMessage(role="tool", content="Historical tool result"),
            GenerationMessage(role="user", content="Use that context"),
        ]
    )

    assert engine.calls[0][0] == [
        {"role": "system", "content": "Tested runtime prompt"},
        {"role": "system", "content": "Historical system note"},
        {"role": "tool", "content": "Historical tool result"},
        {"role": "user", "content": "Use that context"},
    ]


def test_first_bounded_repair_passes_and_stops_after_one_retry() -> None:
    generator, engine, _ = _generator_with_engine(
        [
            GenerationOutput("Only two", 10, 2),
            GenerationOutput("One two three", 12, 3),
        ]
    )

    result = generator.generate_response(
        [GenerationMessage(role="user", content="Answer in exactly 3 words.")]
    )

    assert len(engine.calls) == 2
    assert engine.calls[1][0][0] == {
        "role": "system",
        "content": "Tested runtime prompt",
    }
    assert len(engine.calls[1][0]) == 2
    assert engine.calls[1][0][-1]["role"] == "user"
    assert "Original user request:\nAnswer in exactly 3 words." in engine.calls[1][0][-1]["content"]
    assert "Previous answer:\nOnly two" in engine.calls[1][0][-1]["content"]
    assert result.response == "One two three"
    assert result.validator["retry_attempted"] is True
    assert result.validator["retry_passed"] is True
    assert result.validator["first_retry_passed"] is True
    assert result.validator["retry_count"] == 1
    assert result.validator["final_validation"]["passed"] is True
    assert result.input_tokens == 22
    assert result.output_tokens == 5


def test_second_repair_uses_latest_failure_and_aggregates_all_token_usage() -> None:
    generator, engine, _ = _generator_with_engine(
        [
            GenerationOutput("one two three", 100, 20),
            GenerationOutput("one two three four", 130, 15),
            GenerationOutput("one two three four five", 125, 12),
        ]
    )
    history = [
        GenerationMessage(role="user", content="Earlier question"),
        GenerationMessage(role="assistant", content="Earlier answer"),
        GenerationMessage(role="user", content="Write exactly 5 words."),
    ]

    result = generator.generate_response(history)

    assert len(engine.calls) == 3
    for retry_call in engine.calls[1:]:
        assert retry_call[0][:3] == [
            {"role": "system", "content": "Tested runtime prompt"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        assert len(retry_call[0]) == 4
    first_retry_prompt = engine.calls[1][0][-1]["content"]
    second_retry_prompt = engine.calls[2][0][-1]["content"]
    assert "Previous answer:\none two three" in first_retry_prompt
    assert "2 words short" in first_retry_prompt
    assert "add exactly 2 words" in first_retry_prompt
    assert "Previous answer:\none two three four" in second_retry_prompt
    assert "1 word short" in second_retry_prompt
    assert "add exactly 1 word" in second_retry_prompt
    assert "contains 3 whitespace-separated words" not in second_retry_prompt
    assert result.response == "one two three four five"
    assert result.validator["retry_attempted"] is True
    assert result.validator["first_retry_passed"] is False
    assert result.validator["retry_passed"] is True
    assert result.validator["retry_count"] == 2
    assert result.validator["final_validation"]["passed"] is True
    assert set(result.validator) == {
        "retry_attempted",
        "retry_passed",
        "retry_count",
        "parsed_constraints",
        "final_validation",
        "first_retry_passed",
    }
    assert result.input_tokens == 355
    assert result.output_tokens == 47


def test_exhausted_exact_word_repairs_fail_instead_of_returning_final_candidate() -> None:
    generator, engine, _ = _generator_with_engine(
        [
            GenerationOutput("Paris is the capital", 10, 4),
            GenerationOutput("France has Paris capital", 12, 4),
            GenerationOutput("Paris remains the capital", 14, 4),
        ]
    )

    with pytest.raises(ChatGenerationError, match="Assistant generation failed"):
        generator.generate_response(
            [
                GenerationMessage(
                    role="user",
                    content="What is the capital of France? Answer in exactly 3 words.",
                )
            ]
        )

    assert len(engine.calls) == 3


def test_latency_covers_the_complete_original_and_repair_flow() -> None:
    engine = SequenceEngine(
        [
            GenerationOutput("Only two", 10, 2),
            GenerationOutput("One two three", 12, 3),
        ]
    )
    clock_values = iter([10.0, 10.5])
    generator = TransformersChatGenerator(
        _runtime_config(),
        engine_factory=lambda _model, _seed: engine,
        clock=lambda: next(clock_values),
    )

    result = generator.generate_response(
        [GenerationMessage(role="user", content="Answer in exactly 3 words.")]
    )

    assert len(engine.calls) == 2
    assert result.latency_ms == 500


def test_lazy_engine_is_initialized_once_and_reused_between_requests() -> None:
    generator, engine, factory_calls = _generator_with_engine(
        [
            GenerationOutput("First answer", 5, 2),
            GenerationOutput("Second answer", 6, 2),
        ]
    )

    generator.generate_response([GenerationMessage(role="user", content="Hello")])
    generator.generate_response([GenerationMessage(role="user", content="Again")])

    assert len(factory_calls) == 1
    assert len(engine.calls) == 2


def test_failed_lazy_initialization_can_be_retried() -> None:
    engine = SequenceEngine([GenerationOutput("Recovered", 5, 1)])
    factory_calls = 0

    def factory(_model_config, _seed):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("temporary load failure")
        return engine

    generator = TransformersChatGenerator(_runtime_config(), engine_factory=factory)

    with pytest.raises(RuntimeError, match="temporary load failure"):
        generator.generate_response([GenerationMessage(role="user", content="Hello")])

    result = generator.generate_response([GenerationMessage(role="user", content="Try again")])
    assert factory_calls == 2
    assert result.response == "Recovered"


def test_lazy_initialization_and_generation_are_serialized_across_threads() -> None:
    class ConcurrentEngine:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def generate_detailed(self, _messages, _generation_config):
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return GenerationOutput("Serialized answer", 5, 2)

    engine = ConcurrentEngine()
    factory_calls = 0

    def factory(_model_config, _seed):
        nonlocal factory_calls
        factory_calls += 1
        time.sleep(0.01)
        return engine

    generator = TransformersChatGenerator(_runtime_config(), engine_factory=factory)
    start = threading.Barrier(3)

    def generate(message: str):
        start.wait()
        return generator.generate_response([GenerationMessage(role="user", content=message)])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(generate, message) for message in ("One", "Two")]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    assert [result.response for result in results] == [
        "Serialized answer",
        "Serialized answer",
    ]
    assert factory_calls == 1
    assert engine.maximum_active == 1


def test_runtime_mode_selection_keeps_mock_default_and_real_path_lazy(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AMITAI_GENERATOR", raising=False)
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return object()

    assert select_response_generator(generator_factory=factory) is None
    assert select_response_generator(mode="mock", generator_factory=factory) is None
    assert factory_calls == []

    selected = select_response_generator(
        mode="transformers",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
        generator_factory=factory,
    )
    assert selected is not None
    assert len(factory_calls) == 1
    assert factory_calls[0].model["revision"] == EXPECTED_MODEL_REVISION


def test_runtime_config_path_can_be_selected_from_the_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selected_path = tmp_path / "selected-runtime.yaml"
    selected_path.write_text(
        DEFAULT_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("AMITAI_GENERATOR", "transformers")
    monkeypatch.setenv("AMITAI_RUNTIME_CONFIG", str(selected_path))
    configs = []

    def factory(config):
        configs.append(config)
        return object()

    select_response_generator(generator_factory=factory)

    assert len(configs) == 1
    assert configs[0].source_path == selected_path


def test_runtime_mode_selection_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported AMITAI_GENERATOR"):
        select_response_generator(mode="surprise")


def test_runtime_app_injects_only_the_explicitly_selected_generator(
    tmp_path: Path,
) -> None:
    fake_generator = object()
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return fake_generator

    mock_app = create_runtime_app(
        f"sqlite+pysqlite:///{(tmp_path / 'mock.sqlite3').as_posix()}",
        mode="mock",
        generator_factory=factory,
    )
    real_app = create_runtime_app(
        f"sqlite+pysqlite:///{(tmp_path / 'real.sqlite3').as_posix()}",
        mode="transformers",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
        generator_factory=factory,
    )

    assert mock_app.state.generator is None
    assert real_app.state.generator is fake_generator
    assert len(factory_calls) == 1


def test_runtime_app_runs_bounded_repair_and_persists_only_final_chat_messages(
    tmp_path: Path,
) -> None:
    engine = SequenceEngine(
        [
            GenerationOutput("Only two", input_tokens=10, output_tokens=2),
            GenerationOutput("One two three", input_tokens=12, output_tokens=3),
        ]
    )

    def factory(config: RuntimeConfig) -> TransformersChatGenerator:
        return TransformersChatGenerator(
            config,
            engine_factory=lambda _model, _seed: engine,
        )

    application = create_runtime_app(
        f"sqlite+pysqlite:///{(tmp_path / 'runtime.sqlite3').as_posix()}",
        mode="transformers",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
        generator_factory=factory,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Answer in exactly 3 words."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "One two three"
        assert body["metadata"]["input_tokens"] == 22
        assert body["metadata"]["output_tokens"] == 5
        assert body["metadata"]["validator"]["retry_count"] == 1
        assert body["metadata"]["validator"]["retry_passed"] is True

        detail = client.get(f"/api/conversations/{body['conversation_id']}").json()
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
        ]
        assert [message["content"] for message in detail["messages"]] == [
            "Answer in exactly 3 words.",
            "One two three",
        ]

    assert len(engine.calls) == 2


def test_runtime_app_rejects_final_validation_failure_without_persistence(
    tmp_path: Path,
) -> None:
    failed_attempts = [
        GenerationOutput("Paris is the capital", input_tokens=10, output_tokens=4),
        GenerationOutput("France has Paris capital", input_tokens=12, output_tokens=4),
        GenerationOutput("Paris remains the capital", input_tokens=14, output_tokens=4),
    ]
    engine = SequenceEngine([*failed_attempts, *failed_attempts])

    def factory(config: RuntimeConfig) -> TransformersChatGenerator:
        return TransformersChatGenerator(
            config,
            engine_factory=lambda _model, _seed: engine,
        )

    application = create_runtime_app(
        f"sqlite+pysqlite:///{(tmp_path / 'runtime-validation-failure.sqlite3').as_posix()}",
        mode="transformers",
        config_path=DEFAULT_RUNTIME_CONFIG_PATH,
        generator_factory=factory,
    )
    payload = {
        "message": "What is the capital of France? Answer in exactly 3 words."
    }

    with TestClient(application, raise_server_exceptions=False) as client:
        new_conversation_response = client.post("/api/chat", json=payload)

        assert new_conversation_response.status_code == 500
        assert new_conversation_response.json() == {
            "detail": "Assistant generation failed"
        }
        assert client.get("/api/conversations").json() == []

        existing = client.post(
            "/api/conversations",
            json={"title": "Keep unchanged"},
        ).json()
        before = client.get(f"/api/conversations/{existing['id']}").json()
        existing_response = client.post(
            "/api/chat",
            json={"conversation_id": existing["id"], **payload},
        )
        after = client.get(f"/api/conversations/{existing['id']}").json()

        assert existing_response.status_code == 500
        assert existing_response.json() == {"detail": "Assistant generation failed"}
        assert after == before

    assert len(engine.calls) == 6


def test_unconstrained_runtime_streams_multiple_exact_deltas_with_full_history() -> None:
    engine = StreamingSequenceEngine(
        [
            [
                "Second",
                " streamed",
                " answer",
                GenerationOutput(
                    "Second streamed answer",
                    input_tokens=20,
                    output_tokens=3,
                ),
            ]
        ]
    )
    factory_calls: list[tuple[dict, int]] = []

    def factory(model_config, seed):
        factory_calls.append((model_config, seed))
        return engine

    generator = TransformersChatGenerator(_runtime_config(), engine_factory=factory)
    cancel_event = threading.Event()
    stream_items = list(
        generator.stream_response(
            [
                GenerationMessage(role="user", content="First question"),
                GenerationMessage(role="assistant", content="First answer"),
                GenerationMessage(role="user", content="Now answer normally"),
            ],
            cancel_event=cancel_event,
        )
    )

    deltas = [
        item.delta for item in stream_items if isinstance(item, ChatGenerationDelta)
    ]
    final_results = [
        item for item in stream_items if isinstance(item, ChatGenerationResult)
    ]
    assert deltas == ["Second", " streamed", " answer"]
    assert len(deltas) > 1
    assert len(final_results) == 1
    final = final_results[0]
    assert "".join(deltas) == final.response == "Second streamed answer"
    assert final.model == EXPECTED_MODEL_NAME
    assert final.input_tokens == 20
    assert final.output_tokens == 3
    assert final.validator == {
        "retry_attempted": False,
        "retry_passed": None,
        "retry_count": 0,
        "parsed_constraints": [],
        "final_validation": {
            "passed": True,
            "checks": [],
            "failures": [],
            "normalized_response": None,
        },
    }
    assert engine.calls[0][0] == [
        {"role": "system", "content": "Tested runtime prompt"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Now answer normally"},
    ]
    assert engine.cancel_events == [cancel_event]
    assert factory_calls == [(_runtime_config().model, 3407)]


def test_constrained_runtime_stream_hides_failed_candidate_and_emits_only_validated_final() -> None:
    failed_candidate = "LEAKED FAILED CANDIDATE TEXT"
    generator, engine, _ = _generator_with_engine(
        [
            GenerationOutput(failed_candidate, input_tokens=10, output_tokens=3),
            GenerationOutput("One two three", input_tokens=12, output_tokens=3),
        ]
    )

    stream_items = list(
        generator.stream_response(
            [GenerationMessage(role="user", content="Answer in exactly 3 words.")],
            cancel_event=threading.Event(),
        )
    )

    deltas = [
        item.delta for item in stream_items if isinstance(item, ChatGenerationDelta)
    ]
    final_results = [
        item for item in stream_items if isinstance(item, ChatGenerationResult)
    ]
    assert len(engine.calls) == 2
    assert deltas == ["One two three"]
    assert len(final_results) == 1
    final = final_results[0]
    assert "".join(deltas) == final.response == "One two three"
    assert failed_candidate not in repr(stream_items)
    assert final.input_tokens == 22
    assert final.output_tokens == 6
    assert final.validator["retry_attempted"] is True
    assert final.validator["retry_passed"] is True
    assert final.validator["retry_count"] == 1
    assert final.validator["final_validation"]["passed"] is True


def test_constrained_runtime_stream_exhaustion_emits_no_failed_candidate() -> None:
    failed_candidates = [
        "Paris is the capital",
        "France has Paris capital",
        "Paris remains the capital",
    ]
    engine = StreamingSequenceEngine(
        [
            [
                candidate[:8],
                candidate[8:],
                GenerationOutput(candidate, input_tokens=10, output_tokens=4),
            ]
            for candidate in failed_candidates
        ]
    )
    generator = TransformersChatGenerator(
        _runtime_config(),
        engine_factory=lambda _model, _seed: engine,
    )
    stream = generator.stream_response(
        [
            GenerationMessage(
                role="user",
                content="What is the capital of France? Answer in exactly 3 words.",
            )
        ],
        cancel_event=threading.Event(),
    )

    with pytest.raises(ChatGenerationError, match="Assistant generation failed"):
        next(stream)

    assert len(engine.calls) == 3


def test_constrained_runtime_disconnect_cancels_buffered_candidate_before_retry() -> None:
    class CancellingStreamingEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate_detailed_stream(
            self,
            _messages,
            _generation_config,
            *,
            cancel_event,
        ):
            self.calls += 1
            yield "failed candidate"
            cancel_event.set()
            yield GenerationOutput("failed candidate", 10, 2)

    engine = CancellingStreamingEngine()
    generator = TransformersChatGenerator(
        _runtime_config(),
        engine_factory=lambda _model, _seed: engine,
    )
    cancel_event = threading.Event()

    with pytest.raises(RuntimeError, match="cancelled"):
        list(
            generator.stream_response(
                [GenerationMessage(role="user", content="Answer in exactly 3 words.")],
                cancel_event=cancel_event,
            )
        )

    assert cancel_event.is_set() is True
    assert engine.calls == 1


def test_streaming_gpu_generation_remains_serialized_across_threads() -> None:
    class ConcurrentStreamingEngine:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def generate_detailed_stream(
            self,
            _messages,
            _generation_config,
            *,
            cancel_event,
        ):
            assert cancel_event.is_set() is False
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                yield "Serialized"
                time.sleep(0.03)
                yield " stream"
                yield GenerationOutput("Serialized stream", 5, 2)
            finally:
                with self.lock:
                    self.active -= 1

    engine = ConcurrentStreamingEngine()
    factory_calls = 0

    def factory(_model_config, _seed):
        nonlocal factory_calls
        factory_calls += 1
        return engine

    generator = TransformersChatGenerator(_runtime_config(), engine_factory=factory)
    start = threading.Barrier(3)

    def generate(message: str):
        start.wait()
        return list(
            generator.stream_response(
                [GenerationMessage(role="user", content=message)],
                cancel_event=threading.Event(),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(generate, message) for message in ("One", "Two")]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    assert [
        "".join(
            item.delta for item in result if isinstance(item, ChatGenerationDelta)
        )
        for result in results
    ] == ["Serialized stream", "Serialized stream"]
    assert factory_calls == 1
    assert engine.maximum_active == 1


def test_transformers_engine_streams_chunks_and_terminal_token_metadata_without_model_load() -> None:
    class FakeInputIds:
        shape = (1, 3)
        device = "cuda:0"

    class FakeBatch(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    class FakeCompletionIds:
        shape = (1, 2)

    class FakeGenerated:
        def __getitem__(self, item):
            assert item == (slice(None), slice(3, None))
            return FakeCompletionIds()

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Prompt"},
            ]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "rendered prompt"

        def __call__(self, prompt, **kwargs):
            assert prompt == "rendered prompt"
            assert kwargs == {"return_tensors": "pt"}
            return FakeBatch(input_ids=FakeInputIds())

    stream_end_sentinel = object()
    streamer_calls = []

    class FakeStreamer:
        def __init__(self, tokenizer, **kwargs):
            assert isinstance(tokenizer, FakeTokenizer)
            streamer_calls.append(kwargs)
            self.items: queue.Queue[object] = queue.Queue()

        def __iter__(self):
            while True:
                item = self.items.get(timeout=1)
                if item is stream_end_sentinel:
                    return
                yield item

        def on_finalized_text(self, text, *, stream_end: bool):
            if text:
                self.items.put(text)
            if stream_end:
                self.items.put(stream_end_sentinel)

    class FakeStoppingCriteria:
        pass

    class FakeStoppingCriteriaList(list):
        pass

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        bool = "bool"

        @staticmethod
        def inference_mode():
            return InferenceMode()

        @staticmethod
        def full(shape, value, *, dtype, device):
            assert shape == (1,)
            assert dtype == "bool"
            assert device == "cuda:0"
            return value

    generation_calls = []

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            generation_calls.append(kwargs)
            streamer = kwargs["streamer"]
            stopping_criteria = kwargs["stopping_criteria"]
            assert len(stopping_criteria) == 1
            assert stopping_criteria[0](FakeInputIds(), None) is False
            streamer.on_finalized_text("Stream", stream_end=False)
            streamer.on_finalized_text(" output", stream_end=True)
            return FakeGenerated()

    engine = object.__new__(TransformersGenerator)
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    engine.torch = FakeTorch()
    engine.StoppingCriteria = FakeStoppingCriteria
    engine.StoppingCriteriaList = FakeStoppingCriteriaList
    engine.TextIteratorStreamer = FakeStreamer

    stream_items = list(
        engine.generate_detailed_stream(
            [
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Prompt"},
            ],
            {
                "max_new_tokens": 32,
                "enable_thinking": False,
                "do_sample": False,
                "repetition_penalty": 1.15,
            },
            cancel_event=threading.Event(),
        )
    )

    assert stream_items == [
        "Stream",
        " output",
        GenerationOutput("Stream output", input_tokens=3, output_tokens=2),
    ]
    assert streamer_calls == [
        {
            "skip_prompt": True,
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
    ]
    assert len(generation_calls) == 1
    generation_kwargs = generation_calls[0]
    assert isinstance(generation_kwargs["input_ids"], FakeInputIds)
    assert generation_kwargs["max_new_tokens"] == 32
    assert generation_kwargs["do_sample"] is False
    assert generation_kwargs["use_cache"] is True
    assert generation_kwargs["repetition_penalty"] == 1.15
