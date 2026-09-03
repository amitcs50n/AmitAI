"""Production prompt composition and delivery; fake engines, no GPU or network."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from threading import Event
from uuid import uuid4

import pytest
import yaml

from backend.chat_service import ChatGenerationDelta, GenerationMessage
from backend.memory import format_memory_context
from backend.vision_grant import RemoteVisionGrant
from evaluation.run_baseline import load_config as load_eval_config
from runtime import config as configuration
from runtime.app import select_response_generator
from runtime.config import (
    DEFAULT_PRODUCTION_PROFILE_PATH,
    DEFAULT_RUNTIME_CONFIG_PATH,
    EXPECTED_MODEL_NAME,
    load_production_runtime_config,
    load_runtime_config,
)
from runtime.context import (
    HISTORY_OMISSION_NOTICE,
    MAX_HISTORY_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
    compile_model_messages,
)
from runtime.generator import TransformersChatGenerator
from tests.test_assets import image_bytes
from tests.test_remote_vision import ORIGIN, TOKEN, Harness
from tests.test_vision import VisionEngine

EPISTEMIC_RULES = (
    "correct weak assumptions or false premises",
    "Be confident when evidence suffices",
    "reasonable everyday inferences without unnecessary hedging",
    "Never invent unavailable facts, citations, evidence, memories, conversation",
    "tool results, files inspected, actions completed, schema, or configuration",
    "Use explicit trusted context over older assistant guesses",
    "Do not agree just to follow the question's framing",
    "Explicit facts beat plausible unstated assumptions",
    "never substitute plausible alternatives",
    "Do not reconstruct missing or truncated history",
    "a reference has no usable referent or materially different possible referents",
    "ask a concise clarification",
    "before agreeing or acting",
    "give a useful qualified answer when partial reasoning suffices",
    "do not escape contradictions by redefining terms or inventing exceptions",
    "Distinguish facts from hypotheses",
    "an unverified cause is not an established root cause",
    "Missing evidence does not prove global nonexistence",
    "an explicitly complete list establishes absence from that stated set",
)


@pytest.fixture(autouse=True)
def clean_runtime_selection(monkeypatch):
    for name in ("AMITAI_RUNTIME_CONFIG", "AMITAI_INFERENCE_PROVIDER", "AMITAI_GENERATOR"):
        monkeypatch.delenv(name, raising=False)


def test_production_profile_is_prompt_only_and_establishes_identity():
    profile = yaml.safe_load(DEFAULT_PRODUCTION_PROFILE_PATH.read_text(encoding="utf-8"))
    assert set(profile) == {"schema_version", "runtime_system_prompt"}
    assert EXPECTED_MODEL_NAME not in profile["runtime_system_prompt"]
    production = load_production_runtime_config()
    prompt = production.runtime_system_prompt
    assert prompt.startswith("You are Aevon,")
    assert (
        "AmitAI is the software project/platform that contains you, not your assistant name."
        in prompt
    )
    assert f"Your underlying configured model is {production.model['name']}." in prompt
    assert "Never claim that you are named Qwen." in prompt
    assert "When asked your name, answer Aevon." in prompt
    assert "report the exact configured model identifier above" in prompt
    assert "or mention AmitAI, Qwen, or model identity unless relevant" in prompt
    assert "You are AmitAI" not in prompt and "${model_name}" not in prompt
    assert len(prompt.split()) < 390
    for principle in (
        "actual question first",
        "concise by default",
        "correct weak assumptions",
        "solution, supported causes, and verification",
        "latest explicit user correction overrides older",
        "hard constraints",
        '"exactly", "at most", and "code only"',
        "humor or profanity only when it fits",
        "fake praise",
    ) + EPISTEMIC_RULES:
        assert principle in prompt


def test_production_settings_are_shared_and_frozen_eval_prompt_is_unchanged():
    before = DEFAULT_RUNTIME_CONFIG_PATH.read_bytes()
    # Pin the baseline's Git blob at b82bd44; normalize checkout-only CRLF.
    tracked_bytes = before.replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(tracked_bytes)).encode() + b"\0" + tracked_bytes
    assert (
        hashlib.sha1(blob, usedforsecurity=False).hexdigest()
        == "3d4910c7f4437e6aa75021aeb7f50b6af3668bd0"
    )
    baseline = load_runtime_config()
    production = load_production_runtime_config()
    literal, composed = asdict(baseline), asdict(production)
    assert literal.pop("runtime_system_prompt") != composed.pop("runtime_system_prompt")
    assert literal == composed  # includes revision, BF16, ALL generation and validator settings
    evaluation = load_eval_config(DEFAULT_RUNTIME_CONFIG_PATH)
    assert evaluation["runtime_system_prompt"] == baseline.runtime_system_prompt
    assert evaluation["runtime_system_prompt"].startswith("You are AmitAI,")
    assert evaluation["model"] == production.model
    assert evaluation["generation"] == production.generation
    assert DEFAULT_RUNTIME_CONFIG_PATH.read_bytes() == before


def test_model_identity_is_taken_from_loaded_settings_not_duplicated(monkeypatch):
    baseline = load_runtime_config()
    configured_name = "Test/Configured-Qwen-Identity"
    configured = replace(baseline, model={**baseline.model, "name": configured_name})
    monkeypatch.setattr(configuration, "load_runtime_config", lambda _path: configured)
    prompt = load_production_runtime_config().runtime_system_prompt
    assert f"Your underlying configured model is {configured_name}." in prompt
    assert EXPECTED_MODEL_NAME not in prompt


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("model", {"name": "other"}),
        ("generation", {"do_sample": True}),
        ("runtime_system_prompt", ""),
        ("runtime_system_prompt", "Identity without configured model"),
        ("runtime_system_prompt", "${model_name} ${AMITAI_REMOTE_INFERENCE_TOKEN}"),
        ("runtime_system_prompt", "${model_name} ${broken"),
    ],
)
def test_invalid_profile_fails_without_falling_back_to_frozen_identity(tmp_path, field, value):
    profile = yaml.safe_load(DEFAULT_PRODUCTION_PROFILE_PATH.read_text(encoding="utf-8"))
    profile[field] = value
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_production_runtime_config(profile_path=path)


def test_missing_production_profile_does_not_fall_back(tmp_path):
    with pytest.raises(ValueError, match="Unable to read production runtime profile"):
        load_production_runtime_config(profile_path=tmp_path / "missing.yaml")


def test_environment_settings_override_still_uses_production_identity(tmp_path, monkeypatch):
    selected = tmp_path / "settings.yaml"
    original = DEFAULT_RUNTIME_CONFIG_PATH.read_bytes()
    selected.write_bytes(original)
    monkeypatch.setenv("AMITAI_RUNTIME_CONFIG", str(selected))
    monkeypatch.setenv("AMITAI_INFERENCE_PROVIDER", "transformers")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", "IDENTITY_TOKEN_CANARY_918273")
    monkeypatch.chdir(tmp_path)  # profile location does not depend on the settings file's directory
    captured = []
    result = select_response_generator(
        generator_factory=lambda config: captured.append(config) or config
    )
    assert result is captured[0] and result.source_path == selected
    assert result.runtime_system_prompt.startswith("You are Aevon,")
    assert "IDENTITY_TOKEN_CANARY_918273" not in result.runtime_system_prompt
    assert selected.read_bytes() == original


def _local_runtime(engine):
    def engine_factory(_model, _seed):
        return engine

    engine_factory.supports_vision = True
    return select_response_generator(
        mode="transformers",
        generator_factory=lambda config: TransformersChatGenerator(
            config, engine_factory=engine_factory
        ),
    )


@contextmanager
def runtime_pair(outputs):
    engine = VisionEngine(outputs)
    local = _local_runtime(engine)
    harness = Harness(outputs)
    remote = select_response_generator(
        mode="remote",
        remote_endpoint=ORIGIN,
        remote_token=TOKEN,
        remote_provider_factory=lambda **_: harness.provider,
    )
    try:
        assert not engine.calls and not harness.engine.calls  # still lazy
        yield (local, engine), (remote, harness.engine), harness
    finally:
        harness.provider.close()
        harness.client.close()


def _messages(current):
    return [
        GenerationMessage(
            "system",
            format_memory_context([
                {"category": "preference", "key": "style", "value": "concise"},
                {"category": "project", "key": "database", "value": "PostgreSQL"},
            ]),
        ),
        GenerationMessage(
            "system", 'MEMORY_COMMAND_V1\n<memory_command>{"operation":"none"}</memory_command>'
        ),
        GenerationMessage("user", "OLD_DROPPED_IDENTITY_HISTORY " * 2000),
        *(
            GenerationMessage("user" if i % 2 == 0 else "assistant", f"recent-{i} " + "x" * 1200)
            for i in range(40)
        ),
        GenerationMessage("user", current),
    ]


def _text_messages(messages):
    return [
        {
            "role": item["role"],
            "content": item["content"]
            if isinstance(item["content"], str)
            else next(part["text"] for part in item["content"] if part["type"] == "text"),
        }
        for item in messages
    ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_production_identity_compilation_provider_retry_and_vision_parity(
    monkeypatch, streaming, vision, retry
):
    current = "What is your name?" if not retry else "Identify yourself. Answer in exactly 3 words."
    messages = _messages(current)
    outputs = (
        ["Aevon is here."]
        if not retry
        else ["This candidate has too many words.", "Aevon is here."]
    )
    compiler_inputs = []

    def record_compilation(source, **kwargs):
        compiler_inputs.append(kwargs)
        return compile_model_messages(source, **kwargs)

    monkeypatch.setattr("runtime.generator.compile_model_messages", record_compilation)
    with runtime_pair(outputs) as (local, remote, harness):
        for generator, engine in (local, remote):
            kwargs = {"remote_grant": RemoteVisionGrant(str(uuid4()), True)} if vision else {}
            args = (messages, image_bytes()) if vision else (messages,)
            if streaming:
                method = generator.stream_vision_response if vision else generator.stream_response
                items = list(method(*args, cancel_event=Event(), **kwargs))
                assert (
                    "".join(item.delta for item in items if isinstance(item, ChatGenerationDelta))
                    == "Aevon is here."
                )
                result = items[-1]
            else:
                method = (
                    generator.generate_vision_response if vision else generator.generate_response
                )
                result = method(*args, **kwargs)
            assert result.response == "Aevon is here." and result.model == EXPECTED_MODEL_NAME
            assert result.validator["retry_count"] == int(retry)
            assert len(engine.calls) == len(outputs)
            for index, (model_messages, generation) in enumerate(engine.calls):
                compiled = _text_messages(model_messages)
                assert compiled[0]["role"] == "system"
                assert compiled[0]["content"].startswith(
                    generator.config.runtime_system_prompt + "\n\nTOOLS\n"
                )
                assert EXPECTED_MODEL_NAME in compiled[0]["content"]
                assert all(rule in compiled[0]["content"] for rule in EPISTEMIC_RULES)
                assert compiled[0] == _text_messages(engine.calls[0][0])[0]
                assert compiled[0]["content"].count(HISTORY_OMISSION_NOTICE) == 1
                assert compiled[1:3] == [
                    {"role": m.role, "content": m.content} for m in messages[:2]
                ]
                assert '"value":"PostgreSQL"' in compiled[1]["content"]
                assert compiled[3]["role"] == "user"  # no orphan assistant
                assert len(compiled[3:-1]) <= MAX_HISTORY_MESSAGES
                assert sum(len(m["content"]) for m in compiled[3:-1]) <= MAX_HISTORY_CONTEXT_CHARS
                assert compiled[-2]["content"] == messages[-2].content
                assert current in compiled[-1]["content"]
                if index == 0:
                    assert compiled[-1] == {"role": "user", "content": current}
                assert "OLD_DROPPED_IDENTITY_HISTORY" not in json.dumps(compiled)
                assert "You are AmitAI" not in compiled[0]["content"]
                assert generation == load_runtime_config().generation
        local_calls = [_text_messages(call[0]) for call in local[1].calls]
        remote_calls = [_text_messages(call[0]) for call in remote[1].calls]
        assert local_calls == remote_calls
        assert len(compiler_inputs) == len(outputs) * 2  # every repair recompiles
        assert all(
            item["runtime_system_prompt"] == local[0].config.runtime_system_prompt
            for item in compiler_inputs
        )
        expected_route = "/v1/vision" if vision else "/v1/generate"
        if streaming:
            expected_route += "/stream"
        assert all(request.url.path == expected_route for request in harness.requests)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_tools_and_retries_keep_memory_and_omission_notice_in_every_provider_path(streaming, vision):
    messages = _messages("Calculate 2+3, then identify yourself. Answer in exactly 3 words.")
    tool = '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
    outputs = [tool, "This candidate has too many words.", tool, "Aevon is here."]
    with runtime_pair(outputs) as (local, remote, _harness):
        for generator, engine in (local, remote):
            kwargs = {"remote_grant": RemoteVisionGrant(str(uuid4()), True)} if vision else {}
            args = (messages, image_bytes()) if vision else (messages,)
            if streaming:
                method = generator.stream_vision_response if vision else generator.stream_response
                result = list(method(*args, cancel_event=Event(), **kwargs))[-1]
            else:
                method = generator.generate_vision_response if vision else generator.generate_response
                result = method(*args, **kwargs)
            assert result.validator["retry_count"] == 1
            assert len(result.tools) == 2 and all(tool["success"] for tool in result.tools)
            assert len(engine.calls) == 4
            first = _text_messages(engine.calls[0][0])
            for model_messages, _config in engine.calls:
                compiled = _text_messages(model_messages)
                assert compiled[:3] == first[:3]  # Canonical system, exact memory, memory command.
                assert compiled[0]["content"].count(HISTORY_OMISSION_NOTICE) == 1
                assert compiled[1]["content"] == messages[0].content
                assert "OLD_DROPPED_IDENTITY_HISTORY" not in repr(compiled)
            for index in (1, 3):
                assert _text_messages(engine.calls[index][0])[-1]["content"].startswith("<tool_result>")
        assert [_text_messages(call[0]) for call in local[1].calls] == [
            _text_messages(call[0]) for call in remote[1].calls
        ]


def test_production_unconstrained_stream_still_yields_before_generation_finishes():
    engine = VisionEngine(["Aevon is here."])
    generator = _local_runtime(engine)
    allowed_to_finish = Event()

    def after_first():
        assert allowed_to_finish.is_set(), "Production stream buffered the complete response"

    engine.after_first = after_first
    stream = generator.stream_response(
        [GenerationMessage("user", "Who are you?")], cancel_event=Event()
    )
    first = next(stream)
    assert first == ChatGenerationDelta("Aevon") and engine.closed == 0
    allowed_to_finish.set()
    rest = list(stream)
    assert rest[-1].response == "Aevon is here." and engine.closed == 1
