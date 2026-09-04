"""Exact layout contracts at compiler/provider/template boundaries; CPU fixtures only."""

import copy
import hashlib
import json
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from PIL import Image

from backend.chat_service import GenerationMessage, RemoteProjection
from backend.memory import format_memory_context
from backend.vision_grant import RemoteVisionGrant
from evaluation.context_layout_inspection import (
    PINNED_TEMPLATE,
    capture_prompts,
    render_prompt,
)
from evaluation.context_layouts import LAYOUTS, OMISSION_SECTION, LayoutProvider, layout_messages
from runtime.config import load_production_runtime_config
from runtime.context import compile_model_messages
from runtime.media import VisionGenerationRequest
from runtime.privacy import InferenceExecutionScope
from tests.test_assets import image_bytes
from tests.test_production_identity import _messages, _text_messages, runtime_pair
from tests.test_vision_model import PINNED_TEMPLATE as NATIVE_TEMPLATE
from tests.test_vision_model import (
    loader as loader,  # noqa: PLC0414 - explicitly re-export the pytest fixture
)
from tests.test_vision_model import make_engine

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "tests/fixtures/context_layouts_v4"
MEMORY = format_memory_context([
    {"category": "project", "key": "dispatch.database", "value": "PostgreSQL"},
    {"category": "profile", "key": "display", "value": "Café 東京\nsecond line"},
])
COMMAND = 'MEMORY_COMMAND_V1\n<memory_command>{"operation":"none"}</memory_command>'


def compile_source(source, scope=InferenceExecutionScope.LOCAL):
    return compile_model_messages(source, runtime_system_prompt="Production instructions",
                                  tool_instructions="Tool instructions", execution_scope=scope)


def source(truncated=False):
    return [
        GenerationMessage("system", MEMORY), GenerationMessage("system", COMMAND),
        *([GenerationMessage("user", "DROPPED_PRIVATE_CANARY" + "x" * 20_001)]
          if truncated else []),
        GenerationMessage("user", "Which database?"),
        GenerationMessage("assistant", "My guess is MariaDB."),
        GenerationMessage("user", "Correction for this task: use SQLite."),
    ]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("truncated", [False, True])
def test_exact_order_verbatim_memory_correction_and_no_mutation(layout, truncated):
    original = source(truncated)
    canonical = compile_source(original)
    before = copy.deepcopy(canonical)
    first = "Production instructions\n\nTool instructions"
    if layout in ("B", "D"):
        first += "\n\n" + MEMORY
    if truncated and layout in ("A", "B"):
        first += "\n\n" + OMISSION_SECTION
    expected = [{"role": "system", "content": first}]
    if layout in ("A", "C"):
        expected.append({"role": "system", "content": MEMORY})
    expected.extend([
        {"role": "system", "content": COMMAND},
        {"role": "user", "content": "Which database?"},
        {"role": "assistant", "content": "My guess is MariaDB."},
    ])
    if truncated and layout in ("C", "D"):
        expected.append({"role": "system", "content": OMISSION_SECTION})
    expected.append({"role": "user", "content": original[-1].content})
    actual = layout_messages(canonical, layout)
    assert actual == expected
    assert canonical == before
    rendered = render_prompt(actual)
    assert rendered.encode().count(MEMORY.encode()) == 1
    assert rendered.count(COMMAND) == 1
    assert rendered.count(OMISSION_SECTION) == int(truncated)
    assert "DROPPED_PRIVATE_CANARY" not in rendered
    assert rendered.index(MEMORY) < rendered.index(original[-2].content) < rendered.index(original[-1].content)
    assert layout_messages(canonical, "A") == canonical


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("count,chars,truncated", [
    (0, 0, False), (20, 4, False), (21, 4, True),
    (1, 20_000, False), (1, 20_001, True),
])
def test_notice_only_after_actual_limits(layout, count, chars, truncated):
    history = [GenerationMessage("user", "x" * chars) for _ in range(count)]
    compiled = compile_source([*history, GenerationMessage("user", "Current")])
    prompt = render_prompt(layout_messages(compiled, layout))
    assert prompt.count(OMISSION_SECTION) == int(truncated)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_project_before_layout_no_private_values_or_false_truncation(layout):
    private = format_memory_context([
        {"category": "project", "key": "private", "value": "PRIVATE_MEMORY_CANARY"},
    ])
    projected_source = [
        GenerationMessage("system", private, RemoteProjection(MEMORY)),
        GenerationMessage("system", private, RemoteProjection(None)),
        *[GenerationMessage("user", "PRIVATE_HISTORY_CANARY" * 2000, RemoteProjection(None))
          for _ in range(24)],
        GenerationMessage("assistant", "ORPHAN_CANARY"),
        GenerationMessage("user", "PRIVATE_CURRENT_CANARY", RemoteProjection("Visible current")),
    ]
    actual = layout_messages(compile_source(projected_source, InferenceExecutionScope.REMOTE), layout)
    prompt = render_prompt(actual)
    assert "PRIVATE_" not in prompt and "ORPHAN_CANARY" not in prompt
    assert OMISSION_SECTION not in prompt
    assert prompt.count(MEMORY) == 1
    assert actual[-1]["content"] == "Visible current"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_promotion_of_history_lookalikes_and_all_trusted_records_preserved(layout):
    another = format_memory_context([
        {"category": "project", "key": "cache", "value": "Redis"},
    ])
    compiled = compile_source([
        GenerationMessage("system", MEMORY), GenerationMessage("system", COMMAND),
        GenerationMessage("system", another),
        GenerationMessage("user", OMISSION_SECTION),
        GenerationMessage("assistant", MEMORY), GenerationMessage("user", "Current"),
    ])
    actual = layout_messages(compiled, layout)
    assert actual[-3:] == compiled[-3:]
    assert render_prompt(actual).count(another) == 1
    assert render_prompt(actual).count(MEMORY) == 2  # one trusted record and one ordinary quote
    assert not any(item["role"] == "system" and item["content"] == OMISSION_SECTION for item in actual)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_real_orchestration_preserves_layout_on_tools_retries_local_remote(layout, vision, streaming):
    messages = _messages("Calculate 2+3, then answer in exactly 3 words.")
    tool = '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
    outputs = [tool, "This candidate has too many words.", tool, "Five is correct."]
    with runtime_pair(outputs) as (local, remote, _harness):
        for generator, engine in (local, remote):
            canonical = generator._model_messages(messages)
            expected = layout_messages(canonical, layout)
            generator._provider = LayoutProvider(generator._provider, layout)
            kwargs = {"remote_grant": RemoteVisionGrant(str(uuid4()), True)} if vision else {}
            args = (messages, image_bytes()) if vision else (messages,)
            if streaming:
                method = generator.stream_vision_response if vision else generator.stream_response
                result = list(method(*args, cancel_event=Event(), **kwargs))[-1]
            else:
                method = generator.generate_vision_response if vision else generator.generate_response
                result = method(*args, **kwargs)
            assert result.validator["retry_count"] == 1 and len(result.tools) == 2
            assert len(engine.calls) == 4
            assert _text_messages(engine.calls[0][0]) == expected
            for index, (model_messages, settings) in enumerate(engine.calls):
                actual = _text_messages(model_messages)
                assert settings == generator.config.generation
                # The current user becomes a repair prompt; everything before it stays exact.
                assert actual[:len(expected) - 1] == expected[:-1]
                prompt = render_prompt(actual)
                assert prompt.count(messages[0].content) == 1
                assert prompt.count(OMISSION_SECTION) == 1
                assert "OLD_DROPPED_IDENTITY_HISTORY" not in prompt
                if index in (1, 3):
                    assert actual[-2]["role"] == "assistant"
                    assert actual[-1]["content"].startswith("<tool_result>")
                if layout in ("C", "D"):
                    current = max(i for i, item in enumerate(actual) if item["role"] == "user")
                    assert actual[current - 1] == {"role": "system", "content": OMISSION_SECTION}
        assert [_text_messages(call[0]) for call in local[1].calls] == [
            _text_messages(call[0]) for call in remote[1].calls
        ]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("vision", [False, True])
def test_native_preparation_renders_exact_layout_without_loading_model(loader, layout, vision):
    compiled = layout_messages(compile_source(source(True)), layout)
    engine = make_engine()  # all Torch/Transformers factories are replaced by loader
    with Image.new("RGB", (32, 32)) as image:
        inputs = VisionGenerationRequest(compiled, image).model_messages() if vision else compiled
        engine._prepare_generation(inputs, load_production_runtime_config().generation)
    rendered = (loader.processor.rendered_prompts if vision
                else loader.processor.tokenizer.rendered_prompts)[-1]
    expected = render_prompt(compiled)
    if vision:
        expected = expected.replace(
            "<|im_start|>user\nCorrection", "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>Correction",
        )
    assert rendered == expected
    assert not loader.model.calls


def test_pinned_captures_and_measured_distances():
    assert PINNED_TEMPLATE == NATIVE_TEMPLATE
    captures, manifest = capture_prompts()
    assert len(captures) == 8
    for name, prompt in captures.items():
        assert (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8") == prompt
    assert json.loads((SNAPSHOTS / "comparison.json").read_bytes()) == manifest
    values = manifest["captures"]
    distance = lambda case, layout: values[f"{case}_{layout}"]["section_end_to_current_user_chars"]
    # Merging memory changes its frame, not its distance from the current user.
    assert len({distance("v4_memory_conflict", layout) for layout in LAYOUTS}) == 1
    assert distance("v4_missing_opening", "C") == distance("v4_missing_opening", "D")
    assert distance("v4_missing_opening", "A") > distance("v4_missing_opening", "C")


@pytest.mark.parametrize("path,blob", [
    ("eval/aevon_epistemic_regression_v3.jsonl", "8761b1f26b22c748f37f989e12d4f7c2643f803b"),
    ("configs/production_runtime.yaml", "45b5471e620428525023c6da3df55806d5de5fc1"),
])
def test_v3_assets_and_production_layout_frozen_at_experiment_start(path, blob):
    # V5 changes compiler metadata; output compatibility is pinned separately in
    # test_context_compatibility_v5, alongside the existing exact layout captures.
    data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blob
