"""CPU-only contracts for the production loader, processor and shared stream worker."""

import queue
import sys
from contextlib import nullcontext
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

from backend.chat_service import GenerationMessage
from backend.memory import format_memory_context
from evaluation.hf_backend import GenerationOutput
from runtime.config import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    load_production_runtime_config,
    load_runtime_config,
)
from runtime.context import HISTORY_OMISSION_NOTICE, compile_model_messages
from runtime.media import MAX_VISION_PIXELS, MIN_VISION_PIXELS, VisionGenerationRequest
from runtime.model import NativeQwenGenerator, _execution_device, vision_chat_template
from runtime.privacy import InferenceExecutionScope
from runtime.providers import LocalTransformersInferenceProvider
from scripts.vision_smoke import synthetic_image

# Exact template published at the pinned checkpoint, metadata only (no weights).
PINNED_TEMPLATE = """{%- for message in messages %}
{%- if message.role == "system" %}
<|im_start|>system
{{ message.content }}<|im_end|>
{%- elif message.role == "user" %}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is true %}
<think>
{%- else %}
<think>

</think>

{%- endif %}
{%- endif %}
"""


class Tensor:
    def __init__(self, shape=(1, 5), *, data=None, device="cpu"):
        self.shape, self.data, self.device = shape, data, device

    def to(self, device):
        return Tensor(self.shape, data=self.data, device=device)

    def tolist(self):
        return self.data

    def cpu(self):
        return self.to("cpu")

    def __eq__(self, other):
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1024))

    def __getitem__(self, key):
        if isinstance(key, int):
            return Tensor(self.shape[1:], data=self.data[key] if self.data else None)
        return Tensor((1, self.shape[1] - key[1].start))


def render_template(template, **kwargs):
    # Match Transformers' _compile_jinja_template whitespace/sandbox settings.
    # This pinned template needs no Transformers-specific filters or extensions.
    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        template,
    ).render(**kwargs)


class Tokenizer:
    chat_template = PINNED_TEMPLATE

    def __init__(self):
        self.text_calls = []
        self.rendered_prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.text_calls.append((messages, kwargs))
        return render_template(self.chat_template, messages=messages, **kwargs)

    def __call__(self, prompt, **kwargs):
        self.rendered_prompts.append(prompt)
        assert "PIL" not in prompt and "image_pad" not in prompt
        return {"input_ids": Tensor(), "attention_mask": Tensor()}

    def batch_decode(self, output, **kwargs):
        assert output.shape == (1, 3)
        return ["Red square shown."]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return "".join({100: "Red ", 101: "square ", 102: "shown."}[i] for i in ids)

    def convert_ids_to_tokens(self, token_id):
        return {248053: "<|vision_start|>", 248054: "<|vision_end|>"}[token_id]


class Qwen3VLProcessor:
    image_token_id = 248056
    image_token = "<|image_pad|>"
    chat_template = PINNED_TEMPLATE

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.image_processor = SimpleNamespace(patch_size=16, merge_size=2, temporal_patch_size=2)
        self.calls = []
        self.rendered_prompts = []
        self.grid = [[1, 64, 64]]

    def apply_chat_template(self, messages, *, processor_kwargs=None, **kwargs):
        self.calls.append((messages, kwargs))
        assert kwargs["tokenize"] and kwargs["return_dict"] and kwargs["return_tensors"] == "pt"
        prompt = render_template(kwargs["chat_template"], messages=messages, **kwargs)
        self.rendered_prompts.append(prompt)
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 1
        assert "PIL" not in prompt
        assert not set(processor_kwargs).intersection(kwargs)
        assert processor_kwargs == {
            "min_pixels": MIN_VISION_PIXELS,
            "max_pixels": MAX_VISION_PIXELS,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        }
        return {
            # 1024 expanded image tokens plus five text/special tokens, batch 1.
            "input_ids": Tensor((1, 1029)),
            "attention_mask": Tensor((1, 1029)),
            "pixel_values": Tensor((4096, 1536)),
            "image_grid_thw": Tensor((1, 3), data=self.grid),
        }


class Module:
    def __init__(self, device):
        self._hf_hook = SimpleNamespace(execution_device=device)

    def parameters(self):
        yield SimpleNamespace(device="meta", numel=lambda: 100)


class Model:
    def __init__(self):
        self.model = SimpleNamespace(visual=Module("cuda:1"))
        self.embedding = Module("cuda:0")
        self.calls = []
        self.release = Event()
        self.terminal = Event()
        self.failure = None
        self.fail_after_delta = False
        self.worker = None
        self.cancellation_seen = False

    @property
    def device(self):
        pytest.fail("Sharded generation must not assume model.device covers every input")

    def eval(self):
        pass

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["input_ids"].device == "cuda:0"
        assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
        assert kwargs["attention_mask"].device == "cuda:0"
        assert kwargs["use_cache"] is True
        assert "cache_position" not in kwargs  # Qwen's generation code owns cache positions.
        if "pixel_values" in kwargs:
            assert kwargs["pixel_values"].device == "cuda:1"
            assert kwargs["pixel_values"].shape == (4096, 1536)
            assert kwargs["image_grid_thw"].tolist() == [[1, 64, 64]]
            assert kwargs["image_grid_thw"].device == "cuda:0"
        streamer = kwargs.get("streamer")
        if streamer:
            self.worker = current_thread()
            if self.failure is not None and not self.fail_after_delta:
                raise self.failure
            criteria, = kwargs["stopping_criteria"]
            stopped = criteria(kwargs["input_ids"], None)
            assert stopped.shape == (1,) and stopped.device == "cuda:0"
            assert stopped.tolist() == [False]
            streamer.put(kwargs["input_ids"].cpu())  # skip the multimodal prompt
            streamer.put(Tensor((1,), data=[100]))
            assert self.release.wait(5), "consumer never received the first chunk"
            if self.failure is not None:
                raise self.failure
            self.cancellation_seen = criteria(kwargs["input_ids"], None).tolist() == [True]
            if not self.cancellation_seen:
                streamer.put(Tensor((1, 2), data=[[101, 102]]))
            streamer.end()
            self.terminal.set()
        return Tensor((1, kwargs["input_ids"].shape[1] + 3))


class Streamer:
    def __init__(self, tokenizer, **kwargs):
        self.queue = queue.Queue()
        assert kwargs.pop("skip_prompt") is True
        self.tokenizer, self.decode_kwargs = tokenizer, kwargs
        self.first = True

    def put(self, value):
        assert value.shape[0] == 1
        if self.first:
            self.first = False
            return
        if len(value.shape) > 1:
            value = value[0]
        self.on_finalized_text(self.tokenizer.decode(value.tolist(), **self.decode_kwargs))

    def end(self):
        self.on_finalized_text("", stream_end=True)

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.queue.put(text)
        if stream_end:
            self.queue.put(None)

    def __iter__(self):
        while (value := self.queue.get(timeout=6)) is not None:
            yield value


@pytest.fixture
def loader(monkeypatch):
    state = SimpleNamespace(
        processor=Qwen3VLProcessor(),
        model=Model(),
        calls=[],
        loading={},
        config=SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            language_model_only=False,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            vision_config=SimpleNamespace(
                patch_size=16, spatial_merge_size=2, temporal_patch_size=2
            ),
        ),
    )

    def factory(kind):
        def load(name, **kwargs):
            state.calls.append((kind, name, kwargs))
            assert name == EXPECTED_MODEL_NAME and kwargs["revision"] == EXPECTED_MODEL_REVISION
            assert kwargs["trust_remote_code"] is False
            if kind == "model":
                return state.model, state.loading
            return getattr(state, kind)

        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fake",
            bfloat16="BF16",
            bool=bool,
            full=lambda shape, value, *, dtype, device: Tensor(shape, data=[value], device=device),
            manual_seed=lambda _: None,
            cuda=SimpleNamespace(is_available=lambda: False),
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="fake",
            AutoConfig=factory("config"),
            AutoProcessor=factory("processor"),
            Qwen3_5ForConditionalGeneration=factory("model"),
            StoppingCriteria=object,
            StoppingCriteriaList=list,
            TextIteratorStreamer=Streamer,
        ),
    )
    return state


def make_engine():
    config = load_runtime_config()
    return NativeQwenGenerator(config.model, config.generation["seed"])


def test_native_loader_one_model_one_processor_shared_tokenizer_text_and_vision(loader):
    engine = make_engine()
    config = load_runtime_config()
    text_messages = [{"role": "user", "content": "Describe a square"}]
    assert engine.generate_detailed(text_messages, config.generation).text == "Red square shown."
    assert "pixel_values" not in loader.model.calls[0]
    with Image.new("RGB", (128, 128)) as image:
        request = VisionGenerationRequest(text_messages, image)
        assert (
            engine.generate_detailed(request.model_messages(), config.generation).text
            == "Red square shown."
        )
    assert [call[0] for call in loader.calls] == ["config", "processor", "model"]
    assert loader.calls[-1][2]["dtype"] == "BF16"
    assert loader.calls[-1][2]["device_map"] == "auto"
    assert loader.calls[-1][2]["output_loading_info"] is True
    assert engine.tokenizer is engine.processor.tokenizer
    assert engine.tokenizer.chat_template == PINNED_TEMPLATE
    assert loader.processor.calls[0][0][-1]["content"][0]["image"] is image
    assert len(loader.processor.tokenizer.text_calls) == 1


def test_native_worker_streams_pixels_before_terminal_output(loader):
    engine = make_engine()
    with Image.new("RGB", (128, 128)) as image:
        request = VisionGenerationRequest([{"role": "user", "content": "What is shown?"}], image)
        stream = engine.generate_detailed_stream(
            request.model_messages(),
            load_runtime_config().generation,
            cancel_event=Event(),
        )
        try:
            assert next(stream) == "Red "
            assert not loader.model.terminal.is_set()
        finally:
            loader.model.release.set()
        remainder = list(stream)
        assert remainder[0] == "square shown."
        assert remainder[-1].text == "Red square shown."
        assert remainder[-1].input_tokens == 1029 and remainder[-1].output_tokens == 3
        assert sum(isinstance(item, GenerationOutput) for item in remainder) == 1
        assert not loader.model.worker.is_alive()


@pytest.mark.parametrize("after_delta", [False, True])
def test_native_worker_relays_original_valueerror_and_joins(loader, after_delta):
    engine = make_engine()
    failure = ValueError("VISION_STREAM_CANARY")
    loader.model.failure, loader.model.fail_after_delta = failure, after_delta
    loader.model.release.set()
    items, errors = [], []

    def consume(messages):
        try:
            items.extend(engine.generate_detailed_stream(messages, {}, cancel_event=Event()))
        except ValueError as exc:
            errors.append(exc)

    with synthetic_image() as image:
        request = VisionGenerationRequest([{"role": "user", "content": "Describe"}], image)
        consumer = Thread(target=consume, args=(request.model_messages(),), daemon=True)
        consumer.start()
        consumer.join(timeout=7)
        assert not consumer.is_alive(), "worker exception deadlocked the stream consumer"
    assert errors == [failure] and errors[0] is failure
    assert items == (["Red "] if after_delta else [])
    assert not loader.model.worker.is_alive()


@pytest.mark.parametrize("mode", ["before", "after_prepare", "during", "close"])
def test_native_stream_cancellation_has_no_terminal_and_joins(loader, monkeypatch, mode):
    engine = make_engine()
    signal = Event()
    if mode == "before":
        signal.set()
    elif mode == "after_prepare":
        prepare = engine._prepare_generation

        def cancel_after_prepare(*args):
            output = prepare(*args)
            signal.set()
            return output

        monkeypatch.setattr(engine, "_prepare_generation", cancel_after_prepare)
    with synthetic_image() as image:
        request = VisionGenerationRequest([{"role": "user", "content": "Describe"}], image)
        stream = engine.generate_detailed_stream(request.model_messages(), {}, cancel_event=signal)
        if mode in {"before", "after_prepare"}:
            assert list(stream) == [] and not loader.model.calls
            assert bool(loader.processor.calls) is (mode == "after_prepare")
        else:
            assert next(stream) == "Red "
            if mode == "during":
                signal.set()
            # The worker may finish before close(), but closing still joins it
            # without publishing the completion it may already have queued.
            loader.model.release.set()
            if mode == "close":
                stream.close()
            assert list(stream) == [] and signal.is_set()
            assert not loader.model.worker.is_alive()
            if mode == "during":
                assert loader.model.cancellation_seen


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("stop", ["event", "close"])
def test_native_provider_cancel_then_generate_repeatedly_reuses_model_and_joins(loader, vision, stop):
    config = load_runtime_config()
    provider = LocalTransformersInferenceProvider(config.model, config.generation["seed"])
    with synthetic_image() as image:
        messages = [{"role": "user", "content": "Describe"}]
        def stream(signal):
            if vision:
                return provider.stream_vision(VisionGenerationRequest(messages, image), {}, cancel_event=signal)
            return provider.stream(messages, {}, cancel_event=signal)
        for _ in range(8):
            signal = Event()
            loader.model.release.clear()
            first = stream(signal)
            assert next(first) == "Red "
            if stop == "event":
                signal.set()
                loader.model.release.set()  # Let the fake model finish its current forward step.
                assert list(first) == []
            else:
                def next_forward_step(cancelled=signal):
                    cancelled.wait(2)
                    loader.model.release.set()
                unblock = Thread(target=next_forward_step, daemon=True)
                unblock.start()
                first.close()
                unblock.join(2)
            assert signal.is_set() and loader.model.cancellation_seen
            assert not loader.model.worker.is_alive()
            assert not provider._generation_lock.locked()
            second = list(stream(Event()))
            assert second[-1].text == "Red square shown."
            assert not loader.model.worker.is_alive()
        assert [call[0] for call in loader.calls] == ["config", "processor", "model"]


@pytest.mark.parametrize("after_delta", [False, True])
def test_native_provider_worker_failure_then_second_generation_succeeds(loader, after_delta):
    config = load_runtime_config()
    provider = LocalTransformersInferenceProvider(config.model, config.generation["seed"])
    loader.model.failure = ValueError("synthetic model failure")
    loader.model.fail_after_delta = after_delta
    loader.model.release.set()
    messages = [{"role": "user", "content": "Describe"}]
    with pytest.raises(ValueError, match="synthetic model failure"):
        list(provider.stream(messages, {}, cancel_event=Event()))
    assert not loader.model.worker.is_alive() and not provider._generation_lock.locked()
    loader.model.failure = None
    assert list(provider.stream(messages, {}, cancel_event=Event()))[-1].text == "Red square shown."
    assert not loader.model.worker.is_alive()
    assert [call[0] for call in loader.calls] == ["config", "processor", "model"]


def test_same_native_instance_handles_all_four_modes(loader):
    engine = make_engine()
    loader.model.release.set()
    messages = [{"role": "user", "content": "Describe"}]
    with synthetic_image() as image:
        for inputs in (messages, VisionGenerationRequest(messages, image).model_messages()):
            expected = engine.generate_detailed(inputs, {})
            events = list(engine.generate_detailed_stream(inputs, {}, cancel_event=Event()))
            assert events[-1] == expected
            assert "".join(events[:-1]) == expected.text
    assert len(loader.model.calls) == 4
    assert [kind for kind, *_ in loader.calls] == ["config", "processor", "model"]


def test_legacy_processor_api_keeps_exact_same_limits_and_options(loader, monkeypatch):
    modern = loader.processor.apply_chat_template

    def legacy(messages, **kwargs):
        assert "processor_kwargs" not in kwargs
        options = {key: kwargs.pop(key) for key in (
            "min_pixels", "max_pixels", "return_token_type_ids", "return_mm_token_type_ids"
        )}
        return modern(messages, processor_kwargs=options, **kwargs)

    monkeypatch.setattr(loader.processor, "apply_chat_template", legacy)
    engine = make_engine()
    with synthetic_image() as image:
        request = VisionGenerationRequest([{"role": "user", "content": "Describe"}], image)
        assert engine.generate_detailed(request.model_messages(), {}).input_tokens == 1029
    assert len(loader.processor.calls) == 1


@pytest.mark.parametrize(
    "change", ["processor", "vision_config", "language_model_only", "template", "tokens"]
)
def test_loader_fails_before_model_allocation_for_incompatible_metadata(loader, change):
    if change == "processor":
        loader.processor = object()
    elif change == "template":
        loader.processor.chat_template = "incompatible"
    elif change == "tokens":
        loader.processor.image_token_id = 0
    else:
        setattr(loader.config, change, None)
    with pytest.raises(RuntimeError, match="Native Qwen initialization failed"):
        make_engine()
    assert not any(call[0] == "model" for call in loader.calls)


@pytest.mark.parametrize(
    "key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
)
def test_loader_rejects_incomplete_or_incompatible_vision_weights(loader, key, caplog):
    loader.loading[key] = ["model.visual.PRIVATE_PATH_CANARY"]
    with pytest.raises(RuntimeError) as error:
        make_engine()
    assert "PRIVATE_PATH_CANARY" not in str(error.value) + caplog.text
    assert "vision_capable=true" not in caplog.text


def test_missing_native_class_never_falls_back_to_causal_model(loader, monkeypatch):
    monkeypatch.delattr(sys.modules["transformers"], "Qwen3_5ForConditionalGeneration")
    with pytest.raises(RuntimeError, match="Native Qwen initialization failed"):
        make_engine()
    assert not loader.calls


@pytest.mark.parametrize(
    "grid", [[[1, 128, 128]], [[2, 64, 64]], [], [[1, 0, 1]], [[1, 63, 64]], [[1, 32, 32]]]
)
def test_vision_token_budget_checked_before_model_generate(loader, grid):
    engine = make_engine()
    loader.processor.grid = grid
    with Image.new("RGB", (128, 128)) as image:
        request = VisionGenerationRequest([{"role": "user", "content": "Describe"}], image)
        with pytest.raises(ValueError, match="tensor layout"):
            engine.generate_detailed(request.model_messages(), {})
    assert not loader.model.calls


@pytest.mark.parametrize("thinking", [True, False])
def test_template_adapter_preserves_every_text_byte_and_renders_only_current_image(thinking):
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Current question"},
    ]
    kwargs = {"messages": messages, "add_generation_prompt": True, "enable_thinking": thinking}
    render = lambda template: render_template(template, **kwargs)
    assert render(vision_chat_template(PINNED_TEMPLATE)) == render(PINNED_TEMPLATE)
    with Image.new("RGB", (32, 32)) as image:
        request = VisionGenerationRequest(messages, image)
        assert "Current question" not in repr(request) and "PIL" not in repr(request)
        multimodal = request.model_messages()
        assert multimodal[:-1] == messages[:-1]
        assert isinstance(messages[-1]["content"], str)
        result = render_template(
            vision_chat_template(PINNED_TEMPLATE),
            **{**kwargs, "messages": multimodal},
            vision_start_token="<|vision_start|>",
            image_token="<|image_pad|>",
            vision_end_token="<|vision_end|>",
        )
        assert result.count("<|image_pad|>") == 1 and "PIL" not in result
        assert "Current question" in result


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_exact_native_render_keeps_memory_roles_correction_and_omission_signal(
    loader, vision, truncated, correction,
):
    # loader replaces all Torch/Transformers factories with CPU fixtures; this
    # exercises NativeQwenGenerator's real preparation method without model inference.
    config = load_production_runtime_config()
    memory = format_memory_context([
        {"category": "project", "key": "audit.database", "value": "PostgreSQL"},
    ])
    current = ("Correction: use SQLite for this task." if correction
               else "Which database is specified in trusted context?")
    source = [GenerationMessage("system", memory)]
    if truncated:
        source.extend([
            GenerationMessage("user", "OMITTED_RENDER_CANARY"),
            GenerationMessage("assistant", "x" * 20_001),
        ])
    source.extend([
        GenerationMessage("user", "What database does the service use?"),
        GenerationMessage("assistant", "My earlier guess was MariaDB."),
        GenerationMessage("user", current),
    ])
    compiled = compile_model_messages(
        source, runtime_system_prompt=config.runtime_system_prompt,
        tool_instructions="Tool instructions", execution_scope=InferenceExecutionScope.LOCAL,
    )
    assert compiled[1] == {"role": "system", "content": memory}
    assert [item["role"] for item in compiled] == ["system", "system", "user", "assistant", "user"]
    engine = make_engine()
    with Image.new("RGB", (32, 32)) as image:
        model_messages = VisionGenerationRequest(compiled, image).model_messages() if vision else compiled
        engine._prepare_generation(model_messages, config.generation)
    rendered = (loader.processor.rendered_prompts if vision
                else loader.processor.tokenizer.rendered_prompts)[-1]
    expected = ""
    for index, item in enumerate(compiled):
        content = item["content"]
        if vision and index == len(compiled) - 1:
            content = "<|vision_start|><|image_pad|><|vision_end|>" + content
        expected += f"<|im_start|>{item['role']}\n{content}<|im_end|>"
    # Preserve the published template's exact generation prefix, including whitespace.
    expected += "<|im_start|>assistant<think>\n\n</think>"
    assert rendered == expected  # Every role frame and content byte reaches tokenization.
    assert f"<|im_start|>system\n{memory}<|im_end|>" in rendered
    assert rendered.count('"value":"PostgreSQL"') == 1
    assert rendered.index(memory) < rendered.index("My earlier guess was MariaDB.") < rendered.index(current)
    assert rendered.count(HISTORY_OMISSION_NOTICE) == int(truncated)
    assert "OMITTED_RENDER_CANARY" not in rendered


def test_execution_device_requires_real_device_or_dispatch_hook():
    with pytest.raises(ValueError, match="execution device"):
        _execution_device(
            SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="meta")]))
        )


def test_smoke_image_is_synthetic_bounded_and_in_memory():
    with synthetic_image() as image:
        assert image.size == (1024, 768)
        assert image.getpixel((200, 400)) == (255, 0, 0)
        assert image.getpixel((740, 440)) == (0, 0, 255)
        assert not getattr(image, "filename", None)
