"""Native vision orchestration contracts without importing Torch or model weights."""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from backend.chat_service import ChatGenerationDelta, ChatGenerationError, GenerationMessage
from evaluation.hf_backend import GenerationOutput
from runtime.config import EXPECTED_MODEL_NAME, load_runtime_config
from runtime.generator import TransformersChatGenerator
from runtime.media import MAX_VISION_PIXELS, decoded_vision_image
from tests.test_assets import image_bytes


class VisionEngine:
    supports_vision = True

    def __init__(self, outputs=("Red square shown.",)):
        self.outputs = iter(outputs)
        self.calls = []
        self.images = []
        self.before_generate = lambda: None
        self.after_first = lambda: None
        self.closed = 0

    def _output(self, messages, config):
        self.before_generate()
        self.calls.append((messages, config))
        for message in messages:
            if isinstance(message["content"], list):
                image = message["content"][0]["image"]
                assert image.mode == "RGB" and not image.info
                assert not getattr(image, "filename", None)
                self.images.append(image)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output

    def generate_detailed(self, messages, config):
        return GenerationOutput(self._output(messages, config), 100, 5)

    def generate_detailed_stream(self, messages, config, *, cancel_event):
        try:
            output = self._output(messages, config)
            first, _, rest = output.partition(" ")
            yield first
            self.after_first()
            if cancel_event.is_set():
                return
            if rest:
                yield " " + rest
            yield GenerationOutput(output, 100, 5)
        finally:
            self.closed += 1


def vision_generator(engine):
    loads = []

    def factory(config, seed):
        loads.append((config, seed))
        return engine

    factory.supports_vision = True
    generator = TransformersChatGenerator(load_runtime_config(), engine_factory=factory)
    return generator, loads


def assert_closed(image):
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


def test_lazy_single_engine_shared_by_text_vision_and_streaming():
    engine = VisionEngine(["Text answer.", "Red square shown.", "Next vision answer."])
    generator, loads = vision_generator(engine)
    assert generator.supports_vision and not loads
    assert generator.generate_response([GenerationMessage("user", "Hi")]).response == "Text answer."
    result = generator.generate_vision_response(
        [GenerationMessage("user", "Describe")], image_bytes()
    )
    assert result.response == "Red square shown." and result.model == EXPECTED_MODEL_NAME
    assert_closed(engine.images[0])
    output = list(
        generator.stream_vision_response(
            [GenerationMessage("user", "Describe")],
            image_bytes(),
            cancel_event=Event(),
        )
    )
    assert len(loads) == 1 and output[-1].response == "Next vision answer."
    assert_closed(engine.images[-1])
    assert all(isinstance(m["content"], str) for m in engine.calls[0][0])


def test_text_and_vision_use_the_same_serial_generation_gate():
    engine = VisionEngine(["Text answer.", "Red square shown."])
    generator, loads = vision_generator(engine)
    text_started, release_text, vision_started = Event(), Event(), Event()

    def block_first():
        if not text_started.is_set():
            text_started.set()
            assert release_text.wait(5)
        else:
            vision_started.set()

    engine.before_generate = block_first
    with ThreadPoolExecutor(max_workers=2) as pool:
        text = pool.submit(generator.generate_response, [GenerationMessage("user", "Hi")])
        assert text_started.wait(5)
        vision = pool.submit(
            generator.generate_vision_response,
            [GenerationMessage("user", "Describe")],
            image_bytes(),
        )
        try:
            assert not vision_started.wait(0.1)
        finally:
            release_text.set()
        assert text.result(timeout=5).response == "Text answer."
        assert vision.result(timeout=5).response == "Red square shown."
    assert len(loads) == 1 and vision_started.is_set()


def test_vision_genuinely_streams_before_engine_is_allowed_to_finish():
    engine = VisionEngine(["Python dictionaries shown."])
    allowed_to_finish = Event()
    terminal = Event()

    def after_first():
        assert allowed_to_finish.is_set(), "regression: candidate was buffered before first delta"
        terminal.set()

    engine.after_first = after_first
    generator, _ = vision_generator(engine)
    stream = generator.stream_vision_response(
        [GenerationMessage("user", "Describe")],
        image_bytes(),
        cancel_event=Event(),
    )
    first = next(stream)
    assert first == ChatGenerationDelta("Python") and not terminal.is_set()
    allowed_to_finish.set()
    items = [first, *stream]
    assert (
        "".join(item.delta for item in items if isinstance(item, ChatGenerationDelta))
        == items[-1].response
    )
    assert terminal.is_set()


@pytest.mark.parametrize("streaming", [False, True])
def test_vision_retry_reuses_image_and_minimized_history_then_succeeds(streaming):
    engine = VisionEngine(["The red square shown.", "Red square shown."])
    generator, _ = vision_generator(engine)
    history = [GenerationMessage("user", "OLD_PRIVATE_CANARY" * 3000)]
    history += [
        GenerationMessage("user" if i % 2 == 0 else "assistant", f"recent {i}") for i in range(30)
    ]
    messages = [*history, GenerationMessage("user", "Describe this. Answer in exactly 3 words.")]
    if streaming:
        items = list(
            generator.stream_vision_response(messages, image_bytes(), cancel_event=Event())
        )
        assert [i.delta for i in items if isinstance(i, ChatGenerationDelta)] == [
            "Red square shown."
        ]
        result = items[-1]
    else:
        result = generator.generate_vision_response(messages, image_bytes())
    assert result.response == "Red square shown." and result.validator["retry_passed"] is True
    assert result.validator["retry_count"] == 1
    assert engine.images[0] is engine.images[1]
    assert_closed(engine.images[0])
    for messages, _ in engine.calls:
        assert "OLD_PRIVATE_CANARY" not in str(messages)
        assert len(messages) <= 22
        assert "recent 29" in str(messages)


@pytest.mark.parametrize("streaming", [False, True])
def test_vision_exhausted_validation_never_returns_or_streams_invalid_candidate(streaming):
    engine = VisionEngine(["Invalid four word response"] * 3)
    generator, _ = vision_generator(engine)
    messages = [GenerationMessage("user", "Answer in exactly 3 words.")]
    visible = []
    with pytest.raises(ChatGenerationError, match="Assistant generation failed"):
        if streaming:
            visible.extend(
                generator.stream_vision_response(messages, image_bytes(), cancel_event=Event())
            )
        else:
            generator.generate_vision_response(messages, image_bytes())
    assert not visible and len(engine.calls) == 2
    assert_closed(engine.images[0])


def test_vision_tool_loop_remains_bounded_and_uses_same_image():
    call = '<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'
    engine = VisionEngine([call, "Result is 1411."])
    generator, _ = vision_generator(engine)
    items = list(
        generator.stream_vision_response(
            [GenerationMessage("user", "Read the calculation. Answer in exactly 3 words.")],
            image_bytes(),
            cancel_event=Event(),
        )
    )
    assert [i.delta for i in items if isinstance(i, ChatGenerationDelta)] == ["Result is 1411."]
    assert items[-1].tools[0]["success"] is True
    assert engine.images[0] is engine.images[1]
    assert "<tool_result>" in engine.calls[1][0][-1]["content"]
    assert items[-1].validator["final_validation"]["passed"] is True


def test_cancel_closes_image_and_underlying_stream_without_final_result():
    engine = VisionEngine()
    generator, _ = vision_generator(engine)
    signal = Event()
    stream = generator.stream_vision_response(
        [GenerationMessage("user", "Describe")],
        image_bytes(),
        cancel_event=signal,
    )
    assert next(stream) == ChatGenerationDelta("Red")
    signal.set()
    assert list(stream) == []
    assert engine.closed == 1
    assert_closed(engine.images[0])


def test_close_without_explicit_cancel_still_closes_image_and_stream():
    engine = VisionEngine()
    generator, _ = vision_generator(engine)
    signal = Event()
    stream = generator.stream_vision_response(
        [GenerationMessage("user", "Describe")],
        image_bytes(),
        cancel_event=signal,
    )
    next(stream)
    stream.close()
    assert engine.closed == 1 and signal.is_set()
    assert_closed(engine.images[0])


def test_cancel_before_start_neither_loads_model_nor_decodes():
    generator, loads = vision_generator(VisionEngine())
    signal = Event()
    signal.set()
    assert list(generator.stream_vision_response([], b"not PNG", cancel_event=signal)) == []
    assert not loads


def test_model_resize_in_ram_preserves_original_bytes_and_aspect_ratio():
    png = image_bytes(size=(2400, 1800))
    original = bytes(png)
    with decoded_vision_image(png) as image:
        width, height = image.size
        assert width * height <= MAX_VISION_PIXELS
        assert abs(width / height - 4 / 3) < 0.002
        assert png == original
    assert_closed(image)


@pytest.mark.parametrize(
    "data", [b"unsafe PNG canary", image_bytes(size=(201, 1)), image_bytes()[:35]]
)
def test_decode_and_processor_failures_are_generic_and_not_logged(data, caplog):
    engine = VisionEngine([RuntimeError("CACHE_PATH_HF_TOKEN_CANARY")])
    generator, _ = vision_generator(engine)
    with caplog.at_level(logging.DEBUG), pytest.raises(ChatGenerationError) as error:
        generator.generate_vision_response([GenerationMessage("user", "PRIVATE_PROMPT")], data)
    assert str(error.value) == "Assistant generation failed"
    assert all(
        text not in caplog.text for text in ["PRIVATE_PROMPT", "canary", "CACHE_PATH_HF_TOKEN"]
    )


def test_model_failure_closes_image_and_sanitizes_exception(caplog):
    engine = VisionEngine([RuntimeError("CACHE_PATH_HF_TOKEN_CANARY")])
    generator, _ = vision_generator(engine)
    with pytest.raises(ChatGenerationError) as error:
        generator.generate_vision_response(
            [GenerationMessage("user", "PRIVATE_PROMPT")], image_bytes()
        )
    assert "CANARY" not in str(error.value) + caplog.text
    assert_closed(engine.images[0])
