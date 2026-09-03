"""Opt-in context layout experiments, downstream of the production compiler.

Only already-projected, bounded production messages enter this adapter. It moves
existing sections verbatim; it never selects history, retrieves memory or adds rules.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from runtime.context import HISTORY_OMISSION_NOTICE
from runtime.media import VisionGenerationRequest
from runtime.privacy import InferenceExecutionScope

Layout = Literal["A", "B", "C", "D"]
LAYOUTS = ("A", "B", "C", "D")
OMISSION_SECTION = "CONTEXT_AVAILABILITY\n" + HISTORY_OMISSION_NOTICE
_OMISSION_SUFFIX = "\n\n" + OMISSION_SECTION


def layout_messages(
    messages: Sequence[Mapping[str, str]], layout: Layout,
) -> list[dict[str, str]]:
    """Transform a canonical production request, including any appended tool tail.

    This is not a general chat normalizer and must not be applied twice. The
    omission suffix is compiler-owned; mentions inside history are ordinary text.
    """
    if layout not in LAYOUTS:
        raise ValueError("Unknown experimental context layout")
    result = [dict(message) for message in messages]
    if not result or result[0]["role"] != "system":
        raise ValueError("Expected compiled production context")
    if layout == "A":
        return result
    truncated = result[0]["content"].endswith(_OMISSION_SUFFIX)
    if truncated:
        result[0]["content"] = result[0]["content"][:-len(_OMISSION_SUFFIX)]
    if layout in ("B", "D"):
        end = 1
        while end < len(result) and result[end]["role"] == "system" and (
            result[end]["content"].startswith(("MEMORY_CONTEXT_V1\n", "MEMORY_COMMAND_V1\n"))
        ):
            end += 1
        trusted = result[1:end]
        memory = [item for item in trusted if item["content"].startswith("MEMORY_CONTEXT_V1\n")]
        for item in memory:
            result[0]["content"] += "\n\n" + item["content"]
        result[1:end] = [item for item in trusted if not item["content"].startswith(
            "MEMORY_CONTEXT_V1\n",
        )]
    if truncated:
        if layout == "B":
            result[0]["content"] += _OMISSION_SUFFIX
        else:
            current = max(index for index, item in enumerate(result) if item["role"] == "user")
            result.insert(current, {"role": "system", "content": OMISSION_SECTION})
    return result


class LayoutProvider:
    """Evaluation-only adapter; production providers and their policies stay intact.

    Tools append to canonical messages and retries recompile them. Applying the
    layout at each provider entry keeps those paths consistent without changing
    the orchestrator. Vision images/grants are forwarded through existing APIs.
    """

    def __init__(self, provider, layout: Layout):
        if layout not in LAYOUTS:
            raise ValueError("Unknown experimental context layout")
        self.provider, self.layout = provider, layout
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.execution_scope = provider.execution_scope
        self.supports_vision = getattr(provider, "supports_vision", False) is True

    def generate(self, messages, generation_config):
        return self.provider.generate(layout_messages(messages, self.layout), generation_config)

    def stream(self, messages, generation_config, *, cancel_event):
        yield from self.provider.stream(
            layout_messages(messages, self.layout), generation_config, cancel_event=cancel_event,
        )

    def _vision_input(self, request):
        if self.execution_scope is InferenceExecutionScope.LOCAL:
            return VisionGenerationRequest(
                layout_messages(request.messages, self.layout), request.image, request.content_type,
            )
        return layout_messages(request, self.layout)

    def generate_vision(self, request, generation_config, *args, **kwargs):
        return self.provider.generate_vision(
            self._vision_input(request), generation_config, *args, **kwargs,
        )

    def stream_vision(self, request, generation_config, *args, **kwargs):
        yield from self.provider.stream_vision(
            self._vision_input(request), generation_config, *args, **kwargs,
        )

    def close(self):
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
