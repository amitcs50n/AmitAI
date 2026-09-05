"""Replaceable, stateless model-inference providers for the local control plane."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from threading import Event
from typing import Literal, Protocol
from uuid import uuid4

import httpx

from backend.vision_grant import RemoteVisionGrant, require_remote_vision_grant
from evaluation.hf_backend import GenerationOutput

from .inference_auth import validate_inference_token
from .media import VisionGenerationRequest
from .model import NativeQwenGenerator
from .privacy import InferenceExecutionScope, guarded_request_body
from .remote_transport import (
    DNSPolicyError,
    RemoteTransportPolicy,
    Resolver,
    create_remote_ssl_context,
    resolve_addresses,
)
from .vision_wire import encode_vision_body

LOGGER = logging.getLogger(__name__)
# Poll only where threading.Event cannot wait on a lock/socket too. This bounds
# cancellation observation latency, not model runtime or worker shutdown duration.
CANCELLATION_POLL_SECONDS = 0.05


@contextmanager
def _interruptible_response(response: httpx.Response, cancelled: Event):
    """Interrupt this HTTP/1 request's blocked read without closing the shared client."""
    finished = Event()

    def watch() -> None:
        while not finished.wait(CANCELLATION_POLL_SECONDS):
            if not cancelled.is_set():
                continue
            try:
                try:
                    network = response.extensions.get("network_stream")
                    connection = network.get_extra_info("socket") if network is not None else None
                    if connection is not None:
                        try:
                            # close() alone need not wake recv blocked in another thread.
                            connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass  # Already closed/disconnected.
                finally:
                    response.close()
            except Exception as exc:  # noqa: BLE001 - never expose request/socket contents.
                LOGGER.warning("Remote stream cleanup failed failure=%s", type(exc).__name__)
            return

    watcher = threading.Thread(target=watch, name="aevon-remote-cancel", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join()


class InferenceProviderError(RuntimeError):
    """A safe provider failure that contains no prompt, response, or credential data."""


class InferenceProvider(Protocol):
    """Low-level stateless generation boundary used by local chat orchestration."""

    provider_name: str
    model_name: str
    execution_scope: InferenceExecutionScope

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
    ) -> GenerationOutput: ...

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
        *,
        cancel_event: Event,
    ) -> Iterator[str | GenerationOutput]: ...


class DetailedGenerationEngine(Protocol):
    def generate_detailed(
        self,
        messages: list[dict[str, str]],
        generation_config: dict[str, object],
    ) -> GenerationOutput: ...


EngineFactory = Callable[[dict[str, object], int], DetailedGenerationEngine]
InitializationState = Literal["unloaded", "loading", "ready", "failed"]


class LocalTransformersInferenceProvider:
    """Lazy, single-instance, serialized local Hugging Face inference provider."""

    provider_name = "local-transformers"
    execution_scope = InferenceExecutionScope.LOCAL

    def __init__(
        self,
        model_config: Mapping[str, object],
        seed: int,
        *,
        engine_factory: EngineFactory = NativeQwenGenerator,
    ) -> None:
        self.model_name = str(model_config["name"])
        self._model_config = dict(model_config)
        self._seed = seed
        self._engine_factory = engine_factory
        self.supports_vision = getattr(engine_factory, "supports_vision", False) is True
        self._engine: DetailedGenerationEngine | None = None
        self._initialization_lock = threading.Lock()
        # Never hold the state lock during loading or while waiting for initialization.
        self._state_lock = threading.Lock()
        self._initialization_state: InitializationState = "unloaded"
        self._generation_lock = threading.Lock()

    @property
    def initialization_state(self) -> InitializationState:
        """Cheap, read-only state; no model access or exception details."""
        with self._state_lock:
            return self._initialization_state

    def preload(self) -> None:
        """Explicitly load this provider's shared engine without running inference."""
        self._get_engine()

    def _get_engine(self) -> DetailedGenerationEngine:
        with self._state_lock:
            engine = self._engine
        if engine is not None:
            return engine
        with self._initialization_lock:
            with self._state_lock:
                if self._engine is not None:
                    return self._engine
                self._initialization_state = "loading"
            try:
                engine = self._engine_factory(dict(self._model_config), self._seed)
            except BaseException:
                # Do not retain the exception/traceback (it can own partial model tensors).
                with self._state_lock:
                    self._initialization_state = "failed"
                raise
            with self._state_lock:
                self._engine = engine
                self._initialization_state = "ready"
            return engine

    @staticmethod
    def _copy_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
    ) -> GenerationOutput:
        engine = self._get_engine()
        with self._generation_lock:
            return engine.generate_detailed(
                self._copy_messages(messages),
                dict(generation_config),
            )

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
        *,
        cancel_event: Event,
    ) -> Iterator[str | GenerationOutput]:
        if cancel_event.is_set():
            return
        engine = self._get_engine()
        stream_method = getattr(engine, "generate_detailed_stream", None)
        if not callable(stream_method):
            output = self.generate(messages, generation_config)
            if cancel_event.is_set():
                return
            yield output.text
            yield output
            return

        yield from self._serialized_stream(
            lambda: stream_method(self._copy_messages(messages), dict(generation_config),
                                  cancel_event=cancel_event),
            cancel_event,
        )

    def _serialized_stream(self, factory, cancel_event: Event):
        while not cancel_event.is_set():
            if self._generation_lock.acquire(timeout=CANCELLATION_POLL_SECONDS):
                break
        else:
            return
        stream = None
        completed = False
        try:
            if cancel_event.is_set():
                return
            stream = iter(factory())
            # Explicit iteration lets us signal BEFORE close; yield-from closes its
            # delegate first, which can deadlock a delegate waiting for cancellation.
            for item in stream:
                if cancel_event.is_set():
                    return
                yield item
            completed = True
        finally:
            try:
                if not completed:
                    cancel_event.set()
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            finally:
                # Includes iterator creation, iteration and close failures. Native
                # model close joins its worker before we allow the next model call.
                self._generation_lock.release()

    def generate_vision(
        self, request: VisionGenerationRequest, generation_config: Mapping[str, object],
    ) -> GenerationOutput:
        if not self.supports_vision:
            raise InferenceProviderError("Native vision is unavailable")
        engine = self._get_engine()
        with self._generation_lock:
            return engine.generate_detailed(request.model_messages(), dict(generation_config))

    def stream_vision(
        self, request: VisionGenerationRequest, generation_config: Mapping[str, object],
        *, cancel_event: Event,
    ) -> Iterator[str | GenerationOutput]:
        if cancel_event.is_set():
            return
        if not self.supports_vision:
            raise InferenceProviderError("Native vision is unavailable")
        engine = self._get_engine()
        stream_method = getattr(engine, "generate_detailed_stream", None)
        if not callable(stream_method):
            raise InferenceProviderError("Native vision streaming is unavailable")
        yield from self._serialized_stream(
            lambda: stream_method(request.model_messages(), dict(generation_config),
                                  cancel_event=cancel_event),
            cancel_event,
        )


class LocalVisionSession:
    """Request-local adapter; the existing orchestrator still owns tools and retries.

    Borrows one PIL image and the SAME provider/cache/lock. No separate model,
    persisted state, media serialization, or remote implementation is involved.
    """

    execution_scope = InferenceExecutionScope.LOCAL
    provider_name = "local-transformers-vision"

    def __init__(self, provider: LocalTransformersInferenceProvider, image) -> None:
        if provider.execution_scope is not InferenceExecutionScope.LOCAL:
            raise InferenceProviderError("Remote vision disclosure is not enabled")
        self._provider = provider
        self._image = image
        self.model_name = provider.model_name

    def generate(self, messages, generation_config):
        return self._provider.generate_vision(
            VisionGenerationRequest(messages, self._image), generation_config,
        )

    def stream(self, messages, generation_config, *, cancel_event):
        yield from self._provider.stream_vision(
            VisionGenerationRequest(messages, self._image), generation_config,
            cancel_event=cancel_event,
        )


class RemoteVisionSession:
    """Borrow canonical bytes and a live grant; retain REMOTE context projection."""

    execution_scope = InferenceExecutionScope.REMOTE
    provider_name = "remote-vision"

    def __init__(self, provider: RemoteInferenceProvider, image_png: bytes, grant: RemoteVisionGrant):
        require_remote_vision_grant(grant)
        if provider.execution_scope is not InferenceExecutionScope.REMOTE:
            raise InferenceProviderError("Invalid vision provider scope")
        self._provider, self._image_png, self._grant = provider, image_png, grant
        self.model_name = provider.model_name

    def generate(self, messages, generation_config):
        return self._provider.generate_vision(
            messages, generation_config, self._image_png, remote_grant=self._grant,
        )

    def stream(self, messages, generation_config, *, cancel_event):
        yield from self._provider.stream_vision(
            messages, generation_config, self._image_png,
            remote_grant=self._grant, cancel_event=cancel_event,
        )


def _generation_output(
    payload: object,
    *,
    request_id: str,
    model_name: str,
) -> GenerationOutput:
    if not isinstance(payload, dict):
        raise InferenceProviderError("Remote inference returned an invalid response")
    if payload.get("request_id") != request_id or payload.get("model") != model_name:
        raise InferenceProviderError("Remote inference returned mismatched metadata")
    text = payload.get("text")
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    if not isinstance(text, str) or not text.strip():
        raise InferenceProviderError("Remote inference returned an invalid response")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        raise InferenceProviderError("Remote inference returned invalid token metadata")
    return GenerationOutput(text, input_tokens, output_tokens)


class RemoteInferenceProvider:
    """Authenticated client for the stateless AmitAI inference service."""

    provider_name = "remote"
    execution_scope = InferenceExecutionScope.REMOTE
    supports_vision = True

    def __init__(
        self,
        endpoint: str,
        token: str,
        model_name: str,
        *,
        timeout_seconds: float = 600.0,
        transport: httpx.BaseTransport | None = None,
        allowed_origins: str | Sequence[str] | None = None,
        resolver: Resolver = resolve_addresses,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        self._transport_policy = RemoteTransportPolicy.from_config(endpoint, allowed_origins)
        validated_token = validate_inference_token(token)
        if timeout_seconds <= 0:
            raise ValueError("Remote inference timeout must be positive")
        self._resolver = resolver
        self.model_name = model_name
        self._transport_token = validated_token
        self._headers = {
            "Authorization": f"Bearer {validated_token}",
            "Content-Type": "application/json",
        }
        self._client = client_factory(
            verify=create_remote_ssl_context(),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    @property
    def endpoint(self) -> str:
        return self._transport_policy.origin.url

    def _check_dns(self, request_id: str) -> None:
        try:
            self._transport_policy.validate_dns(self._resolver)
        except DNSPolicyError:
            self._log_failure(request_id, "dns_policy")
            raise InferenceProviderError("Remote inference failed") from None

    @staticmethod
    def _request_payload(
        request_id: str,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "request_id": request_id,
            "messages": [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in messages
            ],
            "generation_config": dict(generation_config),
        }

    def _log_success(
        self,
        *,
        request_id: str,
        started_at: float,
        output: GenerationOutput,
    ) -> None:
        LOGGER.info(
            "Inference completed request_id=%s provider=%s latency_ms=%d "
            "input_tokens=%d output_tokens=%d",
            request_id,
            self.provider_name,
            max(0, int((time.perf_counter() - started_at) * 1000)),
            output.input_tokens,
            output.output_tokens,
        )

    @staticmethod
    def _log_failure(request_id: str, failure: str) -> None:
        LOGGER.warning(
            "Inference failed request_id=%s provider=remote failure=%s",
            request_id,
            failure,
        )

    def generate(
        self, messages, generation_config,
    ) -> GenerationOutput:
        return self._generate_request(messages, generation_config)

    def generate_vision(self, messages, generation_config, image_png, *, remote_grant):
        require_remote_vision_grant(remote_grant)
        if not isinstance(image_png, bytes) or not image_png:
            raise InferenceProviderError("Invalid vision input")
        return self._generate_request(
            messages, generation_config, image_png=image_png, remote_grant=remote_grant,
        )

    def stream(self, messages, generation_config, *, cancel_event):
        yield from self._stream_request(messages, generation_config, cancel_event=cancel_event)

    def stream_vision(
        self, messages, generation_config, image_png, *, remote_grant, cancel_event,
    ):
        require_remote_vision_grant(remote_grant)
        if not isinstance(image_png, bytes) or not image_png:
            raise InferenceProviderError("Invalid vision input")
        yield from self._stream_request(
            messages, generation_config, cancel_event=cancel_event,
            image_png=image_png, remote_grant=remote_grant,
        )

    def _wire_request(self, request_id, messages, generation_config, image_png, remote_grant):
        payload = self._request_payload(request_id, messages, generation_config)
        if image_png is not None:
            require_remote_vision_grant(remote_grant)
            payload["version"] = 1
        body = guarded_request_body(payload, transport_token=self._transport_token)
        if image_png is None:
            return "/v1/generate", self._headers, body
        body, content_type = encode_vision_body(body, image_png)
        require_remote_vision_grant(remote_grant)
        return "/v1/vision", {**self._headers, "Content-Type": content_type}, body

    def _generate_request(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
        *, image_png: bytes | None = None, remote_grant: RemoteVisionGrant | None = None,
    ) -> GenerationOutput:
        request_id = str(uuid4())
        started_at = time.perf_counter()
        try:
            path, headers, body = self._wire_request(
                request_id, messages, generation_config, image_png, remote_grant,
            )
            self._check_dns(request_id)
            response = self._client.post(
                f"{self.endpoint}{path}",
                headers=headers,
                content=body,
            )
            if response.status_code != 200:
                failure = (
                    "redirect" if 300 <= response.status_code < 400
                    else f"http_{response.status_code}"
                )
                self._log_failure(request_id, failure)
                raise InferenceProviderError("Remote inference failed")
            output = _generation_output(
                response.json(),
                request_id=request_id,
                model_name=self.model_name,
            )
        except InferenceProviderError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self._log_failure(request_id, type(exc).__name__)
            raise InferenceProviderError("Remote inference failed") from None
        self._log_success(request_id=request_id, started_at=started_at, output=output)
        return output

    def _stream_request(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
        *,
        cancel_event: Event,
        image_png: bytes | None = None,
        remote_grant: RemoteVisionGrant | None = None,
    ) -> Iterator[str | GenerationOutput]:
        if cancel_event.is_set():
            return
        request_id = str(uuid4())
        started_at = time.perf_counter()
        chunks: list[str] = []
        final_output: GenerationOutput | None = None
        event_name: str | None = None
        data_lines: list[str] = []

        def process_event() -> tuple[str, object] | None:
            nonlocal event_name, data_lines
            if event_name is None:
                data_lines = []
                return None
            name = event_name
            serialized = "\n".join(data_lines)
            event_name = None
            data_lines = []
            try:
                data: object = json.loads(serialized or "{}")
            except json.JSONDecodeError as exc:
                raise InferenceProviderError("Remote inference stream was invalid") from exc
            return name, data

        try:
            path, headers, body = self._wire_request(
                request_id, messages, generation_config, image_png, remote_grant,
            )
            self._check_dns(request_id)
            if cancel_event.is_set():
                return
            with self._client.stream(
                "POST",
                f"{self.endpoint}{path}/stream",
                headers=headers,
                content=body,
            ) as response, _interruptible_response(response, cancel_event):
                if response.status_code != 200:
                    failure = (
                        "redirect" if 300 <= response.status_code < 400
                        else f"http_{response.status_code}"
                    )
                    self._log_failure(request_id, failure)
                    raise InferenceProviderError("Remote inference failed")
                for line in response.iter_lines():
                    if cancel_event.is_set():
                        return
                    if line.startswith(":"):
                        continue
                    if line == "":
                        event = process_event()
                        if event is None:
                            continue
                        name, data = event
                        if name == "delta":
                            if not isinstance(data, dict) or not isinstance(
                                data.get("delta"), str
                            ):
                                raise InferenceProviderError(
                                    "Remote inference stream was invalid"
                                )
                            delta = data["delta"]
                            if delta:
                                chunks.append(delta)
                                yield delta
                        elif name == "final":
                            if final_output is not None:
                                raise InferenceProviderError(
                                    "Remote inference stream returned multiple final events"
                                )
                            final_output = _generation_output(
                                data,
                                request_id=request_id,
                                model_name=self.model_name,
                            )
                        elif name == "error":
                            raise InferenceProviderError("Remote inference failed")
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                trailing = process_event()
                if trailing is not None:
                    raise InferenceProviderError("Remote inference stream was unterminated")
        except InferenceProviderError:
            if cancel_event.is_set():
                return
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            if cancel_event.is_set():
                return
            self._log_failure(request_id, type(exc).__name__)
            raise InferenceProviderError("Remote inference failed") from None

        if cancel_event.is_set():
            return
        if final_output is None or "".join(chunks) != final_output.text:
            self._log_failure(request_id, "invalid_terminal_output")
            raise InferenceProviderError("Remote inference stream was invalid")
        self._log_success(
            request_id=request_id,
            started_at=started_at,
            output=final_output,
        )
        yield final_output

    def close(self) -> None:
        self._client.close()
