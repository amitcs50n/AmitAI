"""Replaceable, stateless model-inference providers for the local control plane."""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from threading import Event
from typing import Protocol
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from evaluation.hf_backend import GenerationOutput, TransformersGenerator

LOGGER = logging.getLogger(__name__)


class InferenceProviderError(RuntimeError):
    """A safe provider failure that contains no prompt, response, or credential data."""


class InferenceProvider(Protocol):
    """Low-level stateless generation boundary used by local chat orchestration."""

    provider_name: str
    model_name: str

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


class LocalTransformersInferenceProvider:
    """Lazy, single-instance, serialized local Hugging Face inference provider."""

    provider_name = "local-transformers"

    def __init__(
        self,
        model_config: Mapping[str, object],
        seed: int,
        *,
        engine_factory: EngineFactory = TransformersGenerator,
    ) -> None:
        self.model_name = str(model_config["name"])
        self._model_config = dict(model_config)
        self._seed = seed
        self._engine_factory = engine_factory
        self._engine: DetailedGenerationEngine | None = None
        self._initialization_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _get_engine(self) -> DetailedGenerationEngine:
        engine = self._engine
        if engine is not None:
            return engine
        with self._initialization_lock:
            if self._engine is None:
                self._engine = self._engine_factory(dict(self._model_config), self._seed)
            return self._engine

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

        with self._generation_lock:
            stream = iter(
                stream_method(
                    self._copy_messages(messages),
                    dict(generation_config),
                    cancel_event=cancel_event,
                )
            )
            try:
                yield from stream
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()


def _validate_remote_url(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Remote inference endpoint must be an HTTP(S) base URL")
    hostname = parsed.hostname
    if parsed.scheme == "http" and not _is_loopback_hostname(hostname):
        raise ValueError(
            "Remote inference requires HTTPS; plaintext HTTP is allowed only for loopback development"
        )
    return normalized


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


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

    def __init__(
        self,
        endpoint: str,
        token: str,
        model_name: str,
        *,
        timeout_seconds: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_token = token.strip()
        if not normalized_token:
            raise ValueError("Remote inference token must be configured")
        if timeout_seconds <= 0:
            raise ValueError("Remote inference timeout must be positive")
        self.endpoint = _validate_remote_url(endpoint)
        self.model_name = model_name
        self._headers = {"Authorization": f"Bearer {normalized_token}"}
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

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
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
    ) -> GenerationOutput:
        request_id = str(uuid4())
        started_at = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/generate",
                headers=self._headers,
                json=self._request_payload(request_id, messages, generation_config),
            )
            if response.status_code != 200:
                self._log_failure(request_id, f"http_{response.status_code}")
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

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, object],
        *,
        cancel_event: Event,
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
            with self._client.stream(
                "POST",
                f"{self.endpoint}/v1/generate/stream",
                headers=self._headers,
                json=self._request_payload(request_id, messages, generation_config),
            ) as response:
                if response.status_code != 200:
                    self._log_failure(request_id, f"http_{response.status_code}")
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
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
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
