"""Stateless, authenticated inference-only service for replaceable GPU compute."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from typing import Any, Literal
from uuid import UUID

import anyio
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.security import environment_flag
from backend.streaming import ClosingStreamingResponse, stream_in_worker
from evaluation.hf_backend import GenerationOutput

from .config import DEFAULT_RUNTIME_CONFIG_PATH, load_runtime_config
from .inference_auth import validate_inference_token
from .providers import InferenceProvider, LocalTransformersInferenceProvider

LOGGER = logging.getLogger(__name__)


class InferenceMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    messages: list[InferenceMessage] = Field(min_length=1)
    generation_config: dict[str, Any]


class InferenceResponse(BaseModel):
    request_id: UUID
    model: str
    text: str
    input_tokens: int
    output_tokens: int


def _sse(event: str, data: dict[str, object]) -> str:
    encoded = json.dumps(
        jsonable_encoder(data),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {encoded}\n\n"


def _validated_output(output: object) -> GenerationOutput:
    if not isinstance(output, GenerationOutput):
        raise TypeError("Inference provider returned an invalid output type")
    if not isinstance(output.text, str) or not output.text.strip():
        raise ValueError("Inference provider returned an empty response")
    for value in (output.input_tokens, output.output_tokens):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError("Inference provider returned invalid token metadata")
    return output


def create_inference_app(
    *,
    provider: InferenceProvider | None = None,
    auth_token: str | None = None,
    config_path: str | Path | None = None,
    enable_dev_docs: bool = False,
) -> FastAPI:
    """Create a compute-only app with no database or application-state dependencies."""

    selected_config = load_runtime_config(
        config_path
        or os.getenv("AMITAI_RUNTIME_CONFIG", str(DEFAULT_RUNTIME_CONFIG_PATH))
    )
    selected_provider = provider or LocalTransformersInferenceProvider(
        selected_config.model,
        int(selected_config.generation["seed"]),
    )
    selected_token = auth_token if auth_token is not None else os.getenv(
        "AMITAI_INFERENCE_AUTH_TOKEN"
    )
    if selected_token is not None:
        selected_token = validate_inference_token(selected_token)

    preload_executor: ThreadPoolExecutor | None = None
    preload_future: Future[None] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal preload_executor, preload_future
        # Executor construction starts no thread and loads no model.
        preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aevon-preload")
        try:
            yield
        finally:
            executor = preload_executor
            preload_executor = None
            # A client disconnect must not cancel a shared load. Shutdown owns the join;
            # there is no timeout that abandons a loader still allocating model tensors.
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            preload_future = None

    application = FastAPI(
        title="AmitAI Stateless Inference",
        lifespan=lifespan,
        docs_url="/docs" if enable_dev_docs else None,
        redoc_url="/redoc" if enable_dev_docs else None,
        openapi_url="/openapi.json" if enable_dev_docs else None,
    )
    application.state.inference_provider = selected_provider

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not selected_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference authentication is not configured",
            )
        prefix = "Bearer "
        if (
            authorization is None
            or not authorization.startswith(prefix)
            or not secrets.compare_digest(
                authorization[len(prefix) :].encode("utf-8"), selected_token.encode("ascii"),
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    def local_provider() -> LocalTransformersInferenceProvider:
        # Injected/custom providers must not be reported ready based on a capability flag.
        if not isinstance(selected_provider, LocalTransformersInferenceProvider):
            raise HTTPException(status_code=503, detail="Model readiness is unavailable")
        return selected_provider

    @application.get("/ready")
    async def ready(authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        state = local_provider().initialization_state
        return JSONResponse(
            {"state": state}, status_code=200 if state == "ready" else 503,
            headers={"Cache-Control": "no-store"},
        )

    def run_preload(provider: LocalTransformersInferenceProvider) -> None:
        try:
            provider.preload()
        except BaseException:  # noqa: BLE001 - keep failures out of background tracebacks/futures.
            LOGGER.error("Model preload failed")

    @application.post("/preload")
    async def preload(authorization: str | None = Header(default=None)) -> JSONResponse:
        nonlocal preload_future
        authorize(authorization)
        provider = local_provider()
        if preload_executor is None:
            raise HTTPException(status_code=503, detail="Model preload is unavailable")
        state = provider.initialization_state
        if state == "ready":
            return JSONResponse({"state": "ready"}, headers={"Cache-Control": "no-store"})
        # No await between checking and submitting: concurrent requests on this event
        # loop cannot queue repeated loads. The provider lock also covers generation.
        if state != "loading" and (preload_future is None or preload_future.done()):
            preload_future = preload_executor.submit(run_preload, provider)
        return JSONResponse(
            {"status": "accepted"}, status_code=202, headers={"Cache-Control": "no-store"},
        )

    @application.post("/v1/generate", response_model=InferenceResponse)
    def generate(
        payload: InferenceRequest,
        authorization: str | None = Header(default=None),
    ) -> InferenceResponse:
        authorize(authorization)
        request_id = str(payload.request_id)
        started_at = time.perf_counter()
        try:
            output = _validated_output(
                selected_provider.generate(
                    [message.model_dump() for message in payload.messages],
                    payload.generation_config,
                )
            )
        except Exception as exc:
            LOGGER.error(
                "Inference failed request_id=%s provider=%s failure=%s",
                request_id,
                selected_provider.provider_name,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Inference failed",
            ) from exc
        LOGGER.info(
            "Inference completed request_id=%s provider=%s latency_ms=%d "
            "input_tokens=%d output_tokens=%d",
            request_id,
            selected_provider.provider_name,
            max(0, int((time.perf_counter() - started_at) * 1000)),
            output.input_tokens,
            output.output_tokens,
        )
        return InferenceResponse(
            request_id=payload.request_id,
            model=selected_provider.model_name,
            text=output.text,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
        )

    @application.post("/v1/generate/stream")
    def generate_stream(
        payload: InferenceRequest,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        authorize(authorization)

        async def body():
            request_id = str(payload.request_id)
            started_at = time.perf_counter()
            cancel_event = Event()
            provider_stream = stream_in_worker(
                lambda: selected_provider.stream(
                    [message.model_dump() for message in payload.messages],
                    payload.generation_config,
                    cancel_event=cancel_event,
                ),
                cancel_event,
            )
            chunks: list[str] = []
            try:
                output: GenerationOutput | None = None
                async for item in provider_stream:
                    if isinstance(item, str):
                        if output is not None:
                            raise TypeError("Inference provider streamed after final output")
                        if item:
                            chunks.append(item)
                            yield _sse("delta", {"delta": item})
                        continue
                    if output is not None:
                        raise TypeError("Inference provider returned multiple final outputs")
                    output = _validated_output(item)
                if output is None or "".join(chunks) != output.text:
                    raise ValueError("Inference stream did not reconstruct final output")
                LOGGER.info(
                    "Inference completed request_id=%s provider=%s latency_ms=%d "
                    "input_tokens=%d output_tokens=%d",
                    request_id,
                    selected_provider.provider_name,
                    max(0, int((time.perf_counter() - started_at) * 1000)),
                    output.input_tokens,
                    output.output_tokens,
                )
                yield _sse(
                    "final",
                    {
                        "request_id": payload.request_id,
                        "model": selected_provider.model_name,
                        "text": output.text,
                        "input_tokens": output.input_tokens,
                        "output_tokens": output.output_tokens,
                    },
                )
            # Streaming has already opened, so every provider failure becomes a safe event.
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Inference failed request_id=%s provider=%s failure=%s",
                    request_id,
                    selected_provider.provider_name,
                    type(exc).__name__,
                )
                yield _sse("error", {"detail": "Inference failed"})
            finally:
                cancel_event.set()
                await provider_stream.aclose()

        return ClosingStreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    from .vision_api import register_vision_routes

    register_vision_routes(application, selected_provider, authorize)
    return application


app = create_inference_app(
    enable_dev_docs=environment_flag(
        "AMITAI_ENABLE_DEV_DOCS",
        environ=os.environ,
    )
)
