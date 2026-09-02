"""Authenticated inference-only media routes; request RAM, never files or databases."""

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event, Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.assets import normalize_image
from evaluation.hf_backend import GenerationOutput

from .media import VisionGenerationRequest, decoded_vision_image
from .privacy import InferenceExecutionScope
from .vision_wire import MAX_VISION_BODY_BYTES, decode_vision_body

LOGGER = logging.getLogger(__name__)


def register_vision_routes(application: FastAPI, provider, authorize) -> None:
    from .inference_app import _sse, _validated_output

    async def read_input(request: Request):
        # Authorization precedes all body reads, parsing and decoding.
        authorize(request.headers.get("authorization"))
        if request.query_params or request.headers.get("content-encoding"):
            raise HTTPException(422, "Invalid vision request")
        if (
            getattr(provider, "execution_scope", None) is not InferenceExecutionScope.LOCAL
            or getattr(provider, "supports_vision", False) is not True
        ):
            raise HTTPException(503, "Vision inference is unavailable")
        body = bytearray()
        try:
            length = request.headers.get("content-length")
            if length is not None and (
                not length.isdecimal() or int(length) > MAX_VISION_BODY_BYTES
            ):
                raise ValueError("Invalid vision request")
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_VISION_BODY_BYTES:
                    raise ValueError("Invalid vision request")
                body.extend(chunk)
            metadata, png = decode_vision_body(bytes(body), request.headers.get("content-type", ""))
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid vision request")
            # Strict CRC/container, single-frame and dimensional validation, no metadata.
            normalized = await asyncio.to_thread(normalize_image, png, "image/png")
            # Validate aspect ratio / decode before accepting the SSE response as well.
            with decoded_vision_image(normalized.content):
                pass
            return metadata, normalized.content
        except Exception:  # noqa: BLE001 - decoder and validation details are private
            raise HTTPException(422, "Invalid vision request") from None
        finally:
            body.clear()

    async def events(metadata, png, *, request: Request, streaming: bool):
        """Interruptible async consumer; the worker owns and closes its PIL input."""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=8)
        cancelled = Event()
        end = object()

        def publish(item):
            if cancelled.is_set():
                return
            operation = queue.put(item)
            try:
                pending = asyncio.run_coroutine_threadsafe(operation, loop)
            except RuntimeError:
                operation.close()
                cancelled.set()
                return
            try:
                while not cancelled.is_set():
                    try:
                        pending.result(timeout=0.1)
                        return
                    except FutureTimeout:
                        continue
            finally:
                pending.cancel()

        def produce():
            stream = None
            try:
                with decoded_vision_image(png) as image:
                    vision = VisionGenerationRequest(
                        [m.model_dump() for m in metadata.messages], image
                    )
                    stream = iter(
                        provider.stream_vision(
                            vision,
                            metadata.generation_config.model_dump(exclude_none=True),
                            cancel_event=cancelled,
                        )
                    )
                    chunks = []
                    output = None
                    for item in stream:
                        if cancelled.is_set():
                            return
                        if output is not None:
                            raise ValueError("Invalid vision stream")
                        if isinstance(item, str):
                            if item:
                                chunks.append(item)
                                if streaming:
                                    publish(("delta", {"delta": item}))
                        else:
                            output = _validated_output(item)
                    if not isinstance(output, GenerationOutput) or "".join(chunks) != output.text:
                        raise ValueError("Invalid vision stream")
                    publish(
                        (
                            "final",
                            {
                                "request_id": str(metadata.request_id),
                                "model": provider.model_name,
                                "text": output.text,
                                "input_tokens": output.input_tokens,
                                "output_tokens": output.output_tokens,
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - no body, exception text, paths or tokens
                LOGGER.warning("Vision inference failed failure=%s", type(exc).__name__)
                publish(("error", {"detail": "Inference failed"}))
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:  # noqa: BLE001 - never leak worker cleanup errors
                        LOGGER.warning("Vision cleanup failed failure=%s", type(exc).__name__)
                publish(end)

        worker = Thread(target=produce, name="vision-inference", daemon=True)
        worker.start()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.25)
                except TimeoutError:
                    # StreamingResponse monitors disconnects; the JSON route must do so too.
                    if not streaming and await request.is_disconnected():
                        return
                    if streaming:
                        yield "heartbeat", {}
                    continue
                if item is end:
                    return
                yield item
        finally:
            cancelled.set()
            # Do not forcibly kill CUDA or close an image still in use by the worker.
            # The shared model checks cancellation at generation steps and then releases it.

    @application.post("/v1/vision")
    async def vision(request: Request):
        metadata, png = await read_input(request)
        stream = events(metadata, png, request=request, streaming=False)
        try:
            async for event, data in stream:
                if event == "final":
                    return JSONResponse(data, headers={"Cache-Control": "no-store"})
                if event == "error":
                    raise HTTPException(502, "Inference failed")
            raise HTTPException(502, "Inference failed")
        finally:
            await stream.aclose()

    @application.post("/v1/vision/stream")
    async def vision_stream(request: Request):
        metadata, png = await read_input(request)

        async def body():
            stream = events(metadata, png, request=request, streaming=True)
            try:
                async for event, data in stream:
                    yield ": keep-alive\n\n" if event == "heartbeat" else _sse(event, data)
            finally:
                await stream.aclose()

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
