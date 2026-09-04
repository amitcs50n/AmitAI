"""Authenticated inference-only media routes; request RAM, never files or databases."""

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event, Thread

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.assets import normalize_image
from backend.streaming import ClosingStreamingResponse
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
        finished = asyncio.Event()
        end = object()
        terminal_error = False

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
            nonlocal terminal_error
            stream = None
            try:
                with decoded_vision_image(png) as image:
                    vision = VisionGenerationRequest(
                        [m.model_dump() for m in metadata.messages], image
                    )
                    if cancelled.is_set():
                        return
                    config = metadata.generation_config.model_dump(exclude_none=True)
                    if not streaming:
                        # A JSON request must exercise genuine non-streaming inference.
                        # An in-flight synchronous call cannot be forcibly cancelled;
                        # publish() discards its result after a disconnect instead.
                        output = _validated_output(provider.generate_vision(vision, config))
                    else:
                        stream = iter(
                            provider.stream_vision(vision, config, cancel_event=cancelled)
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
                                    publish(("delta", {"delta": item}))
                            else:
                                output = _validated_output(item)
                        if (
                            not isinstance(output, GenerationOutput)
                            or "".join(chunks) != output.text
                        ):
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
                terminal_error = True
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:  # noqa: BLE001 - never leak worker cleanup errors
                        LOGGER.warning("Vision cleanup failed failure=%s", type(exc).__name__)
                # A model/iterator error may set cancelled; completion must still
                # wake a connected consumer independently of cancellable data.
                def mark_finished():
                    finished.set()
                    if queue.empty():
                        queue.put_nowait(end)
                loop.call_soon_threadsafe(mark_finished)

        worker = Thread(target=produce, name="vision-inference", daemon=True)
        worker.start()
        try:
            while True:
                if finished.is_set() and queue.empty():
                    if terminal_error:
                        yield "error", {"detail": "Inference failed"}
                    return
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
                    if terminal_error:
                        yield "error", {"detail": "Inference failed"}
                    return
                yield item
        finally:
            cancelled.set()
            # Do not forcibly kill CUDA or close an image still in use by the worker.
            # Streaming checks cancellation between generation steps. Synchronous
            # inference keeps its image alive until generate_vision actually returns.
            if streaming:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(worker.join)

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

        return ClosingStreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
