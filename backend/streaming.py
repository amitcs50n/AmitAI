"""HTTP stream ownership: close iterators explicitly and keep blocking work off ASGI."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event, Thread
from typing import TypeVar

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

T = TypeVar("T")


class ClosingStreamingResponse(StreamingResponse):
    """Own async iterator cleanup on disconnect, failed send, and task cancellation.

    Monitor receive even on ASGI 2.4: a silent model may not produce a send on which
    the server can report the disconnected socket. All callers supply async bodies.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as tasks:
                async def stream() -> None:
                    try:
                        await self.stream_response(send)
                    except OSError:
                        # A closed downstream socket is cancellation, not generation failure.
                        pass
                    finally:
                        tasks.cancel_scope.cancel()

                tasks.start_soon(stream)
                await self.listen_for_disconnect(receive)
                tasks.cancel_scope.cancel()
        finally:
            # async-for does not close a generator suspended at yield when send fails.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
        if self.background is not None:
            await self.background()


async def stream_in_worker(factory: Callable[[], Iterator[T]], cancelled: Event) -> AsyncIterator[T]:
    """One owner thread drives AND closes the iterator; completion includes cleanup."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[T | object] = asyncio.Queue(maxsize=8)
    finished = asyncio.Event()
    end = object()
    failures: list[BaseException] = []

    def mark_finished() -> None:
        finished.set()
        if queue.empty():
            queue.put_nowait(end)

    def publish(item: T | object) -> None:
        if cancelled.is_set():
            return
        pending = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        try:
            while not cancelled.is_set():
                try:
                    # Cancellation polling under backpressure, not a generation deadline.
                    pending.result(timeout=0.1)
                    return
                except FutureTimeout:
                    continue
        finally:
            pending.cancel()

    def produce() -> None:
        iterator = None
        try:
            if not cancelled.is_set():
                iterator = iter(factory())
                for item in iterator:
                    if cancelled.is_set():
                        break
                    publish(item)
        except BaseException as exc:  # noqa: BLE001 - relay worker failures, never orphan the consumer.
            failures.append(exc)
        finally:
            try:
                if iterator is not None:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
            except BaseException as exc:  # noqa: BLE001 - cleanup must still signal completion.
                failures.append(exc)
            finally:
                # Completion/control cannot share publish's cancellation filter.
                # A worker failure may set cancelled while the consumer still needs
                # to learn that the producer ended.
                loop.call_soon_threadsafe(mark_finished)

    worker = Thread(target=produce, name="aevon-inference-stream", daemon=True)
    worker.start()
    try:
        while True:
            if finished.is_set() and queue.empty():
                if failures:
                    raise failures[0]
                return
            item = await queue.get()
            if item is end:
                if failures:
                    raise failures[0]
                return
            yield item
    finally:
        cancelled.set()
        # Do not release request ownership or overlap model calls while cleanup runs.
        # There is intentionally no timeout that abandons a model worker.
        with anyio.CancelScope(shield=True):
            await finished.wait()
            await anyio.to_thread.run_sync(worker.join)
