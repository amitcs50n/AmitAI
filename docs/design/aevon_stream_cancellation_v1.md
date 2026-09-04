# Aevon V1 stream cancellation

## Findings

The reported A100 incident has no captured server stack trace, so its exact cause is
not established. CPU reproductions prove two defects consistent with the symptoms:

1. `/v1/generate/stream` passed a synchronous generator to Starlette's
   `StreamingResponse`. The installed threadpool iterator adapter does not explicitly
   close that generator. On disconnect or a failed/blocked send, the generator can
   remain suspended at a yield while its provider holds `_generation_lock`. Its
   `finally` owns the cancellation event, so the model does not receive Stop and the
   lock can remain held even after model computation finishes.
2. The remote client tested `cancel_event` only after `response.iter_lines()` returned
   another line. A silent stream could therefore retain the backend's single stream
   worker and semaphore until further network activity or the existing read timeout.
   Later chat requests then waited behind that cancelled request.
3. Worker failures can set the same cancellation event used to suppress data. The
   vision producer also filtered its terminal marker through that event, so an exited
   worker could leave its connected consumer emitting heartbeats forever. Completion
   and error state now use a separate control channel that cannot be discarded as data.

Six ASGI disconnect/send-failure variants plus an idle remote-read reproduction all
failed before the fix and passed after it. Further tests exercise real loopback HTTP,
real blocked socket reads, native model-worker fakes, and the persistence boundary.

## Ownership and cleanup

- `ClosingStreamingResponse` monitors disconnects on both ASGI 2.3 and 2.4 and explicitly
  closes its async body in a shielded `finally`. A failed send and cancellation of the
  request task use the same cleanup path. It is used for backend chat and inference
  text/vision streaming; SSE payloads, authentication, and response headers stay intact.
- Text inference drives the provider iterator in a dedicated owner thread, feeding a
  bounded eight-item async queue. The owner also closes the iterator. The async body
  sets cancellation and waits for thread completion, including cleanup, instead of
  closing an iterator concurrently with `next()` or relying on garbage collection.
- The local provider checks cancellation while waiting for the generation lock and
  again after acquiring it. On early exit or failure, it signals cancellation before
  closing the engine iterator. Lock release has its own `finally`, including when
  iterator creation, iteration, or cleanup throws. Normal completion leaves the caller's
  event unset so tool continuations and mechanical retries retain their existing behavior.
- NativeQwen inherits the existing Transformers stream worker. Its stopping criterion
  observes the event between model steps; iterator cleanup sets the event and joins the
  model thread. That implementation and the cached model are unchanged. The provider
  releases its lock only after this close/join returns; it never permits concurrent calls
  on a model whose previous generation is still running.
- A request-scoped remote watcher interrupts the active HTTP/1 socket's blocked read
  with shutdown and closes that response. It neither closes the shared HTTP client nor
  changes DNS, TLS, origin, authentication, or privacy policy. Cancellation-induced read
  failures emit no successful terminal output. The watcher is stopped and joined on exit.
- Streaming vision waits for its owner thread to finish before returning from cleanup,
  preserving the existing rule that an image is not closed while a worker uses it.
- Backend cleanup publishes its terminal queue marker even if iterator close raises.
  Its existing semaphore remains owned until the producer exits, then releases for the
  next request. Incomplete generation still cannot cross the chat commit boundary.

The 50 ms lock/socket cancellation checks and 100 ms queue-backpressure checks are
polling intervals because a standard threading Event cannot wait simultaneously on a
lock/socket/queue operation. They are not generation deadlines. There is no timeout
that abandons a model worker, no forced lock release, and no process restart mechanism.

## Validation and remaining limits

CPU tests cover explicit Stop, closed iterators, disconnects, failed and blocked sends,
ASGI 2.3/2.4, idle reads, queued cancellation, provider creation/iteration/close failures,
native model exceptions, repeated text/vision cancel/start cycles, cached-model reuse,
no partial persistence, and successful non-cancelled streams. A loopback HTTP test
repeats backend cancellation followed immediately by a second remote request using the
same inference server, provider, and fake model.

A real A100 retest is still required. Stop is cooperative at model-step boundaries;
Python cannot safely interrupt a wedged CUDA kernel or an arbitrary engine that ignores
cancellation. Such a worker is intentionally not abandoned so another generation can
overlap it. Likewise, connection setup before response headers and non-streaming model
calls retain their existing transport/completion limits. Test the deployed GPU with
repeated mid-stream Stop, immediate follow-up requests, and client disconnects for text
and vision, checking worker exit, lock availability, unchanged model identity, and no
partial chat commit. No model download, GPU execution, or training was used for this fix.
