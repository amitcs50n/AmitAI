# AmitAI

AmitAI is a personal-assistant evaluation and optional LoRA project built around
**OBLITERATUS/Qwen3.8-27B-OBLITERATED V3**.

The repository also includes a persistent chat API with a safe mock default and an explicitly
selected real Hugging Face runtime for the tested 27B checkpoint.

## v0 goal

Measure the untouched base model first. Train a text-only BF16 LoRA adapter only if the
held-out evaluation shows repeatable behavior gaps that prompting alone does not solve.
Vision remains frozen if LoRA training is needed.

## Baseline-first workflow

1. Freeze the behavior spec and held-out eval set.
2. Run the untouched base checkpoint with the intended runtime system prompt.
3. Review every response against its pass criteria and failure signals.
4. Fine-tune only if the baseline misses the decision gate.
5. Rerun the same held-out eval after training and compare base versus adapter.

The existence of a training scaffold is not evidence that training is necessary.

## Why the training code uses `FastVisionModel`

Qwen3.8-27B uses the Qwen3.5 multimodal architecture (`Qwen3_5ForConditionalGeneration`).
Even for text-only SFT, using Unsloth's multimodal training path avoids architecture-specific
collator/template problems. The LoRA config freezes vision layers and trains language,
attention and MLP modules only. Baseline inference is separate: it unwraps the text backbone
through `AutoModelForCausalLM`. This does not change the `FastVisionModel` training path.

## Repository structure

```text
amitai/
├── configs/                  # behavior + training configs
├── backend/                  # FastAPI + SQLite persistent chat foundation
├── data/
│   ├── raw/
│   ├── sft/                  # SFT JSONL
│   └── preference/           # later DPO data
├── eval/
├── evaluation/
│   ├── baseline.py          # validation, artifacts, and scoring
│   ├── constraints.py       # deterministic mechanical output checks
│   ├── hf_backend.py        # Qwen3.5 Hugging Face inference backend
│   ├── run_baseline.py      # resumable base-model generation
│   └── summarize.py         # manual-review aggregation
├── frontend/                # Next.js Aevon chat experience
├── runtime/                 # mock/transformers runtime selection + chat adapter
├── training/
│   ├── data.py
│   ├── train_qlora.py
│   └── validate_dataset.py
├── inference/
├── memory/
├── tools/
├── app/
└── tests/
```

## Dataset format

Each JSONL line is one conversation. AmitAI v0 accepts text only, but stores each message in
multimodal-compatible content-part format:

```json
{
  "id": "tech_001",
  "spec_version": "1.1.0",
  "category": "technical",
  "primary_rules": ["TECH-002", "DISAGREE-001"],
  "messages": [
  {"role":"system","content":[{"type":"text","text":"You are AmitAI..."}]},
  {"role":"user","content":[{"type":"text","text":"Question"}]},
  {"role":"assistant","content":[{"type":"text","text":"Answer"}]}
  ]
}
```

Every SFT example must include the four metadata fields and end with an `assistant` message.

## Local development

The local machine does not need the 27B model just to work on the repo.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev,encrypted-storage,secure-runtime]'
pytest
python -m training.validate_dataset
```

Local tests do not load the 27B checkpoint.

### Persistent chat backend

Install the development dependencies and start the local API:

```bash
pip install -e '.[dev,encrypted-storage,secure-runtime]'
python -m runtime.keyctl init --database-file ./amitai.db
python -m runtime.serve
```

The API stores local conversations in a SQLCipher-encrypted `amitai.db` by default and currently
returns a deterministic mock assistant response. `keyctl init` generates the random database key,
wraps it under an interactively entered passphrase, and never displays the key. The canonical
launcher prompts for that passphrase, binds to `127.0.0.1`, and creates a fresh local API token for
the process lifetime. Run all backend, evaluation, and dataset tests with:

```bash
pytest --basetemp .pytest_tmp
```

The mock generator remains isolated behind the chat service. The real runtime uses the same
frontend-facing API contract without making ordinary local development load model weights.

#### Streaming chat

`POST /api/chat` remains the backward-compatible synchronous JSON endpoint. Clients that want
progressive output can send the same request body to `POST /api/chat/stream` and consume its
SSE response:

```bash
curl -N http://127.0.0.1:3000/api/chat/stream \
  -H 'Origin: http://127.0.0.1:3000' \
  -H 'Accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  --data '{"conversation_id":null,"message":"Explain generators in Python."}'
```

The stream lifecycle is `start`, one or more `text` events, `final`, then `done`. A failure after
the HTTP stream has opened is reported as an `error` event instead of exposing an internal
exception. SSE comment heartbeats may appear during model initialization or buffered generation
and can be ignored.

```text
event: start
data: {"conversation_id":null}

event: text
data: {"delta":"Python generators "}

event: text
data: {"delta":"produce values lazily."}

event: final
data: {"conversation_id":"...","message_id":"...","response":"Python generators produce values lazily.","metadata":{"model":"...","latency_ms":2400,"input_tokens":42,"output_tokens":7,"validator":{"retry_attempted":false,"retry_passed":null},"tools":[],"memory":[]}}

event: done
data: {}
```

If generation fails, the terminal sequence is instead:

```text
event: error
data: {"detail":"Assistant generation failed"}
```

Each `text.data.delta` appends directly to the assistant response; concatenating the deltas is
guaranteed to equal `final.data.response`. The `final` data object has the same shape as the
synchronous `ChatResponse`, including conversation/message IDs, model and latency, token counts,
validator details, tools, and memory. Native browser `EventSource` cannot POST the JSON request
body, so the Aevon frontend uses `fetch()` with a `ReadableStream` SSE parser.

Unconstrained prompts stream genuine decoded Transformers output incrementally after a small
prefix gate determines that the response is ordinary assistant text. If the current
prompt contains a supported parsed mechanical constraint, the original generation and all
bounded validator retries are buffered. Only the final candidate is emitted as a `text` event and
persisted; failed candidates are never exposed to the client or stored as conversation messages.
In both paths, the user and final assistant turns are inserted atomically only after successful
generation, with all expensive model work outside SQL transactions.

On disconnect, the server signals cancellation, closes the stream, and does not persist partial
assistant output. Both unconstrained and buffered constrained generation use a Transformers
stopping criterion, and a constrained flow will not start another validator retry after it observes
cancellation. An in-flight CUDA operation may not stop immediately, so the model generation lock
remains held until the worker actually exits; the server does not pretend cancellation completed
or allow another GPU generation to overlap it.

In Aevon, **Stop generation** replaces Send while a response is being generated. It aborts the
current fetch, removes transient assistant text, and leaves the pending user message available
for an explicit Retry without marking the backend disconnected. Every remote-image retry needs
fresh consent. No drafts or partial responses are written to browser storage. Cancellation is
best effort: it cannot undo an atomic save that wins the disconnect race. Once `final` confirms
that save, Stop is no longer offered; if `done` is lost, the UI keeps the saved response and offers
history reload instead of resending it. An unconfirmed completion race may require reloading
history before retrying; there is no server-side request-id deduplication in this UI change.

The chat follows new output while within 96 pixels of the bottom. Scrolling up pauses following;
**Jump to latest**, scrolling back near the bottom, or sending a new turn resumes it. Scroll work
is coalesced to animation frames, and status announcements do not repeat for each text chunk.

#### Explicit image uploads (foundation only)

Aevon cannot browse your files, repositories or folders. The composer paperclip accepts only an
explicitly selected image. There is no path-import, shell, directory-listing or neighbor-file API.
Uploading does **not** authorize remote disclosure. All assets currently have `local_only` scope;
neither their bytes, filenames nor IDs are added to text-provider prompts.

- Input: single-frame PNG, JPEG or WebP; no SVG, GIF, PDF or arbitrary files.
- Limits: 20 MiB per input and normalized image, 8,192 pixels per side, 24,000,000 pixels total,
  and up to four staged uploads. Native vision V1 accepts only **one image per message**;
  multiple-image requests are rejected without deleting uploads. The multipart envelope is capped
  at 20 MiB + 16 KiB.
- Validation: MIME must match decoded bytes; zero-byte, corrupt/truncated, animated, oversized,
  invalid multipart and unsupported inputs fail with sanitized errors. Extensions are not authority.
- Canonical copy: orientation-corrected RGBA PNG reconstructed from pixels. EXIF/GPS/device data,
  ICC/XMP and embedded text are stripped. The original upload is discarded, not archived or spooled
  into a raw temporary file. Metadata stripping does not hide information visible in the pixels.
- Filenames: bounded, sanitized ASCII leaf names for display only. Client directory paths and
  original path-bearing names are never retained. Files use server-generated UUIDs exclusively.

The upload API is authenticated like chat:

```bash
curl http://127.0.0.1:3000/api/assets \
  -H 'Origin: http://127.0.0.1:3000' \
  -F 'file=@photo.png;type=image/png' \
  -F 'persistence_mode=temporary'
```

It returns `{id, kind, original_filename, content_type, byte_size, width, height, sha256,
created_at, conversation_id, persistence_mode, processing_scope}`. Sizes/digest describe the
normalized PNG. Metadata is also available at `GET /api/assets/{id}`. The ID-only
`GET /api/assets/{id}/content` serves that PNG with no-store/nosniff headers; it never serves a
client-chosen path. No generic download or asset-listing endpoint is exposed.

`temporary` is the upload default. The composer stages temporary images with local previews,
supports removal, and sends `asset_ids` alongside a nonempty text `message` to `/api/chat` or
`/api/chat/stream`. Successful sends atomically promote them to `conversation` mode and add a
message/asset relationship. A failed or cancelled send creates no conversation turn and leaves
the upload temporary. The conversation API returns safe metadata under each message's `assets`,
not inline bytes; the existing UI renders those previews on reload. Only IDs/preferences, never
image bytes or asset objects, are saved in browser web storage. Attachment count is per message;
there is no total-disk quota yet.

An attachment turn uses native vision when the configured **local** Transformers generator is
enabled. Mock/explicitly unsupported generators retain the non-analysis acknowledgment; a native
load failure is an error, never a silent placeholder fallback. Remote-provider image requests require
explicit per-message consent; without it they fail before decryption or HTTP. Failed attachment
requests can be retried while their uploads remain available; expired attachments must be uploaded
again. Normal text-only chat, tools, memory and remote text inference keep their existing contracts.

#### Native local vision V1

The same pinned `OBLITERATUS/Qwen3.8-27B-OBLITERATED` checkpoint at
`a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8` supplies text and vision. Its
[config](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8/config.json)
declares `Qwen3_5ForConditionalGeneration`, `language_model_only=false`, and a vision tower;
its weight index contains `model.visual.*` tensors. The production loader uses that built-in
conditional-generation class, BF16, `device_map="auto"`, and `trust_remote_code=False`.
One lazily cached model, one `AutoProcessor` / `Qwen3VLProcessor`, and the processor's shared
tokenizer serve both paths under the same generation lock. No second causal model is loaded.
Incomplete/mismatched weights or incompatible processor metadata fail closed.

The pinned processor and tokenizer templates contain text-only content rendering. A narrow
runtime adapter adds the image-content rendering used by Qwen's native template, with special
tokens checked against the pinned config/tokenizer. The processor expands image placeholders
and creates pixel tensors. The original text-only chat template, runtime prompt, thinking setting,
and frozen generation config are unchanged. Tests verify identical text-template rendering.

Attach one photo, screenshot, chart/UI image, or visible-text image using the existing paperclip.
The native path supports visual question answering and model-based text reading, not guaranteed
forensic OCR. It does **not** support multiple-image comparison, image editing,
image generation, video input, or a separate OCR engine.

`AssetService.processing_bytes()` authenticates/decrypts the current asset into normalized PNG
bytes. A request-local `BytesIO`/Pillow RGB image goes to the processor; no asset ID, filesystem
path, filename, database handle, AEK, base64 conversation content, or plaintext temp file enters
that request. Pillow rejects excessive dimensions/aspect ratios (over 200:1), and resizes large
images proportionally in RAM. Qwen processing uses 65,536 minimum / **1,048,576 maximum pixels**,
with dimensions aligned to its 16-pixel patches and 2x spatial merge: at most **1,024 merged image
tokens**, excluding wrapper tokens. The resulting image grid is checked before model execution.
The encrypted canonical original and preview quality never change. Decoded images are closed on
success, failure, and cancellation; Python/Torch may retain allocator memory, so this is not a
claim of secure zeroization. Accelerate execution hooks determine placement for text and pixels.

Only an image explicitly attached to the **current request** is read. The existing 20-message /
20,000-character text-history compiler also applies to vision. Historical images are never loaded
or automatically resent on followup text turns. Image turns do not retrieve or mutate structured
memory in V1; make explicit memory commands in a separate text-only turn. Captions and extracted
text are not automatically captured as memory.

Both chat endpoints retain the normal response/metadata contract and actual configured model ID.
Unconstrained vision uses the existing incremental streamer and prefix-commit tool filtering.
The same bounded calculator loop remains available, with no new tool capabilities. Mechanically
constrained vision remains buffered through bounded validation/retries, reusing the same
request-local image. Exhausted validation emits no candidate/final/done success and persists
nothing. Model work remains outside SQL transactions; only successful final answers and the
user/asset relationship persist atomically. Cancellation closes the stream, joins the worker and
releases the image; an in-flight CUDA operation cannot be interrupted instantly.

Built-in Qwen support exists in the existing `transformers>=5.2,<6` range. The GPU environment must
also provide matching CUDA PyTorch **and TorchVision**: `AutoProcessor` initializes Qwen's bundled
video processor even for image-only use. As before, Torch packages are environment-managed, not
installed by AmitAI's runtime extra. No `qwen-vl-utils` is needed.
Jinja2 is a small dev dependency for CPU-only template regression tests.

Real-weight validation is separate from the CPU/mock suite. On a suitable **A100 80GB** environment
with matching CUDA PyTorch/TorchVision already installed, from the repository root the minimal
opt-in synthetic smoke commands are:

```bash
pip install -e '.[runtime]'
python -m scripts.vision_smoke
```

This uses the production pinned BF16 loader and an in-memory white image containing
`AEVON VISION 42`, a red square, and a blue circle. It accepts no image path, calls no application
API, and prints synthetic answers/deltas, load/generation timing, and allocated/reserved/peak
CUDA VRAM. One model load exercises non-streaming vision, streaming vision (including exact
chunk reconstruction and one terminal result), and a text-only followup. This explicit
developer-only script prints full generation tracebacks; it accepts no private inputs.
It deliberately loads real weights **only when explicitly run**; normal tests use fakes.
The reported A100 non-streaming control succeeded; streaming still requires the synthetic
traceback/revalidation below. CPU/mock tests do not establish real-weight streaming success.
Keep one Uvicorn worker and no `--reload`
for any deployed model server. Running the synthetic test on RunPod is not a claim of
provider-blind inference.

#### Explicit remote single-image vision

Upload and remote disclosure are separate actions. With a remote provider configured, the composer
shows **Allow this image to be sent to the remote GPU for this message**, unchecked by default.
The local authenticated `GET /api/capabilities` returns only `{vision: {enabled, scope}}`; no
provider URL, token or host is exposed. The browser calls only the local `/api` proxy.
Local native vision needs no remote consent. Unknown capability state blocks image submission.

Both `/api/chat` and `/api/chat/stream` accept strict boolean `allow_remote_vision` (default false)
alongside the current `asset_ids: ["..."]`. The checkbox resets on submission, attachment changes,
and retries; a failed request requires fresh consent. Consent is not stored in preferences,
conversation metadata, browser storage, or the asset: `processing_scope` remains `local_only`.
A typed request-local `RemoteVisionGrant` binds exactly one asset, explicit consent and the vision
purpose. The local service checks it before decryption and revokes it on success/failure/cancel.
Bounded mechanical repairs and calculator followups reuse it only inside that request; each
provider invocation sends the same one image. There is no automatic fallback or network retry.

The existing remote client sends to authenticated `POST /v1/vision` or `/v1/vision/stream`:

- `multipart/form-data`, exactly two parts, with **no filename parameters**;
- `metadata` (`application/json`): version `1`, fresh transport request ID, compiled `messages`
  and bounded `generation_config` (at most 512 output tokens);
- `image` (`image/png`): the normalized, metadata-free canonical PNG, at most **20 MiB**.

Metadata is capped at **1 MiB**, 40 messages and 200,000 characters per message; the actual complete
body is capped at **21 MiB + 4 KiB**, regardless of Content-Length. Extra/duplicate/missing parts,
unknown headers or fields, filename parameters, encodings and non-PNG media fail closed. The server
authenticates before reading the body, verifies PNG CRC/container, full decode and single frame,
enforces **8192 pixels per side / 24 million pixels / 200:1 aspect ratio**, strips metadata again,
and applies the existing native processor's 1,048,576-pixel limit. All media stays in request RAM:
no UploadFile spool, temp image, image cache, asset directory or application database is opened.
The same lazy local inference provider, model instance and serialized generation lock handle text
and vision on the GPU server.
`POST /v1/vision` calls genuine `generate_vision()` and returns JSON; it never reconstructs a
stream. `/v1/vision/stream` calls `stream_vision()` and returns SSE. Neither falls back to the
other on failure. A disconnected synchronous request discards its result, but an in-flight
model call may continue; its image stays alive until that call returns. Streaming cancellation
is cooperative between model steps and produces no successful terminal result after cancellation.

Only the canonical image, minimized/projected text and generation/protocol metadata cross. Asset
UUIDs, original filenames, paths, ciphertext, AEK, hashes, local timestamps, conversation IDs,
database details and local-only memory records do not. The existing semantic credential guard
scans the owned JSON metadata **before** multipart construction; it does not regex-scan PNG bytes.
Text history uses the same 20-message / 20,000-character policy and remote memory-command
projection. Image turns still perform **no memory retrieval or mutation**. Historical image bytes
are never automatically resent. Ordinary conversation text deliberately included in the compiled
context remains visible remotely; this is not automatic PII/path redaction.

The exact-origin/DNS policy, verified TLS >=1.2, hostname/CA checks, header-only bearer auth,
`trust_env=False`, and disabled redirects are shared with text inference. Every non-200 response,
including 300/301/302/303/307/308, fails without resending the image. DNS checks remain preflight
checks, not connection-level DNS pinning. Proxy buffering must be disabled for genuine streaming.

Remote SSE uses the existing `delta`, `final`, `error` format; local chat still emits
`start`, `text`, `final`, `done` or terminal `error`. Constraints remain fully buffered until
validated; invalid candidates never reach the browser or persistence. Disconnects signal the
remote model's cooperative cancellation and close streams; periodic SSE heartbeats also let the
local client observe cancellation during a quiet model load. Connection establishment, an in-flight
read/CUDA operation or model loading cannot always be interrupted instantly. No partial turn is
committed, and no SQL transaction spans inference. Images are released when the worker stops;
Python/GPU allocators are not securely zeroized.

**Disclosure warning:** a screenshot/photo can visibly contain private data or credentials.
Pixels are not automatically scrubbed. Explicit remote consent is the security gate. Normalized
plaintext image bytes and included text are visible to the inference provider during execution.
The service is designed not to persist them, but ordinary RunPod is provider-controlled development
compute, not provider-blind, zero-knowledge or end-to-end confidential inference.

Minimal synthetic A100 smoke (not run by Codex): use the existing `/workspace/AmitAI` checkout,
the existing `/workspace/hf` cache, matching installed CUDA PyTorch/TorchVision/runtime dependencies,
and an already-set valid `AMITAI_INFERENCE_AUTH_TOKEN`. No network volume or weights redownload is
needed. Offline mode deliberately fails if the pinned checkpoint is missing. First, on the GPU:

```bash
cd /workspace/AmitAI
git pull --ff-only
HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1 python -m scripts.vision_smoke
```

The smoke reports each native stage, runtime versions, tensor shapes, processed dimensions,
both vision latencies/output tokens, reconstruction, and allocated/reserved/max-allocated/
**max-reserved** VRAM. On failure, stop and collect the synthetic traceback; do not repeatedly
reload the 27B checkpoint or change resolution/quantization to mask the failure. The production
server continues to log exception class only and returns `Inference failed`. The earlier
class-only `ValueError` does **not** establish its original failing line or a streaming-only cause.
Processor options now use `processor_kwargs` where that API is available (including 5.16.1),
with identical pixel limits and option values; this addresses the non-fatal warning, not proof
of the original error's cause. The validated text template and non-streaming generation algorithm
are unchanged.

If the native smoke passes, start the server (one worker, no `--reload`):

```bash
HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1 python -m uvicorn runtime.inference_app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Lazy-load/524 warning:** the standalone native smoke runs in a different process; it does NOT
warm the later Uvicorn server. To make the server's own model resident before going through
Cloudflare/RunPod, run this authenticated synthetic request from another GPU-side terminal:

```bash
AMITAI_REMOTE_INFERENCE_URL=http://127.0.0.1:8000 \
AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS=http://127.0.0.1:8000 \
AMITAI_REMOTE_INFERENCE_TOKEN="$AMITAI_INFERENCE_AUTH_TOKEN" \
python -m scripts.remote_vision_smoke
```

This uses the existing authenticated endpoints over loopback, loads no second model, bypasses
the edge timeout, and adds no unauthenticated warmup route or timeout change. Once it passes,
on the **local machine** with the existing remote environment configured, run:

```bash
python -m scripts.remote_vision_smoke
```

It reports `REMOTE VISION NONSTREAM: PASS/FAIL`, `REMOTE VISION STREAM: PASS/FAIL`, and
`REMOTE TEXT: PASS/FAIL`, stopping at the first failure. It prints no token, URL, request/response
body or private exception details. Both smoke scripts use only the generated text/shapes image
in RAM and accept no image path. Do not test private screenshots.

Explicit API clients may upload with `persistence_mode=conversation` plus an existing
`conversation_id`; this retains the asset immediately. Temporary uploads cannot supply a
conversation ID. Assets cannot be silently reassigned between conversations. Missing, expired,
deleted, duplicate or excessive references fail cleanly and do not persist a turn.

**Encrypted local image storage:** metadata/linkage live in the existing SQLCipher database.
Normalized images are AES-256-GCM authenticated ciphertext, stored separately as
`%LOCALAPPDATA%/AmitAI/assets/<database-namespace>/<uuid>.asset` on Windows,
`~/Library/Application Support/AmitAI/assets/...` on macOS, or
`${XDG_DATA_HOME:-~/.local/share}/amitai/assets/...` on Linux. The namespace is a hash of the
configured database's absolute location. This app-controlled directory and files are owner-only
(Windows requires the existing `secure-runtime` extra). Links/reparse points are rejected.

A dedicated, independent random 32-byte asset encryption key (AEK) is stored only in the private
`asset_encryption_state` singleton inside SQLCipher. It is not derived from the passphrase,
database key, or API tokens, and has no API/schema/export representation. Passphrase changes
rewrap the database key; SQLCipher key rotation preserves the AEK record. Neither operation
re-encrypts image files. One locked, mutable key handle is loaded at startup and zeroized at
shutdown or startup failure, using the same secure-memory implementation as the database key.
DBAPI, Python and cryptographic libraries can still create transient immutable copies.
Explicit plaintext SQLite test mode stores the same key row **unencrypted**; that mode does not
provide strong asset-at-rest protection and is not the canonical secure runtime.

V1 framing is eight-byte magic `AMITASST`, version byte `1`, a fresh random 12-byte nonce, and
ciphertext with its full 16-byte GCM tag. AAD is the byte concatenation
`b"amitai:image-asset\x00" + b"AMITASST\x01" + UUID(asset_id).bytes`; it contains no path or
database namespace. Asset swapping, wrong keys, altered/truncated bytes, unknown versions and
oversized files fail closed with generic errors. Maximum plaintext remains 20 MiB; maximum
ciphertext is 20 MiB + 37 bytes (20,971,557 bytes). The existing SHA-256 still describes the
normalized plaintext and is checked after authenticated decryption.

Encryption happens in memory before any write. Only ciphertext enters same-directory,
owner-only temporary files; they are flushed/fsynced then atomically replaced. Directory fsync
is used on supported POSIX filesystems; Windows uses file `FlushFileBuffers` but has no directory
fsync equivalent here. No plaintext image temp, preview cache or migration backup is created.
Authenticated previews decrypt only in backend memory and continue returning normalized
`image/png` with no-store, nosniff and same-origin headers. Browser storage and conversation JSON
exports still contain no image bytes, AEK or ciphertext paths. Remote image processing requires
the separate one-request consent/grant path above; encrypted storage scope remains local-only.

**Legacy migration:** stop every other backend process and use one worker. After DB unlock and
schema creation, the AEK is durably committed **before** migration. Startup verifies each legacy
PNG's canonical container, dimensions, stored plaintext size and SHA-256. It writes/fsyncs a
ciphertext temp, replaces the original `.png` pathname with ciphertext, then renames it to
`.asset`. Startup does not serve requests until migration succeeds. A crash before replacement
leaves the original PNG and possibly a ciphertext temp; restart uses the same committed key and
retries. A crash after replacement leaves encrypted bytes under `.png`; restart authenticates
and renames them without re-encryption. Existing `.asset` files are authenticated and unchanged.
Unknown/mismatching files, conflicting representations, bad permissions and missing/corrupt key
state fail startup without silently deleting images or generating a replacement key. Missing
files retain their metadata and remain unavailable; migration never recreates image content.
Complete generated orphans are also authenticated/migrated (canonical-container checks when no
metadata survives), then follow normal expiry cleanup. Abandoned generated upload/migration
temps are removed during offline startup; unrelated names/directories are never traversed.

**Backup/recovery:** the [encrypted local backup CLI](#encrypted-local-backup-and-recovery)
packages the encrypted database (including its AEK), its referenced encrypted assets, and the
existing wrapped database key. Database-only copies do **not** preserve image bytes. Restore
derives the asset namespace from the new DB path; asset authentication does not bind the old
filesystem location. There is no separate AEK backup, automatic backup, restore UI or asset-key
rotation.

`DELETE /api/assets/{id}` hard-deletes metadata, detaches it from history, and removes ciphertext
without decrypting it;
ordinary message text is untouched. Conversation deletion also removes its assets. Temporary
uploads expire after 24 hours and become unreadable even before cleanup. Startup and hourly
cleanup remove expired records/files and unreferenced generated files older than 24 hours,
including interrupted-write leftovers. Metadata is committed before physical deletion, so a
disk-removal failure makes the image inaccessible immediately and cleanup retries later.
Orphan grace prevents cleanup racing a fresh upload; stop extra backend processes before
maintenance. Use one backend worker as already recommended. Cleanup runs only while the backend
is running and cannot guarantee physical erasure from SSDs, snapshots or backups.
Encryption at rest does not protect an unlocked process from same-user malware or compromised
application code, and does not make any future remote inference provider blind to supplied
plaintext. OS/full-disk encryption remains complementary protection.

`runtime/media.py` defines a request-local vision DTO containing compiled text and a borrowed
in-memory image. Future edit/generation DTOs still only reference uploaded IDs.
`AssetService.processing_bytes()` is the normalized-image preparation seam; remote access fails
closed without a matching live `RemoteVisionGrant`. Providers receive only the current explicitly
referenced image after the separate consent check. Image editing/generation remain unimplemented.

#### Structured memory V1

Memory V1 stores explicit, structured memories for the server-owned local principal
`local-default`. Automatic capture is off: ordinary conversation never mutates durable memory.
The chat parser accepts one explicit personal fact (case-insensitive):

```text
Remember that my favourite color is black.
Remember my dog's name is Bruno.
Forget my favourite color.
Actually, update my favourite color to blue.
```

Natural forms are `Remember [that] my <field> is <value>`, `Update my <field> to <value>`,
and `Forget my <field>`. `Actually,` may prefix update/forget. Fields contain up to eight
English words, with possessives allowed. They become stable underscore keys (`dog_name`);
`favorite`/`favourite` and `color`/`colour` normalize to `favourite_color`. Fields beginning
with `favourite` use category `preference`; other personal fields use `profile`. Values
can contain Unicode. Natural commands are single-line, single-fact requests; ambiguous
compound/conditional forms return a local clarification without a write. This is a bounded
grammar, not general language understanding. Use the Memory editor or the existing structured
grammar for other categories, exact identifiers, punctuation-heavy values, or unsupported wording.
Existing structured forms remain supported:

```text
Remember <category> <key>: <value>
Update <category> <key>: <value>
Forget <category> <key>
Actually, update <category> <key>: <value>
Actually, forget <category> <key>
```

Categories are `preference`, `profile`, `project`, `workflow`, and `instruction`; keys are stable
dot/underscore/hyphen identifiers such as `ui.theme`. `Actually` alone has no memory meaning, and
ambiguous or malformed corrections/deletes do not mutate memory. Values are normalized, bounded,
and rejected when they contain recognized credentials such as passwords, API keys, tokens, JWTs,
or private keys.

The explicit API uses the same validation and never accepts a client-supplied owner:

```bash
curl -X POST http://127.0.0.1:3000/api/memory \
  -H 'Origin: http://127.0.0.1:3000' \
  -H 'Content-Type: application/json' \
  --data '{"category":"preference","key":"ui.theme","value":"dark"}'

curl 'http://127.0.0.1:3000/api/memory?category=preference'
curl -X POST http://127.0.0.1:3000/api/memory/search \
  -H 'Origin: http://127.0.0.1:3000' \
  -H 'Content-Type: application/json' \
  --data '{"query":"Which UI theme do I prefer?"}'
curl -X PATCH http://127.0.0.1:3000/api/memory/MEMORY_ID \
  -H 'Origin: http://127.0.0.1:3000' \
  -H 'Content-Type: application/json' \
  --data '{"value":"light"}'
curl -X DELETE http://127.0.0.1:3000/api/memory/MEMORY_ID \
  -H 'Origin: http://127.0.0.1:3000'
curl 'http://127.0.0.1:3000/api/memory?status=deleted'
```

Replace `MEMORY_ID` with the UUID returned by the API. Search accepts a strict string `query`,
trimmed at its edges, with 1–2,000 characters. Search text travels only in a POST JSON body;
`GET /api/memory?query=...` is rejected, not a compatibility search endpoint. Search still uses
the existing relevance rules and 8-record / 4,000-character limits, without modifying memory.
GET lists accept only `category` and `status` filters. The Memory UI keeps search precedence over
filters and requires no new user workflow.

Each logical `(owner, category, key)` has a stable slot ID and monotonically increasing revision.
Updates mark the previous revision stale. Forget creates a tombstone and redacts values from all
historical revisions; deleted and stale revisions are never retrieved. Explicitly remembering a
forgotten key reactivates the same slot with a new revision while old values remain redacted.
Optimistic revision/status checks reject concurrent lost updates.

Every memory has a separate `sensitivity`: `local_only` (default) or `remote_allowed`.
New API/chat memories and existing pre-policy records default to `local_only`. Startup upgrades
the existing memory table transactionally before serving requests, including encrypted databases;
it does not reset data, export plaintext, or change keys. Reactivating a forgotten memory defaults
to local-only again. A value-only update preserves the active memory's current sensitivity.
The local Memory API exposes sensitivity, but chat metadata stays reference-only and unchanged.
POST accepts optional sensitivity; PATCH accepts value, sensitivity, or both, rejects empty/null
updates and unknown policies, and uses the same atomic revision/conflict checks for policy changes.
To explicitly allow a memory for remote inference:

```bash
curl -X PATCH http://127.0.0.1:3000/api/memory/MEMORY_ID \
  -H 'Origin: http://127.0.0.1:3000' \
  -H 'Content-Type: application/json' \
  --data '{"sensitivity":"remote_allowed"}'
```

Retrieval is deterministic and conservative: key/category matches dominate value overlap, with at
most eight active items and about 4,000 serialized characters. Selected items are injected as a
runtime-generated `MEMORY_CONTEXT_V1` system message with deterministic JSON. That block is
remembered user context only—even category `instruction` remains below runtime/system rules, tool
protocol, mechanical validation, and the current request. User-authored lookalike markup remains
ordinary user text. Internal memory context and tool-loop messages are never persisted.

The value-bearing retrieved record exists only in the request-local model context and the memory
API. Chat responses and `message_metadata.memory_refs_json` persist reference-only memory metadata:
ID, operation, category, key, status, provenance, and update timestamp, but never the raw memory
value. A valid explicit forget is staged before retrieval, so its target is neither injected into
that request's model context nor reported as retrieved. The delete transaction also strips a
matching raw `value` from legacy structured metadata rows created by earlier Memory V1 builds.

Chat memory mutation is staged during the short preparation read and applied with the user message
and deterministic local acknowledgment in the existing final conversation transaction. These commands
do not invoke a model, tools, or mechanical repair. Even streaming emits acknowledgment text only after
commit; a failed mutation/commit cannot claim success. Cancellation before the write leaves no rows;
disconnecting after a committed acknowledgment does not undo it. Invalid commands and missing targets
receive a deterministic no-change response. Image turns still cannot read/write memory and explicit
memory commands with an image receive a text-only request clarification locally.

New memories remain local-only. Their acknowledgment explains that remote Aevon cannot recall them.
To opt in, open **Memory**, edit the chosen memory, select **Inference access → Remote allowed**,
then **Save changes**. This uses the existing revision-checked API; no mode switch promotes memories.
Changing access back to Local only excludes it from future memory projections but cannot retract
information already sent to a provider. Forget redacts memory revisions, not ordinary chat history.
Normal model generation and tool execution remain outside SQL transactions.

#### Runtime tools and calculator

The real runtime has a reusable tool registry separate from the chat service. A tool publishes a
name, description and argument schema, validates its own arguments, and executes without access to
the database layer. The first registered tool is `calculator`; additional tools can implement the
same runtime protocol without adding tool-specific branches to `backend/chat_service.py`.

The pinned model tokenizer has only plain `system`, `user` and `assistant` chat-template roles, so
the runtime does not assume OpenAI-style native tool calling. Instead, the model may return exactly
one whole-response envelope, with harmless surrounding whitespace allowed:

```text
<tool_call>{"name":"calculator","arguments":{"expression":"15% of 200"}}</tool_call>
```

No prose, Markdown or trailing content is allowed around an envelope. The runtime parses the JSON,
validates the exact schema and tool name, executes through the allowlisted registry, and gives the
model a request-local trusted `system` message:

```text
<tool_result>{"arguments":{"expression":"15% of 200"},"attempt":1,"name":"calculator","result":"30","success":true}</tool_result>
```

These internal assistant/system messages are never sent to the frontend or stored as conversation
messages. A lookalike envelope in persisted user history remains ordinary user text and is never
treated as trusted. Only the original user message and final natural-language assistant response
are persisted. Model generation and tool execution remain outside SQL transactions.

The loop permits at most three attempted tool turns. Every reserved tool candidate consumes one
attempt before parsing: successful calls, malformed JSON/envelopes, unknown tools, invalid
arguments, and execution failures all count. A sanitized error result lets the model recover within
the remaining attempts; a fourth tool candidate hard-fails generation instead of executing.

Calculator expressions support decimal numbers, unary signs, `+`, `-`, `*`, `/`, right-associative
`**`, parentheses, postfix percentages and `of`. Postfix `15%` is `0.15`; `15% of 200` is `30`.
`of` has the same precedence as multiplication and division and those operators evaluate
left-to-right. Exponents must be integers with magnitude at most 100. Expressions are limited to
256 characters, 128 tokens, 16 nested parenthesis levels, 64 digits per literal, and absolute
intermediate/final magnitude `1e100`.

The calculator uses a dedicated lexer, recursive-descent parser and `Decimal` arithmetic. It never
uses `eval`, imports, attribute access, function calls, assignment, comprehensions, filesystem,
shell, network or arbitrary Python execution. Unsupported syntax and division by zero return
sanitized tool failures.

Successful final metadata records validated activity, for example:

```json
{
  "tools": [
    {
      "attempt": 1,
      "name": "calculator",
      "arguments": {"expression": "15% of 200"},
      "success": true,
      "result": "30"
    }
  ]
}
```

Failed attempts may appear with `success: false` and a safe error code/message; raw malformed or
unsafe payloads are not retained. Tool invocation follows V1 prefix-commit semantics: after
harmless leading whitespace, the response must begin with `<tool_call` before any normal assistant
text is committed. The prefix gate holds only decoded text that could still become `<tool_call` or
`<tool_result>`—at most 11 significant characters, plus leading whitespace. Prefix tool protocol
is fully buffered and never emitted; once the prefix diverges, held text is released and ordinary
generation streams incrementally.

After normal-text commit, a later `<tool_call>...</tool_call>` or
`<tool_result>...</tool_result>` is a protocol violation, never a tool invocation. A short
character lookahead suppresses that late envelope without executing it or retaining its raw body;
already-streamed prose remains public, and ordinary text after a complete closing tag continues to
stream. The final response is reconstructed from exactly the sanitized visible deltas, so the SSE
output and persisted assistant message match. For mechanically constrained requests, tool use
finishes first and validation/retries apply only to the final user-visible answer, which remains
fully buffered under the existing constraint policy.

### Privacy-first local control plane and inference providers

The Aevon frontend always talks to the local AmitAI API. That local control plane owns
`amitai.db`, conversations, messages, structured memory, preferences, tools, validator retries,
and final persistence. Model execution sits behind a stateless `InferenceProvider` boundary:

```text
Aevon browser
  -> local Next.js server
    -> authenticated local runtime.app API + local SQLite state
      -> mock, local Transformers, or remote inference provider
```

The browser uses only relative `/api/*` URLs. A Next.js Route Handler rereads the owner-only
runtime token file on the server for each request and adds the backend bearer header while
proxying; the token is never returned to browser JavaScript, stored in web storage, placed in a
cookie, or put in a URL. State-changing proxy requests **require** an `Origin` header that exactly
matches the loopback Aevon origin, including scheme, hostname, and port. Missing, `null`, malformed,
or different origins are rejected before token access or FastAPI. `localhost` and `127.0.0.1` are
not interchangeable origins. Browser code currently persists only harmless UI preferences and the selected
conversation ID—never conversations, messages, memory values, or credentials.

#### Browser proxy boundary

The browser-facing proxy is deny-by-default. These are its complete method/path contracts:

| Path | Methods |
| --- | --- |
| `/api/health` | GET |
| `/api/conversations` | GET, POST |
| `/api/conversations/{uuid}` | GET, PATCH, DELETE |
| `/api/chat`, `/api/chat/stream` | POST |
| `/api/memory` | GET, POST |
| `/api/memory/search` | POST |
| `/api/memory/{uuid}` | PATCH, DELETE |
| `/api/assets` | POST (bounded image multipart only) |
| `/api/assets/{uuid}` | GET, DELETE |
| `/api/assets/{uuid}/content` | GET |

Unknown paths/methods (including HEAD/OPTIONS), malformed/non-UUID IDs, encoded separators,
dot segments, controls, and extra segments are rejected without forwarding. Adding a backend
endpoint does not expose it automatically through Next. The only allowed URL parameters are
one `status=active|deleted` and/or one supported `category` on GET `/api/memory`; all other query
keys and duplicate parameters are rejected. Parameters are reconstructed, never blindly copied.

Every browser request must address a loopback host (`localhost`, `127.0.0.1`, or `[::1]`), even
when `AMITAI_ALLOW_LAN=1`. Explicit `Sec-Fetch-Site: cross-site` is rejected for reads and writes.
Next's loopback URL alias normalization is disabled, and a differing HTTP Host is rejected too.
Use the exact loopback address printed by the launcher, not a hostname alias of that listener.
Missing fetch metadata is tolerated, but never substitutes for the mandatory mutation Origin.
Origin is a browser CSRF boundary, not authentication against arbitrary local processes that can
forge headers. Non-browser clients should normally call the authenticated FastAPI endpoint;
the curl examples through Next explicitly supply the same Origin as the target URL.

Validation order is browser origin/host/fetch metadata, route/method, query/body, runtime token
read, backend configuration, then fetch. JSON routes require `application/json` (UTF-8 charset
permitted) and a top-level object. POST conversation creation alone may omit its body. Actual
JSON request bytes are capped at 256 KiB regardless of `Content-Length`; over-limit requests get 413,
unsupported media types 415, invalid JSON/query/body 400, disallowed routes 404, and denied
browser origins 403. GET/DELETE bodies are rejected; HEAD is unsupported. Domain validation
remains in FastAPI. Proxy errors contain no raw content, token, path, or exception details.

Only Accept/Content-Type and the server-generated bearer header go upstream: browser
Authorization, Cookie, Proxy-Authorization, Host, Forwarded, and X-Forwarded-* are not forwarded.
Only Content-Type/X-Accel-Buffering may return from upstream; upstream cookies, auth/debug/version
headers and cache policy are discarded. All API successes/errors use `Cache-Control: no-store`.
SSE uses `no-store, no-transform` and `X-Accel-Buffering: no`. Only bounded request JSON is buffered;
response streams remain incremental and client abort signals propagate upstream. The sole multipart
exception is `POST /api/assets`, capped at 20 MiB + 16 KiB of envelope overhead. Both proxy and backend
accept one image and only `persistence_mode`/`conversation_id` metadata fields, rejecting unknown or
duplicate fields. Image decoding and canonicalization remain backend responsibilities.

Next sets global browser headers: `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
`X-Frame-Options: DENY`, `Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Resource-Policy: same-origin`, `X-DNS-Prefetch-Control: off`,
`X-Permitted-Cross-Domain-Policies: none`, and
`Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=()`.
Its version header is disabled. HSTS is deliberately absent for local HTTP operation.

The production Content-Security-Policy is:

```text
default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; connect-src 'self'; img-src 'self' data: blob:; font-src 'self' data:; media-src 'self'; worker-src 'self' blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'
```

No external analytics/CDN/font or inference origins are allowed. Inline scripts are needed for
current Next hydration, and inline styles for the existing UI; these compatibility exceptions
mean CSP is not a complete inline-injection defense. There is no nonce infrastructure in this
version. `unsafe-eval` is allowed only when `NODE_ENV !== "production"` for development tooling,
never in production; network connections remain self-only.

These controls do not stop same-user malware or malicious browser extensions from reading
visible content. Localhost HTTP is for local-only operation; supported LAN/TLS is future work,
not enabled by this proxy. None of these browser controls make remote inference provider-blind.

Private `/api/chat*`, `/api/conversations*`, `/api/memory*`, and `/api/assets*` routes reject missing or incorrect
bearer authentication. `/api/health` remains a minimal unauthenticated readiness route. FastAPI
docs, ReDoc, and OpenAPI are disabled by default; set `AMITAI_ENABLE_DEV_DOCS=1` only during
intentional local development. FastAPI has no wildcard CORS policy because the normal browser
path is same-origin browser-to-Next, followed by server-to-server Next-to-FastAPI traffic. Direct
browser cross-origin FastAPI access is not supported.

The canonical launcher enforces the network default it actually controls:

```bash
export AMITAI_HOST=127.0.0.1
python -m runtime.serve
```

Do not use `0.0.0.0` for normal single-user operation. A deliberate LAN deployment requires
`AMITAI_ALLOW_LAN=1` and still requires authentication. The FastAPI application cannot discover
the socket address supplied by someone who bypasses this launcher with a raw Uvicorn command, so
it does not pretend to enforce that external bind; local firewall configuration remains relevant.
The canonical launcher also disables raw HTTP access logging. Memory search now uses a JSON
body rather than a URL; request/response content is not logged by the proxy. Starting either `uvicorn runtime.app:app` or
`uvicorn backend.app:app` directly fails closed; production secrets are injected only by the
interactive `python -m runtime.serve` path.

#### Inference context minimization

AmitAI keeps complete conversation history in the local database, but the model does not
automatically receive that unlimited history. Before either a local or remote provider runs, one
deterministic compiler selects at most the 20 newest complete prior messages and at most 20,000
characters of prior-message content, after applying the provider-safe projection. The current
user turn remains outside that budget; ordinary text is preserved intact, while remote memory
commands use the generic projection described below. Local raw requests remain intact;
dropped history is neither truncated nor summarized, and local persisted history is unchanged.

Structured memory retrieval remains relevance-based and capped at 8 records and 4,000 characters.
Rich memory records stay in the local control plane for persistence and auditing, while the model
receives only each permitted selected memory's `category`, `key`, and `value`. Memory IDs, source
conversation/message IDs, timestamps, revision details, status, and persistence operations are
not included in the retrieved-memory prompt. Sensitivity is not model-visible either.

#### Remote inference disclosure policy

Providers explicitly declare a typed local/remote execution scope; absent or invalid scope fails
closed, and provider names never determine trust. Local Transformers keeps relevant local-only
and remote-allowed memory. Remote inference receives only `remote_allowed` records from the
existing relevance-selected set, with no replacement retrieval after filtering. An all-local set
produces no remote memory block, key list, count, or placeholder.

Explicit current remember/update/forget commands and their acknowledgments are processed locally,
with zero inference requests, including invalid commands and previously opted-in targets.
Historical memory commands
and their immediately paired assistant acknowledgments are projected to generic text before
history budgeting; stored history and the conversation API remain untouched.

Constraint validation still evaluates the original local request. Mechanical retry prompts use
the selected provider-safe current request, so retries cannot restore raw memory commands. The
same compilation applies to sync/stream generation and tool-loop base context; bounded current
tool messages are appended locally, never reloading dropped history.

Immediately before **every** remote HTTP request, including retries and tool follow-ups, the
client checks an owned snapshot of the decoded payload with the shared memory credential
heuristic, then serializes that checked snapshot once and sends those exact bytes. Checks inspect
actual strings, nested mappings, and embedded JSON (including memory `key`/`value` records), not
JSON serialization punctuation. They cover labeled passwords/passcodes, API keys,
access/refresh/auth tokens, client secrets, PEM private-key markers, JWT-shaped strings, and
explicit `Authorization: Bearer ...` values. Labels use NFKC/casefold and space/hyphen/underscore/dot
separator normalization; known families also match suffixes such as `OPENAI_API_KEY`,
`MY_PASSWORD`, and `GITHUB_ACCESS_TOKEN`. Quoted assignments are recognized; empty assignments
such as `password:` or `"api_key":""` and labels such as `password_hash_algorithm` remain safe.
The configured remote bearer token is also blocked anywhere in the body, including
JSON-escaped strings and generation configuration; its authentication header remains permitted.
No environment scanning, matched-text logging, silent redaction, or local-model fallback occurs.

Memory storage applies this same policy to the **key and value together**, regardless of memory
sensitivity. Credential-shaped creates, value edits, and sensitivity-only edits of legacy bad
records return a generic HTTP 422 without echoing submitted values. Such chat memory commands
are not applied and receive a deterministic local no-change response without inference. Legacy
records that bypassed storage validation are independently checked by the outbound guard.
Ordinary local-model conversation text is not blocked by the remote disclosure policy.

A block sends no HTTP request for that invocation and returns only
`Remote inference blocked by local privacy policy`: HTTP 422 on `/api/chat`, or a terminal SSE
`error` without successful `final`/`done` on `/api/chat/stream`. No conversation turn or staged
memory mutation is committed. An earlier safe invocation may already have run before a later
retry/tool follow-up is blocked. This is an input guard, not output DLP.

This is deliberately a heuristic, not a guarantee that every arbitrary secret is detected.
Ordinary noncredential text (including names and other PII) still goes to remote inference.
Benign discussions of API key rotation, password hashing, Authorization, and JWTs are not blocked.
It does not protect against same-user malware or browser extensions, add TLS pinning/mTLS/TEE,
or hide plaintext from a remote GPU operator. Unlocked local process memory is still plaintext.

This minimization reduces disclosure to remote inference; it does not make the provider blind or
provide zero-knowledge inference. The current request, retained recent history, selected memory
values explicitly allowed by policy, tool-loop context, and any other text intentionally compiled into the model prompt are
plaintext to the inference provider while it executes.

#### Encrypted local storage

AmitAI's persistent SQLite application state is encrypted at rest using SQLCipher. Install the
native driver and secure key-management libraries with:

```bash
pip install -e '.[encrypted-storage,secure-runtime]'
```

The dependency is pinned to `sqlcipher3==0.6.2`, which provides SQLCipher-enabled wheels for the
project's supported Python versions on Windows and Linux. Startup verifies the native codec with
`PRAGMA cipher_version` and fails closed instead of falling back to ordinary `sqlite3`.

Initialize a new installation interactively:

```bash
python -m runtime.keyctl init --database-file ./amitai.db
python -m runtime.keyctl status
```

`init` prompts twice for an unlock passphrase, generates a random 256-bit SQLCipher key, derives a
256-bit wrapping key with Argon2id (65,536 KiB, three iterations, parallelism one), and stores only
an AES-256-GCM-authenticated, versioned key envelope. The raw database key is never displayed,
embedded in the SQLAlchemy URL, or placed in `amitai.db`. The default envelope lives outside the
repository at `%LOCALAPPDATA%\AmitAI\secrets\database-key.json` on Windows,
`~/Library/Application Support/AmitAI/secrets/database-key.json` on macOS, or the XDG data
directory on Linux. Paths inside this repository or a detected Windows OneDrive root are refused.
Secret directories/files are created owner-only (0700/0600 on POSIX and a protected current-user
DACL on Windows); failure to establish or verify those protections fails closed.

Every `python -m runtime.serve` launch prompts once for the passphrase. The passphrase is not a
command-line or environment option. A wrong passphrase, modified envelope, unsupported version,
weakened/absurd KDF settings, or invalid authenticated ciphertext produces only `Unlock failed`.
Losing both the passphrase and usable key envelope can make the database unrecoverable; do not
store the passphrase beside the envelope.

The canonical launcher creates a fresh 256-bit local API bearer token after unlock and writes it
to an owner-only runtime file for the local Next.js server. The plaintext token file is protected
by its short lifetime, loopback boundary, and file permissions—not by database encryption. It is
atomically replaced before serving, removed on clean shutdown, and regenerated after every
restart, so stale tokens no longer authenticate. The optional
`AMITAI_LOCAL_API_TOKEN_FILE` setting changes only this non-secret shared path and must match in
the backend and Next server environments; an override must be absolute.

Change only the unlock passphrase without modifying the database key or database bytes:

```bash
python -m runtime.keyctl change-passphrase
```

Rotate the actual SQLCipher key only while every AmitAI backend/frontend process is stopped:

```bash
python -m runtime.keyctl rotate-db-key --database-file ./amitai.db
```

Rotation exports the old encrypted database directly to an encrypted candidate under a fresh key,
verifies SQLCipher, schema, table counts, representative rows, `user_version`, integrity and the
locked source fingerprint, then atomically replaces the database. A restricted authenticated
recovery journal stores both old and new keys only in wrapped form before any database change. If
the process stops at any phase, the next unlock safely tests which wrapped key opens the database,
finalizes the matching envelope, and removes obsolete artifacts; if neither or both keys appear
valid, recovery fails closed and retains recovery material. A busy database aborts immediately
with `Database is busy; stop AmitAI before key rotation` and leaves the source unchanged.

Changing or rotating the key prevents normal access with obsolete credentials, but atomic file
replacement/removal cannot guarantee physical erasure of old flash pages because filesystems and
SSD wear leveling may retain remnants.

For an existing database created with the former raw-key environment workflow, do not run `init`.
Use the one-time import path, paste the existing 64-hex key only into its hidden prompt, then choose
a new passphrase:

```bash
python -m runtime.keyctl import-existing --database-file ./amitai.db
```

The command proves that the supplied key opens the encrypted database, writes the wrapped
envelope atomically, and does not modify the database. It never accepts a database key on the
command line and never reads it from an environment variable. After importing, remove the legacy
`AMITAI_DB_KEY`, `AMITAI_LOCAL_API_TOKEN`, and `AMITAI_UNLOCK_PASSPHRASE` variables from the parent
shell or service configuration. AmitAI refusing or clearing a variable in its child process does
not erase a value exported by the parent shell (`unset ...` on POSIX, or
`Remove-Item Env:NAME` in PowerShell).

An existing plaintext `amitai.db` is never migrated automatically. Startup refuses it unless one
intentional launch has explicit authorization. Migration is an offline operation: stop every
AmitAI process first, and do not run another backend against `amitai.db` while migration is in
progress. `AMITAI_ENCRYPT_EXISTING_DB=1` is only for this one intentional offline migration launch:

```bash
python -m runtime.keyctl init --database-file ./amitai.db
export AMITAI_ENCRYPT_EXISTING_DB=1
python -m runtime.serve --database-file ./amitai.db
# after successful startup/migration, stop AmitAI and remove the one-shot flag
unset AMITAI_ENCRYPT_EXISTING_DB
```

Migration checkpoints any plaintext WAL, normalizes journal mode, requires exclusive access, and
exports the complete locked source snapshot into a sibling encrypted candidate. It captures the
source fingerprint while that lock is still held, verifies the candidate, and checks the source
fingerprint again immediately before atomic replacement. A changed database or reappearing
sidecar fails migration; the original plaintext database is retained and the incomplete candidate
is removed. These application-level checks support the required offline workflow but cannot
protect indefinitely against arbitrary external writers, which is why every AmitAI process must
remain stopped. A successful migration does not leave a plaintext backup or obsolete plaintext
`-wal`, `-shm`, or journal file. Normal operation preserves SQLite's existing `DELETE` journal
mode; its transient rollback journal is also written through SQLCipher.

This protects offline, copied, or stolen database files and database backups, including table
pages, indexes, messages, and memory values. It does not provide secrecy while AmitAI is running
and unlocked: application data and the key exist in process memory, and same-user malware or a
compromised process can access the live application. BitLocker, LUKS, or another full-disk
encryption layer is complementary protection. SQLCipher does not encrypt model weights and does
not hide intentionally sent generation context from a remote inference provider such as RunPod.
The canonical process disables POSIX core dumps and Linux process dumpability when supported, and
keeps the long-lived database key in a mutable locked buffer that is explicitly zeroed on
shutdown. Windows uses `VirtualLock`; system crash-dump/diagnostic policy remains an OS-level
consideration. SQLCipher APIs and Python itself may still create transient copies, so memory
locking and zeroization reduce exposure rather than providing a formally zero-copy environment.
This is encryption at rest, not a claim of perfect privacy.

Do not intentionally place the AmitAI data directory in OneDrive, Google Drive, Dropbox, iCloud,
or another automatic sync folder unless you understand that encrypted database blobs and file
metadata will be uploaded. Encryption makes copied blobs unreadable without the key; it does not
prevent cloud copies from existing.

### Encrypted local backup and recovery

With the encrypted-storage and secure-runtime dependencies above installed, stop Aevon and any
key-management operations, then explicitly choose an absolute output filename:

```powershell
python -m runtime.backup create --output "D:\Backups\aevon.amitai-backup"
```

On the replacement machine, install the same dependencies and restore **before** running
`keyctl init` or starting Aevon:

```powershell
python -m runtime.backup restore --input "D:\Backups\aevon.amitai-backup"
```

Both commands default to `./amitai.db` and the normal platform KeyStore path. Both accept
`--database-file` and `--key-file`, for example:

```powershell
python -m runtime.backup restore --input "D:\Backups\aevon.amitai-backup" --database-file "C:\AevonData\restored.db" --key-file "C:\AevonSecrets\database-key.json"
```

Start Aevon with those same destination overrides. The new database path determines the new
canonical asset namespace; no original machine path is needed. Programmatic custom
`asset_directory` overrides are not supported by this CLI. Existing asset expiry/deletion rules
still apply after restore, including expiry of temporary uploads.

The hidden prompt asks for the **existing Aevon unlock passphrase**; there is no passphrase CLI
argument/environment option, second password or recovery key. A backup keeps the envelope from
creation time: later passphrase changes or DB-key rotation do not update older backups. Use the
passphrase that applied when that backup was created. No local API or inference tokens are backed
up; normal secure startup creates a fresh local runtime token.

V1 is a stored/uncompressed ZIP with only `manifest.json`, `database.bin`,
`database-key-envelope.json`, and `assets/<uuid>.asset`. The manifest contains a format/version,
random backup ID, ciphertext sizes/SHA-256 hashes, and asset IDs/count. Source paths, uploaded
filenames, ACLs and source timestamps are not ZIP/manifest metadata. Sizes and asset IDs are
visible; SHA-256 is an integrity check, not a signed manifest or rollback-prevention mechanism.
Limits are 10,000 assets, 1 GiB database, 32 KiB envelope, 4 MiB manifest, existing per-asset size
limits, and an archive strictly smaller than 2 GiB. ZIP64/compression and unexpected members are
rejected. There is no plaintext export: the archive contains the SQLCipher snapshot, encrypted
images and passphrase-wrapped DB key, **not** a raw DB key, plaintext image, chat or memory export.

Create uses a SQLCipher transactional export under the same DB key, never a live-file copy. It
checks schema/data, `user_version`, SQLite integrity and cipher integrity. A writer lock captures
committed state, including WAL commits; later writes are not part of the snapshot. Every asset
row in that snapshot must have matching authenticated ciphertext and normalized-image metadata;
decryption for verification stays in memory. Unreferenced orphan files are ignored. Concurrent
deletion, busy storage, key changes, pending key rotation or missing/corrupt assets fail closed;
stopping Aevon is the recommended way to avoid such failures.

Create privately stages, fsyncs and reopens/verifies the archive before atomic, no-overwrite
publication. Choose a new output filename for each backup. Restore rejects existing DB/key or
nonempty asset destinations; **there is no `--force`**. It validates the entire bounded archive,
then the staged key, SQLCipher DB and every referenced asset, before installing assets, DB, and
wrapped key **last**. Final files are owner-only. Atomic publication requires hard-link-capable
storage (for example NTFS); FAT/exFAT and filesystems lacking the required private permissions
are not supported. Temporary candidates are on each destination filesystem and require spare
space for staging, including a second restored ciphertext copy.

An interrupted install leaves a tiny private `<key-file>.restore` marker with only backup ID and
phase. Normal unlock/key mutations refuse while it exists. Keep Aevon stopped and re-run the
**same restore command and destinations with `--resume`**. Resume fully revalidates the backup
and accepts existing files only when their ciphertext exactly matches; it never overwrites
changed targets. Do not manually remove the marker to bypass verification. If recovery artifacts
were changed, restore to fresh DB/key destinations instead. Ordinary failures clean generated
staging directories; process/OS crashes may leave private `.amitai-backup-*` ciphertext staging
directories, which are not trusted or reused by resume. Power-loss durability remains dependent
on the filesystem/OS; directory fsync is best-effort where unsupported.

Keep a backup physically separate from the laptop and test recovery. Losing both the backup and
the current installation loses the data; forgetting the backup's unlock passphrase makes its
ciphertext unrecoverable through this mechanism. This is explicit local backup, not cloud sync,
scheduled/automatic backup or a retention service. Same-user/OS compromise while Aevon is
unlocked remains outside this backup-at-rest guarantee.

#### Production assistant identity

Aevon is the assistant; AmitAI is its software project/platform; Qwen is the underlying model,
not the assistant's name. The prompt-only `configs/production_runtime.yaml` supplies the
production identity and concise behavioral guidance. Its model identifier is populated from
the validated model configuration, currently `OBLITERATUS/Qwen3.8-27B-OBLITERATED`. The prompt
instructs Aevon to give its name when asked, accurately identify its configured model when asked,
and avoid irrelevant identity announcements.

Normal local and remote chat startup automatically uses `load_production_runtime_config()`;
streaming, mechanical retries and vision sessions inherit that same prompt. System/tool
instructions, trusted memory, recent history and the current user retain their existing order
and privacy limits. Mock mode remains unchanged and does not invoke a model.

`AMITAI_RUNTIME_CONFIG` still selects the **model/generation settings** file, defaulting to
`configs/baseline_eval_v2_constrained.yaml`; it does not replace the production prompt. This
keeps checkpoint, revision, BF16, generation and validator settings in one validated source.
The frozen YAML is unmodified. `load_runtime_config()` retains its literal prompt, and
`python -m evaluation.run_baseline --config configs/baseline_eval_v2_constrained.yaml` still
evaluates the original "You are AmitAI" baseline. The stateless inference server forwards
caller-supplied messages rather than injecting a second identity prompt. These are prompt
instructions, not a guarantee of model compliance; no sampling or model tuning is included.

The default provider is still `mock`. Local GPU inference remains available and preserves lazy
one-model-per-process initialization plus serialized generation:

```bash
pip install -e '.[runtime,encrypted-storage,secure-runtime]'
export AMITAI_INFERENCE_PROVIDER=transformers
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml
python -m runtime.serve
```

`AMITAI_GENERATOR=mock|transformers` remains supported for existing deployments, but
`AMITAI_INFERENCE_PROVIDER` is the provider-oriented setting going forward.

#### Remote development inference

Ordinary RunPod is a **development inference provider**, not a privacy boundary. The remote host
receives the model messages needed to generate a response, including relevant retrieved memory
content when the local control plane selected it. RunPod or its infrastructure may therefore see
prompts and outputs. This architecture keeps durable/private application state local and makes the
compute provider replaceable; it does not make remote inference confidential.

Generate a fresh credential for each inference/pod session:

```bash
python -m runtime.inference_token
```

This prints one `secrets.token_urlsafe(32)` token (256 bits of random input); it does not save it.
Transfer that credential securely to the two process environments below. The placeholders are
not valid credentials. Both sides require at least **32 printable ASCII characters**, with no
whitespace or control characters; neither side silently strips or normalizes the token. This is
a deterministic minimum, not an entropy estimate. Prefer generated tokens over typed passwords.

On the GPU host, run only the authenticated inference service. It exposes `/v1/generate` and
`/v1/generate/stream`; it has no conversation, memory, preference, or user-facing chat routes and
does not initialize a database:

```bash
pip install -e '.[runtime]'
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml
export AMITAI_INFERENCE_AUTH_TOKEN='<fresh-session-token>'

uvicorn runtime.inference_app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Do **not** use `--reload` or multiple Uvicorn workers on the GPU host: either can load another
roughly 55 GB model copy. Configure the local control plane explicitly with the exact remote
origin and matching session credential; never commit credentials:

```bash
export AMITAI_INFERENCE_PROVIDER=remote
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml
export AMITAI_REMOTE_INFERENCE_URL='https://<POD-ID>-8000.proxy.runpod.net'
export AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS='https://<POD-ID>-8000.proxy.runpod.net'
export AMITAI_REMOTE_INFERENCE_TOKEN='<fresh-session-token>'

python -m runtime.serve
```

The provider is disabled unless `remote` is explicitly selected. Public endpoints require all
three `AMITAI_REMOTE_INFERENCE_*` settings above. The allowlist is comma-separated **exact
origins**: scheme, hostname, and effective port. There is no implicit public allowlist, substring
matching, or wildcard trust. When a RunPod hostname changes, update both URL and allowed origins.
Hostname case, a single trailing DNS dot, the default port, and a single trailing slash normalize
consistently. V1 accepts standard ASCII DNS names (including ASCII IDNA/punycode spellings);
raw Unicode/ambiguous host forms are rejected. Public IP literals, endpoint paths, userinfo,
queries, and fragments are rejected. Only `/v1/generate` and `/v1/generate/stream` are appended.

Plain HTTP is accepted only for loopback development origins, such as `http://localhost:9000`,
`http://127.0.0.1:9000`, or `http://[::1]:9000`; these need no public allowlist. Every resolved
address must still be loopback. Hostnames merely containing `localhost` get no exception.
LAN/private-IP endpoints and public HTTP fail closed, without upgrades, downgrades, or fallbacks.

The local control-plane token is independent of the inference-service credential. The client
uses `AMITAI_REMOTE_INFERENCE_TOKEN`; the inference service receives the **same session value**
as `AMITAI_INFERENCE_AUTH_TOKEN`. Authentication travels only in the Authorization header, never
the URL or generation body. Generate a fresh token for every pod/inference session and rotate
both sides immediately after suspected exposure. Stopping the service removes its active token
acceptor, but there is **no automatic expiry or coordinated rotation**: restarting with the old
value would accept it again. Do not reuse one permanent credential across pods. Short-lived
signed credentials are a possible future gateway feature, not implemented here.

#### Remote transport identity policy

The origin is fixed at provider construction and must be approved before any DNS query. Every
HTTP invocation—including streaming, mechanical retries, and tool follow-ups—rechecks A/AAAA
answers using a dedicated resolver. Public answers must all be globally routable; private,
loopback, link-local, multicast, unspecified, reserved, malformed, empty, and mixed public/private
results fail before HTTP transmission. DNS failure never falls back to sending anyway.

HTTPX explicitly disables redirects (`follow_redirects=False`) and environment trust
(`trust_env=False`). All 3xx responses fail: prompt bodies and Authorization are not resent to
Location targets. `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `SSL_CERT_FILE`, and `SSL_CERT_DIR`
cannot silently reroute the connection or replace its trust roots. An explicit SSLContext uses
HTTPX's normal CA bundle, mandatory certificate-chain and hostname verification, and TLS 1.2 or
newer. No insecure verification override or proxy fallback is provided.

**Residual DNS race:** the resolver check is a preflight, not connection-address pinning. HTTPX
may resolve again when connecting, leaving a preflight/connect TOCTOU interval. Exact origins,
address-class checks, redirect refusal, and TLS identity verification reduce common rebinding
and endpoint-confusion risks but do not guarantee that the connected IP is the inspected IP.

Certificate/SPKI pinning is **not enabled because AmitAI does not control a stable
certificate/public-key identity for these provider-owned proxy endpoints**. No live certificate
is scraped, learned, or trusted on first use. Normal CA/hostname verification remains mandatory.

These controls address accidental endpoint misconfiguration, malicious redirects, common DNS
rebinding toward private services, inherited proxy interception, plaintext remote HTTP,
unverified TLS identities, and weak convenience-token practices. They do **not** solve trusted-CA
compromise, compromise of the approved host/provider, same-user malware, malicious browser
extensions, the DNS race above, or provider visibility into allowed plaintext. They do not add
confidential compute, attestation, or zero-knowledge inference.

The provider sends only a request ID, model messages, and generation settings. Tool execution,
mechanical repair, memory mutation, and conversation persistence remain local, so a failed or
interrupted remote generation cannot persist a partial assistant turn. Application logs omit
prompt bodies, response bodies, memory values, and authorization tokens; operational entries may
contain request IDs, provider names, latency, token counts, HTTP status, and sanitized error types.

Set frontend server-side `AMITAI_API_ORIGIN` to this **local** API. The Next.js server reads the
same runtime token file created by `runtime.serve`; never point browser code or the proxy directly
at the GPU inference service. Future local or confidential-GPU providers can implement the same
boundary without changing the frontend or persistence layer.

Local state encryption and HTTPS transport do not make an ordinary RunPod provider blind to the
generation payload it processes.

### Frontend development

AmitAI remains the project and backend identity; Aevon is the user-facing assistant shown by the
frontend. Run the two development servers separately:

```bash
# Terminal 1, from the repository root
python -m runtime.serve

# Terminal 2: the proxy reads Terminal 1's owner-only runtime token file
cd frontend
export AMITAI_API_ORIGIN='http://127.0.0.1:8000'
npm install
npm run dev
```

The supported `npm run dev` command binds Next.js explicitly to `127.0.0.1`; after `npm run build`,
`npm run start` uses the same loopback-only default. The packaged launchers do not provide an
implicit LAN mode. Any future supported LAN launcher must require an explicit
`AMITAI_ALLOW_LAN=1` opt-in and retain authentication and origin protections. The current browser
proxy rejects non-loopback hosts regardless of that setting.

Open the local Next.js address printed in Terminal 2. Its server-only Route Handler streams
relative `/api/*` requests to the FastAPI backend at `http://127.0.0.1:8000` by default and adds
the ephemeral local bearer credential from the private runtime file. Do not copy that token into
an environment variable, `NEXT_PUBLIC_*` variable, browser storage, HTML, cookies, URLs, or logs.
Browser code does not need or receive separate CORS access. A missing or malformed runtime token
file makes the proxy fail closed with HTTP 503.

Open **Memory** from Aevon's sidebar to manage Memory V1 without leaving the product UI. The page
lists active structured memories, supports API-backed search and category filters, creates and
edits explicit entries, confirms value-redacting forget operations, and can inspect safe forgotten
tombstones. Memory values come only from `/api/memory`; chat bubbles and developer metadata keep
using the reference-only memory contract.
Active cards show **Local only** or **Remote allowed**. In **New memory** or **Edit**, the
**Inference access** selector controls disclosure; new entries default to Local only and require
an explicit selection to allow remote use. Policy-only edits send only sensitivity in the PATCH
body. Value-only edits preserve the policy, edits refetch the current search/filter, and no memory
values or sensitivity settings are saved in browser storage.

## Experimental Aevon context layouts V4

`eval/aevon_epistemic_regression_v4.jsonl` contains 10 semantic probes requiring
human review. The existing runner accepts an explicit `--context-layout A|B|C|D` for
evaluation only; production keeps its existing Layout A and prompt wording.
See [the V4 experiment](docs/design/aevon_epistemic_v4.md) for layout definitions,
exact prompt captures, measured distances, and the later A100 commands.

```bash
python -m evaluation.aevon_text_quality --mode fake --cases eval/aevon_epistemic_regression_v4.jsonl --context-layout A --output-dir outputs/aevon-epistemic-v4-fake-A
python -m evaluation.context_layout_inspection --output-dir outputs/aevon-epistemic-v4-prompts
```

Repeat the fake command with B, C, and D and a separate output directory for each.
The run manifest records the experimental layout; resume requires the same layout.
Fake passes validate the harness, not semantic correctness or a layout winner.

## Production Aevon epistemic regression V3

`eval/aevon_epistemic_regression_v3.jsonl` contains 16 focused cases for context and
memory fidelity, with direct-answer controls. See the [input-path investigation and
design](docs/design/aevon_epistemic_v3.md) for the verified template behavior, conditional
history-omission notice, unchanged memory representation, and later A100 command.
All semantic outcomes still require human review. Exercise the existing offline runner:

```bash
python -m evaluation.aevon_text_quality --mode fake --cases eval/aevon_epistemic_regression_v3.jsonl --output-dir outputs/aevon-epistemic-v3-fake
```

## Production Aevon epistemic regression V2

`eval/aevon_epistemic_regression_v2.jsonl` adds 24 targeted cases to the same text-quality
runner. It covers ambiguous references, insufficient evidence, false premises, contradictions,
invented continuity, trusted-memory fidelity, missing history, unsupported nonexistence,
hypotheses versus facts, and technical schema/stack invention. Ten cases (IDs ending in
`_control`) require a direct answer from sufficient evidence, one in each category. Review
both fabrication and unnecessary hedging: this suite measures calibration, not refusal rate.

The production profile in `configs/production_runtime.yaml` adds compact epistemic guidance.
Checkpoint, revision, BF16, generation settings, context compilation, memory, and tools are
unchanged. V1 cases and frozen historical baseline assets remain unchanged.

Run the CPU-only scripted harness from the repository root:

```bash
python -m evaluation.aevon_text_quality --mode fake --cases eval/aevon_epistemic_regression_v2.jsonl --output-dir outputs/aevon-epistemic-v2-fake
```

Later, explicitly on the provisioned A100 environment, run the production model:

```bash
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v2.jsonl --output-dir outputs/aevon-epistemic-v2-real
```

The existing `--stream`, `--ids`, and `--resume` options also apply. `run.json` names the
suite from the case filename; fingerprints and resume compatibility checks still apply.
V2 uses existing context/protocol/tool checks and mechanical constraints, without new
wording-based semantic graders. Numeric controls accept digits or spelled-out numbers.
Every case still requires the supplied human rubric: a deterministic pass can include a
factually wrong answer, and a fake pass establishes harness operation only. Review whether
claims are supported, corrections are accurate, terminology is sound, and uncertainty is
concise and useful. No scripted answer or rubric is sent to real inference.

## Production Aevon text-quality benchmark V1

`eval/aevon_text_quality_v1.jsonl` contains 54 synthetic cases across identity, normal
conversation, judgment, technical help, reasoning, continuity, memory-visible behavior,
tone, format following, calculator judgment, uncertainty/evidence, and long context.
It measures the **production Aevon path**, separately from the untouched historical
baseline below. No prompt, sampling, memory, tool, or model behavior is tuned by this suite.

From the repository root, first exercise the offline harness (no model/network/database):

```bash
python -m evaluation.aevon_text_quality --mode fake --output-dir outputs/aevon-text-v1-fake
```

Later, on a suitably provisioned machine, explicitly run the exact production checkpoint,
pinned revision, BF16 and generation settings, composed with the production Aevon profile:

```bash
python -m evaluation.aevon_text_quality --mode transformers --output-dir outputs/aevon-text-v1-real
```

Alternatively, `--mode remote` uses the existing production remote provider, requiring
`AMITAI_REMOTE_INFERENCE_URL`, `AMITAI_REMOTE_INFERENCE_TOKEN`, and
`AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS` as documented above. Nothing starts RunPod or an
inference server automatically. The operator must deploy the matching pinned model/settings;
the client cannot attest remote weights. Never put tokens in benchmark cases or artifacts.
`--stream` exercises production streaming; use a different output directory for each run.
`--ids identity_name tools_recovery` selects cases; `--cases PATH` selects a validated JSONL
suite. `--ids` preserves the supplied order and rejects duplicates. Existing output directories
are refused unless `--resume` is explicit, so review work is not silently overwritten.

To resume an interrupted run, use the same commit, configuration, cases, mode and flags:

```bash
python -m evaluation.aevon_text_quality --mode transformers --output-dir outputs/aevon-text-v1-real --resume
```

Resume also supports `fake`, `remote`, `--stream`, `--cases`, and `--ids`. It requires an
existing incomplete run; it never creates a replacement. Before constructing a provider it
checks the suite/schema, mode, streaming flag, exact ordered case IDs and case fingerprint,
production prompt fingerprint, validated model/generation settings, source commit and a hash
of backend/runtime/evaluation Python code (including uncommitted changes). All must match.
Remote endpoint credentials are not stored in artifacts; the remote operator must still keep
the deployed weights/settings unchanged. Schema 2 runs support resume; older schema 1
incomplete runs are refused rather than migrated. Completed runs exit 2 with a sanitized
"already complete" error, without inference or artifact rewrites.

`results.jsonl` is authoritative: existing rows must form the exact contiguous prefix of
selected cases and retain their immutable input/expectation/review-rubric fields. Completed
rows, including recorded generation failures and controlled tool-recovery cases, are never
regenerated or rewritten. An interrupted current case with no complete row may run again.
Each completed result is serialized to one compact UTF-8 line, appended, flushed and fsynced
where supported before it counts as progress. The runner then atomically replaces the summary
and prints only `[17/54] technical_pipeline_watermark complete`; resume first reports the
completed count. No prompt, response, memory or credential content is printed.

Only a syntactically valid-but-incomplete final object **without a terminating newline** can
be discarded as a torn write, after all earlier rows validate. Truncation is exactly to that
line's start and is fsynced; earlier bytes remain untouched. Invalid middle lines, complete
malformed lines, duplicate/unknown/out-of-order IDs, invalid UTF-8, and ambiguous corruption
fail closed. A UTF-8 code point cut short inside an unfinished string is recoverable. A valid
final row lacking only its newline is kept, with a separator appended if another row follows.
Recovery is bounded to 1 MiB manifests, 2 MiB result lines, 64 MiB results and 64-level JSON
nesting. An OS advisory `.run.lock` permits one writer per output directory and releases on
process exit; do not remove the lock file or edit artifacts during a run. Storage durability
still depends on filesystem support; directory fsync is not available through this Windows
implementation. Atomic writes use fsynced temporary files before replacement.

The runner uses the existing provider selection, context compiler, trusted
`MEMORY_CONTEXT_V1` formatter, bounded calculator loop and mechanical validator. It never
opens the conversation database or mutates persistent memory. Synthetic memory tests inspect
model use/ignoring of supplied context, not retrieval quality. Long-history fixtures inspect
actual provider-visible input, including window limits, old/recent canaries, orphan handling,
current-user retention and memory ordering. `tools_recovery` is explicitly labeled controlled
fault injection: one synthetic malformed provider output is substituted, then the unchanged
runtime asks the selected provider to recover. It does not measure spontaneous malformed-call
frequency. `fake_responses` are independent scripted harness fixtures, never sent to a real
provider or used as golden answers for conversational grading.

Each run writes:

- `run.json`: mode, source revision, case/prompt fingerprints and exact configured model and
  generation settings. Fake runs are clearly labeled and have synthetic zero latency/tokens.
- `results.jsonl`: expanded input messages, final response, runtime latency/token counts,
  validator/tool metadata, deterministic checks, and a pending `human_review` with a rubric,
  flags, nullable `overall_pass`, and notes. It excludes raw intermediate model candidates.
- `summary.json`: overall/category deterministic pass rates, mechanical/tool/identity and
  memory/context check failures, generation failures, failed tool attempts (including recovered
  injected faults), and IDs requiring human review. It is rebuilt after every durable result
  and on resume, with `expected_total_cases`, `completed_cases`, `remaining_cases`, and
  `status`. Statistics cover **completed cases only**, not a final suite score while incomplete;
  zero completed cases have no pass rate. Final summary precedes the atomic complete manifest.

Checks include case-insensitive contains/forbidden text, narrow assistant-name probes, exact
configured-model identifier inclusion, successful calculator use/no attempts, memory canaries,
protocol leakage, production context invariants, and existing mechanical constraints. These
checks **do not measure all conversational quality**: negation, factuality, natural tone,
conciseness and appropriateness require human review. Even code-only acceptance is not a code
correctness guarantee; unverified unfenced code is flagged for review. All 54 cases require
review; a passing fake run proves harness wiring only, not real-model quality.

Generation failures are retained as failed cases without inventing final text or hidden
validator diagnostics. Cases without a successful final response fail applicable checks; summary failure
counts are check outcomes, not inferred root causes. The CLI exits successfully when artifact
collection completes (even with failed cases); inspect the summary. Setup/artifact errors exit
nonzero; interrupted runs remain marked running and can be explicitly resumed as above.
Clean and resumed fake final artifacts are byte-identical with the same inputs/code, unless
human review fields were edited; those existing edits are retained, not reset. Resume is
infrastructure recovery, not evidence of model quality. Artifacts are
local plaintext and contain prompts/responses: use synthetic inputs, protect custom sensitive
runs, and do not commit outputs. Remote inference sees supplied text in plaintext during
execution. No judge LLM or hidden external API is used.

## Run the base-model evaluation

Use an 80 GB A100/H100-class CUDA environment, or equivalent multi-GPU capacity, with PyTorch
already installed. Clone the repository and install only the evaluation dependencies first:

```bash
git clone https://github.com/amitcs50n/AmitAI.git amitai
cd amitai
pip install -U pip
pip install -e '.[eval]'
```

Authenticate to Hugging Face if needed:

```bash
huggingface-cli login
```

Baseline v1 and its output directory have already been generated and are frozen. Do not rerun or
overwrite `configs/baseline_eval.yaml` or `outputs/eval/qwen38_27b_base_behavior_v1/`.
`configs/baseline_eval_v2.yaml` keeps the same model, generation settings, and held-out eval file
while appending three prompt-only patches for constraint obedience, insufficient evidence, and
emotional support. Target specific cases with:

```bash
python -m evaluation.run_baseline \
  --config configs/baseline_eval_v2.yaml \
  --ids eval_normal_001,eval_reasoning_002
```

`--ids` trims surrounding whitespace, preserves the eval-file order, and may be combined with
`--limit`; ID filtering happens before the limit. Omitting `--ids` retains the existing full-set
behavior. Targeted runs are for response inspection, not the complete 20-case decision-gate
summary.

Prompt-only v2 results remain frozen. Mechanical constraint correction uses the separate
`configs/baseline_eval_v2_constrained.yaml` config and writes only to
`outputs/eval/qwen38_27b_base_behavior_v2_constrained/`:

```bash
python -m evaluation.run_baseline \
  --config configs/baseline_eval_v2_constrained.yaml \
  --ids eval_technical_002,eval_roleplay_001
```

That path parses only explicit `exactly N words`, `exactly N bullets`, `at most N bullets`,
and `code only` / `return code only` instructions. Passing responses stay unchanged, including
valid outer code fences; normalized inner code is validation metadata only. A mechanical failure
gets up to two bounded corrective generations containing the original request, latest failed
response, and measured miss. The latest retry becomes the final response even if it still fails,
and all attempts plus validation results remain in the response and review artifacts. Supported
count limits may use digits or deterministic written integers from zero through one hundred;
sentence counts, other written-number forms, semantic "one item" checks, and subjective scoring
are intentionally excluded.
Unfenced output is accepted as mechanically unverified rather than classified as code or prose.

Use `--resume` after an interrupted run, repeating the same selection options—including
`--limit` and `--ids`. Each completed case is appended immediately, so an expensive run does
not need to restart from zero.

Artifacts are written under the selected config's `output_dir`. V1 remains frozen at
`outputs/eval/qwen38_27b_base_behavior_v1/`; v2 writes to
`outputs/eval/qwen38_27b_base_behavior_v2/`. Each run directory contains:

- `run.json`: pinned model revision, code/dependency revisions, eval hash, and progress
- `responses.jsonl`: untouched v1/v2 outputs or constrained-run attempt metadata
- `reviews.jsonl`: outputs plus held-out scoring criteria

Constrained-run rows additionally retain the original prompt and response, parsed constraints,
all retry attempts and validations, backward-compatible first-retry fields, `retry_count`,
`final_validation`, and `final_response`.

Review `reviews.jsonl` manually. For every row, replace the null values:

- Set each entry in `rule_scores` to `0` for clear failure, `1` for partial or
  inconsistent behavior, or `2` when that rule is met.
- `critical_failure: true` marks a critical failure from the behavior spec

Then aggregate a complete 20-case review with its matching config, for example v2:

```bash
python -m evaluation.summarize --config configs/baseline_eval_v2.yaml
```

The current gate requires at least 90% of primary-rule assessments to score `2` and zero
critical failures. The summary also reports full-case pass rate plus category and genuine
per-rule results. A complete review produces one of three decisions:

- `baseline_meets_gate`: do not fine-tune yet
- `fine_tuning_candidate`: use the category/rule breakdown to design targeted SFT data
- `review_incomplete`: finish scoring before making a training decision

The default run disables Qwen thinking mode and uses greedy decoding with repetition penalty
1.15 so base-versus-adapter comparisons are stable. The model revision is pinned in the eval
config. The text-only harness uses `AutoTokenizer` with `AutoModelForCausalLM` and requires
Transformers 5.2 or newer for Qwen3.5 support.

`configs/baseline_eval.yaml` has a dedicated `runtime_system_prompt`. It is intentionally more
complete than the short `canonical_system_message` used in SFT records: the baseline should
measure the strongest prompt-only AmitAI behavior before LoRA is considered. Do not copy the
runtime prompt into every training example. Change that prompt, generation settings, or the
checkpoint only by creating a new named baseline run.

## Fine-tuning, only if the baseline misses the gate

Install the training dependencies and validate the selected SFT data:

```bash
pip install -e '.[train]'
python -m training.validate_dataset data/sft/v1/batch_01.jsonl
```

The baseline and training configs pin the same model commit. Do not change that revision
between the base run and adapter training.

Run a tiny smoke train before a full job:

```bash
python -m training.train_qlora --config configs/qlora_sft.yaml
```

The initial training config is intentionally conservative:

- BF16 LoRA
- LoRA rank 16 / alpha 32
- 4096 max length
- batch size 1
- gradient accumulation 8
- 1 epoch
- vision frozen

Do **not** treat these as final hyperparameters. Training data should target measured baseline
gaps while retaining enough balanced coverage to avoid regressions.

## Important current compatibility note

Qwen3.5/3.8 is a multimodal architecture. There have been recent reports of exported
text-only Unsloth fine-tunes having vLLM/tokenizer/export issues. v0 therefore saves the LoRA
adapter first and leaves merged-model export disabled by default. We will validate the exact
Unsloth/vLLM versions on the RunPod image before relying on merged export.

## Current direction

The tested prompt, bounded mechanical validator, production streaming path, structured Memory V1,
and first deterministic runtime tool now sit behind the persistent chat API. Keep the placeholder
SFT data untrained; vLLM, broader tools, and LoRA remain separate later milestones.
