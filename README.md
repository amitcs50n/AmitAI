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

#### Structured memory V1

Memory V1 stores explicit, structured memories for the server-owned local principal
`local-default`. Automatic capture is off: ordinary conversation never mutates durable memory.
The chat parser accepts exactly one of these narrow command forms (case-insensitive):

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

Chat memory mutation is staged during the short preparation read, but it is applied only after
successful generation inside the final conversation transaction, after the real user message
exists. The assistant metadata reports `stored`, `updated`, or `deleted` only for the mutation that
committed; generation failure, exhausted validation, disconnect/cancellation, optimistic conflict,
or transaction rollback leaves memory and conversation rows unchanged. Model generation and tool
execution remain outside SQL transactions.

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
request bytes are capped at 256 KiB regardless of `Content-Length`; over-limit requests get 413,
unsupported media types 415, invalid JSON/query/body 400, disallowed routes 404, and denied
browser origins 403. GET/DELETE bodies are rejected; HEAD is unsupported. Domain validation
remains in FastAPI. Proxy errors contain no raw content, token, path, or exception details.

Only Accept/Content-Type and the server-generated bearer header go upstream: browser
Authorization, Cookie, Proxy-Authorization, Host, Forwarded, and X-Forwarded-* are not forwarded.
Only Content-Type/X-Accel-Buffering may return from upstream; upstream cookies, auth/debug/version
headers and cache policy are discarded. All API successes/errors use `Cache-Control: no-store`.
SSE uses `no-store, no-transform` and `X-Accel-Buffering: no`. Only bounded request JSON is buffered;
response streams remain incremental and client abort signals propagate upstream.

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

Private `/api/chat*`, `/api/conversations*`, and `/api/memory*` routes reject missing or incorrect
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

Explicit current remember/update/forget commands are processed locally. Remote inference gets
only a deterministic generic operation/status acknowledgment request, never the command's raw
category, key, value, or identifiers. Detected but unapplied commands get a generic not-applied
request. Remote command acknowledgments receive neither the secondary memory-command system
block nor retrieved memories (including previously opted-in targets). Historical memory commands
and their immediately paired assistant acknowledgments are projected to generic text before
history budgeting; stored history and the conversation API remain untouched.

Constraint validation still evaluates the original local request. Mechanical retry prompts use
the selected provider-safe current request, so retries cannot restore raw memory commands. The
same compilation applies to sync/stream generation and tool-loop base context; bounded current
tool messages are appended locally, never reloading dropped history.

Immediately before **every** remote HTTP request, including retries and tool follow-ups, the
client scans the exact outgoing JSON body and decoded fields with the shared memory credential
heuristic. It covers labeled passwords/passcodes, API keys, access/refresh/auth tokens, client
secrets, PEM private-key markers, JWT-shaped strings, and explicit `Authorization: Bearer ...`
values. The configured remote bearer token is also blocked anywhere in the body, including
JSON-escaped strings and generation configuration; its authentication header remains permitted.
No environment scanning, matched-text logging, silent redaction, or local-model fallback occurs.

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

On the GPU host, run only the authenticated inference service. It exposes `/v1/generate` and
`/v1/generate/stream`; it has no conversation, memory, preference, or user-facing chat routes and
does not initialize a database:

```bash
pip install -e '.[runtime]'
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml
export AMITAI_INFERENCE_AUTH_TOKEN='replace-with-a-long-random-development-token'

uvicorn runtime.inference_app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Do **not** use `--reload` or multiple Uvicorn workers on the GPU host: either can load another
roughly 55 GB model copy. Configure the local control plane explicitly with the remote base URL
and the matching service credential; never commit either value:

```bash
export AMITAI_INFERENCE_PROVIDER=remote
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml
export AMITAI_REMOTE_INFERENCE_URL='https://your-development-endpoint.example'
export AMITAI_REMOTE_INFERENCE_TOKEN='replace-with-the-same-development-token'

python -m runtime.serve
```

The endpoint is disabled unless `remote` is explicitly selected and both variables are present.
Remote inference requires HTTPS before any prompt or credential is sent. Plain HTTP is accepted
only for explicit loopback development endpoints using `localhost`, `127.0.0.1`, or `::1`; it is
rejected for LAN addresses and non-loopback hosts without silently upgrading the URL. The
ephemeral local control-plane token is independent of the inference-service credential. The
remote client uses `AMITAI_REMOTE_INFERENCE_TOKEN`, while the inference service receives the
matching credential as `AMITAI_INFERENCE_AUTH_TOKEN`.

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
