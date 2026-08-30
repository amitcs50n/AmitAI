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
pip install -e '.[dev]'
pytest
python -m training.validate_dataset
```

Local tests do not load the 27B checkpoint.

### Persistent chat backend

Install the development dependencies and start the local API:

```bash
pip install -e '.[dev]'
uvicorn backend.app:app --reload
```

The API stores local conversations in `amitai.db` by default and currently returns a deterministic
mock assistant response. Run all backend, evaluation, and dataset tests with:

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
curl -N http://127.0.0.1:8000/api/chat/stream \
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

### Real GPU runtime

Use the CUDA/PyTorch environment supplied by the GPU host. The runtime extra intentionally does
not install or replace PyTorch:

```bash
pip install -e '.[runtime]'
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export AMITAI_GENERATOR=transformers
export AMITAI_RUNTIME_CONFIG=configs/baseline_eval_v2_constrained.yaml

uvicorn runtime.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Do **not** use `--reload` or multiple Uvicorn workers with the 27B runtime: either can load
another roughly 55 GB model copy. The default `AMITAI_GENERATOR=mock` remains safe for local
development. The transformers mode lazily loads one model per process on the first chat request,
then reuses it while serializing GPU generation calls.

The real path loads its system prompt, pinned model revision, BF16 settings, generation settings,
and mechanical-validator flag directly from the selected runtime YAML. To point the existing
frontend proxy at a deployed GPU API, set `AMITAI_API_ORIGIN` in the frontend server environment;
do not commit a RunPod URL or credentials.

### Frontend development

AmitAI remains the project and backend identity; Aevon is the user-facing assistant shown by the
frontend. Run the two development servers separately:

```bash
# Terminal 1, from the repository root
uvicorn backend.app:app --reload

# Terminal 2
cd frontend
npm install
npm run dev
```

Open the local Next.js address printed in Terminal 2. The development server proxies relative
`/api/*` requests to the FastAPI backend at `http://127.0.0.1:8000` by default, so browser code
does not need a separate CORS configuration.

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

The tested prompt, bounded mechanical validator, production streaming path, and first deterministic
runtime tool now sit behind the persistent chat API. Keep the placeholder SFT data untrained;
memory, vLLM, broader tools, and LoRA remain separate later milestones.
