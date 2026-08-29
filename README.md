# AmitAI

AmitAI is a personal-assistant evaluation and optional LoRA project built around
**OBLITERATUS/Qwen3.8-27B-OBLITERATED V3**.

The repository also includes a small persistent chat API whose generator is mocked until the
real runtime orchestration layer is ready.

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

The mock generator is isolated behind the chat service so a later AmitAI orchestration layer can
replace it without changing the frontend-facing API contract.

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
gets exactly one corrective generation containing the original request, previous response, and
measured miss. The retry becomes the final response even if it still fails, and both attempts
plus validation results remain in the response and review artifacts. Sentence counts,
spelled-out numeric limits, semantic "one item" checks, and subjective scoring are intentionally
excluded.
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
both validation attempts, retry reason and response, `retry_passed`, and `final_response`.

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

## Next milestone

Do not train the eight placeholder examples. They only validate the pipeline/schema.

The next real task is the base checkpoint evaluation:

1. Run the 20 held-out prompts against the untouched base.
2. Complete the manual review and generate `summary.json`.
3. If the baseline meets the gate, stop and use prompting/runtime controls.
4. If it misses, finish SFT v1 around the measured gaps, run a tiny smoke train, and compare
   the adapter against this saved baseline.
