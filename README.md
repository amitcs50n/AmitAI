# AmitAI

AmitAI is a personal assistant fine-tune built on **OBLITERATUS/Qwen3.8-27B-OBLITERATED V3**.

## v0 goal

Train a text-only BF16 LoRA adapter that changes behavior/personality while preserving the base
model's general capability. Vision is deliberately frozen in v0.

## Why the training code uses `FastVisionModel`

Qwen3.8-27B uses the Qwen3.5 multimodal architecture (`Qwen3_5ForConditionalGeneration`).
Even for text-only SFT, using Unsloth's multimodal training path avoids architecture-specific
collator/template problems. The LoRA config freezes vision layers and trains language,
attention and MLP modules only.

## Repository structure

```text
amitai/
├── configs/                  # behavior + training configs
├── data/
│   ├── raw/
│   ├── sft/                  # SFT JSONL
│   └── preference/           # later DPO data
├── eval/
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
  "spec_version": "1.0.0",
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

## RunPod training setup

Use a CUDA RunPod image with PyTorch already installed, then install the training extras:

```bash
git clone https://github.com/amitcs50n/AmitAI.git amitai
cd amitai
pip install -U pip
pip install -e '.[train]'
```

Authenticate to Hugging Face if needed:

```bash
huggingface-cli login
```

Validate data before paying GPU money to discover a broken JSON line:

```bash
python -m training.validate_dataset data/sft/amitai_sft_v0.jsonl
```

Then start training:

```bash
python -m training.train_qlora --config configs/qlora_sft.yaml
```

The first config is intentionally conservative:

- BF16 LoRA
- LoRA rank 16 / alpha 32
- 4096 max length
- batch size 1
- gradient accumulation 8
- 1 epoch
- vision frozen

Do **not** treat these as final hyperparameters. We first need a real dataset and a smoke test.

## Important current compatibility note

Qwen3.5/3.8 is a multimodal architecture. There have been recent reports of exported
text-only Unsloth fine-tunes having vLLM/tokenizer/export issues. v0 therefore saves the LoRA
adapter first and leaves merged-model export disabled by default. We will validate the exact
Unsloth/vLLM versions on the RunPod image before relying on merged export.

## Next milestone

Do not train the eight placeholder examples. They only validate the pipeline/schema.

The next real task is **AmitAI SFT dataset v1**:

1. Freeze the behavior specification.
2. Define dataset categories and quality rules.
3. Produce the first 100 high-quality conversations manually/curated.
4. Run a tiny smoke train.
5. Only then scale toward 500-2,000+ examples.
