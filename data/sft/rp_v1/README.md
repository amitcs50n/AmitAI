# RP extension candidate preparation

This area is separate from `data/sft/v1`, the frozen 95-example core plan, and the
existing training configuration. This pipeline prepares and reviews existing
conversations. It does not train, generate conversations, alter inference, or
merge anything into core SFT. No AO3 or Literotica dumps are included.

## Bounded smoke build

Run from the repository root with Python 3.11–3.13 and the existing project
dependencies (`httpx`, `PyYAML`; `datasets` for loader validation):

```powershell
python -m training.prepare_rp_dataset --output-dir data/sft/rp_v1/generated/smoke_01
```

The default inspects PIPPA only. It stops at 100 accepted conversations, 500 raw
records, or 8 MiB read per source, whichever comes first. Unmet acceptance targets
are reported, never filled by weakening filters. This is an ordered prefix smoke
test, not a representative or randomized quality benchmark. New output directories
are mandatory; the CLI only permits output below `data/sft/rp_v1/generated`.

Explicitly inspect the disabled candidates, still subject to provenance gates:

```powershell
python -m training.prepare_rp_dataset --source pippa --source gryphe_sonnet --source openerotica --max-candidates-per-source 100 --max-bytes-per-source 2097152 --output-dir data/sft/rp_v1/generated/source_review_01
```

These commands use pinned JSONL file URLs, HTTP prefix ranges, a client byte cap,
bounded record buffers, and connection closure when a target is reached. No remote
dataset scripts run and no whole-file Hub cache is created. A server ignoring
ranges is still cut off at the client limit (at most one 16 KiB transport chunk of
prefetch). There are no automatic retries. Oversized rows receive a reason code;
a partial final line at the byte cap is not parsed. Each request has a 30-second
network timeout. Source/network failures yield partial reports and CLI exit 1.
Configuration/path errors exit 2. Cap exhaustion is an expected smoke outcome.

Local snapshots can be processed without network access:

```powershell
python -m training.prepare_rp_dataset --offline --source pippa --local-input pippa=data/sft/rp_v1/raw/pippa_sample.jsonl --output-dir data/sft/rp_v1/generated/local_01
```

Local files are not asserted to match the upstream revision. Metadata identifies
local inputs, hashes the bytes read and each parsed raw record, and records source
line numbers. HTTP reports also hash only the prefix read, not the entire file.
The stats manifest includes pinned source metadata, config/code/protected-prompt
hashes, limits, output hashes, stop reasons, errors, and unmet targets. Identical
inputs/config/code produce byte-identical outputs; no timestamps or absolute
machine paths are injected. Source manifest order decides duplicate precedence.

## Outputs and review gates

| File | Meaning |
| --- | --- |
| `rp_sft.jsonl` | Mechanically accepted, schema-compatible candidates from provenance-reviewed sources. Every row still has `human_review: pending`. Never automatically training-ready. |
| `review_candidates.jsonl` | Otherwise eligible conversations blocked only on source provenance. Schema-compatible inspection artifact; not approved SFT. |
| `decisions.jsonl` | One content-free locator, status and reason list per processed row, with metrics/hashes where normalization succeeded. |
| `review_manifest.jsonl` | Deterministic bounded locator sample per source/status/reason for human review. Contains no conversation text. |
| `stats.json` | Source counts, rejection/quarantine reasons, exact duplicates, length/turn/classification distributions, input limits and reproducibility evidence. |

All source provenance starts **pending**. Therefore an initial build may have an
empty `rp_sft.jsonl` even when many rows pass structural and quality checks. PIPPA
is enabled for inspection; Gryphe and openerotica are disabled by default. Explicit
`--source` selects inspection only; it never approves provenance. A reviewed source
requires `provenance_status: reviewed`, `reviewed_on`, and substantive `review_notes`
in a separately reviewed manifest. Do not change these merely to meet a quota.
`manual_review_required` records upstream review obligations; all imported rows
require human review regardless, as required by behavior-spec `DATASET-010`.

Rejected text and safety/PII quarantine text are not copied to output. Review those
through their source revision/line or the local input; do not publish that material
in a report. Mechanically eligible review candidates may still contain undetected
sensitive content. Raw/generated folders are Git-ignored, but they are ordinary
local files, not encrypted storage. This path may be synchronized by the user's
existing OneDrive configuration.

Reason counters are multi-label: a row can contribute to more than one reason.
Status totals count each row exactly once. Duplicate matching uses canonical
role/content JSON, ignoring metadata, across every selected source. It does not
perform fuzzy deduplication, whitespace stripping, or paraphrase matching. Exact
user-prompt overlap with repository core SFT and held-out eval files is rejected.
Top-level distributions describe accepted rows; `normalized_distributions_by_status`
also reports rejected/quarantined normalized rows, so an empty accepted set does
not hide the turn depths and lengths of the inspected candidates.

## Schema and source differences

The pipeline calls `training.data.normalize_example`, preserving text-part content:
`messages: [{role: user, content: [{type: text, text: ...}]}]`. It supplies valid
`rpv1_...` IDs, spec version `1.1.0`, category `creative_roleplay`, and existing rule
IDs as **coverage targets**, not verified annotations. It retains additional source,
length, filtering, and review metadata. The existing `load_sft_dataset` maps the
normalizer over these rows and preserves the extra metadata column. No loader or
trainer changes are necessary.

- PIPPA uses `conversation[].message/is_human`. The adapter preserves character
  name, description and definitions as system context, plus the existing opening
  assistant greeting. It does not invent a preceding user turn, duplicate the
  separate `bot_greeting` field, or reinterpret definition examples as dialogue.
  PIPPA's `bot_id` identifies a character, not a unique conversation; record hashes
  and revision/line locators distinguish conversations.
- Gryphe uses `conversations[].from/value` with `system/human/gpt`.
- openerotica uses the same shape, including the `assistant` alias.
- Missing/empty turns, unknown speakers, non-text parts, internal system turns,
  repeated roles and trailing user turns are rejected rather than silently fixed.
  The valid PIPPA opening assistant turn is an explicit source-specific exception.
- Approximate tokens are `ceil(character_count / 4)`, not Qwen tokenization.
  Long records are quarantined instead of being silently truncated. The current
  trainer's 4096-token limit needs a tokenizer-aware review before any experiment.

The existing `train_qlora.py` filename does not mean the checked-in config currently
uses quantization: `configs/qlora_sft.yaml` has `load_in_4bit: false` and points to
the v0 placeholder. This task intentionally does not alter that configuration or
assert model/collator compatibility from a CPU schema test alone.

## Provenance evidence

The manifest pins the files and dataset cards inspected on 2026-09-12:

- [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA): declares Apache-2.0;
  describes submitter permission to redistribute and attempted PII redaction.
  Includes both real and fictional personas. Further rights and content review
  remains necessary.
- [Gryphe Sonnet 3.5 character roleplay](https://huggingface.co/datasets/Gryphe/Sonnet3.5-Charcard-Roleplay):
  declares an unknown license. Describes synthetic dialogues based on character
  cards; character rights and applicable generation terms remain unresolved.
- [openerotica long roleplay](https://huggingface.co/datasets/openerotica/long-roleplay-v0.1):
  declares Apache-2.0, but describes scraping Chub/Janitor/other character cards.
  The metadata license is not evidence of rights to every underlying card.

## Heuristic limits and next gate

Clear structural failures, model self-identification at assistant-message openings,
dataset control tokens, injection phrases, corrupted text, and severe repeated
replies/phrases are rejected. Email/phone/address/secret-like strings and repeated
assistant control of the user's speech or decisions are quarantined. These checks
have both false positives and false negatives, especially for quoted instructions,
fictional contact details, unusual prose, and languages other than English.

Sexual keyword signals combined with explicit minor or real-person indicators are
rejected. Real-person indicators include descriptors and a small configurable seed
list of known names; that list is explicitly incomplete. Missing/ambiguous age
information is quarantined. Even a global assertion
that everyone is an adult does not establish age, identity, or consent: every
detected sexual candidate remains quarantined for review. These regexes are not
age verification, entity recognition, consent detection, or a comprehensive sexual
content classifier. No keyword match does **not** establish SFW content. All
automatic adult/SFW labels therefore remain `unknown`; signal metadata is separate.

Before curating any training dataset, review provenance and every selected row for
fictional adults, consent, privacy, agency, stable character facts, continuity,
repetition and purple prose. Review false negatives as well as flagged samples.
Create tokenizer-aware length and character/scenario-grouped held-out splits,
check semantic overlap, and design a balanced SFW/adult mix with explicit exits to
normal assistant behavior. Do not infer those behavior gains from mechanical
smoke checks. Any adapter configuration, data promotion, or training is a later,
separately authorized milestone.

## Tests

```powershell
python -m pytest tests/test_rp_dataset.py tests/test_dataset.py tests/test_dataset_plan.py tests/test_sft_batch_01.py tests/test_spec.py
python -m ruff check training/rp_data.py training/prepare_rp_dataset.py tests/test_rp_dataset.py
```

Fixtures are synthetic, non-graphic examples. Tests cover source adapters, shared
loader compatibility, malformed input, ordering, length, identity contamination,
age/real-person signals, repetition, PII, agency, provenance gates, duplicates,
protected prompt overlap, determinism, bounded I/O, failure reporting, and output
isolation. No GPU, model, tokenizer download, or training is involved.
