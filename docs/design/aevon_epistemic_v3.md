# Aevon epistemic V3: context-input investigation

Investigated the production path at `a0acc410dba3dab940b979473824d15bd82d8d16`
with synthetic inputs, rendered prompts, and fake engines. No GPU or real model
inference was used. Only two small public metadata files were fetched: the pinned
[chat template](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8/chat_template.jinja)
(506 bytes) and
[tokenizer configuration](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8/tokenizer_config.json)
(7,675 bytes). Both templates match `PINNED_TEMPLATE` in `tests/test_vision_model.py`.

## Findings before implementation

1. `format_memory_context` serializes exact retrieved category/key/value records as
   JSON under `MEMORY_CONTEXT_V1` and `<memory_context>`. It explicitly identifies
   the records as trusted retrieved context, subordinate to system instructions and
   the current explicit user request. It contains no statement giving older
   assistant guesses priority over memory.
2. `compile_model_messages` emits production rules plus tool instructions as the
   first system message, then projected trusted memory and memory-command messages,
   then bounded conversation history, then the current user message. The V2 memory
   failure's PostgreSQL record reaches the model intact. The SQLite guess remains
   an assistant message; no serialization or replacement defect was observed.
3. The pinned template renders **every** system message as its own
   `<|im_start|>system` frame, preserving order and content. It does not merge,
   discard, or promote the older assistant guess. It does not encode an additional
   hierarchy among system messages; that hierarchy is expressed in instructions.
   Correct serialization does not guarantee that the model follows it.
4. Native text preparation renders that tokenizer template before tokenization.
   Vision uses the processor's matching template with the existing image-content
   adapter; text remains unchanged. New tests capture the exact string at these
   boundaries, including memory, an older guess, and a current correction.
   The old fake tokenizer used default Jinja whitespace settings. The fixture now
   uses the immutable sandbox and `trim_blocks=True, lstrip_blocks=True`, verified
   in the installed Transformers utility and the
   [upstream implementation](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py).
   This corrects test fidelity; the production renderer itself was already correct.
   Separate CPU-only execution of the installed Jinja compilation function matched
   all five before-change and sixteen after-change synthetic prompt captures exactly.
   The exact published generation prefix is
   `<|im_start|>assistant<think>\n\n</think>` with thinking disabled; this patch
   preserves it. Its semantic effect has not been measured or attributed as a cause.
5. Earlier history was removed at the 20-message / 20,000-character limits without
   an explicit notice. The oversized V2 case reached the model as only system rules
   and the current question. The model could report unavailable information, but
   was not told that prior turns had actually been omitted.
6. Repairs recompile from the original prior history with a corrected current
   prompt. Tool followups append assistant calls and trusted system results to the
   compiled messages. Both preserve the canonical prefix. Text/vision and
   local/remote share this compiler; privacy projection happens before selection.

The missing omission signal is a demonstrated structural gap. The cause of the
model's semantic memory-precedence failures remains unproven. Merging memory into
the first system message would change a functioning representation without evidence
that multiple system frames caused the failure, so memory serialization and order
remain unchanged. An earlier assistant guess still cannot structurally replace a
retrieved value; whether the model uses that value correctly needs real human review.

## Minimal changes

The compiler reports whether selection stopped because of a history limit. Only
then it appends this fixed section to the first system message, after tool rules:

```text
CONTEXT_AVAILABILITY
Earlier conversation turns were omitted from the available context. Do not infer or reconstruct their contents.
```

There are no summaries, removed values, counts, topics, or canaries in this notice.
It appears once per compiled request and survives repairs and tool followups. It is
absent for no history, exact-limit history, orphan cleanup alone, or privacy-only
omission. The decision uses projected history, so private omitted turns do not
trigger disclosure. A notice does not mean retained facts are unavailable: answers
should still use any sufficient retained evidence. History limits are unchanged.

The existing production wording is refined, rather than adding another policy
block: trusted facts outrank older assistant guesses and unstated assumptions;
materially unresolved references require clarification before agreement/action;
false premises should be corrected without affirmative framing; missing evidence
does not prove global absence, while an explicitly complete set establishes absence
from that set. Current explicit user correction precedence remains unchanged.

## Regression coverage and later model review

V3 has 16 cases across ambiguous references, memory fidelity, missing history, false
premises, open/closed-world reasoning, and technical evidence/configuration fidelity.
It includes five explicitly named confidence controls plus the bounded-inventory
case. NULL coverage separates a missing timestamp from proof an event did not occur.
V1, V2, and historical baseline assets stay frozen; V2 now has a pinned blob test.
The existing runner, categories, graders, and fake/resume machinery are reused.

All cases need human review. Mechanical passes establish context/tool/format
contracts, not factual correctness, proper agreement, or calibrated confidence.
After CPU checks, run separately on the provisioned A100 from the repository root:

```bash
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v3.jsonl --output-dir outputs/aevon-epistemic-v3-real
```

For an offline harness check, use `--mode fake` and a new output directory such as
`outputs/aevon-epistemic-v3-fake`. This does not start a model or access the memory DB.
