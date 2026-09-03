# Aevon V4: experimental context layouts

Starting production revision: `67c9a013c2dc32db46fe1cceba1290a1d390f302`.
The V3 runtime delivered memory and truncation notices correctly, but real human
review still found semantic failures. This experiment changes layout only. It
does not establish a cause or select a winner.

## Isolation and exact layouts

`evaluation.context_layouts.LayoutProvider` adapts **already compiled** messages
at the evaluation provider boundary. The production compiler still owns privacy
projection, trusted-context recognition, history limits, omission detection and
current-user selection. No production runtime or configuration file is modified.
The normal runner has no adapter unless `--context-layout` is explicitly supplied.
There is no production environment variable, runtime config switch or default change.

Here `P` is the unchanged production instruction text plus tool instructions,
`M` is the exact projected `MEMORY_CONTEXT_V1` block, `H` is retained history,
`U` is the current user, and `N` is the existing fixed `CONTEXT_AVAILABILITY` section.
Each bracket is one model-visible message; adjacent sections inside a bracket
are separated by two newlines.

| Layout | Model-visible order |
| --- | --- |
| A, control | `[system: P + optional N] [system: M] H [user: U]` |
| B | `[system: P + M + optional N] H [user: U]` |
| C | `[system: P] [system: M] H [system: optional N] [user: U]` |
| D | `[system: P + M] H [system: optional N] [user: U]` |

Absent memory contributes no block or separator. Memory-command context remains
in separate trusted system frames ahead of history. Multiple leading retrieved
memory blocks are preserved in order; only those blocks merge in B/D. Memory
quotes inside conversation are never promoted. The current correction stays
after memory/history; instruction wording and its precedence remain unchanged.

All memory category/key/value JSON and the surrounding trust instructions remain
byte-for-byte equivalent UTF-8 substrings, with one copy per retrieved block.
Only frame boundaries and section separators change. A supplied assistant guess
keeps its assistant role. A/B/C/D never retrieve, rank, edit or persist memory.

The notice is moved only if the compiler actually appended its exact terminal
omission section. C/D remove that copy and put one trusted system frame directly
before the current user. No removed content, counts, inferred topic or summary
is included. Privacy projection and orphan removal alone do not create notices.
History size limits and the projection logic are unchanged.

Tool followups still append the existing assistant call/system result after the
current user. The adapter finds that current user before the tool tail. Repairs
recompile from the original source and receive the same layout. The adapter also
forwards the existing local/remote vision APIs, images and grants; tests cover
text/vision, streaming/nonstreaming, tools and retries. The benchmark CLI remains
a text benchmark; this ticket does not add a vision benchmark or change consent.

## Exact pinned-template inspection

`python -m evaluation.context_layout_inspection --output-dir <new-directory>`
renders the memory-conflict and missing-opening cases in all four layouts. It
uses Jinja's immutable sandbox with the same whitespace settings as Transformers,
and the 506-byte template verified during V3 for checkpoint
`OBLITERATUS/Qwen3.8-27B-OBLITERATED`, revision
`a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8`.
No tokenizer/model is loaded or downloaded. The template hash is
`4c2b3eabcdd87d5fef156833b055d520a8185f73580c5a14c75d140a16acbc14`.
The published generation suffix remains `<|im_start|>assistant<think>\n\n</think>`.

Eight exact captures and their hashes/role sequences are checked into
`tests/fixtures/context_layouts_v4/`. Fake native text and vision preparation
tests independently verify that the model preparation boundary renders the
same role frames and content, including the current image where applicable.

Distances below run from the end of the named section to the beginning of the
current user's ChatML frame. These are Unicode character distances, **not tokens**.

| Capture | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Memory conflict: memory-to-user distance | 161 | 161 | 161 | 161 |
| Missing opening: notice-to-user distance | 1240 | 1240 | 10 | 10 |
| Memory conflict: total prompt characters | 4536 | 4509 | 4536 | 4509 |
| Missing opening: total prompt characters | 5260 | 5260 | 5287 | 5287 |

For the memory conflict, B/D change membership in the first instruction frame,
not proximity to the user. They replace a 29-character frame transition with a
two-newline separator. C/D move the truncation notice past retained history,
adding a separate frame. A=C and B=D for a memory-only case; A=B and C=D for a
truncation-only case. The previous-task case also supplies retained database
memory, so the suite includes an interaction where all four layouts differ.

## Suite and interpretation

V4 uses the existing strict JSONL schema, runner, storage, resume and fake mode.
Its 10 cases cover trusted PostgreSQL versus an older MariaDB guess; a latest
SQLite correction; actually truncated opening and previous-task history;
unresolved action; unknown configuration variable; SQL NULL versus delivery;
COUNT with explicit NULLs; a clear rewrite referent; and an exhaustive inventory.
Every case requires human semantic review. Written-out counts and equivalent
wording are accepted by the rubric; no keyword grader decides semantic success.

Layout experiments use the same case content, production prompt, model and
generation settings. The layout is recorded in `run.json`, and a different or
missing layout fails resume validation before any provider is constructed.
The deterministic context check compares the actual initial provider input
against the selected layout exactly; only then do existing canonical-order
checks run on its known canonical representation. Content-presence checks still
inspect actual calls, including tool followups and retries. These are plumbing
checks, not scores that choose a better layout.

## CPU validation for this patch

- Targeted pytest: 347 passed.
- Full pytest: 1706 passed, 1 skipped, 1 existing dependency deprecation warning.
- Ruff on all changed Python files: passed. Repository-wide Ruff: 45 existing
  findings, all in files unchanged from the starting revision; no unrelated fixes.
- Fake V4 CLI: 10/10 mechanical passes for each of A, B, C and D, with zero
  generation/tool failures. All semantic reviews remain pending. The four
  manifests have identical case/code/prompt fingerprints and model/generation
  settings, differing in the explicit experimental layout.
- Eight pinned-template snapshots reproduce exactly. Tests also verify the
  previous-task case has both retained memory and actual history truncation.
- V1/V2/V3 suites, production prompt/configuration, runtime, backend and frozen
  historical baselines are untouched. Production still compiles Layout A.

No GPU, real model inference, model loading/download, training or network calls
were used. Validation used the existing CPU test environment. Generated run
artifacts remain under ignored `outputs/`; checked-in snapshots contain only
synthetic model-visible context.

## Later A100 runs (not executed in this ticket)

Use the same committed source and provisioned environment for all four commands,
with no overrides to the pinned BF16 checkpoint or generation settings. Each
command uses every case, in the same order, and a fresh artifact directory:

```bash
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v4.jsonl --context-layout A --output-dir outputs/aevon-epistemic-v4-real-A
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v4.jsonl --context-layout B --output-dir outputs/aevon-epistemic-v4-real-B
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v4.jsonl --context-layout C --output-dir outputs/aevon-epistemic-v4-real-C
python -m evaluation.aevon_text_quality --mode transformers --cases eval/aevon_epistemic_regression_v4.jsonl --context-layout D --output-dir outputs/aevon-epistemic-v4-real-D
```

Recommended order: A, B, C, then D. A establishes the same-suite control; B isolates
memory-frame placement; C isolates notice placement. Review those before spending
on D's combination. If neither isolated change helps, defer D rather than infer
that the combination will work. For a complete four-layout comparison run D too.
Each invocation loads one model for the full suite; these commands do not share
an in-memory model across processes. Resume an interrupted invocation with the
identical arguments plus `--resume` instead of rerunning completed cases.
Review responses with layout labels hidden where practical, checking confidence
controls as well as failures. Ten probes are diagnostic evidence, not a reliable
estimate of broad semantic quality or grounds alone for a production default.
