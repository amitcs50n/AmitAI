# Aevon V5.2.1: fail-safe mechanical repair

Starting revision: `39a707266709d28376bd5498f30a111706fb2921` (clean `main`).
This is the last mechanical-repair correctness patch, not an epistemic iteration.

## Trace recorded before code changes

1. `validate_with_bounded_retries` parses supported mechanical constraints and
   validates the original. A passing/unconstrained answer makes no repair call.
2. `build_retry_prompt` includes the user request, original candidate, measured
   failures and the V5.2 preservation instructions. Its text will remain unchanged.
3. `ProviderChatGenerator` makes one direct provider repair call with the existing
   projected/bounded context and obtained tool results. Repair never re-enters the
   tool loop. Empty output, provider errors and tool-protocol output are rejected.
4. `validate_repair_literals` checks number sets, backtick identifiers and leading
   Yes/No reversal. It does not check ordinary token substitutions or order.
5. The helper mechanically validates the repair and combines literal failures
   with that result. A correct word count can therefore admit relationship changes.
6. The generator raises `ChatGenerationError` for every failed final validation,
   including an unsafe repair; there is no original-answer fallback.
7. Constrained streaming buffers both candidates, runs the same validation path,
   then emits only the selected final response. Unconstrained streaming is separate.

## Scoped implementation decision

For exact-word repairs, accept only a case-insensitively equal sequence of retained
whitespace tokens, with deletion limited to bare, unquoted `a`, `an`, `the` tokens.
Preserve punctuation as part of each token. Ambiguous uppercase `A` is retained
as a possible identifier. This intentionally rejects broader
rewrites, including synonyms and content-word deletion; it is not a semantic judge.
Do not delete articles inside quotes/code, or when `code_only` is requested.

After rejecting an unsafe repair, a too-long exact-word original may receive one
deterministic attempt: remove only the required number of eligible articles from
left to right. Retain every other token verbatim and in order, then require all
mechanical and literal checks to pass. Insufficient articles or an unsatisfied
constraint preserves the nonempty original with truthful failed mechanical
validation and `semantic_fallback_used: true`. A successful deterministic edit is
identified separately. No extra provider call or tool execution is allowed.

Only a rejected unsafe repair may trigger original-answer fallback. Passing initial
answers, safe-but-mechanically-invalid repairs, malformed output, tool recovery,
privacy failures and cancellation keep their existing behavior. Production metadata
contains fixed reason codes/flags, never hidden candidates or repair prompts.

Planned changes: shared constraint helper, generator final-selection/metadata,
focused regression tests, necessary fake formatting fixtures, and this document.
Production prompt/model/generation configuration, context, V5/V5.1 guards, memory,
calculator/registry, vision implementation, transport and frontend remain frozen
to the starting revision. The earlier GitHub prompt change is part of this starting
revision; older prompt-specific test failures will be measured before edits.

## Validation

| Check | Result |
| --- | --- |
| Pre-edit full pytest baseline | 1,910 passed, 14 failed, 1 skipped, 1 warning |
| Targeted constraints and V5.2.1 repair regressions | 154 passed |
| Targeted generator (`tests/test_runtime.py`) | 63 passed |
| Regression/evaluator/resume/vision/memory/privacy integration | 525 passed |
| Fresh V5.2 killer fake run, synchronous | 12/12 passed; zero generation/tool failures; 13 provider calls |
| Fresh V5.2 killer fake run, streaming | 12/12 passed; zero generation/tool failures; 13 provider calls |
| Final full pytest | 1,971 passed, 14 failed, 1 skipped, 1 warning in 62.20 seconds |
| Ruff across all changed Python files | Passed |
| Diff whitespace and frozen-scope checks | Passed |

All 14 full-suite failure identities match the pre-edit baseline exactly:
the old production-prompt hash in the V5.2 tests (one), old prompt/layout captures
and hash in context-layout tests (two), and old identity/prompt assertions in
production-identity tests (eleven). The GitHub prompt change already present in
the starting revision caused those failures. This patch neither edits that prompt
nor refreshes its historical assertions/snapshots. The current production-config
SHA-256 remains `db72d43e847f43fcc834bfb2cd3203fa9d5c67e9766093e091c6bc0138092b49`.
The existing POSIX permission test is skipped on Windows; Windows ACL checks run.
The warning is the existing Starlette/httpx TestClient deprecation.

Fake formatting candidates in older tests were adjusted to obey the new contract,
while safe-but-invalid candidates still test the existing error/no-persistence path.
Two checked-in fake sequences changed: the killer capital repair now proposes
`capital is Paris.`, and the original calculator formatting case uses
`The answer is 1411.` followed by `answer is 1411.`. Its resume fingerprint was
updated accordingly. Every ID, real request, expectation and human-review rubric
in both suites remains byte-logically unchanged, including all reasoning and 502
controls. These are harness results, not evidence of real-model semantic quality.

The new 61-case CPU regression module covers ordered article deletion, preserved
negation/uncertainty/literals, too-short originals, insufficient articles, protected
quoted/code tokens, conflicting constraints, safe-but-invalid/malformed responses,
single retry, tool non-reexecution, cancellation, and actual backend persistence.
Sync and streaming both reject `Capital of Paris` as a repair of
`The capital is Paris.` and select `capital is Paris.`. When no safe edit exists,
the original is the only persisted/streamed answer, even with failed mechanical
validation. No general factual judge is added to initially passing responses.

`repair_safety` continues to describe the model repair; its new fixed failure
code is `repair_changed_exact_word_tokens`. Existing numeric/identifier/polarity
codes remain. A deterministic recovery adds `deterministic_repair_used: true`;
an original-answer fallback adds `semantic_fallback_used: true`. `first_retry_passed`
is false for a rejected model candidate; `retry_passed` and `final_validation`
describe the selected final answer. No hidden candidate or prompt is added to
production metadata.

Tests used the bundled CPU Python interpreter and existing installed dependencies
through the ignored `.cache/v5_cpu_runner.py` bridge. No dependency installation,
GPU, A100, RunPod, model download, real inference, or training occurred. No frontend
files changed and no frontend tests ran. The production prompt, repair-prompt
construction, model, generation settings, original literal/mechanical validators,
context, epistemic guards, memory, tools, vision implementation and remote transport
are unchanged. Production edits are confined to shared repair selection/safety and
the generator's explicit fallback/metadata handling.

Real validation is limited to `v52_capital_repair` and `v52_no_repair_control`, with
human semantic review, followed by final real UI testing. No full real twelve-case
run is planned without a new runtime defect. The pre-existing prompt-test drift
remains visible and outside this patch.
