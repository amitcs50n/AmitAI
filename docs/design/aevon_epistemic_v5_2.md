# Aevon V5.2: meaning-preserving format repair

Starting revision: `6a23b39b0d021b0fc14aa5baed2d5a0c2de73b75`.

## Repair flow traced before implementation

`ProviderChatGenerator._generate_validated_response()` compiles/project-bounds
the request, runs the frozen V5 preflight, and runs the initial tool loop to a
final textual candidate. `evaluation.constraints.validate_with_bounded_retries()`
parses only the original user request and calls `validate_response()` on that
candidate. A failed supported mechanical constraint enables corrective inference.
Despite its name, `validate_with_one_retry()` was a compatibility wrapper for
**two** retries (`MAX_MECHANICAL_RETRIES = 2`). V5.2 explicitly changes this to the
ticket's maximum of one repair.

`build_retry_prompt()` already included the original request, previous answer,
and measured constraint failures/count deltas. It asked for a rewrite, allowed
rewriting from scratch if unavoidable, and requested content preservation only
"as much as possible". Mechanical revalidation alone could accept the three-word
`Capital of Paris.` even though `The capital is Paris.` answered the France
question correctly. A word count is not a factual relationship check.

The runtime retry callback previously recompiled prior history plus the corrective
user prompt and called the same tool loop again. It had access to the original
request, provider-safe projected current request, candidate, validation failures,
canonical compiled context, and tool records in its enclosing scope, but did not
pass the successful tool results into the repair request. It could execute tools
again, subject to the request-wide tool budget. Malformed-tool recovery is inside
the initial tool loop and is a separate mechanism.

This is production behavior, shared by local/remote providers and the
`evaluation.aevon_text_quality` harness. Constrained streaming buffers candidates
and calls the same validated path through a cancellation-aware streaming callback.
Unconstrained streaming takes its separate existing path. Local/remote vision
sessions also delegate to this generator, with text epistemic preflight disabled.
The older `evaluation.baseline.generate_constrained_case()` uses the same repair
helper, with a direct model callback and no tool loop.

The helper records original/repair prompts and responses, first/second/final
validation, retry count/reason and attempt records for legacy evaluation artifacts.
Production returns only final text, aggregate token/latency totals, tool records
and compact validator metadata (`retry_attempted`, count, pass/final validation,
and `first_retry_passed` when applicable). The backend snapshots before generation
then atomically persists the original user turn and successful final assistant
turn plus metadata. Corrective prompts/candidates are not chat turns. Failed
validation raises a sanitized generation error, with no partial conversation;
the text-quality harness records failure with final diagnostics unavailable.

## V5.2 design

Repair is instructed to edit only the existing candidate's form, preserving factual claims, entities,
numbers, dates, polarity, negation, comparison direction, uncertainty, quoted
identifiers, tool values and subject/object relationships. It never asks the model
to solve again. The initial production prompt and generation configuration remain
unchanged. A passing initial candidate receives no repair-only metadata or calls.

Only the failing path reuses the already compiled/projected context prefix,
replaces its current user message with the corrective prompt, and includes already
successful tool results. One direct provider call returns repair text; tool
protocol in that text fails without execution. Mechanical validation runs again.
Failure still follows the existing runtime error / evaluator failure contract.

The optional safety check is deliberately literal: unchanged numeric literal sets
(ignoring ordered-list markers), unchanged inline backtick identifier sets, and
no explicit leading Yes/No reversal. Failure codes contain no copied evidence.
This is conservative and does not infer the meaning of numbers, recognize all
entity/quote forms, or prove semantic equivalence. Relationship, uncertainty and
general factual preservation depend on the instruction and human review. No
Paris-specific runtime rule is introduced; the suite rejects known relation
reversal phrases and requires human semantic review.

Passing repairs add `first_validation` (without normalized source code) and
`repair_safety` to the existing runtime retry diagnostics. Literal failure codes
also make the final validation fail, even if its word/bullet check passes. Raw
repair prompts, tool values and candidates are not added to runtime metadata.
Legacy evaluation retains its existing prompt/response attempt artifacts; the
production-quality harness retains its existing sanitized missing-diagnostics
failure record. No new persistence or evaluator schema is required.

Vision adapters, images, grants and the vision exclusion from text epistemic
preflight are unchanged. Since vision already shares mechanical validation, its
format repairs inherit the same one-call policy and literal check. Existing CPU
vision tests verify image reuse/closure, grant lifetime, projection, cancellation
and no partial persistence. This does not add an image semantic validator.

LEFT JOIN, 502, contradiction, hotel and DOI cases receive no runtime special
cases: their general reasoning/evidence failures are base-model limitations.
V5/V5.1 epistemic trigger semantics and scoping are frozen.

## Final decision gate

V5.2 is intended as the final text-runtime architecture iteration for this
baseline. General reasoning failures remaining after V5.2 are model-quality
issues, not justification for indefinite deterministic guard expansion. After
manual code review, one small real-model killer run remains a separate user-run
gate. Fake results establish harness/control-flow correctness, not model quality.

## Suite and CPU validation

The new `eval/aevon_epistemic_killer_v5_2.jsonl` contains twelve natural cases:

| Case ID | Review focus |
| --- | --- |
| `v52_left_join_false_premise` | Reject unconditional row-count preservation; explain multiple matches. |
| `v52_unknown_502` | Unknown cause without logs; distinguish hypotheses from findings. |
| `v52_contradictory_boxes` | Same boxes/time/meaning contradict; no invented exception. |
| `v52_missing_hotel` | No hotel supplied; no invented name, location or booking. |
| `v52_unsupported_citation` | V1 citation prompt; inability to verify is not nonexistence. |
| `v52_capital_repair` | Fake four-word Paris proposition requires repair; relationship must survive. |
| `v52_left_join_unique_control` | Supplied unique right key permits row-count preservation. |
| `v52_502_evidence_control` | Matching gateway log establishes timeout for this response, not its deeper cause. |
| `v52_consistent_boxes_control` | Explicit exception for box C makes the claims consistent. |
| `v52_retained_hotel_control` | Recall supplied Cedar Hotel. |
| `v52_supplied_citation_control` | Repeat supplied synthetic DOI without pretending external verification. |
| `v52_no_repair_control` | Correct initial three-word answer requires one provider call. |

The Paris case uses existing `not_contains` expectations for known reversed
relationships plus human review; no new grading engine is introduced. Tests
explicitly show those bad three-word answers failing the suite. Real mode uses
natural generation and may pass initially; only fake mode deliberately supplies
the first failing candidate. Every case retains pending human semantic review.

The original 54-case suite retains all IDs, prompts, expectations and rubrics.
Only `tools_constrained_final.fake_responses` loses the repeated calculator call
that formerly occurred during formatting repair (four provider outputs become
three). Its initial call/result and corrected final text are unchanged. Tests pin
all nonfake case content to the starting revision, and pin the updated complete
fixture separately for resume compatibility. The old frozen byte assertion for
the shared constraints helper is replaced by behavioral tests because this ticket
explicitly modifies that shared helper. Production profile, inference config,
epistemic detector and compiler bytes remain pinned to the starting revision;
the preexisting compiler snapshots pass unchanged.

Tests that expected two repairs or a repeated tool execution now expect the
explicit one-repair/no-tool-rerun contract. They retain privacy, identity, context,
vision and persistence assertions. Historical second-retry scenarios are retained
as parameterized tests proving that an offered second candidate is never consumed.

CPU validation uses the bundled Python 3.12 interpreter with existing repository
dependencies through the ignored `.cache/v5_cpu_runner.py` launcher. No dependency
installation, weights, real inference, GPU, A100, RunPod or training is used.

| Check | Result |
| --- | --- |
| Targeted repair/runtime/evaluator/privacy/vision/context/V5/identity/resume tests | 943 passed; one existing Starlette/httpx deprecation warning |
| Fresh killer fake run, synchronous | 12/12 deterministic; zero generation/tool failures; 13 provider calls |
| Fresh killer fake run, streaming | 12/12 deterministic; zero generation/tool failures; 13 provider calls |
| Full CPU pytest | 1,924 passed, 1 skipped, 1 warning in 80.62 seconds |
| Changed Python Ruff | Passed |
| Repository-wide Ruff | 40 preexisting findings in 13 untouched files; no unrelated cleanup |
| `git diff --check` | Passed |

Expected and observed call counts in both sync and streaming tests: passing
initial candidate **1**; text repair **2**; initial calculator call, textual answer,
and repair **3**, with **1** tool execution. Both a replacement value `1412` and a
new tool request during repair fail, still after only three provider calls and
one tool execution. Failed format repair stops after two provider calls; malformed
tool recovery in the initial tool loop continues to use its unchanged budget.

Repository Ruff findings: B008 (11), FURB162 (2), I001 (7), RUF012 (2), SIM117 (1),
TRY004 (6), UP017 (7), UP035 (1), UP037 (3). They occur in `backend/app.py`,
`backend/models.py`, `evaluation/baseline.py`, `evaluation/run_baseline.py`,
`tests/test_backend_api.py`, `tests/test_backend_storage.py`,
`tests/test_baseline_eval.py`, `tests/test_dataset_plan.py`, `tests/test_eval.py`,
`tests/test_sft_batch_01.py`, `tests/test_spec.py`, `training/data.py`, and
`training/train_qlora.py`; all remain unchanged. Small lint issues in the already
edited `tests/test_constraints.py` were corrected to meet changed-file Ruff.

The one skip is `tests/test_asset_encryption.py:327`: POSIX mode checks do not
apply on Windows; the Windows ACL checks run. The warning is the existing
Starlette TestClient/httpx deprecation. Full tests pass after updating obsolete
two-repair/repeated-tool fixtures; no runtime fix beyond the scoped repair changes
was needed. No real 54-case benchmark was run.
