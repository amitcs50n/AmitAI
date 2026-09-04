# Aevon V5: deterministic epistemic preflight

CPU implementation based on `75702d68148248cec7efb4451fa78edad150641e`.
Real-model validation follows code review; fake outputs are not quality evidence.

V5.1 tightens only env-var evidence scope, starting from
`13eb4379910121f558098ef5cebf1c84cb80e33b`. The original two history/reference
guards, compiler, generator, evaluator, metadata, privacy and vision paths are
unchanged.

## Evidence and investigation

The ticket reports that V1 completed 54/54 real BF16 cases without generation
crashes, but human review found epistemic failures. V2 prompt rules improved
some categories without resolving missing history, ambiguity, and invented
details. V3 supplied an explicit history-limit omission notice and confirmed
trusted memory reached the model. V4's real A100 Layout A and C runs each remained
approximately 6/10 under strict human review. Moving the warning next to the
current request did not reliably prevent reconstruction. These are the ticket's
historical findings, not measurements made by this patch. Further prompt-layout
tuning stopped; production remains Layout A.

Inspection at the starting revision confirmed all eight ticket assumptions:

1. `runtime/context.py` detects actual count/character truncation during selection.
2. `compile_model_messages()` returned only the message list.
3. Remote projection precedes selection intentionally.
4. Privacy-only omissions and orphan-assistant cleanup do not set truncation.
5. Production places the omission notice in the initial runtime/tool system frame.
6. A/B/C/D adapters live in evaluation and are not production configuration.
7. `ProviderChatGenerator` owns compilation, provider calls, tools, and validation.
8. Local and remote vision construct sessions using that same generator.

No architectural assumption needed changing. The V4 source-hash assertion for
`runtime/context.py` is replaced by compiler-output compatibility checks, since
metadata is now intentionally added to that file. Frozen evaluation datasets,
production prompt/configuration, layout captures, and layout adapters remain intact.
Legacy V2/V3 provider-capture assertions now distinguish local compilation from
delivery, the V2 missing-opening formatting test expects bypass, and the identity
test observes the new structured compiler entry without changing its assertions.

## Available context, not reconstructed context

`compile_model_context()` returns a frozen `CompiledModelContext` containing:

| Field | Meaning |
| --- | --- |
| `messages` | The unchanged production list of role/content dictionaries. |
| `history_truncated` | The existing selector observed a count/character overflow after projection. |
| `retained_history_count` | Retained user/assistant messages, excluding current user and trusted frames. |
| `retained_user_turn_count` | Retained prior user messages. |
| `latest_prior_turn_retained` | The latest source user/assistant position survived projection and selection. |
| `latest_prior_user_turn_retained` | The latest source user position survived projection and selection. |
| `trusted_context_count` | Number of projected leading trusted runtime frames. |

The old `compile_model_messages()` API returns `.messages`. Twenty fingerprints
captured from the unchanged starting compiler pin serialized list output for ten
fixtures in both scopes: empty, normal, count, character, oversized, orphan,
memory, memory command, projection, and privacy-only omission. Existing exact
layout/template captures independently check message order and content.

Source positions and roles establish which recent turn survived. This avoids
substituting an older visible turn when privacy projection removed the latest
one. The truncation flag still means history-limit truncation only. There is no
new privacy omission notice. Neither omitted content nor a summary is retained
in the compiled result. The detector accepts only that compiled result, never
raw history, database access, a retrieval callback, or an external tool.

## Exact guard boundaries

`runtime/epistemic.py` implements a deliberately limited English grammar. It
normalizes whitespace/case for matching and returns a frozen decision with
`kind`, `deterministic_response`, and a fixed `reason` code.

| Kind | Trigger | Non-trigger controls |
| --- | --- | --- |
| `missing_history` | A recognized recall request for this conversation's first/opening/earliest message with observed truncation or no retained prior user; a turn-content request before the oldest/visible/retained window with observed truncation; or a recognized previous-turn recall request whose requested prior user/conversational position is unavailable. | Complete first-message history; a retained immediately previous user despite older truncation; semantic remembered facts such as the database; requests to compose an opening; ordering inside queues/algorithms. |
| `ambiguous_reference` | No retained conversational turns and a complete short request matching supported actions on `this/that/it/these/those/them`, optionally with a simple weekday/today/tomorrow date; `Should I take it tomorrow?`; short meaning/identity queries including an unexplained error. | Any retained conversational history; inline object, quoted text, code, URL, or extra explanatory clause; ambiguous multiple referents, left to the model. Memory system frames alone do not establish the target. |
| `unknown_internal_env_var` | A what/which request for an exact/actual/used/read/configuring environment-variable name scoped to the user's app/project/service, with no identifier candidate bound to the queried target. Requests for suggestions/recommendations/examples are excluded. | General technical knowledge such as PATH; other identifier types; relevant current/retained user evidence or a trusted memory value scoped to the target. |

Supported deictic actions are shift, move, reschedule, cancel, rewrite, revise,
summarize, explain, translate, and delete. The grammar full-matches the short
request to avoid swallowing an object supplied elsewhere in the same message.
It deliberately misses unsupported paraphrases rather than guessing semantics.

Env-var candidates are conventional uppercase underscore identifiers, bare
uppercase identifiers explicitly associated with an environment variable/config,
or quoted identifier lookups through `environ`/`getenv` (including lowercase).
Trusted memory parsing reads values inside leading `MEMORY_CONTEXT_V1` frames;
a bare identifier value needs an env-specific memory key. A key alone, the frame
label, runtime/tool instructions, `MEMORY_COMMAND_V1`, and assistant messages
cannot authorize an identifier. `PostgreSQL`/`POSTGRESQL` alone do not qualify.
Malformed memory payloads do not become evidence. Existing memory retrieval,
ranking, storage, sensitivity and formatting are unchanged.

V5.1 fixes the original scope bug: V5 accepted any retained user candidate or
trusted-memory candidate, so an unrelated `REDIS_URL` could permit the model to
invent a dispatch variable. Candidate recognition itself is unchanged; a small
lexical binding now follows recognition:

1. Extract exact normalized target words from the current env-var question's
   sentence/clause. A named service/project/configuration such as `dispatch`
   takes precedence over shared component terms such as `database`. If there is
   no name, use the small component set database/cache/frontend/backend, treating
   `DB` as `database`. A generic current question may use the one named scope
   explicitly supplied elsewhere in that same current prompt. No prior or
   assistant message participates in target extraction.
2. Prior user evidence needs a candidate and an independent target word in the
   same sentence/clause. Sentence boundaries followed by whitespace and newlines
   separate evidence, so a dispatch sentence cannot lend scope to a separate
   Redis sentence. Strip candidate expressions before matching target words:
   `DISPATCH_DATABASE_URL` alone is not independent dispatch association.
   Matching is case-insensitive whole-token equality, including separator-based
   memory-key tokens; no fuzzy matching or entity model is used.
3. Trusted-memory keys bind scope and values supply identifiers.
   `dispatch.env_var` and `dispatch.database_env` qualify; `cache.env_var` and
   `frontend.api_env` do not qualify for a dispatch query. A generic key composed
   only of env/config/name/key/value/note words can qualify only when its value
   explicitly associates a candidate with the target in prose. Thus generic
   `env_var = DISPATCH_DATABASE_URL` does not qualify, while
   `env_var = Dispatch uses DISPATCH_DATABASE_URL.` can. An unrelated specific
   key cannot be overridden by words in its value.
4. Current-prompt evidence follows the same binding rule, with a narrow exception
   for a simple unscoped assertion such as `Our config says DISPATCH_DATABASE_URL.`
   next to the query. Only the existing short config/code/env assertion forms,
   with no additional scope words, qualify for that exception. It does not apply
   to prior history. Explicit Redis/frontend statements in the current prompt
   still cannot authorize dispatch.

Target words, matched evidence and memory values exist only as local temporary
values inside the detector. They are neither returned nor logged nor added to
validator metadata. Assistant guesses remain excluded. Unknown or unrecognized
associations conservatively retain the guard.

Lexically bound evidence allows normal reasoning; it does not certify a
candidate as correct, current, or consistent. Compound clauses, negation, shared
names, and nuanced project associations remain limitations of this small
matcher; it is not an entity-resolution engine. This is not a general
implementation-detail checker. Unusual identifiers or bare lowercase names
without an explicit code/memory association may fall outside recognition.

Reasons are `opening_unavailable`, `before_window_unavailable`,
`previous_unavailable`, `no_conversational_referent`, and
`internal_env_name_unavailable`. Responses are fixed content-free unavailability
or clarification sentences; they include neither canaries nor guessed topics,
identifiers, memory keys, omitted content or regex details.

## Generator, persistence, streaming, and privacy

Text sync/stream entry compiles context, then runs preflight before provider
generation, provider streaming, tool-loop entry, or mechanical repair. A guard
returns the configured provider model identity with zero input/output provider
tokens and empty tools/memory lists. Normal chat service processing subsequently
adds its existing memory references and atomically persists the user/final
assistant pair with normal title/timestamp handling. Backend/schema changes are
unnecessary.

Normal validator fields are preserved. Guarded results add:

```json
{
  "epistemic_guardrail": {
    "triggered": true,
    "kind": "missing_history",
    "reason": "opening_unavailable",
    "provider_bypassed": true,
    "mechanical_override": false
  }
}
```

The unchanged mechanical parser/validator still reports the deterministic
response's actual constraint result. If it fails, `mechanical_override` is true
and `final_validation.passed` remains false: epistemic correctness wins without
a provider call or formatting retry. Ordinary generations retain the old
validation/retry path and receive no extra guard metadata.

Guarded streaming emits one exact response delta followed by the result. The
backend produces `start`, `text`, `final`, `done`. Cancellation is checked before
compilation, before the delta, and again before the terminal result. Cancellation
or closure after the delta cannot cause backend partial-turn persistence. No
provider stream or worker exists for these requests. Normal streaming and
prefix-gated tool streaming still use the existing path.

Remote decisions use projected context and run locally before provider entry,
DNS, HTTP, or any fallback. Tests cover raising providers, a real remote-provider
object with mock transport plus forbidden HTTP/DNS, and successful persisted
remote requests with local-only retrieved memory. Zero-call requests send neither
the prompt nor retrieved context. Existing disclosure detection is untouched:
requests that do reach a provider still pass the same privacy checks. A locally
handled request is persisted under the normal local chat contract.

Both vision session constructors explicitly pass `_text_epistemic_guards=False`.
All three guards are excluded from those sessions. Image decoding, processor
input, consent grants, transport, tool/repair behavior and image lifetime remain
unchanged. Tests use synthetic images and fake engines to prove local and remote
`What is this?`/`Explain that error` still invoke vision, while remote calls without
consent still fail before transport.

## Evaluation and validation

V5.1's 22 cases declare `expectations.epistemic_guardrail`: one of the three kinds,
or explicit `null` to require normal provider execution and absent metadata.
The original 18 cases are unchanged; two unrelated-evidence guards and two
related-evidence controls are appended. There are eleven guarded cases and eleven
controls. A guarded expectation checks a
final response, matching kind, zero calls/tokens/tools, no protocol leakage, and
exact stream reconstruction when streaming. Context checks compile independently
and label their source `compiled_locally_provider_bypassed`. Ordinary cases keep
checking provider-visible context. A bypass does not imply layout delivery.

Frozen V1-V4 cases without this field keep their historical expectation
dictionaries. They can now be handled by V5, but did not historically declare a
guard expectation; they still undergo response/context/actual execution-contract
checks. This compatibility distinction is preserved through serialization.
Result schema version remains 2. Existing source revision, code fingerprint,
case fingerprint, and invocation matching reject old incomplete runs under
changed code before provider creation or file repair. V5 tests verify interrupted
sync and stream runs resume to byte-identical fresh-run artifacts.

CPU validation includes detector boundaries, frozen compiler outputs, canary
independence, latest-turn availability, provider/tool/transport bypass, lazy
engine non-initialization, persistence, cancellation, formatting precedence,
vision exclusion, existing text/tools/privacy/vision tests, fake suites, full
pytest, changed-file Ruff, and `git diff --check`. Benchmark artifacts and the
local test launcher remain outside the committed source. No new dependency,
GPU, RunPod, model load/download, layout experiment, or training is needed.

## Limits

V5 does **not** solve arbitrary model hallucinations. It only handles cases where
runtime evidence can deterministically establish that the requested answer or
referent is unavailable within the recognized narrow request forms. Retained
history may still contain no usable referent or several possible referents;
these situations intentionally go to the model. Other languages, complicated
paraphrases, arbitrary exact identifiers, contradictions, false premises, and
open-ended unsupported claims remain outside the guards.

The underlying 27B model will still sometimes make unsupported claims in
open-ended situations. That is a model-quality problem, not something V5 claims
to eliminate. Passing CPU/fake checks demonstrates integration and control flow,
not real-model quality improvement. The small real A100 V5 benchmark must be run
manually after patch review.
