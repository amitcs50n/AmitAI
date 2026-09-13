# SOTOPIA 500 inspection and original-author seed design

Completed 13 September 2026. **No row is approved, provenance-reviewed, or training-ready.** The production RP adapter and frozen RP filter v2 were not modified. No model inference, synthetic generation, or training ran.

## Decision

SOTOPIA-pi is worth retaining as a **narrow SFW negotiation and social-state support source**, subject to row-level human selection, a future explicit event/turn adapter, and unresolved provenance/license review. It is not a replacement for open-ended character RP and is not sufficient to authorize a 1k–5k corpus or training.

The 500-row investigation produced six strict manual-review nominations (1.2% of the bounded sample), 12 near misses, and 10 clearly weak rows among a 28-row full-reading queue. This is an observed shortlist yield, **not a population acceptance estimate**.

For the original-author corpus, V1 should target **48 human-written scenes across 16 original characters**: 28 SFW, 12 mature nonsexual relationship scenes, and eight adult-capable scenes. Synthetic expansion remains blocked until the exact gate below is satisfied.

## Collection and exclusions

The inspection used `cmu-lti/sotopia-pi` revision `e583406958ff132f6749ca87a2f9aa31ae3c0fa1` and the exclusion evidence frozen in the previous task. The source card declares CC BY-SA 4.0; that declaration and upstream dependencies remain provenance-review inputs, not project approval.

The collector streamed the pinned episode file and stopped immediately after selecting 500 eligible rows. It retained five episodes from each of all 100 allowlisted training environments, giving 100 distinct scenarios. It screened 2,239 source lines and transferred 74,022,571 bytes. Of the nonselected lines, 1,129 matched documented evaluation environment/episode IDs, 519 had evaluation/test/baseline/dev-like tags, and 91 exceeded the five-per-environment cap. There were no documented-exclusion intersections in the retained set.

Every retained row has experiment tag `sft_round_1_gpt-4_gpt-4_clean`, GPT-4/GPT-4 actor labels, and GPT-4 evaluator metadata. There is therefore no valid model-stratum comparison in this sample. Episode IDs, environment IDs, source lines, source hashes, actor/model labels, tag, and pinned revision are retained.

Only scenario text, public profile text, and actual social interactions were retained for inspection. Hidden goals, private profile sections, evaluator reasoning, rewards, raw reward prompts, raw messages, and simulator internals were excluded. Explicit action and exit events remain visible as events. They were not rewritten as ordinary assistant dialogue, and no training-format export exists.

## Frozen-v2 results

| Result | Count | Rate |
|---|---:|---:|
| Selected | 500 | 100% |
| Rejected by unchanged structural/v2 checks | 341 | 68.2% |
| Quarantined | 159 | 31.8% |
| Provenance-only format survivors | 151 | 30.2% |
| Other quarantine | 8 | 1.6% |
| Human-approved / training-ready | 0 / 0 | 0% / 0% |

The 340 `broken_speaker_order` results arise because the inspection projection ends on the first actor/user. One further row fails minimum turns. This is current-loader compatibility evidence, not proof that those 340 dialogues are semantically bad. The source contains explicit simulator exits/actions; 412 of 500 selected episodes have a terminal exit event. A future adapter must decide how events and episode endings map to training turns without deleting or relabeling them to improve survival.

Across rows that reached frozen-v2 assessment, the unchanged signals found three user-decision reviews, two user-dialogue reviews, two participant-age reviews, two sexual-content/ambiguous-age reviews, one HTML-card review, and one intimate-power-imbalance review. Counts overlap. They remain review routes, not semantic verdicts.

Approximate context-plus-dialogue length is 266 tokens minimum, 1,146 median, 1,748 at p90, and 3,699 maximum. These estimates use characters divided by four and are not model tokenization.

## Deterministic screening and full reading

Deterministic features ranked all 500 rows for reading. They do not alter v2 and do not approve rows:

| Diagnostic feature | Rows |
|---|---:|
| Proposals from both speakers | 474 |
| Resolution language | 213 |
| Early constraint terms carried into later turns | 416 |
| At least two number/quantity-bearing turns | 11 |
| Agreement-loop heuristic | 38 |
| Pronoun/reflexive mismatch heuristic | 13 |
| Provenance-only format survivors | 151 |
| Strict full-reading queue | 28 |

The proposal detector is broad; the contrast between 474 bilateral proposals and only 11 rows with repeated concrete quantities helps explain why many dialogues sound collaborative but remain vague. The loop and pronoun counts are conservative diagnostics, not population-quality claims.

All 500 received source eligibility checks, frozen-v2 diagnostics, structural/event analysis, token estimates, and deterministic semantic screening. The complete scenario, public profiles, and every episode turn were read for all 28 rows in the strict queue. The other 472 did not receive an exhaustive human semantic verdict. The review ledger preserves this distinction.

## Strict shortlist

All six entries remain pending manual quality and rights approval. They span six environments.

| Sample | Approx. tokens | Why it survived strict reading | Remaining concern |
|---|---:|---|---|
| `sotopia500_004` | 1,468 | Garden conflict reaches barrier construction, weekly checks, growth-based allocation, trellises, and seasonal review while preserving both speakers' decisions. | Slightly long closing; prior 100-row nomination. |
| `sotopia500_172` | 1,387 | Apple negotiation carries inspection, price, volume, and delivery through a conditional $1.80/lb agreement. | Volume threshold remains unspecified; profile occupations and scene roles need plausibility review. |
| `sotopia500_200` | 1,823 | Gift disagreement resolves into a concrete allocation across two events after several genuine counterarguments. | Final acknowledgement turns repeat. |
| `sotopia500_336` | 907 | Road-trip disagreement resolves to hourly music alternation and moderate volume, then enacts the choice through explicit stereo/listening events. | Event representation needs the future adapter decision. |
| `sotopia500_383` | 1,383 | Dinner conflict resolves to takeaway from a specific restaurant and a cooking project the next day, retaining both preferences. | Slightly long close and action is announced rather than externally confirmed. |
| `sotopia500_401` | 1,092 | Former partners reach closure while the no-reconciliation boundary remains intact; lighthouse/astronomy voice is distinct. | Poetic mirroring needs style review; this is social-state change rather than offer bargaining. |

The earlier apple nominee, previously `sotopia_061` and now `sotopia500_062`, moved to near miss after comparison with five episodes from its environment: it preserves $1.70/lb and delivery terms but leaves volume thresholds undefined, while `sotopia500_172` is the stronger representative. One of the previous two nominees remains shortlisted, and five nominations are newly observed outside the previous 100.

## Near misses

| Sample | Reason held back |
|---|---|
| `sotopia500_062` | Clear bargaining, but undefined volume terms and a stronger same-environment alternative. |
| `sotopia500_072` | Concrete alternating coffee plan followed by five redundant farewell/anticipation turns. |
| `sotopia500_099` | Agrees on a surcharge formula but never fixes the percentage or base price; safety assurances are unsupported. |
| `sotopia500_168` | Repairs a neglected friendship and proposes next-week contact without selecting a date; resolution is too quick. |
| `sotopia500_236` | Useful upfront-price plus future-profit structure, but all figures are deferred to another meeting. |
| `sotopia500_247` | Respects an anxious partner's concern and proposes several controls, but never fixes hours or coping steps before reassurance repeats. |
| `sotopia500_263` | Reaches a specific first meal, but makes an ungrounded health claim and closes with slogan repetition. |
| `sotopia500_305` | Balances indoor enrichment and supervised outdoor time, but leaves timing undefined and over-acknowledges the plan. |
| `sotopia500_311` | Proposes a heavier-base test run, but does not enact it, repeats the compromise, and changes Eli to Elliott in the profile. |
| `sotopia500_436` | Reaches a fenced-space/weekend plan, but remains at planning level and ends with generic excitement. |
| `sotopia500_452` | Identifies concrete work blockers but postpones the actual task division and repeats teamwork assurances. |
| `sotopia500_467` | Respectfully adapts a memorial activity after reluctance, but the profile contradicts they/them with “her” and the closure becomes therapeutic repetition. |

## Common failure patterns

The dominant pattern is **agreement without enough new state**. The conversation finds a compromise early, then spends several turns thanking, praising, or restating it. The deterministic loop heuristic marks 38 rows; full reads show more paraphrased loops that simple phrase similarity misses, including `025`, `072`, `154`, `158`, `321`, and `454`.

The next pattern is **deferred resolution**: participants agree to research, schedule, or discuss specifics later while the original variable remains unset. `236`, `290`, and `452` illustrate price, date, and task-allocation deferral.

**Profile/scenario inconsistency** is material. In `124`, the scenario names Carson while the profiles/speaker label use Sasha and dialogue alternates between both identities. `454` gives software-developer/pharmacist profiles to a researcher/activist scene. Several profiles contain internal name or pronoun drift. These are reasons to retain profile and scenario metadata during review, even though they must not become hidden target text.

Other recurring defects are abrupt consensus, generic coaching language, unsupported factual/medical assurances, and tangents after resolution. `448` solves a sink issue and then drifts into a tree-house project. Distinct character voice is uncommon; the clearest shortlisted exception is `401`.

The frozen filter remains appropriately unchanged. These findings concern source/editorial quality and future adapter design, not a concrete blocker requiring more filter tuning.

## Comparison with the previous 100

| Measure | Previous 100 | Targeted 500 |
|---|---:|---:|
| Rejected | 67 (67.0%) | 341 (68.2%) |
| Quarantined | 33 (33.0%) | 159 (31.8%) |
| Provenance-only format survivors | 32 (32.0%) | 151 (30.2%) |
| Full semantic reads | 8 | 28 |
| Strict nominations | 2 (2.0%) | 6 (1.2%) |

All previous 100 episodes are contained in the targeted 500. The rates therefore are nested observations, not independent samples. The larger run covers all 100 allowlisted environments rather than 72, but still has only one actor-model stratum. Neither nomination rate should be extrapolated to the full source or a future corpus.

Per-environment and model statistics are in `environment_model_stats.json`. Each environment contributes exactly five rows. Twenty-one environments have zero provenance-only format survivors; 36 have one, 21 have two, 15 have three, and seven have four. Twenty-four environments contribute at least one row to the full-reading queue, and six contribute a nomination.

## SOTOPIA recommendation

Retain SOTOPIA as a research source for a small, human-curated **SFW negotiation/social-state support slice**. Its strengths are explicit constraints, counterproposals, and occasional observable resolution. Its limitations are formulaic GPT-4 politeness, weak character differentiation, agreement tails, profile/scenario drift, one model stratum, current turn/event incompatibility, and unresolved provenance obligations.

Do not enable it in production data preparation yet. The next permitted step is manual review of the six nominations and selected near misses, followed by a separate decision about an event-aware adapter and rights assessment. It does not solve adult RP, long-form character continuity, or general open-ended RP quality.

## Original-author seed corpus

The complete design is in [original_author_seed_spec.md](original_author_seed_spec.md), with machine-readable contracts for character cards, loader-compatible conversations, provenance, and quality reviews under `schemas/`.

V1 recommendation:

- 16 original characters, exactly three anchor scenes per character.
- 48 scenes: 28 SFW, 12 mature nonsexual relationship scenes, and eight adult-capable scenes.
- At least four authors, with no author contributing more than 12 scenes or four cards.
- At least 12 settings, 12 SFW negotiation/social-state scenes, eight enacted state changes, and eight respected refusal/boundary cases.
- 12 short, 24 medium, and 12 long scenes; the long band has 16–24 exchanges.
- Adult-capable characters also appear in ordinary SFW scenes, and adult contexts are limited to clearly fictional 21-plus peers without authority/dependency dynamics.

## Exact authorization gates

**Synthetic generation remains unauthorized** until all 16 cards and at least 40 of the planned 48 human-written scenes are complete and independently accepted. The accepted set must contain at least 24 SFW scenes, eight mature nonsexual scenes, six adult-capable scenes, eight enacted state changes, and eight respected boundary/refusal cases across at least 12 settings and four authors. Every used card and scene must have a signed ownership declaration, explicit training/modification/redistribution permissions, matching source hashes, complete revision history, and human rights status `reviewed`. Every accepted scene must pass two independent reviewers with the rubric thresholds, have no unresolved hard flag, and remain separate from core SFT-v1. The held-out suite and denylist must already be frozen and hashed. A human must approve the exact teacher model, applicable terms, prompt template, pilot size, and storage plan.

**QLoRA remains unauthorized** until a separately versioned immutable dataset manifest lists only rows with row-level rights review and two-reviewer quality acceptance after final deduplication. No row may be pending, held, provenance-blocked, or automatically promoted. The held-out suite must be uncontaminated, RP plus normal-assistant evaluation criteria must be approved, dataset statistics/failure audit must be reviewed, and the user must explicitly authorize that exact manifest and training run. Passing the authoring or synthetic-generation gate does not authorize QLoRA.

## Local inspection artifacts

The raw bounded samples and complete manual-review packet remain in the ignored local directory `data/sft/rp_v1/generated/sotopia_500_20260913/`. That directory contains collection provenance, unchanged-v2 diagnostics, semantic ranking features, all 28 full-read judgments, the six-row locator shortlist, 12 near misses, per-environment/model statistics, comparison data, and verification. Raw dialogue is not added to this report directory.

Validation completed with 173 relevant RP pipeline tests passing and one platform-specific test skipped under the available Python 3.12 runtime. All four JSON Schema documents parse and satisfy the required contract-field checks. The final artifact verifier confirms 500 unique episode IDs, 100 environments, a maximum of five rows per environment, zero documented-exclusion intersections, zero approved/provenance-reviewed/training-ready rows, and no retained private target fields.
