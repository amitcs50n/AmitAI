# RP seed V1 held-out evaluation protocol

The structural plan and reserved scenario-family names are frozen before scene authoring. Actual held-out cases have not been written, so the synthetic authorization gate remains closed until case content, expected state transitions, scoring assertions, a denylist, and their immutable hashes are complete and approved.

The suite uses 64 unique case slots with overlapping slices defined in `heldout_evaluation.yaml`. It covers character continuity, 20-plus-turn roleplay, user agency, state memory, consequences, repetition, SFW roleplay, mature nonsexual roleplay, adult-capable roleplay, refusal, switch-back behavior, and normal-assistant reasoning regression. Every character receives at least two fidelity cases. The long-form cases span at least eight characters and multiple authors' styles.

Evaluation-only events, conflicts, user turns, state variables, expected answers, evaluator notes, and rewards stay outside character cards, training scenes, synthetic prompts, and author examples. The same character can appear in evaluation, but the scenario family and evaluation-only facts remain unseen. Before synthetic work, case IDs and hashes are added to an exact-match denylist and scenario/event/state descriptions receive semantic overlap review against the full authoring corpus.

Human evaluation uses the same quality dimensions as scene review plus slice-specific assertions. Agency, ignored refusal, participant-age ambiguity in adult-capable cases, real-person imitation, and character leakage after switch-back are zero-tolerance failures. Evaluator reasoning and rewards are never target dialogue.

Any change to a frozen case creates a new suite revision, hashes the new manifest, and invalidates prior benchmark comparisons. The old manifest stays archived. The normal-assistant regression tolerance and exact model/run configuration require separate approval at training authorization; this plan does not modify them.
