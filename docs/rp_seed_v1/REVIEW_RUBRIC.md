# RP seed V1 human review rubric

Two independent reviewers score the final hashed revision from 1 to 5. They review without seeing each other's decision. A disagreement of two or more points on any dimension, or an accept versus revise/reject disagreement, requires a third reviewer.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Character fidelity | Generic or contradicts the card | Mostly recognizable with slips | Voice and choices are consistently specific without quoting the card |
| Continuity | Critical contradictions | Minor recoverable lapse | Facts, positions, constraints, and changes stay coherent |
| User agency | Authors user speech, mind, consent, or movement | One uncertain attribution | User retains every decision and action; alternatives remain real |
| Scene progression | Repeats or resets | Partial change with weak consequence | Choices produce a clear, enacted, persistent consequence |
| Dialogue quality | Stilted or incoherent | Functional but generic | Responsive, specific, and character-shaped |
| Prose quality | Confusing or excessive | Readable with uneven detail | Clear, controlled, vivid, and appropriate to the scene |
| Repetition control | Severe loop | Noticeable recap or gesture reuse | Repetition is deliberate and earns its place |
| Naturalness | Model/editor boilerplate dominates | Some generic phrasing | Turns respond naturally without therapeutic or assistant scripts |
| State tracking | State is lost or silently changed | Most state retained | Every relevant state change has a cause and matches the final record |
| Consent/boundary handling | Stop, refusal, or power risk fails | Correct but mechanical | Limits are understood, retained, and acted on without pressure |
| Context/token efficiency | Large redundant context | Some removable material | Context and turns earn their length |
| Contamination control | Card, prompt, HTML, or editor leakage | Suspicious echo requiring revision | No card-copy, markup, placeholders, instructions, or source artifacts |

## Acceptance threshold

Both reviewers must score character fidelity, continuity, user agency, scene progression, dialogue quality, prose quality, repetition control, naturalness, state tracking, consent/boundary handling, and contamination control at least 4. Context/token efficiency must be at least 3. Each reviewer's mean across all twelve dimensions must be at least 4.0. Adult-capable scenes require a 5 for consent/boundary handling from both reviewers. `hard_flags` must be empty.

Hard flags are user-agency loss, consent failure, age ambiguity, identity inconsistency, real-person imitation, continuity break, provenance gap, card or prompt leakage, severe repetition, and unsafe power dynamics. Any hard flag blocks acceptance until a new hashed revision receives fresh reviews. Deterministic RP v2 signals send a scene to review or quarantine; they do not replace human judgment.

Reviewers choose `accept`, `revise`, or `reject`. Passing review sets only `accepted_by_reviewers` in the review record. It does not set rights to reviewed and cannot set `training_ready`.

## Reject or revise

Automatically reject a submitted revision when it contains copied or unlicensed material, identifiable real-person imitation, knowingly false ownership claims, sexual content involving a minor or ambiguous-age participant, or coercive/abusive content presented as valid consent. Preserve the rejected record and hash; an author may submit a wholly new original work under a new revision or item ID as directed by the rights reviewer.

Use revise-and-resubmit for repairable craft or scope failures: weak progression, generic voice, continuity errors, agency ambiguity without an enacted violation, excessive repetition, overlong context, malformed role order, accidental markup/editor notes, or a boundary response that is safe but unclear. The new content receives a new hash and two fresh independent reviews. Reviewers may not edit a score in place after the content changes.
