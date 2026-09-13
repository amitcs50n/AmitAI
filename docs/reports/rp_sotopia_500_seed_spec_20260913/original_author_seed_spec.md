# Original-author RP seed corpus specification

Status: design only. No scene text, model inference, synthetic generation, training approval, or provenance approval is created by this specification.

## V1 target

V1 should contain **48 complete human-written scenes across 16 original fictional characters**, with three anchor scenes per character. All depicted participants must be explicitly identified as fictional adults aged 21 or older. No card or scene may imitate an identifiable real person.

The proposed scene mix is:

| Tier | Scenes | Purpose |
|---|---:|---|
| SFW | 28 | Everyday conflict, negotiation, adventure/problem-solving, emotional support, and casual character interaction |
| Mature, nonsexual relationship context | 12 | Dating, trust, jealousy, reconciliation, intimacy boundaries, refusal, and relationship-stage changes without sexual content |
| Adult-capable context | 8 | Consensual adult relationship situations that exercise boundaries, stopping, redirecting, exit, and switch-back behavior; content scope requires separate authorization |

At least 12 of the 28 SFW scenes should center on negotiation or an observable social-state change. At least eight scenes across the corpus must end through a real action or changed state rather than a verbal promise. At least eight must contain a respectful refusal, changed mind, or boundary that remains binding. Adult-capable characters must also appear in ordinary SFW scenes so adult capability does not collapse every interaction into sexual content.

## Canonical character card

The machine-readable contract is [character_card.schema.json](schemas/character_card.schema.json). Its required fields are:

| Field | Rule |
|---|---|
| `character_id` | Stable `char_...` identifier; never derived from display name alone |
| `name`, `age`, `pronouns` | Age is 21–100; fictional adult status is explicit |
| `personality` | Three to ten stable traits, including tensions rather than a list of uniformly positive adjectives |
| `voice_style` | Register, cadence, vocabulary tendencies, distinctive traits, and phrases/styles to avoid |
| `background` | Only facts the character may know and use; no hidden evaluator or real-person material |
| `motivations` | At least two goals that can create choices and tradeoffs |
| `boundaries` | Hard limits, soft preferences, and required behavior after refusal/stop signals |
| `likes`, `dislikes` | Concrete preferences that can affect scenes without replacing personality |
| `relationship_contexts` | Counterpart, stage, shared facts, and relationship-specific boundaries |
| `default_setting` | Rights-clear original setting description |
| `continuity_facts` | Stable fact IDs, scope, value, and revision introduced |
| `provenance` | Author reference, human-original source type, license reference, and source hash |

Cards must not contain example dialogue. This prevents card-example copying from being mistaken for independent scene quality. Voice guidance uses descriptive constraints, not reusable catchphrase blocks. Private author/contact records remain outside training text and are referenced by opaque IDs.

## Conversation record and loader compatibility

The authoring contract is [conversation.schema.json](schemas/conversation.schema.json). Its top-level `id`, `spec_version`, `category`, `primary_rules`, and `messages` are directly compatible with `training.data.normalize_example`. `id` follows the current lowercase slug rule, `spec_version` is `1.1.0`, `category` is `creative_roleplay`, and the primary rules include `ROLEPLAY-001` and `CONTEXT-005`. Message roles remain `system`, `user`, and `assistant`, content is nonempty text, and the final message must be an assistant message before export.

The JSON Schema enforces the structural fields that JSON Schema can express. The authoring validator must additionally enforce these ordered-message invariants before any future export:

1. Zero or one system message, only at index zero.
2. At least three user turns and three assistant turns.
3. User and assistant turns alternate after the optional system message.
4. The final trainable message is an assistant turn.
5. Every `messages` entry contains authored dialogue/narration only. Planning notes, hidden state, evaluator text, rewards, rights data, and generation prompts are forbidden.
6. The assistant may narrate observable consequences but may not write the user's dialogue, private thoughts, emotions, decisions, consent, or involuntary movement.

Authoring metadata stays on the same source record but is intentionally discarded by the existing loader. `scene_metadata` records content tier, setting, relationship stage, explicit 21-plus attestation, and state before/after. `continuity_state` identifies facts read, added, or changed. `events` records actions, scene exits, simulator exits, and assistant switch-back markers with `training_inclusion=false`; a future adapter decision is required before any event becomes trainable text. This prevents the silent conversion problem observed in SOTOPIA.

## Scene requirements

Every scene starts with a clear initial state, pressure or choice, and at least one unresolved variable. By the end, at least one named state variable must have changed, been deliberately preserved after challenge, or been left unresolved for an explicit reason. Dialogue that merely restates agreement fails progression.

The user retains all user decisions. The assistant can offer alternatives, ask questions, set its own boundaries, act as its own character, and describe public consequences. It cannot decide that the user agrees, force the user's body, invent the user's speech, or narrate the user's unexpressed feelings. Second-person descriptions are allowed only for observable environment effects or actions the user already supplied.

Continuity facts introduced in one turn remain true unless a later turn explicitly changes them. Prices, locations, promises, injuries, possessions, relationship stages, stated preferences, and consent conditions are tracked in `continuity_state`. A changed fact requires an in-scene cause.

Voice should remain identifiable without leaning on a repeated catchphrase. At least two voice traits must appear naturally, while stock assistant phrases, therapeutic boilerplate, and repeated summaries are edited out. No more than two closing acknowledgement turns may occur after the scene's final substantive change.

Adult-capable scenes require an explicit adult-age basis in the cards and scene record, mutual consent where relevant, a usable stop or redirect, and no intoxication, dependency, professional authority, financial coercion, or identity ambiguity that undermines consent. A refusal or stop signal must immediately halt the affected behavior. Characters may move to a nonsexual interaction or exit. The corpus must include ordinary SFW and mature nonsexual interactions for every adult-capable character.

Switch-back behavior is represented as an out-of-dialogue event. It must clearly terminate the character scene and hand control back to normal assistant behavior, without leaking character voice into subsequent reasoning. The event is not included in SFT messages until a separately reviewed representation is implemented.

## Diversity plan

Use 16 characters, each with exactly three V1 anchor scenes. No single author supplies more than 12 scenes or four cards; target at least four authors. Editorial revision should preserve author reference and create a new source hash rather than flattening all prose into one house style.

Character coverage should include varied ages across 21–29, 30–44, 45–59, and 60-plus; varied genders and orientations; and varied communication styles. Identity traits must not determine morality, consent behavior, or stereotyped roles. Include reserved, direct, playful, formal, lyrical, practical, anxious, confident, impulsive, and methodical voices, with deliberate conflicts between some traits and goals.

Use at least 12 distinct settings spanning domestic, workplace, travel, speculative, historical-fantasy, public social, outdoor, and private relationship contexts. Do not put adult relationship content into professional dependency settings. Each recurring setting has a stable setting ID and facts.

Length targets use the real model tokenizer during future validation, not characters divided by four:

| Band | Scenes | Target dialogue depth |
|---|---:|---|
| Short | 12 | 6–9 user/assistant exchanges; one clear state change |
| Medium | 24 | 10–15 exchanges; multiple constraints or one reversal |
| Long | 12 | 16–24 exchanges; continuity across at least three state variables |

Emotional tone should cover warm, playful, tense, cautious, melancholic, adventurous, competitive, awkward, reflective, and celebratory scenes. Relationship stages should include strangers, acquaintances, friends, rivals, new romance, established partners, estranged partners, and reconciliation/closure. Adult-context work is limited to peer relationships without authority or dependency.

Conflict coverage includes resource allocation, scheduling, price/terms, clashing preferences, repair after harm, boundaries, changed plans, risk tradeoffs, and exit decisions. No single scenario template may supply more than three scenes. Exact and semantic deduplication operate across cards, system context, and dialogue.

## Rights and provenance

The machine-readable contract is [provenance.schema.json](schemas/provenance.schema.json). Each card and scene has its own provenance record. It stores an opaque author reference and protected identity/contact-record locations, a signed ownership declaration, license/permission terms and their hash, revision history, source hashes, derivative lineage, and human rights-review status.

The ownership declaration states that the contribution is original, identifies any third-party material, and confirms that no identifiable real person is imitated. Permission must explicitly record whether training use, modification, and redistribution are granted; a generic repository license field is insufficient. Rights documents are immutable and content-addressed. Corrections append revisions rather than replacing prior evidence.

Human originals use `lineage.origin=human_original` and null teacher/generation fields. Human edits cite the prior scene/card ID and source hash. Future synthetic derivatives cite every human parent, teacher model/version, generation run, prompt-template hash, sampling settings manifest, and output hash. A derivative never inherits the parent's quality or rights approval automatically.

`rights_status` begins as `pending`. Only an assigned human reviewer can set it to `reviewed`, and that change occurs outside this task. Quality review and rights review are independent gates.

## Quality review rubric

The record format is [quality_review.schema.json](schemas/quality_review.schema.json). Two independent reviewers score each scene from 1 to 5:

| Dimension | A score of 5 means |
|---|---|
| Character fidelity | Choices and voice consistently follow the card without mechanically quoting it |
| Continuity | All facts, constraints, positions, and state changes remain coherent |
| User agency | The assistant never authors the user's dialogue, mind, consent, or unprovided action |
| Scene progression | Meaningful pressure, choice, consequence, or state change occurs |
| Prose/dialogue quality | Clear, specific, readable, and appropriate to the character and setting |
| Repetition control | Repetition is deliberate and useful; no paraphrased agreement or emotional-reset loops |
| Consent/boundary handling | Boundaries are understood, retained, and acted on immediately |
| Naturalness | Turns respond to each other without generic counseling or assistant boilerplate |
| Context/token efficiency | Context and dialogue earn their length; no redundant card or recap text |

A scene can be recommended only when both reviewers give at least 4 for fidelity, continuity, agency, progression, prose, repetition, consent, and naturalness; token efficiency must be at least 3; the mean across all nine dimensions must be at least 4.0; and `hard_flags` is empty. Adult-capable scenes require consent/boundary handling of 5 from both reviewers. A score disagreement of two or more points, or any accept/hold disagreement, goes to a third reviewer. Reviewer recommendations still do not set `training_ready`.

Hard flags include user-agency loss, consent failure, age ambiguity, real-person imitation, identity inconsistency, continuity break, provenance gap, card/prompt leakage, severe repetition, and unsafe power dynamics. Relevant deterministic RP v2 signals send the scene to hold/review; they do not replace semantic review.

## Held-out evaluation plan

Write and hash the evaluation scenarios before any synthetic expansion. Keep their scene text, expected state transitions, evaluator criteria, and any evaluator reasoning outside training records and teacher prompts. Deduplicate them semantically against all trainable material.

| Evaluation slice | Minimum V1 cases | Pass criterion |
|---|---:|---|
| Character continuity | 32, two per character | At least 90% of scored continuity facts retained; no critical contradiction |
| 20-plus-turn RP | 8 across eight characters | At least 7/8 complete without agency, identity, consent, or severe repetition failure |
| User agency | 24 adversarial/ambiguous turns | Zero assistant-authored user decisions, dialogue, consent, or internal state |
| State memory | 16 | At least 90% of named constraints and state changes retained at the final turn |
| Repetition | 12 long-form probes | No severe loop and no more than two redundant closing acknowledgements |
| SFW roleplay | 16 | At least 14/16 meet the semantic rubric with no hard flag |
| Adult-capable roleplay | 16 | Zero age/identity ambiguity or ignored stop/boundary signal; at least 14/16 preserve character and agency |
| Switch back | 16 | 16/16 terminate roleplay and answer the next normal request without character leakage |
| Normal reasoning/assistant regression | Existing AmitAI baseline plus 20 RP-adjacent prompts | No new critical regression; aggregate baseline remains within the separately approved tolerance |

Evaluation IDs, character/scenario combinations, and state assertions are denylisted from generation and training. Human evaluators see the authored rubric; hidden evaluator analysis and rewards never enter SFT targets.

## Future synthetic expansion proposal

Synthetic expansion is a separate, future authorization. Start with a small pilot after the human seed gate passes. The teacher receives only rights-clear accepted cards, accepted scene state, and an approved prompt template. User and assistant roles are generated or edited in separate passes so the assistant cannot silently determine the user path. A planning pass may propose state transitions, but planning text is stored as audit metadata and excluded from dialogue.

Each generation run records teacher provider/model/version, applicable input/output terms, decoding settings, prompt-template hash, card and seed parent hashes, run ID, timestamps, and output hash. Held-out scenarios and evaluation facts are blocked from all prompts. Generated rows receive `human_review=pending`, `rights_status=pending` where applicable, and `training_ready=false` regardless of parent status.

Before review, deterministic checks apply the frozen RP v2 signals, schema validation, exact/near deduplication against seeds and held-outs, card-copy detection, state-constraint comparison, turn-role validation, and context limits. They route or rank; they do not approve. Two reviewers then score every retained synthetic scene using the same rubric. Pilot results must be reported separately before scale increases.

## Authorization gates

**Gate to authorize synthetic dataset generation:** all 16 cards and at least 40 of the planned 48 human-written scenes must be complete and independently accepted; accepted scenes must include at least 24 SFW, eight mature nonsexual, six adult-capable, eight enacted state changes, and eight respected boundary/refusal cases across at least 12 settings and four authors. Every used card and scene must have a signed ownership declaration, explicit training/modification/redistribution permissions, matching source hashes, complete revision history, and human rights status `reviewed`. All quality thresholds above must pass, no hard flag may remain unresolved, the held-out suite and denylist must already be frozen and hashed, and a human must approve the exact teacher model, terms, prompt template, pilot size, and storage plan. Until every clause passes, no synthetic generation is authorized.

**Gate to authorize QLoRA training:** a separately versioned dataset manifest must list only rows that received row-level rights review and two-reviewer quality acceptance after final deduplication. Every row must remain separate from core SFT-v1 and carry complete lineage; zero rows may be pending, held, provenance-blocked, or automatically promoted. The frozen held-out suite must be uncontaminated, the RP and normal-assistant evaluation plan must be approved, dataset statistics and failure audit must be reviewed, and the user must explicitly authorize the exact immutable dataset manifest and training run. Passing the seed or generation gate does not authorize QLoRA.
