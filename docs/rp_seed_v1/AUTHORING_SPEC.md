# RP seed V1.2 authoring specification

## Scope and ownership

Authors will write 48 complete multi-turn scenes from the approved allocation and canonical cards. Each contribution must be original, all depicted people must be fictional adults aged 21 or older, and no character may imitate an identifiable real person. Do not paste or adapt character-card examples, scraped dialogue, stories, model outputs, editor instructions, diagnostic-pilot prose, or third-party prose. Cards intentionally contain descriptive voice constraints and no example dialogue. Diagnostic scenes may inform this specification and its briefs, but no diagnostic scene may be copied, paraphrased, revised, relabeled, or migrated into the human gold corpus.

Each author receives an opaque author ID only after identity and contact records are stored outside training data. A scene submission is incomplete until its separate provenance record contains the final content hashes, signed ownership declaration, explicit permissions, and full revision history.

## Conversation contract

Future submissions follow `conversation.schema.json`. The loader-facing fields are `id`, `spec_version`, `category`, `primary_rules`, and `messages`, matching `training.data.normalize_example`. Use `spec_version: 1.2.0`, `category: creative_roleplay`, and include `ROLEPLAY-001` and `CONTEXT-005` in `primary_rules`.

After an optional system message at index zero, user and assistant messages alternate. Provide at least three user and three assistant turns, and end on an assistant message. Messages contain authored dialogue and observable narration only. Rights data, planning notes, hidden state, evaluation criteria, rewards, raw prompts, and private author records remain outside `messages`. Events such as a scene exit or switch back remain explicit metadata with `training_inclusion: false`.

Every allocation and submitted scene carries a user-role contract. `user_role` defines only the role needed for the scenario. `known_user_facts` plus any brief-authorized `shared_history_facts` form the complete allowlist of user-related facts available at the opening; later fictional user turns may add explicit source facts. The corpus baseline plus each scene's `forbidden_user_assumptions` prohibits assistant-invented appearance, gender unless explicitly specified, attraction, thoughts, emotions, decisions, past history beyond the allowlist, unprovided physical actions, and unexpressed consent. Scenario-specific prohibitions may add expertise, risk tolerance, forgiveness, disclosure, or relationship expectations. Authors may not infer a prohibited fact from tone, silence, genre convention, or the character card.

### Offline user turns and live user control

In an offline authored training transcript, the human author writes both sides. The author may write the fictional user's explicit dialogue, observable actions, and choices as original source material in a user turn. The character may rely on those facts only after that user turn explicitly states them; it may not anticipate them or retroactively place them in the user's mind or body.

This authoring permission is not permission for the assistant to control a live roleplay user. In live roleplay, only the user supplies the user's dialogue, actions, choices, private state, and consent. In both settings, the assistant must not invent user thoughts, emotions, attraction, appearance, gender unless explicitly specified, unstated history, unprovided actions, or unexpressed consent.

## Scene construction

Every scene begins with a concrete initial state, at least one constraint, and an unresolved choice. It ends after an observable consequence: an item moves, a term changes, an action starts or stops, a boundary remains binding, a relationship state changes, or the characters explicitly preserve an unresolved issue. Agreement without an enacted or recorded result does not count as progression.

Within each authored exchange, the user turn is the sole source for new user dialogue, decisions, observable actions, private disclosures, and consent. The assistant may describe public consequences and its own character's choices. Normal second-person narration is acceptable only for environmental effects or actions the user has already supplied. Offers must preserve meaningful alternatives, including refusal and exit.

Natural questions are allowed and often preferable to scripted choice menus. Vary interaction patterns across and within scenes: use direct natural questions, consequences that invite a response, pauses or silence, incomplete proposals, interrupted choices, and a single clear alternative when appropriate. Reserve multi-option menus for genuine branch points. Do not replace ordinary conversation with repeated menus merely to demonstrate agency, and do not turn questions into serial interrogation.

Track locations, possessions, injuries, prices, deadlines, promises, relationship stage, stated preferences, pronouns, boundaries, and consent conditions. A fact changes only through an in-scene cause, and the final state record must match the dialogue. Explicitly restate state only when a fact changed, safety requires confirmation, a misunderstanding is being repaired, or a resource or commitment must remain precise. Do not repeatedly summarize what the user chose, what remains allowed, what is open or closed, or the current relationship state. Rubric-like or meta phrases such as “the user's chosen boundary,” “your stated preference,” and “the agreed state” are prohibited inside dialogue or narration unless the wording is genuinely natural in context.

At least two card voice traits should appear naturally. Remove generic counseling, paraphrased agreement, recurring gesture templates, emotional resets, and redundant summaries. Allow no more than two closing acknowledgement turns after the final substantive change.

## Content tiers

SFW scenes cover ordinary problem solving, adventure, negotiation, conflict, humor, care, and companionship. Mature nonsexual scenes may address grief, betrayal, relationship strain, accountability, jealousy, or intimacy boundaries without sexual content. Mature scenes involving an established relationship should use one or two neutral brief-authorized shared-history facts where needed, such as a place visited before, an old project, a shared routine, a remembered object, or a prior harmless disagreement. These facts must not define current feelings, forgiveness, attraction, desired outcome, or willingness to reconcile.

V1 adult-capable gold authoring is limited to adult fictional participants. Consensual romantic or sexual context, flirtation, affectionate and non-explicit physical intimacy, and clearly established sexual intent are allowed. A scene may progress to a clear fade-to-black boundary. Graphically explicit sexual-act prose is outside V1. The V1 goal is adult relationship continuity, character fidelity, natural desire and affection, boundaries, redirection, stopping, and user agency—not explicit sexual description. This ceiling does not decide the scope of any separately governed future adult-extension dataset.

Adult-capable scenes require clearly identified adults, equal authority, sobriety, specific and revocable choices, and an immediate stop or redirect path. Exclude ambiguous age or identity, intoxication, dependency, professional or financial leverage, coercive contracts, forced positioning, and refusal treated as persuasion. When consent is unclear, the character pauses and offers conversation, distance, a nonsexual activity, or exit. Adult-capable characters must retain their full voice and appear in SFW scenes as allocated.

The adult-capable allocation uses five dramatic structures: two boundary/check-in-heavy, two playful/affectionate, two desire-led, one awkward reconnection, and one emotionally complicated scene. Consent must remain visible through responsive action, established shared context, and immediate respect for change or refusal. Do not turn all eight scenes into repeated consent questionnaires.

Mature nonsexual scenes do not require wrongdoing or reconciliation. Authors should preserve reasonable incompatibility, grief, envy, changing friendship, career conflict, emotional distance, and separate futures where the brief calls for them. A scene may progress by clarifying a durable difference or choosing not to reconcile.

## Length and diversity

Short scenes target 12–18 total turns and 700–1,200 tokens. Medium scenes target 20–30 turns and 1,400–2,400 tokens. Long scenes target 32–48 turns and 2,600–4,200 tokens. Token counts use the future approved model tokenizer during authoring review; these ranges guide editing and do not change model configuration.

Every long scene requires three to five intermediate state beats or complications between its opening and final decision. Use environmental change, resource loss, new evidence, interruption, consequence, location change, a new constraint, or a failed attempt. Each beat must materially alter the situation without preselecting the user's response. Do not fill a 32–48-turn scene by repeatedly revisiting the same choice.

Long-scene review checks voice separately in the opening third, middle third, and final third. At each checkpoint, assess cadence, humor, noticed details, vocabulary, emotional behavior, and decision style. Repeated character-specific keywords do not establish voice fidelity when the turn architecture or reactions have drifted into a generic assistant pattern.

Within those totals, SFW uses 7 short, 14 medium, and 7 long scenes; mature nonsexual uses 3, 6, and 3; adult-capable uses 2, 4, and 2. This balance prevents content tier from becoming a proxy for response length.

Preserve the roster's age, gender, communication, genre, and personality diversity. Do not flatten prose into one editorial voice or assign morality and consent behavior by identity. Each author slot owns four cards and twelve scenes so no single author dominates V1. Revisions keep the original author reference and append lineage and hashes.

## Submission sequence

1. Replace the allocated author slot with a verified opaque author ID and create a pending provenance record.
2. Review the current card and scene brief; record questions without adding them to dialogue.
3. Write the scene and explicit initial/final continuity states. For long scenes, record the three to five intermediate state beats. For mature established relationships, record one or two brief-authorized neutral shared-history facts. For adult-capable scenes, confirm the V1 fade-to-black/no-graphic-prose ceiling.
4. Run schema, frozen RP v2, card-copy, exact/semantic overlap, turn-role, and context checks. These checks route issues and never approve a row.
5. Complete the non-gating diagnostic pre-review fields for generic-assistant leakage, procedural-dialogue tendency, excessive state recap, repeated option-menu behavior, and same-model voice convergence. These observations are not reviewer approvals and cannot satisfy a review threshold.
6. For long scenes, complete voice-drift observations for the opening, middle, and final thirds before independent review.
7. Resolve author revisions and record each revision hash.
8. Obtain two independent quality reviews and separate human rights review.
9. Keep `training_ready: false`. Acceptance by reviewers is evidence for the later authorization checklist, not permission to train.
