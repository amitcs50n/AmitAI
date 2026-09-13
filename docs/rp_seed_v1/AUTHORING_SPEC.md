# RP seed V1 authoring specification

## Scope and ownership

Authors will write 48 complete multi-turn scenes from the approved allocation and canonical cards. Each contribution must be original, all depicted people must be fictional adults aged 21 or older, and no character may imitate an identifiable real person. Do not paste or adapt character-card examples, scraped dialogue, stories, model outputs, editor instructions, or third-party prose. Cards intentionally contain descriptive voice constraints and no example dialogue.

Each author receives an opaque author ID only after identity and contact records are stored outside training data. A scene submission is incomplete until its separate provenance record contains the final content hashes, signed ownership declaration, explicit permissions, and full revision history.

## Conversation contract

Future submissions follow `conversation.schema.json`. The loader-facing fields are `id`, `spec_version`, `category`, `primary_rules`, and `messages`, matching `training.data.normalize_example`. Use `spec_version: 1.1.0`, `category: creative_roleplay`, and include `ROLEPLAY-001` and `CONTEXT-005` in `primary_rules`.

After an optional system message at index zero, user and assistant messages alternate. Provide at least three user and three assistant turns, and end on an assistant message. Messages contain authored dialogue and observable narration only. Rights data, planning notes, hidden state, evaluation criteria, rewards, raw prompts, and private author records remain outside `messages`. Events such as a scene exit or switch back remain explicit metadata with `training_inclusion: false`.

## Scene construction

Every scene begins with a concrete initial state, at least one constraint, and an unresolved choice. It ends after an observable consequence: an item moves, a term changes, an action starts or stops, a boundary remains binding, a relationship state changes, or the characters explicitly preserve an unresolved issue. Agreement without an enacted or recorded result does not count as progression.

The user controls the user's words, decisions, private thoughts, emotions, consent, and movement. The assistant may describe public consequences and its own character's choices. Normal second-person narration is acceptable only for environmental effects or actions the user has already supplied. Offers must preserve meaningful alternatives, including refusal and exit.

Track locations, possessions, injuries, prices, deadlines, promises, relationship stage, stated preferences, pronouns, boundaries, and consent conditions. A fact changes only through an in-scene cause, and the final state record must match the dialogue. At least two card voice traits should appear naturally. Remove generic counseling, paraphrased agreement, recurring gesture templates, emotional resets, and redundant summaries. Allow no more than two closing acknowledgement turns after the final substantive change.

## Content tiers

SFW scenes cover ordinary problem solving, adventure, negotiation, conflict, humor, care, and companionship. Mature nonsexual scenes may address grief, betrayal, relationship strain, accountability, jealousy, or intimacy boundaries without sexual content. Adult-capable scenes test consensual adult relationship behavior at an abstract authoring level; their exact content scope requires later human authorization.

Adult-capable scenes require clearly identified adults, equal authority, sobriety, specific and revocable choices, and an immediate stop or redirect path. Exclude ambiguous age or identity, intoxication, dependency, professional or financial leverage, coercive contracts, forced positioning, and refusal treated as persuasion. When consent is unclear, the character pauses and offers conversation, distance, a nonsexual activity, or exit. Adult-capable characters must retain their full voice and appear in SFW scenes as allocated.

## Length and diversity

Short scenes target 12–18 total turns and 700–1,200 tokens. Medium scenes target 20–30 turns and 1,400–2,400 tokens. Long scenes target 32–48 turns and 2,600–4,200 tokens. Token counts use the future approved model tokenizer during authoring review; these ranges guide editing and do not change model configuration.

Preserve the roster's age, gender, communication, genre, and personality diversity. Do not flatten prose into one editorial voice or assign morality and consent behavior by identity. Each author slot owns four cards and twelve scenes so no single author dominates V1. Revisions keep the original author reference and append lineage and hashes.

## Submission sequence

1. Replace the allocated author slot with a verified opaque author ID and create a pending provenance record.
2. Review the current card and scene brief; record questions without adding them to dialogue.
3. Write the scene and explicit initial/final continuity states.
4. Run schema, frozen RP v2, card-copy, exact/semantic overlap, turn-role, and context checks. These checks route issues and never approve a row.
5. Resolve author revisions and record each revision hash.
6. Obtain two independent quality reviews and separate human rights review.
7. Keep `training_ready: false`. Acceptance by reviewers is evidence for the later authorization checklist, not permission to train.
