# RP seed V1 authorization checklist

Current state: **design only**. Every box is intentionally unchecked. Structural validation cannot check a box.

## Gate to authorize synthetic dataset generation

- [ ] All 16 character cards completed and reviewed by humans.
- [ ] At least 40 of 48 human-written scenes accepted by two independent reviewers.
- [ ] Accepted set includes at least 24 SFW scenes.
- [ ] Accepted set includes at least 8 mature nonsexual scenes.
- [ ] Accepted set includes at least 6 adult-capable scenes.
- [ ] Accepted set includes at least 8 enacted state changes.
- [ ] Accepted set includes at least 8 respected boundary/refusal cases.
- [ ] Accepted set spans at least 12 settings and at least 4 verified human authors.
- [ ] Every used card and scene has a signed original-work ownership declaration.
- [ ] Every used card and scene explicitly grants training use, modification, and redistribution.
- [ ] Every final revision has matching content hashes and complete revision history.
- [ ] Human rights review status is `reviewed` for every used card and scene.
- [ ] All review thresholds pass and no hard flag remains unresolved.
- [ ] Held-out cases, scoring assertions, manifest, and denylist are frozen and hashed.
- [ ] Exact and semantic overlap checks show no held-out leakage.
- [ ] A human approves the exact teacher model/version and applicable terms.
- [ ] A human approves the exact prompt template, pilot size, and storage/retention plan.
- [ ] The user explicitly authorizes the bounded synthetic pilot.

`synthetic_generation_authorized: false`

## Gate to authorize QLoRA training

- [ ] Any synthetic expansion used in the dataset has completed its own deterministic checks, two-reviewer quality review, and row-level rights review.
- [ ] A separately versioned immutable dataset manifest exists.
- [ ] The manifest contains only final rows with row-level human rights review and two-reviewer quality acceptance.
- [ ] Final exact, near, card-copy, and semantic deduplication is complete.
- [ ] Every row has complete human or synthetic lineage and immutable source hashes.
- [ ] Zero rows are pending, held, quarantined, provenance-blocked, rejected, or automatically promoted.
- [ ] The RP dataset remains separate from frozen core SFT-v1.
- [ ] The held-out suite is hashed, uncontaminated, and excluded from trainable material.
- [ ] RP, adult-capable, switch-back, and normal-assistant evaluation plans are approved.
- [ ] Dataset counts, source/author distribution, content-tier distribution, token statistics, and failure audit are reviewed.
- [ ] The exact QLoRA configuration and run environment receive separate authorization without changing the frozen baseline by implication.
- [ ] The user explicitly authorizes the exact immutable dataset manifest and exact training run.

`qlora_training_authorized: false`

Passing the authoring plan, scene review, rights review, or synthetic gate does not authorize training. No script in this authoring infrastructure may change either authorization value.
