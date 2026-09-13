# Rights-clear RP seed V1

This is the authoring and governance specification for a future gold RP seed. It creates no training records, contains no scene dialogue, and authorizes neither synthetic generation nor training.

## V1 allocation

- 16 original fictional adults, each assigned three scenes.
- 48 planned human-written scenes: 28 SFW, 12 mature nonsexual, and 8 adult-capable.
- 12 short, 24 medium, and 12 long scenes.
- Four author slots with four cards and twelve scenes each.
- At least 12 settings, with unique scenario families across the allocation.
- Eight adult-capable characters; every one also has an ordinary SFW scene.

Author slots are workload placeholders, not verified identities. All rights fields remain pending, all permission fields remain false or pending, and every quality decision remains pending until an assigned human completes it.

## Source of truth

- [Character roster](../../data/sft/rp_seed_v1/authoring/roster.yaml)
- [Scene allocation](../../data/sft/rp_seed_v1/authoring/scene_allocation.yaml)
- [Held-out evaluation plan](../../data/sft/rp_seed_v1/authoring/heldout_evaluation.yaml)
- [Machine-readable schemas](../../data/sft/rp_seed_v1/authoring/schemas/)
- [Blank authoring records](../../data/sft/rp_seed_v1/authoring/templates/)
- [Authoring specification](AUTHORING_SPEC.md)
- [Review rubric](REVIEW_RUBRIC.md)
- [Held-out protocol](HELD_OUT_EVAL.md)
- [Authorization checklist](AUTHORIZATION_CHECKLIST.md)

Run `python scripts/validate_rp_seed_authoring.py`. The command validates only the plan's deterministic structure. It has no code path that changes rights, review, or training status.
