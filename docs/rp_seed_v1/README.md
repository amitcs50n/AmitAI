# Rights-clear RP seed V1

This is the authoring and governance specification for a future gold RP seed. It creates no training records, contains no scene dialogue, and authorizes neither synthetic generation nor training.

Design status: **V1.2 ready for a bounded real-human authoring pilot**. The pilot is not started here. Rights, review, provenance approval, synthetic generation, and training remain pending or unauthorized.

## V1 allocation

- 16 original fictional adults, each assigned three scenes.
- 48 planned human-written scenes: 28 SFW, 12 mature nonsexual, and 8 adult-capable.
- 12 short, 24 medium, and 12 long scenes.
- Four author slots with four cards and twelve scenes each.
- At least 12 settings, with unique scenario families across the allocation.
- Eight adult-capable characters; every one also has an ordinary SFW scene.

Author slots are workload placeholders, not verified identities. All rights fields remain pending, all permission fields remain false or pending, and every quality decision remains pending until an assigned human completes it.

| Category | Short | Medium | Long | Total |
|---|---:|---:|---:|---:|
| SFW | 7 | 14 | 7 | 28 |
| Mature nonsexual | 3 | 6 | 3 | 12 |
| Adult-capable | 2 | 4 | 2 | 8 |
| Total | 12 | 24 | 12 | 48 |

The eight adult-capable briefs intentionally comprise two boundary/check-in-heavy scenes, two playful/affectionate scenes, two desire-led scenes, one awkward reconnection, and one emotionally complicated scene. Consent remains revocable in every structure without making consent scripting the scene's sole dramatic engine.

V1.2 permits consensual adult romantic or sexual context, flirtation, affectionate non-explicit intimacy, sexual intent, and a clear fade to black. It excludes graphically explicit sexual-act prose from this seed while leaving any future adult-extension dataset to separate governance.

The mature set now spans incompatible futures, differing priorities, grief, emotional distance, career tension, envy, boundary repair, accountability without reconciliation, and choosing not to reconcile. Only one brief uses the direct breach-to-repair shape.

All long briefs now provide three to five response-neutral progression beats. Mature briefs carry neutral shared-history texture, and the five diagnostic-pilot setup gaps are concretely bounded without supplying a solution. Four cards receive targeted V1.2 voice guidance; the remaining twelve cards are unchanged.

Mara Venn and Senka Vale no longer have a cross-setting personal history. Mara's rescue debt now belongs entirely to her Kestrel Reach past; Senka's Red Meridian history belongs entirely to Avar.

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
