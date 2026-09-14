# Aevon RP seed V1 authoring workspace

This directory contains the human-authoring-ready design and governance artifacts for a future rights-clear RP seed. It contains no completed conversations and is not an input to the existing SFT loader or RP preparation pipeline.

- `roster.yaml`: 16 original fictional adult character cards with explicit boundary and user-agency contracts.
- `scene_allocation.yaml`: 48 scene briefs with explicit user roles and no dialogue.
- `heldout_evaluation.yaml`: frozen evaluation families and coverage targets.
- `schemas/`: machine-readable authoring contracts.
- `templates/`: blank submission, review, and provenance forms.

Every card and scene starts with rights/review status `pending` and `training_ready: false`. Human authors must replace author placeholders, sign the provenance record, write the scenes, and pass independent review before any row can enter a gold seed.

Run `python scripts/validate_rp_seed_authoring.py` from the repository root to validate the plan. Validation checks structure and allocation only; it never grants rights or quality approval.
