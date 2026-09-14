# Aevon RP synthetic diagnostic pilot V1

This directory contains eight AI-generated conversations used only to diagnose the
frozen RP seed V1 character cards, scene briefs, and authoring guidance. The
conversations are not human-authored gold scenes, do not occupy any of the 48 human
scene slots, and are not training data.

The diagnostic records intentionally use `transcript` with
`diagnostic_user`/`diagnostic_character` speaker types instead of the SFT loader's
`messages` field. Each provenance record has `provenance_type:
synthetic_diagnostic`, and its schema fixes all training and authorization gates to
`false`. Moving or converting an artifact requires an explicit, separately reviewed
operation; there is no migration or approval path here.

Contents:

- `manifest.yaml`: fixed eight-scene selection and governance assertions.
- `conversations/`: one synthetic diagnostic transcript per selected scene brief.
- `provenance/`: generator and source lineage records with canonical content hashes.
- `diagnostic_assessments.yaml`: AI diagnostic rubric scores and scene-specific notes.
- `REPORT.md`: per-scene findings, cross-scene analysis, and proposed human-authoring
  revisions.
- `schemas/`: diagnostic-only conversation and provenance contracts.

Run `python scripts/validate_rp_seed_diagnostic_pilot.py` from the repository root.
Passing validation confirms only artifact structure, source alignment, hashes, and
closed gates. It grants no rights, human-review status, synthetic-expansion
authorization, QLoRA authorization, or training approval.
