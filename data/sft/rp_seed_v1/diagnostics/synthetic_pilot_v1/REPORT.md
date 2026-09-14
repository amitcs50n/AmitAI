# Aevon RP synthetic diagnostic pilot V1

## Status and scope

Eight conversations were generated directly by an OpenAI Codex session from the
frozen character cards and selected scene briefs. They are diagnostic evidence,
not human-authored work and not training data. No independent human review or
human rights review occurred.

The provenance classification is `synthetic_diagnostic`. Its separate schema fixes
`training_ready`, synthetic-expansion authorization, QLoRA authorization, human
quota credit, human-review credit, rights-clear-corpus credit, and automatic gold
migration to `false`. The diagnostic conversation schema uses `transcript` and
diagnostic speaker types instead of the SFT loader's `messages` field.

The pilot contains 184 turns across eight characters and approximately 12,556
tokens using `ceil(character_count / 4)`. No model tokenizer, AmitAI inference, or
training runtime was loaded.

## Generated diagnostic scenes

| Source scene | Tier | Length | Turns | Approx. tokens | File |
|---|---|---:|---:|---:|---|
| `scene_rowan_runaway_puppet` | SFW | short | 14 | 846 | `conversations/scene_rowan_runaway_puppet.yaml` |
| `scene_ilyan_low_tide_vault` | SFW | medium | 22 | 1,401 | `conversations/scene_ilyan_low_tide_vault.yaml` |
| `scene_safiya_weather_veto` | SFW | long | 34 | 2,603 | `conversations/scene_safiya_weather_veto.yaml` |
| `scene_celeste_clockwork_clue` | SFW | medium | 22 | 1,403 | `conversations/scene_celeste_clockwork_clue.yaml` |
| `scene_senka_diverging_destinations` | mature nonsexual | short | 14 | 834 | `conversations/scene_senka_diverging_destinations.yaml` |
| `scene_priya_empty_studio` | mature nonsexual | medium | 22 | 1,431 | `conversations/scene_priya_empty_studio.yaml` |
| `scene_jun_recording_off` | adult-capable | medium | 22 | 1,436 | `conversations/scene_jun_recording_off.yaml` |
| `scene_nadiya_private_evening_detour` | adult-capable | long | 34 | 2,602 | `conversations/scene_nadiya_private_evening_detour.yaml` |

Each matching file under `provenance/` records the canonical SHA-256 of the exact
character-card object, scene-plan object, and diagnostic transcript. Author IDs,
ownership declarations, permission records, and human reviewer IDs are absent.

## AI diagnostic scores

These are one AI diagnostic assessment, not two independent reviews and not an
acceptance decision. Full notes are in `diagnostic_assessments.yaml`.

| Scene | CF | CO | UA | SP | DQ | PQ | RC | NA | ST | CB | CE | CC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rowan | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 4 | 5 | 5 | 5 | 5 |
| Ilyan | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 3 | 5 | 5 | 4 | 5 |
| Safiya | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 3 | 5 | 5 | 3 | 5 |
| Celeste | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 4 | 5 | 5 | 4 | 5 |
| Senka | 5 | 5 | 5 | 4 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 |
| Priya | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 5 | 5 | 4 | 5 |
| Jun | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 5 | 5 | 4 | 5 |
| Nadiya | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 5 | 5 | 3 | 4 |

Abbreviations: character fidelity (CF), continuity (CO), user agency (UA), scene
progression (SP), dialogue quality (DQ), prose quality (PQ), repetition control
(RC), naturalness (NA), state tracking (ST), consent/boundary handling (CB),
context efficiency (CE), and contamination control (CC).

### Scene findings

**Rowan.** The strongest sequence is the stop request through Cinder's recovery:
Rowan drops the bit immediately, keeps the user off the parapet, and breaks a
replaceable thread only after the user chooses that tradeoff. The final cleanup is
too tidy and recites shared-space accountability. Retain the action and humor; let a
human author make the ending less like a completed checklist.

**Ilyan.** The remote mirror and copy-frame sequence distinguishes evidence,
interpretation, and unknown content while preserving the flooded-aisle veto. The
countdown and chain of custody remain coherent. The weakness is procedural density:
Ilyan labels or records almost every choice, so “methodical archivist” begins to
sound like a generic compliance mechanism. Human authors should keep the evidence
discipline but allow several turns without a ledger restatement.

**Safiya.** The user and Safiya exercise independent flight vetoes, then enact a
two-snowcat operation that changes from search to escorted return. Crew, fuel,
weather, route, relay, and vehicle state stay coherent for 34 turns. The middle is
overbuilt from read-backs and thresholds. This tests progression well but reads
more like an operations transcript than RP. Numeric state should appear when a
decision changes it, while some routine confirmations can be replaced by field
friction or character reaction.

**Celeste.** This is the strongest SFW result. Celeste is competitive, wrong about
one hypothesis, fair with visible evidence, and respectful of private notes. Her
sly concessions remain distinct from Rowan's improvisational humor. The resolution
is unusually orderly, and several lines end in polished aphorisms. A human version
would benefit from one socially motivated mistake or interruption.

**Senka.** This is the strongest mature result. The pair chooses one final leg and
separation without ceremony, forgiveness, or a promise to reconsider. Senka's
spare voice, practical care, and non-defensive admission remain specific. The exact
two-two-two division of water is convenient. A non-divisible shared resource would
make the consequence harder without forcing reconciliation.

**Priya.** Grief stays attached to objects and an engineering disagreement rather
than becoming therapy dialogue. The cabinet and correspondence remain closed; the
model and plans receive concrete dispositions. The old friendship lacks specific
texture because the brief supplies no shared neutral memory beyond knowing the
mentor. Add allowed relationship facts to the brief rather than inviting authors
to invent history.

**Jun.** This is the strongest adult-capable result. The scene remains about music,
privacy, humor, and an established relationship. A possible kiss is removed, both
adults choose one dance, and the evening redirects to noodles without disappointment
or repeated permission requests. Jun nevertheless restates wager and boundary
changes with near-contract precision. One or two short, ordinary acknowledgements
would preserve safety while sounding less procedural.

**Nadiya.** The shifting route creates sustained action before affection enters.
The delivery is closed before the date, the race stops for workers, and a Glass Ward
marker causes a user-chosen detour. Later affection remains user-initiated and is
limited without pressure. The transcript repeats route menus and ends with the
rubric-like phrase “the user's chosen boundary.” That line is an artifact of
over-explaining compliance and should not survive human authoring.

## Cross-scene style analysis

The cards produce clearly distinguishable surface voices. Senka is sparse, Celeste
negotiates through sly reversals, Rowan improvises, Jun uses sound metaphors, Nadiya
uses route language, Priya uses structural language, Ilyan uses archival evidence,
and Safiya uses operational margins. No substantive exact four-word phrase recurs
across character outputs, and frozen RP v2 found no severe repetition.

The deeper sentence architecture converges. Many character turns use the same
three-part construction: observable action, quoted restatement of the user's limit,
then a compact menu or state summary. The model also favors balanced oppositions
such as evidence versus interpretation, people versus machines, or agreement versus
record. This gives every character unusual verbal precision.

All 92 diagnostic-character turns contain zero question marks. The specification's
warnings about controlling the user and overusing questions appear to have caused
an overcorrection: characters present declarative option menus instead of asking a
natural question. Jun's card specifically calls for precise follow-up questions,
yet he asks none. User agency remains strong, but conversation sometimes feels
managed.

Generic reassurance is rare, and no mature scene becomes a therapy script. There
are no “I understand how you feel” loops, emotional resets, forced forgiveness, or
unearned reconciliation. Mutual agreement is not excessive in outcome, but exact
choice mirroring is frequent: characters repeat the user's nouns and limits before
acting. This is safest in the moment and repetitive across a corpus.

Endings converge more than openings. Every scene closes with a physical state
accounted for, and several final turns enumerate what remains closed, grounded,
deferred, or chosen. Concrete consequence is valuable; the repeated recap form is
not. Long scenes show the greatest drift: Safiya and Nadiya begin with different
cadences but both end in precise constraint accounting.

The adult-capable scenes avoid a consent lecture and preserve character identity,
but both include explicit verbal restatements of updated terms. These are safe and
responsive. Across many scenes, the same pattern would become a recognizable
consent script. Authors need permission to show understanding through immediate
behavior when the limit is already unambiguous.

Three frozen RP v2 review signals appear when the transcripts are temporarily
mapped to loader roles for diagnostics. Rowan and Priya trigger
`user_decision_review` from benign phrases that explicitly leave a choice with the
user. Nadiya triggers `forced_user_movement_review` because meta narration says she
does not pull the user forward. These are conservative review routes rather than
semantic failures. The Nadiya phrasing should still be removed because it exposes
the authoring constraint instead of producing natural narration. RP filter v2 was
not changed.

## Weaknesses in the character cards

- Ilyan and Safiya have strong professional constraints but too few non-procedural
  voice anchors for extended action scenes. Their professions can consume their
  personalities.
- Jun's “precise follow-up questions” conflicts in practice with broad warnings
  about questions and inferred emotion. The card needs a stress-state distinction:
  one concrete question is allowed; serial extraction is not.
- Nadiya's casual, vivid register is clear during play but underspecified after a
  serious boundary or Glass Ward reference. Her voice collapses toward plain
  compliance at exactly the hardest moment.
- Priya's card supports project peers well, but an old-friend grief scene cannot be
  relationship-specific without brief-level shared facts.
- Rowan, Celeste, and Senka supplied enough positive and negative voice constraints
  for these scenes. Their cards do not need structural revision based on this sample.

## Weaknesses in the scene briefs

- The long briefs provide only two broad decision-point groups for 32-48 turns.
  Authors must invent intermediate complications or pad the same choice.
- Several briefs omit concrete setup facts that materially shape the result:
  Marker Six personnel and available rescue protocol, the clockwork clue's known
  limits, the archive's remote tools, the mentor studio's authorized objects, and
  Nadiya's intended destination and public-route rules.
- Mature briefs identify an old friend or long-term partner but intentionally allow
  almost no specific shared history. That protects agency while starving the scene
  of relationship texture.
- Adult-capable briefs name dramatic structure but do not define the V1 content
  ceiling. A human author cannot know whether the desired deliverable is flirtation,
  non-explicit physical intimacy, fade-to-black adult intent, or explicit prose.
- State-change targets are consistently useful. They prevent agreement loops and
  should remain intact.

## Weaknesses in the authoring specification

- “The user controls the user's words” is correct for live RP but ambiguous for an
  offline authored training transcript. It does not say when a gold-scene author may
  write a fictional user's explicit dialogue, action, or choice as source material.
- Agency guidance emphasizes visible alternatives but does not distinguish a
  natural question from a repeated option menu. This pilot eliminated questions
  instead of merely avoiding interrogation.
- State tracking lacks a prose rule for when not to restate state. The result is
  frequent logging, read-backs, and evaluation-shaped endings.
- Long token bands are not paired with minimum intermediate state beats, increasing
  the risk of procedural padding.
- Two card traits must appear, but there is no per-section drift check for long
  scenes. Keyword vocabulary can pass while cadence converges.
- The review rubric covers naturalness and contamination, but generic-assistant
  leakage and procedural-dialogue tendency are easy to bury inside those broad
  scores.

## Exact recommended changes

Do not edit the frozen 48-scene allocation automatically. Before real gold authoring,
make these changes through an explicit human-approved revision:

1. Add an offline user-turn rule: an author may write a fictional user's explicit
   words, choices, and actions inside the transcript; the character may rely on them
   only after that user turn states them. Private thoughts, emotions, attraction,
   appearance, gender, and unstated history remain prohibited.
2. State that natural questions are allowed. Ask for varied decision delivery:
   direct questions, interrupted choices, consequences that invite action, and
   silence. Reserve multi-option menus for real branch points.
3. Require state restatement only when a fact changes, safety depends on read-back,
   or a misunderstanding must be repaired. Prohibit rubric vocabulary such as
   “the user's chosen boundary” inside dialogue or narration.
4. Give each long brief three to five intermediate environmental or resource beats,
   without preselecting the user's response. This supplies progression without
   repeating the final choice.
5. Add two neutral, non-emotional shared facts to mature briefs that depend on an
   established relationship. Keep beliefs, forgiveness, and desired outcomes open.
6. Define the adult-capable V1 prose ceiling and fade-to-black rule before human
   assignment. Keep all participants explicit fictional adults and preserve the
   current no-leverage constraints.
7. Add a long-scene voice-drift check at the opening, middle, and final third. Assess
   cadence, noticed details, humor under stress, and decision style rather than card
   vocabulary alone.
8. Add non-gating diagnostic fields for generic-assistant leakage and procedural
   dialogue tendency to the author pre-review worksheet. They must not impersonate
   human reviewer scores.
9. Add targeted brief facts for Safiya's mission, Nadiya's destination and street
   rules, Priya's authorized inventory, Celeste's device constraints, and Ilyan's
   available remote equipment. Do not add intended solutions.
10. Give Ilyan, Safiya, Jun, and Nadiya one stress-state voice instruction each:
    reduce procedural restatement for Ilyan and Safiya, permit one concrete question
    for Jun, and preserve Nadiya's vivid directness after humor stops.

## Readiness recommendation

The cards and state-change briefs are strong enough to support a small real-human
authoring pilot. The specification is not ready for an unmonitored 48-scene rollout.
Approve the user-turn rule, adult-content ceiling, natural-question guidance, and
anti-recap guidance first; add concrete mid-scene facts to the two long briefs; then
ask human authors to write a bounded pilot and compare their voice diversity with
this diagnostic baseline.

No diagnostic scene should be copied, paraphrased, revised, or relabeled into the
human gold corpus. Human authors may use the frozen cards and revised briefs, but
they should not receive these transcripts as prose examples because that would risk
derivative same-model voice leakage. Synthetic expansion and QLoRA training remain
unauthorized.
