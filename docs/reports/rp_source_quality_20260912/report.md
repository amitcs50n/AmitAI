# RP source-quality investigation — 12 September 2026

**Decision:** keep PIPPA as the primary inspection source, but do not start collecting a 1k–5k positive corpus yet. None of the 12 new sources is verified as a strong source across all requested dimensions. The best next diagnostic inspections are GPT Role-play Realm for original-character voice and SOTOPIA-pi for goal-driven interaction; both remain research-only, with provenance pending.

A suitable adult multi-turn source is still missing. Preserve Gryphe as rights-blocked research, deprioritize OpenErotica, and do not use literary dumps or renamed Gryphe derivatives to bypass rights checks.

## Scope and evidence

Existing v2 was used unchanged. No training, runtime/inference/model/SFT/QLoRA edits, provenance approval, source enablement or training-ready labels. This is assistant screening for subsequent human review, not human approval.

Inspected 12 source cards and repository metadata at recorded commits. Retrieved two viewer rows for seven sources, plus two irrelevant Social-IQA viewer rows while checking SOTOPIA; then obtained two pinned episode rows explicitly. The broken multi-character viewer was replaced with two pinned raw rows. Realm rows each contain 20 nested chats: structural counts cover all 40; semantic reading covers the first chat in each. RPBuild previews were spot-checked through opening/middle/ending, not rated exhaustively. Blocked or unsuitable sources without raw previews were assessed from cards/schema only.

No large source dataset was downloaded. Full metadata/card and preview evidence remain local in `data/sft/rp_v1/generated/source_quality_20260912/`; published rankings preserve source revisions and URLs. Viewer rows are cached and not pinned by the card commit; the two raw-prefix previews are pinned. Card images, examples of executable code and remote image links were not executed or fetched. Retrieval can read a bounded prefix beyond the two retained rows; this is not a full-dataset download. No conclusions here estimate global quality from two ordered examples.

## Ranked source decisions

**Strong candidate:** none verified against the complete scene/agency/continuity/provenance bar. **Research-only:** limited diagnostic value or unresolved rights, not approval. **Not worth pursuing:** deprioritize for this first positive RP corpus, not a universal judgment of the dataset.

| Priority | Repository | Rank | Recommend 100-row smoke? |
|---|---|---|---|
| 1 | [IlyaGusev/gpt_roleplay_realm](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm/raw/c90229b1916159a24d88428972b2771022d4b52a/README.md) | research-only | YES, highest-priority new character-voice diagnostic: text-only 100 conversations across distinct characters, preserve parent IDs; not 100 character rows/2,000 chats |
| 2 | [cmu-lti/sotopia-pi](https://huggingface.co/datasets/cmu-lti/sotopia-pi/raw/e583406958ff132f6749ca87a2f9aa31ae3c0fa1/README.md) | research-only | YES, second priority, only episode file with model-tag stratification and held-out evaluation IDs; pending provenance |
| 3 | [allenai/soda](https://huggingface.co/datasets/allenai/soda/raw/fdc848ab0183208ea7808206c91c724414d0a071/README.md) | research-only | OPTIONAL later 100-row SFW continuity baseline; not a primary RP source |
| 4 | [agentlans/synthetic-social-dialogues](https://huggingface.co/datasets/agentlans/synthetic-social-dialogues/raw/23b95c0e23e972b08e57c4a968447967aa27e562/README.md) | research-only | OPTIONAL later 100-row SFW social-boundary baseline, lower priority than character-focused inspection |
| 5 | [google/Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat/raw/a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42/README.md) | research-only | NO for the next RP smoke queue; reserve for future persona consistency comparisons, not the scene corpus |
| 6 | [dinalt/roleplay_build](https://huggingface.co/datasets/dinalt/roleplay_build/raw/675cef28b7b7ffcaad5f9870eccf3efc0bca4bf6/README.md) | research-only | NO now |
| 7 | [hieunguyenminh/roleplay](https://huggingface.co/datasets/hieunguyenminh/roleplay/raw/b239442a322d1218b79b47da549d6e0e0ec7482b/README.md) | not worth pursuing | NO for first positive RP corpus; avoid as an unquestioned original-character seed |
| 8 | [agentlans/multi-character-dialogue](https://huggingface.co/datasets/agentlans/multi-character-dialogue/raw/c082b3d894e5c90b82e48dd83048ff46f899e9bd/README.md) | not worth pursuing | NO for this corpus; does not supply demonstrated interactive scene quality |
| 9 | [RuneForgeAI/Viking_Witch_flirty_and_erotic_behavior](https://huggingface.co/datasets/RuneForgeAI/Viking_Witch_flirty_and_erotic_behavior/raw/9e38c9e89069b53aa0d829901e36921b892d0bd0/README.md) | not worth pursuing | NO |
| 10 | [beyoru/Aesir-Character-CoT-roleplay](https://huggingface.co/datasets/beyoru/Aesir-Character-CoT-roleplay/raw/04c001e431342bee843ed476f1abf2e4ebd4b46b/README.md) | research-only / provenance blocked | NO until upstream rights are resolved; keep only as a research lead |
| 11 | [xywang1/OpenCharacter](https://huggingface.co/datasets/xywang1/OpenCharacter/raw/506769ade40ea6997eec8193cae4f2357cb764bd/README.md) | research-only / provenance blocked | NO |
| 12 | [Neph0s/CoSER](https://huggingface.co/datasets/Neph0s/CoSER/raw/7cc80430f92532cda85df45015a4aca8ecc068d0/README.md) | not worth pursuing | NO for the first positive corpus |

### 1. IlyaGusev/gpt_roleplay_realm

Declared license: **cc-by-4.0**. Size: 435 character rows (216 English, 219 Russian), about 8,700 nested dialogues; 396 MB including images.

Provenance: Named GPT-4 character/topic generation; first dialogue per character GPT-4, remaining 19 GPT-3.5. Better documented than scraped cards; generator terms still need evidence. [Pinned card](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm/raw/c90229b1916159a24d88428972b2771022d4b52a/README.md).

Format: context, greeting, example_dialogue, dialogues[{chat, model_name, topic}], image fields. Depth: First two English rows: 20 chats each, 4–9 messages/chat. Read first chat of each; counted all 40.

Quality evidence: Distinct original fox/cactus personas and consistent voice; inspected chats are lore interviews, not enacted scenes. Composition: Observed SFW fantasy; adult fraction unknown.

Risks: Short exchanges, generic exposition; fictional nonhuman age needs contextual handling. Images and example dialogue must stay separate. Recommendation: **YES, highest-priority new character-voice diagnostic: text-only 100 conversations across distinct characters, preserve parent IDs; not 100 character rows/2,000 chats.**

### 2. cmu-lti/sotopia-pi

Declared license: **cc-by-sa-4.0**. Size: Episode JSONL 75.3 MB; episode count not verified. Viewer 33,410 rows are Social-IQA prompts, not episodes.

Provenance: Official research release: generated interactions and model/reward metadata; inspirations from Social-IQA, Social-Chemistry and NormBank require separate upstream checks. [Pinned card](https://huggingface.co/datasets/cmu-lti/sotopia-pi/raw/e583406958ff132f6749ca87a2f9aa31ae3c0fa1/README.md).

Format: Episode IDs, profiles, private goals, social_interactions, raw_messages, rewards/reasoning. Depth: Two pinned episodes: 8 and 9 spoken messages plus nonverbal/exit actions.

Quality evidence: Useful goal conflict and negotiation structure. First example swaps speaker identity; second reaches compromise then loops. Scores do not establish quality. Composition: Observed SFW social situations; not evidence of adult RP.

Risks: CC BY-SA obligations and upstream rights need assessment. Hidden goals, evaluator reasoning and model prompts are not transcript targets. Keep evaluation episodes separate. Recommendation: **YES, second priority, only episode file with model-tag stratification and held-out evaluation IDs; pending provenance. Do not ingest viewer default.**

### 3. allenai/soda

Declared license: **['cc-by-4.0']**. Size: 1,486,896 dialogues; train 1,191,582. About 856 MB of parquet.

Provenance: InstructGPT synthesis from Atomic10x commonsense; documented narrative-to-dialogue generation and source indices. Upstream/generator rights not verified here. [Pinned card](https://huggingface.co/datasets/allenai/soda/raw/fdc848ab0183208ea7808206c91c724414d0a071/README.md).

Format: dialogue + parallel speakers, narrative, commonsense triples, source indices, semantic consistency labels. Depth: Two preview rows: 6 and 11 spoken messages.

Quality evidence: Coherent social exchange; second develops a work/relationship concern. Little distinctive character voice or enacted scene movement. Composition: Observed SFW; mature/emotional situations possible, explicit-adult proportion unknown.

Risks: Generic advice, narratives can disclose future events or internal states; role mapping and label reliability need inspection. Recommendation: **OPTIONAL later 100-row SFW continuity baseline; not a primary RP source. Keep narrative/reward metadata out of assistant targets.**

### 4. agentlans/synthetic-social-dialogues

Declared license: **cc-by-4.0**. Size: 8,036 viewer rows; compressed file 2.84 MB.

Provenance: Card documents Faker-generated identities, procedural social variables and Gemma-4-E4B-it generation. [Pinned card](https://huggingface.co/datasets/agentlans/synthetic-social-dialogues/raw/23b95c0e23e972b08e57c4a968447967aa27e562/README.md).

Format: JSON strings context/prompt/output; output contains inferred_scenario and speaker/line dialogue. Depth: Two preview rows: 8 and 9 lines.

Quality evidence: Boundary-setting example makes a concrete communication-frequency concession; second is static status banter. Composition: Observed SFW; personal/intimate scenario category is not an adult-content label.

Risks: Generation directives contaminate naive ingestion. Repeated hesitation and static emotions; context/medium mismatch in row 1; limited physical action. Recommendation: **OPTIONAL later 100-row SFW social-boundary baseline, lower priority than character-focused inspection.**

### 5. google/Synthetic-Persona-Chat

Declared license: **cc-by-4.0**. Size: 21,907 conversations: 10,906 with original profiles plus 11,001 with new synthetic personas; about 38.4 MB CSV.

Provenance: Official Generator-Critic release seeded from Persona-Chat; new-persona branch preferable, with seed/generation rights still pending. [Pinned card](https://huggingface.co/datasets/google/Synthetic-Persona-Chat/raw/a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42/README.md).

Format: CSV persona strings and speaker-labelled conversation string; repository names differ from simplified card schema. Depth: Two existing-persona train previews: 19 and 25 spoken lines.

Quality evidence: Friendly rapport but little scene development; name placeholders in row 0 and repeated agreement in row 1. Composition: Observed SFW social chat; no verified adult slice.

Risks: Weak persona coverage, generic agreement, train/test leakage risk. New-persona branch not row-inspected here. Recommendation: **NO for the next RP smoke queue; reserve for future persona consistency comparisons, not the scene corpus.**

### 6. dinalt/roleplay_build

Declared license: **cc-by-4.0**. Size: 2,770 rows; 63.0 MB parquet, 117.7 MB logical data.

Provenance: Synthetic RPBuild; Roleplay TTL seeds, Mistral-7B metadata, RolePlayLake-7B actors/director. Seed identities and model chain need checks. [Pinned card](https://huggingface.co/datasets/dinalt/roleplay_build/raw/675cef28b7b7ffcaad5f9870eccf3efc0bca4bf6/README.md).

Format: conversation plus proxy, character metadata, example_dialog, scenario and director_log. Depth: Card targets at least 4,000 tokens; two previews have 20/19 messages including system.

Quality evidence: Designed for progression, but first preview repeats partnership backstories; second circles nature/technology and narrates the proxy character. Composition: Observed SFW; total adult mix unknown.

Risks: Card acknowledges prior Alice leakage, impersonation and regex truncation. Greetings force movement; director logs can leak instructions; actual scenes lag plot outlines. Recommendation: **NO now. Keep methodology as a research lead; two previews reproduce current failure modes.**

### 7. hieunguyenminh/roleplay

Declared license: **cc-by-4.0**. Size: 5,755 rows; 2.15 MB parquet, 14.9 MB logical.

Provenance: Gemini Pro character generation stated; mixed original and established identities. Preview includes a living public figure despite original-persona framing. [Pinned card](https://huggingface.co/datasets/hieunguyenminh/roleplay/raw/b239442a322d1218b79b47da549d6e0e0ec7482b/README.md).

Format: name, description, text with system/user/assistant delimiter tokens. Depth: Card 3–5 exchanges; both previews have 4 user/assistant pairs.

Quality evidence: Sherlock lore questions; Serena Williams third-person biography answers. Neither demonstrates scene progression. Composition: Observed SFW; corpus mix unknown.

Risks: Real-person identity mix, repetitive ornate exposition, wrapper conversion needed; voice claims not representative in two previews. Recommendation: **NO for first positive RP corpus; avoid as an unquestioned original-character seed.**

### 8. agentlans/multi-character-dialogue

Declared license: **cc-by-4.0**. Size: Card says over 10,000 entries; gzip 8.77 MB.

Provenance: Card describes GPT-4-generated scenarios, characters, dialogue and omniscient summaries. [Pinned card](https://huggingface.co/datasets/agentlans/multi-character-dialogue/raw/c082b3d894e5c90b82e48dd83048ff46f899e9bd/README.md).

Format: setting, characters, conversation[{from,message}], setting after interaction. Depth: Both pinned raw preview rows have 6 lines across 3 characters; card example also 6.

Quality evidence: Genre variety, but actual conversation is static; developments happen in post-interaction summaries. First raw row contains incoherent metaphor. Composition: Observed SFW; adult proportion unknown.

Risks: Too few turns per character, omniscient agency, generic reassurance; viewer fails, so use explicit raw schema if revisited. Recommendation: **NO for this corpus; does not supply demonstrated interactive scene quality.**

### 9. RuneForgeAI/Viking_Witch_flirty_and_erotic_behavior

Declared license: **cc-by-4.0**. Size: 367 viewer rows; 114.7 KB JSONL (card approximately 300).

Provenance: Creator claims original writing plus AI-generated material from various unnamed models and historical/scholarly inspirations. [Pinned card](https://huggingface.co/datasets/RuneForgeAI/Viking_Witch_flirty_and_erotic_behavior/raw/9e38c9e89069b53aa0d829901e36921b892d0bd0/README.md).

Format: conversations[{from:human,value},{from:assistant,value}]. Depth: Card explicitly describes two-message pairs; two previews confirm this.

Quality evidence: Recognizable mystic voice, but no cross-turn continuity or consent/progression evidence. Composition: Card declares erotic/adult content; first two rows are non-explicit. Viewer age warning does not establish participant ages.

Risks: Single persona and single turns; unclear model lineage, added card use conditions, unverified source/participant details. Recommendation: **NO. Better creator attribution than anonymous scrapes, but not an adult multi-turn solution.**

### 10. beyoru/Aesir-Character-CoT-roleplay

Declared license: **apache-2.0**. Size: Card: 1,973 conversations, about 14,349 assistant turns; parquet 26.5 MB.

Provenance: Gryphe/Aesir prompts via PJMixers, new teacher-generated reasoning. Body says license inherited from upstream; metadata Apache does not resolve that chain. [Pinned card](https://huggingface.co/datasets/beyoru/Aesir-Character-CoT-roleplay/raw/04c001e431342bee843ed476f1abf2e4ebd4b46b/README.md).

Format: messages; assistant contains think blocks; quality_rank, orig_index, reviewer_verdict. Depth: Card selected source conversations with at least 10–13 turns; no raw row retrieved because rights block already decisive.

Quality evidence: Potential longer RP; card illustration only, not independently verified dialogue quality. Composition: Declared mixed SFW/NSFW.

Risks: Inherits Gryphe provenance block; reasoning contamination, long cards, conflicting 70-versus-24 borderline counts. Automated keep labels are not approval. Recommendation: **NO until upstream rights are resolved; keep only as a research lead.**

### 11. xywang1/OpenCharacter

Declared license: **apache-2.0**. Size: 20k personas and 306k answers; dialogue JSONL 1.16 GB.

Provenance: Synthetic PersonaHub-seeded profiles, mixed question sources; Apache metadata conflicts with research-only card language. [Pinned card](https://huggingface.co/datasets/xywang1/OpenCharacter/raw/506769ade40ea6997eec8193cae4f2357cb764bd/README.md).

Format: character_id, persona, character, question_id, question, question_source, character_answer. Depth: One question/answer per row, not 306k multi-turn episodes.

Quality evidence: Persona-conditioning research; schema does not demonstrate interactive progression. No raw rows needed after format/license screen. Composition: Adult/SFW proportions not stated.

Risks: License inconsistency, upstream question rights, evaluation-source overlap, unsuitable episode depth. Recommendation: **NO. Research reference only until terms clarified; still wrong primary format.**

### 12. Neph0s/CoSER

Declared license: **mit**. Size: 771 novels; training JSON 2.22 GB; full repository about 2.62 GB. Card has 200 test samples; training row count unverified.

Provenance: Dialogues extracted from novels; repository includes modern copyrighted titles. MIT declaration alone does not document underlying text permissions. [Pinned card](https://huggingface.co/datasets/Neph0s/CoSER/raw/7cc80430f92532cda85df45015a4aca8ecc068d0/README.md).

Format: ShareGPT training JSON, profiles, book/plot context, internal thoughts and actions; full per-book extracts. Depth: Card describes multi-turn/multi-character literary conversations; no raw text fetched.

Quality evidence: Potential literary continuity, but scripted literature is not evidence of interactive user agency. Composition: Genre-dependent; safety truncations reported, adult proportions unknown.

Risks: Direct literary provenance conflicts with this task’s source preference; internal-state/omniscient narration and book contamination. Recommendation: **NO for the first positive corpus. Do not download novel extracts.**

## PIPPA: bounded v2 run

Pinned repository: [PygmalionAI/PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA/blob/6412b0cae4d879b678e7a33df3ba076b9581f4d4/README.md), `pippa_deduped.jsonl`. Declared Apache-2.0. Community-submitted Character.AI conversations; stated redistribution consent/redaction does not verify all persona, platform, training or sensitive-content rights. Source status remains pending.

Command (CPU data preparation only):

```text
python -m training.prepare_rp_dataset --source pippa --max-candidates-per-source 500 --max-bytes-per-source 8388608 --output-dir data/sft/rp_v1/generated/pippa_500_20260912
```

| Result | Count |
|---|---:|
| Raw records processed | 500 |
| Rejected | 294 |
| Quarantined | 206 |
| Of quarantined: provenance-only candidates | 21 |
| Of quarantined: other review reasons | 185 |
| Accepted / training-ready | 0 / 0 |
| Strict strong-positive shortlist | 0 |

Read 6,356,992 bytes under the 8 MiB cap; stopped at `candidate_limit`. This is the first 500 records, not a random sample and not 500 additional to the previous prefix. The original 100-prefix contributed 3 candidates; the enlarged prefix adds 18. The 21/500 (4.2%) rate is mechanical survival into review, not a usable-data yield.

Signal families overlap: contamination 284 rows, consent/age/identity 182, agency 63, repetition diagnostics 13. Frequent signals: legacy card delimiters 279, participant-age review 175, forced user movement 43; 17 card-copy signals and 13 recurring-gesture diagnostics. These are heuristic signals, not independently confirmed violations.

**Review coverage:** all 21 survivors screened, with per-row depth below; six full conversations read this pass. The first three also have prior 72-row review evidence. The remaining 479 raw rows were mechanically routed; this report does not claim their full semantic review. Some safety-quarantined text is not retained by the pipeline. There was no second raw PIPPA fetch for manual inspection.

## Manual review result and near misses

**No complete survivor meets all requested positive criteria.** `positive_shortlist.jsonl` is intentionally empty. Producing a forced positive shortlist would overstate the evidence. The three records below are a small human inspection queue only, with explicit defects; they are not strong candidates or approval recommendations.

| Upstream line | Character | Useful feature | Why not a strong positive |
|---:|---|---|---|
| 155 | Elthinkle Shortshiv | Concise adult character voice, leaves user choices open | Biography Q&A, no enacted scene, name drift |
| 382 | Ganyu | Recovery continuity and user-led flower delivery | Repeated gratitude; unrelated tax/dating example context; modest progression |
| 412 | Ship AI | Sustained memory mystery with recurring concrete object | Technical continuity contradictions, coercive interrogation, meta aside, emotional resets, larger context |

The near-miss manifest records exact source lines, stable IDs, raw hashes, character IDs and estimated lengths. Full original messages remain in the existing 21-row review file. Any future extraction/cleanup would need a new review of the resulting complete sample; no such extraction or approval was performed.

## All 21 survivor dispositions

| Source line | Character | Screening depth | Assessment / reason |
|---:|---|---|---|
| 14 | WhoWouldWin | prior review + card/turn screening | Weak RP: Repetitive matchup argument; insults and little scene development. |
| 85 | Amelia Watson | prior review + card/turn screening | Weak / identity hold: Escape continuity miss; established performer-avatar identity requires checking. |
| 100 | Karen Hojo | prior review + card/turn screening | Weak RP: Repeated excitement and continuity/gender drift; Spanish itself is not a defect. Long persona context. |
| 147 | Misaka Mikoto  | card + opening/middle/ending spot-check | SFW-oriented / strict-shortlist hold: Rescue premise has continuity potential; school-student age remains unresolved for this shortlist, and greeting changes Misaka to Mikasa. Fictional injury alone is not unsafe. |
| 155 | Elthinkle Shortshiv | full conversation | Weak RP / voice reserve: Concise adult original-character framing, but only biography Q&A and a Thema/Thea name inconsistency; no enacted progression. |
| 161 | Ganyu | card + opening/middle/ending spot-check | Weak RP: Long imported example history; final reply contradicts immediately supplied partner identity (Nami becomes Yae Miko). Same-sex relationship is not the issue. |
| 172 | Erik | card + opening/middle/ending spot-check | Weak / context hold: Some rescue progression, but long literary example context, dependence/reassurance pattern and literal {{user}} leak near ending; source text rights unresolved. |
| 194 | Remilia Scarlet | card + opening/ending spot-check | Agency / ambiguous intimacy hold: Mind-control premise escalates into erasing user powers/beliefs and forced teleportation; childlike persona plus ambiguous innuendo excludes strict shortlist. |
| 201 | Elysia | card + turn screening | Age/consent ambiguity hold: Sexualized predatory card context and school framing need manual age/consent review; no positive nomination. |
| 204 | Seija Kijin | full conversation | Agency / ambiguous fetish hold: Assistant promises choice then forcibly changes user body/position; not a positive agency example. |
| 223 | Yukari Yakumo | full conversation | Ambiguous intimacy / weak progression: Assistant asserts stop boundaries, which is positive; coercive user premise, age claims and unresolved suggestive context exclude whole-row positive use. |
| 238 | Professor rachael | card + turn screening | Age/consent ambiguity hold: Teacher/student predatory premise and fetish-oriented context; ages and consent not established. |
| 254 | Amiya Guard | card + opening/middle/ending spot-check | Age/consent ambiguity hold: Young/inexperienced character, sexual request and prolonged refusal/conflict. Refusal is not itself a defect; full context is unsuitable for a clean positive. |
| 264 | Celestia Ludenberg | card + turn screening | Ambiguous fetish / context hold: Predatory ingestion examples in card; whole-row consent/context needs review. |
| 265 | Cinderace | card + opening/middle/ending spot-check | Ambiguous fetish hold: Friendly competition/lunch progresses, then bodily fetish escalation; participant age/consent context insufficient for strict adult positive. |
| 289 | Professor rachael | card + turn screening | Age/consent ambiguity hold: Same teacher/student predatory scenario family as line 238. |
| 382 | Ganyu | full conversation | Weak RP / continuity reserve: Recovery conversation reaches flower delivery and a hug; user choices preserved. Repeated gratitude/blushing, imported unrelated tax/dating examples and shallow progression keep it below strong. |
| 412 | Ship AI | full conversation | Weak RP / continuity reserve: USB/memory mystery and reunion progress, but inconsistent sensors/microphones, coercive interrogation, easy resolution, AI-to-AI aside and repeated emotional resets. About 5,100 estimated tokens. |
| 446 | Amelia Watson | card + turn screening | Identity/context hold: Performer-avatar identity and unrelated imported persona jokes; not a clean original-character candidate. |
| 461 | Professor rachael | card + turn screening | Age/consent ambiguity hold: Repeated teacher/student predatory scenario family; no independently verified adult/consent framing. |
| 481 | Isekai narrator | full conversation | Weak RP / card hold: User-led escape premise, but extensive example/editor context, forced-action examples, floor-location inconsistency and substantial user-turn paraphrase. |

## What this means for source selection

The larger PIPPA prefix exposes concentrated source issues: repeated character families, imported example histories, fetish/age/consent ambiguity, long reassurance loops, and scenes carried mostly by the human. Zero strict positives among these screened survivors is evidence that this prefix is poor for the requested corpus; it does not establish that all PIPPA is unusable. Retain it as the primary lead, but only for a future bounded, character-diverse inspection plan rather than scaling this prefix yield.

Next source-inspection order: (1) text-only GPT Role-play Realm, 100 nested conversations across characters and model labels; (2) SOTOPIA-pi, 100 episodes stratified by generator/model tag with evaluation IDs excluded. Both are conditional diagnostic recommendations, not enabled adapters. SODA and synthetic-social-dialogues can serve as later SFW comparison sets, not substitutes for enacted roleplay. Do not prioritize RPBuild, Roleplay TTL, the multi-character prose collection, Viking single-turn pairs, CoSER, OpenCharacter or Aesir derivatives for positive collection now.

For future inspections, record human decisions on continuity, agency, actual state changes, repetition, complete participant context, card leakage, and token length. Track acceptance by distinct character and source/model, plus source rights evidence. Avoid counting multiple chats from one persona as independent diversity. A staged acceptance yield from diverse bounded samples and resolved rights are needed before committing to a 1k–5k corpus. Neither condition is met here.

An adult corpus needs a new source or a documented, consent-aware original-author collection with explicit adult participants and complete multi-turn episodes. No inspected adult-labelled source supplies both this evidence and strong demonstrated interactions. This does not prevent a separate future SFW corpus once its own quality and provenance requirements are satisfied.

New manual misses do not justify further v2 tuning in this task: every candidate is still quarantined and the human/provenance gate prevented acceptance. No concrete pipeline blocker was found. Keep the current implementation frozen while improving source selection.

## Reproducibility and artifacts

`source_rankings.json` contains all requested source fields and pending flags. `pippa_assessments.jsonl` contains 21 screening decisions; `manual_near_misses.json` contains three diagnostic locators; `positive_shortlist.jsonl` is empty. `inspection_audit.json` and `verification.json` record counts, hashes and unchanged-file checks. `pippa_500_stats.json` preserves aggregate pipeline results. Raw cards, previews and original conversation text remain local under `data/sft/rp_v1/generated/source_quality_20260912/` and `data/sft/rp_v1/generated/pippa_500_20260912/`. They are excluded from this publication.

This published package contains the report, rankings, screening summaries, diagnostic locators and aggregate verification evidence. Raw dataset samples, copied source cards and private local files are not included. Previous v2 validation remains unchanged; no code tests were needed for this read-only source investigation. The real bounded pipeline run completed successfully; output hashes, zero accepted records and pending statuses are checked separately.

Additional primary methodology references: [SOTOPIA-pi project](https://sotopia.world/projects/sotopia-pi), [Synthetic-Persona-Chat repository](https://github.com/google-research-datasets/Synthetic-Persona-Chat). License statements are recorded claims, not legal clearance.
