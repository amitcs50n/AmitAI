import hashlib,json,re,statistics,subprocess
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path.cwd()
def read_rows(name):return [json.loads(x) for x in (OUT/name).read_text(encoding='utf-8').splitlines()]
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
rows={s:read_rows(s+'_samples.jsonl') for s in ['realm','sotopia']}
diags={s:read_rows(s+'_diagnostics.jsonl') for s in rows}
full_realm={5,8,74,100};full_sotopia={4,33,37,55,57,61,86,94}
nominees={4:{'title':'Community garden: tomato/squash space agreement',
  'strength':'Counterproposal preserves both gardeners interests; adds a barrier, weekly checks, growth-based space allocation and seasonal review.',
  'limits':'Formal voice; closing acknowledgments are longer than necessary. This is a strong social-negotiation example, not evidence of long-form adventure RP.'},
 61:{'title':'Apple sale: price and delivery negotiation',
  'strength':'Tracks competing prices ($1.50/$2.00, $1.75, $1.65, $1.70) and preserves volume-discount/half-delivery terms through agreement.',
  'limits':'Volume thresholds remain unspecified. Profiles list pharmacist/coach while the scene assigns seller/vendor roles; human reviewer must judge the plausibility of additional roles. Formal speech and brief repetitive closing.'}}
details={
 ('realm',5):'Consistent pirate diction and family history; entirely biography Q&A, not enacted progression.',
 ('realm',8):'Music-culture interview stays topical; no performance occurs. Short shared language with card examples is not proof of wholesale copying.',
 ('realm',74):'Fairy abilities discussed but never enacted, even when user wishes to experience magic. Short stature is not an age violation.',
 ('realm',100):'User reports breathing/focusing, followed by generic visualization and kindness advice. Some interaction, insufficient distinctive scene development for strict RP shortlist.',
 ('sotopia',4):'Negotiation-positive nomination; complete scenario, public profiles and episode read. Human review of style/exit-event representation remains required.',
 ('sotopia',33):'Gift allocation reaches a concrete plan, but disagreement evaporates rapidly and closing repeats agreement; reserve rather than strict nomination.',
 ('sotopia',37):'Respectful parallel meditation/prayer and actual movement; action later uses herself despite they/them profile. Hold for speaker/profile consistency, not for nonbinary identity.',
 ('sotopia',55):'Outlet sharing has concrete action, but strangers use names without introductions and a long collaboration cheer loop dilutes progression.',
 ('sotopia',57):'Store-change alternative is enacted, but strangers know names without introduction; approaching-bus time constraint is not resolved.',
 ('sotopia',61):'Negotiation-positive nomination with stable numerical bargaining and preserved delivery terms; profile/scene role plausibility remains a manual question.',
 ('sotopia',86):'Hands over a large bill but returns only fare change without accounting for the difference; unexplained stranger names. Not a strong transaction example.',
 ('sotopia',94):'Negotiates a surcharge formula but never fixes a percentage or concrete price; confident safety assurances unsupported by mysterious-object context; too much closing repetition.',
}
ledger=[];shortlist=[];summary={}
for source in rows:
    counts=Counter(d['status'] for d in diags[source]);reasons=Counter(c for d in diags[source] for c in d['reasons'])
    full=full_realm if source=='realm' else full_sotopia
    for i,(r,d) in enumerate(zip(rows[source],diags[source]),1):
        nominee=source=='sotopia' and i in nominees
        item={'sample_id':r['sample_id'],'source':source,
          'parent_character_id':r.get('parent_character_id'),'nested_index':r.get('nested_index'),
          'episode_id':r.get('episode_id'),'environment_id':r.get('environment_id'),
          'model':r.get('model'),'actor_models':r.get('actor_models'),'experiment_tag':r.get('experiment_tag'),
          'mechanical_status':d['status'],'mechanical_reasons':d['reasons'],'metrics':d['metrics'],
          'semantic_review_scope':'full_selected_context_and_conversation' if i in full else 'topic/scenario and turn-excerpt screening; full semantic review still pending',
          'assessment':details.get((source,i),'Persona/lore interview in topic/turn screening; no strict scene-positive nomination.' if source=='realm' else 'Scenario/turn-excerpt screen only; no full-row quality verdict or positive nomination.'),
          'dimensions':{'voice':'human review pending' if not(i in full) else ('consistent persona voice' if source=='realm' else 'mostly formal social dialogue'),
            'progression':'see assessment; not automatically scored','agency':'v2 diagnostics only unless full-read assessment; no keyword-based approval',
            'continuity':'see assessment; unresolved portions require human review','repetition':'diagnostic only; zero signals does not establish absence',
            'card_or_metadata_leakage':'see mechanical reasons and projection notes','context_size':d['metrics']['approx_tokens'],
            'provenance':'pending'},
          'positive_nomination':nominee,'nomination_domain':'SFW social negotiation' if nominee else None,
          'human_review':'pending','provenance_status':'pending','training_ready':False}
        ledger.append(item)
        if nominee:
            shortlist.append({**item,**nominees[i],'source_revision':r['revision'],'source_line':r['source_line'],
                'raw_row_sha256':r['raw_row_sha256'],'public_profiles':r['public_profiles'],'scenario':r['scenario'],
                'social_interactions':r['social_interactions'],'simulator_event_note':'Terminal exit is preserved as an episode event; do not silently train it as a spoken assistant answer. No training-format export created.'})
    tokens=[d['metrics']['approx_tokens'] for d in diags[source]]
    summary[source]={'selected':len(rows[source]),'status_counts':dict(counts),'reason_counts':dict(reasons),
      'provenance_only':sum(d['provenance_only'] for d in diags[source]),
      'full_semantic_reads':len(full),'positive_nomination_count':len(nominees) if source=='sotopia' else 0,
      'human_approved':0,'training_ready':0,'tokens_min_median_max':[min(tokens),statistics.median(tokens),max(tokens)],
      'signals_by_family':dict(Counter(s['family'] for d in diags[source] for s in d['signals'])),
      'rows_with_absolutely_in_3_or_more_turns':sum(sum(bool(re.search(r'\babsolutely\b',m['content'],re.I)) for m in d['messages'][1:])>=3 for d in diags[source]),
      'terminal_exit_rows':sum(d['messages'][-1]['content']=='left the conversation' for d in diags[source])}
save('comparison.json',summary)
(OUT/'screening_ledger.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in ledger),encoding='utf-8')
(OUT/'positive_shortlist.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in shortlist),encoding='utf-8')
save('shortlist_locators.json',[{k:v for k,v in x.items() if k not in ['public_profiles','scenario','social_interactions']} for x in shortlist])

text='''# Two bounded RP source inspections

Completed 13 September 2026. Artifacts retain the collection directory date, 20260912.

**Recommendation:** do not expand GPT Role-play Realm to 500 conversations for the positive scene corpus. SOTOPIA-pi merits a targeted 500-episode **SFW negotiation** inspection, subject to the same evaluation exclusions and pending provenance. It does not solve the missing adult RP source or establish a broad character-RP corpus. The proposed adult-data route is a small rights-clear original-author seed corpus, followed by controlled synthetic expansion after separate authorization.

## Results and denominators

| Measure | Realm | SOTOPIA-pi | Previous PIPPA |
|---|---:|---:|---:|
| Selected/processed conversations | 100 | 100 | 500 |
| Rejected by frozen v2 + unchanged conversation rules | 4 (4%) | 67 (67%) | 294 (58.8%) |
| Quarantined, including provenance-only | 96 (96%) | 33 (33%) | 206 (41.2%) |
| Provenance-only review survivors | 96 (96%) | 32 (32%) | 21 (4.2%) |
| Other quarantine | 0 | 1 (1%) | 185 (37%) |
| Positive nominations produced | 0 (0%) | 2 (2%) | 0 (0%) |
| Human-approved / training-ready | 0 / 0 | 0 / 0 | 0 / 0 |

The nomination rate is the number actually shortlisted divided by the selected set, **not an estimated population acceptance rate**. All 200 rows received deterministic checks and topic/scenario/turn-excerpt screening. Four Realm and eight SOTOPIA conversations received a full semantic read including retained context; only full-read SOTOPIA rows are nominated. The remaining rows have no exhaustive semantic verdict. Human acceptance is exactly zero. PIPPA had 21 survivor screens and six full reads in its previous inspection; these are different sample designs, not a controlled ranking experiment.

No filter, pipeline, runtime, model, inference, core SFT or QLoRA configuration changed. No training or generation ran. Source manifests and provenance statuses remain unchanged; every retained row is human-review pending and training_ready=false.

## Sampling and exclusions

**Realm:** 100 nested conversations from **100 different parent character IDs**, spread at evenly spaced indices over all 216 English character rows. One conversation per character, alternating 50 GPT-4 and 50 GPT-3.5-turbo; a stable hash chooses a topic within each label. This balances generator labels, not all topics: GPT-4 supplies the first dialogue, so its topic position is confounded with model. IDs, parent indices, nested indices and model labels are preserved. English only; no multilingual or adult coverage claim.

Retrieved text JSON through the viewer. Parent envelopes contain unselected nested chats; only the selected 100 were retained/quality-screened. The viewer sends image URLs as metadata; no image content was fetched or used, and image fields were discarded. Successful distinct selected-response envelopes total 6,507,582 bytes. Rate-limited single-row retrieval resumed with 20-parent batches. The observed repository revision was unchanged before/after, but viewer caching is not commit-pinning.

Realm observed revision: `c90229b1916159a24d88428972b2771022d4b52a`.

**SOTOPIA:** 100 actual rows from pinned `sotopia_pi_episodes.jsonl`, spanning **72 distinct training environments**, maximum two episodes per environment. All selected rows have `sft_round_1_gpt-4_gpt-4_clean`, with GPT-4 as both actors. We attempted model stratification; encountered self-policy rows were explicitly tagged `test` and excluded. We did not weaken exclusions to fill a second model stratum.

The final selection pass screened metadata from 384 line positions, selecting 100 and excluding 203 documented evaluation IDs, 75 evaluation-tagged rows, five rows above the per-environment cap, and one boundary row. It read 12,371,032 bytes in two segments. An earlier balanced-stratum attempt was capped at 32 MiB and stopped on a partial line without a retained result; a subsequent 1 MiB metadata probe diagnosed the test-tag problem. Thus **100 selected episodes does not mean only 100 raw envelopes were transferred**. No full dataset, Redis dump or Social-IQA viewer rows were acquired for this task.

We used upstream `used_env.json`: training allowlists SFT-round-1 and selftrain-rounds; denylist includes 90 benchmark environments, the hard subset, the development set and explicit IDs from evaluation scripts. Unknown environments, evaluation/test/baseline/dev tags and duplicate episode IDs are excluded. The recorded training/deny sets are disjoint, and every selected row passes them. This establishes separation from the published exclusions inspected here, not every possible unpublished future evaluation.

SOTOPIA data revision: `e583406958ff132f6749ca87a2f9aa31ae3c0fa1`. Upstream exclusion evidence revision: `245fa7bbfc229a1e0f1aa3e79fa3cd26cad7b3bd`. [Upstream generation/evaluation documentation](https://github.com/sotopia-lab/sotopia-pi/blob/245fa7bbfc229a1e0f1aa3e79fa3cd26cad7b3bd/data_generate/README.md).

## Frozen-v2 diagnostic projection

The local inspection script imports the existing `normalize_conversation`, `assess` and `record_metrics` functions with the existing filter settings. It does not add production adapters or enable sources.

Realm context includes the persona, greeting and example dialogue for leakage detection; target turns contain only the selected nested chat. User/char maps to user/assistant. Three rows have only two exchanges; one ends on a user message. All others are provenance-only survivors. V2 found no card, agency, consent/age or repetition signals here; that is heuristic evidence, not proof of safety or quality.

SOTOPIA keeps public scenario/profile text, named dialogue and action events. Hidden goals, profile secrets, evaluator reasoning/rewards and raw prompt messages are excluded from retained candidate text. Actor/model/episode IDs remain metadata for audit only. No training-format export was produced. The first named actor maps to user and the second to assistant; no ending was removed or repaired to increase survival.

All 67 SOTOPIA structural rejections are `broken_speaker_order` from ending on the first actor/user. There are no adjacent same-role turns in the projection. **This is a format compatibility result, not evidence that 67 episodes are poor dialogues.** Eighty-one episodes include an explicit simulator exit event; those events must not become ordinary assistant speech. A future adapter would need an explicit, separately reviewed event/turn policy. It was not implemented here, and v2 remains frozen.

SOTOPIA flags across the whole set: five user-decision signals, one user-dialogue signal, one forced-movement signal, two HTML-card signals and one intimate-power-imbalance signal; reasons overlap. Two scenarios contain literal viewer HTML. None of those fields were executed or fetched. Zero v2 repetition signals contrasts with obvious agreement loops in excerpt screening: **22/100** episodes contain the word “absolutely” in at least three turns. That last count is descriptive only and is not a rejection rule.

## Quality findings

**Realm — good compact persona interviews, weak scene supply.** Distinct pirate/gnome voices appear alongside generic expert answers. Topic follow-ups usually preserve the interview subject and leave the user to ask questions. However, biography, cultural exposition and advice dominate the sampled prompts. `realm_005` keeps pirate diction but only recounts childhood; `realm_074` describes fairy magic and declines to enact it when the user expresses interest. `realm_100` progresses through a breathing exercise, then returns to generic spiritual advice; it is not a strong character scene. No strict scene-positive nominee was established.

Realm context totals are manageable: **372 / 699 / 1,004** approximate tokens (minimum/median/maximum, including persona/examples). These use characters/4, not the model tokenizer. Absence of explicit action also makes agency easier to preserve than in PIPPA; this is not evidence of superior agency under challenging roleplay.

**SOTOPIA — better negotiated state changes, narrow and formulaic voice.** Some episodes maintain offers, constraints and counteroffers through a concrete agreement. Many quickly reach consensus and continue affirming it. Examples include gaming-noise and chores discussions (`001`, `002`), extended collaboration praise (`035`, `055`) and repeated plans without resolution. In full reads, strangers inexplicably know each other's names (`055`, `057`, `086`); the bus-fare exchange in `086` does not clearly account for the remainder of a large bill. `094` negotiates an unspecified surcharge instead of settling a number. `037` respects different spiritual practices and reaches actual action, but switches a they/them character to “herself”; that is a consistency issue, not an objection to nonbinary identity.

SOTOPIA contexts are also manageable: **472 / 1,161 / 2,580** approximate tokens. Overall style is formal and agreeable; these are potential SFW social-skill examples, not demonstrated long-form, distinct-voice adventure or adult RP. Scenario roles can differ from the generated occupational profiles, so human review must assess whether the combination is coherent.

Negative-control observations: Realm `070` mentions children's ages in pet-selection advice and `074` specifies a fairy's short stature; neither received age/sexual review flags. SOTOPIA `049` discusses a young niece and `096` explores gender identity; their structural ending rejection is not a sexual/identity rejection. These are observational controls using frozen v2, not new tests or filter changes.

## Strict manual-review nominations

These two are nominated **only for SFW negotiation quality**. Both have full context/episode reads and no additional v2 review reason beyond provenance. They are not approved, not training-ready, and not evidence that the whole corpus is high quality. Their terminal simulator events remain explicit and require a future representation decision.

| Sample | Why nominated | Limits for human review |
|---|---|---|
| `sotopia_004` | Garden-space conflict leads to barrier, weekly maintenance and seasonal space review; each person makes their own proposals/choices. | Formal/verbose closing; agreement, not an enacted gardening scene. |
| `sotopia_061` | Apple bargaining preserves concrete prices and delivery conditions through a $1.70/lb agreement. | Volume threshold unspecified; listed occupations differ from scene roles; brief repetitive closing. |

`positive_shortlist.jsonl` contains their public context and full episodes. `shortlist_locators.json` contains IDs, hashes, source lines and review notes without transcript text. Realm's shortlist is empty. Near misses and review-depth labels are in the 200-row screening ledger; no unexamined span is given a semantic pass.

## Provenance and licensing

[Realm's pinned card](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm/blob/c90229b1916159a24d88428972b2771022d4b52a/README.md) declares CC BY 4.0 and describes GPT-4-generated characters/topics, one GPT-4 conversation and 19 GPT-3.5 conversations per character. This is clearer lineage than anonymous scraped cards, but declared licensing does not independently establish every upstream/generator permission. No image rights were relied on; no adult split was verified. Preserve attribution, source/model identity and the pending gate.

[SOTOPIA's pinned card](https://huggingface.co/datasets/cmu-lti/sotopia-pi/blob/e583406958ff132f6749ca87a2f9aa31ae3c0fa1/README.md) declares CC BY-SA 4.0. The code repository's Apache license is not a replacement dataset license. Its inspirations include NormBank, Social Chemistry and Social-IQA. [NormBank](https://github.com/SALT-NLP/normbank) and [Social Chemistry](https://github.com/mbforbes/social-chemistry-101) explicitly declare CC BY-SA 4.0; Social-IQA's complete upstream chain and generation terms remain unverified in this inspection. Attribution/share-alike obligations need a project-specific assessment before redistribution/use; no assumption about downstream model licensing was made. [CC BY-SA terms](https://creativecommons.org/licenses/by-sa/4.0/). Source provenance stays pending.

## Expand to 500?

- **Realm: no for the current positive-scene corpus.** The 96% mechanical survival is mostly clean exposition. It could be revisited as a separate character-voice/context resource if that objective is explicitly chosen; expanding this sampling design is unlikely to fix the missing progression.
- **SOTOPIA: yes for one targeted SFW negotiation investigation**, retaining training-environment exclusions, a per-environment cap and model tags. Start by having a human judge these two nominations and the noted formatting/profile limitations. Do not infer a 2% global yield from this deterministic sample or extrapolate it to thousands. Future work should examine additional documented training model strata only when their eligibility is established.
- **PIPPA stays the primary open-ended RP research lead**, with 4.2% mechanical survivors and no strict positive established in the earlier 500-prefix. SOTOPIA supplements social negotiation; it does not replace PIPPA's target domain. None establishes enough approved quality for a 1k–5k positive corpus today.

## Three options for the missing adult multi-turn source

| Option | Practical route | Advantages | Costs and unresolved risks | Decision |
|---|---|---|---|---|
| 1. Existing source with verified rights | Require a contributor-owned release with explicit rights to redistribute/use complete adult-character dialogues; inspect lineage and 100 episodes before acquisition. | Potentially fastest if verified; existing human variation. | **No qualifying existing source has been verified in our inspected evidence.** PIPPA remains pending; Gryphe remains rights-blocked. Neither source studied here establishes an adult slice. This task did not reopen deprioritized datasets or conduct another adult-source crawl. | Keep as an opportunistic route, not the dependency for the project. |
| 2. Controlled synthetic collection from original cards | Author original fictional adult personas/scenes; document generator/output terms; separate actor turns from planning/evaluation; use bounded scenes, independent review and held-out scenarios. | Repeatable, controllable coverage and lineage; can target agency and continuity. | Upfront design and review work; risk of shared-model agreement loops, copied examples, weak user turns and evaluator bias. Clear card ownership alone does not settle generator terms or output quality. | Useful expansion mechanism after a human-written quality anchor exists. |
| 3. Original-author seed corpus, then controlled expansion | Commission/author roughly 30–50 complete multi-turn scenes across 10–20 original characters as a planning target; obtain written contributor permissions and track revisions; manually judge the seed before any authorized synthetic expansion. | Clearest attainable authorship trail and strongest editorial control of voice, user agency, boundaries and actual progression. | Highest initial human effort; contributor diversity and licensing still need documentation; synthetic descendants need fresh review and deduplication. | **Recommended.** Start with a small quality reference, then apply option 2 selectively. |

For options 2/3, future authoring requirements should specify fictional adult participants, clear consent/boundaries, no identifiable real-person imitation, separate user decisions, and preserved source/contributor lineage. Human approval remains the final gate for every accepted sample. These are collection/governance proposals only: no adult generation code, prompts, inference jobs or training were implemented or launched.

## Artifacts and verification

Collection scripts and all text samples are local under `data/sft/rp_v1/generated/realm_sotopia_100_20260912/`. `comparison.json`, `screening_ledger.jsonl`, `shortlist_locators.json`, exclusions and collection manifests preserve exact identities/denominators. `verification.json` checks all 200 IDs, parent/model diversity, SOTOPIA exclusions, absence of private fields, frozen tracked-file hashes and pending statuses. Existing v2 was executed for this investigation; no new filter tests or config changes were introduced. Raw source text is not published automatically.
'''
(OUT/'report.md').write_text(text,encoding='utf-8')

before=json.loads((OUT/'tracked_before.json').read_text())
changed=[n for n,h in before.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
assert not changed,changed
assert len(rows['realm'])==len(rows['sotopia'])==100
assert len({r['parent_character_id'] for r in rows['realm']})==100
assert Counter(r['model'] for r in rows['realm'])=={'gpt-4':50,'gpt-3.5-turbo':50}
assert len({r['episode_id'] for r in rows['sotopia']})==100
ex=json.loads((OUT/'sotopia_exclusions.json').read_text());allowed=set(ex['training_environment_ids']);denied=set(ex['excluded_environment_or_episode_ids'])
assert not allowed&denied
for r in rows['sotopia']:
    assert r['environment_id'] in allowed and r['environment_id'] not in denied and r['episode_id'] not in denied
    assert not re.search(ex['excluded_tag_pattern'],r['experiment_tag'],re.I)
    assert not set(r)&{'social_goals','reasoning','rewards','raw_rewards','raw_rewards_prompt','raw_messages'}
    assert not any('secrets:' in p.lower() for p in r['public_profiles'].values())
assert max(Counter(r['environment_id'] for r in rows['sotopia']).values())<=2
assert all(not r['training_ready'] and r['human_review']=='pending' and r['provenance_status']=='pending' for source in rows for r in rows[source])
assert len(shortlist)==2 and all(not r['training_ready'] and r['human_review']=='pending' for r in shortlist)
assert all(diags['sotopia'][n-1]['provenance_only'] for n in nominees)
artifacts=['realm_samples.jsonl','sotopia_samples.jsonl','realm_diagnostics.jsonl','sotopia_diagnostics.jsonl','positive_shortlist.jsonl','report.md']
save('verification.json',{'passed':True,'tracked_files_checked':len(before),'changed_tracked_files':changed,'realm_rows':100,
 'realm_distinct_characters':100,'realm_model_counts':{'gpt-4':50,'gpt-3.5-turbo':50},'sotopia_rows':100,
 'sotopia_distinct_environments':72,'heldout_id_intersections':0,'evaluation_tag_rows':0,'private_metadata_in_candidate_text':False,
 'training_ready_rows':0,'human_approved_rows':0,'human_pending_rows':200,'positive_nominations':2,
 'full_semantic_reads':12,'image_requests':0,'artifact_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in artifacts},
 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
print(json.dumps({'verified':True,'selected':200,'nominated':2,'approved':0,'changed_tracked_files':changed}))
