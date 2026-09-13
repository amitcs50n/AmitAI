"""Isolated bounded inspection. No source adapters/config changes; no generation."""
import concurrent.futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
import urllib.error
import time
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'source_quality_20260912'
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def get(url,cap):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=30) as response:
                raw=response.read(cap+1)
                headers=dict(response.headers)
            break
        except urllib.error.HTTPError as exc:
            if exc.code!=429 or attempt==3:raise
            time.sleep(10*(attempt+1))
    if len(raw)>cap:raise ValueError('response_cap_exceeded')
    return raw,headers
def realm():
    repo='IlyaGusev/gpt_roleplay_realm'
    before=json.loads(get('https://huggingface.co/api/datasets/'+repo,32768)[0])['sha']
    indices=[round(i*215/99) for i in range(100)]
    def one(pair):
        i,index=pair
        cached=OUT/f'realm_selected_{i:03d}.json'
        if cached.exists():return json.loads(cached.read_text(encoding='utf-8'))
        url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=repo,config='default',split='en',offset=index,length=1))
        raw,headers=get(url,262144)
        payload=json.loads(raw)
        assert payload['num_rows_total']==216
        wrapper=payload['rows'][0]
        assert not wrapper['truncated_cells']
        r=wrapper['row']
        # One conversation per parent. Balance teacher labels independently of content.
        preferred='gpt-4' if i%2==0 else 'gpt-3.5'
        choices=[j for j,d in enumerate(r['dialogues']) if d['model_name'].startswith(preferred)]
        assert choices
        j=choices[int(hashlib.sha256(r['char_id'].encode()).hexdigest(),16)%len(choices)]
        chosen=r['dialogues'][j]
        result=dict(source='realm',repo=repo,parent_index=index,parent_character_id=r['char_id'],
          sample_id=f'realm_{i+1:03d}',character=r['name'],context=r['context'],greeting=r['greeting'],
          example_dialogue=r['example_dialogue'],nested_index=j,model=chosen['model_name'],topic=chosen['topic'],
          chat=chosen['chat'],retrieval_url=url,response_bytes=len(raw),response_sha256=hashlib.sha256(raw).hexdigest(),
          source_revision_observed=before,revision_binding='Viewer cache not commit-pinned; live repository revision checked before/after.',
          human_review='pending',provenance_status='pending',training_ready=False)
        # The viewer supplies image URLs, not pixels. Do not retain/follow them.
        dump(cached.name,result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        results=list(pool.map(one,enumerate(indices)))
    after=json.loads(get('https://huggingface.co/api/datasets/'+repo,32768)[0])['sha']
    assert before==after
    assert len(results)==len({x['parent_character_id'] for x in results})==100
    (OUT/'realm_samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8')
    dump('realm_collection.json',dict(rows=100,distinct_characters=100,split='en',model_counts=dict(Counter(x['model'] for x in results)),
        response_bytes=sum(x['response_bytes'] for x in results),revision=before,image_requests=0,
        sampling='100 evenly spaced parent indices across 216 English characters; one chat per parent, alternating GPT-4/GPT-3.5 labels, hashed topic selection within label.',
        retrieval_note='100 parent envelopes contain unselected nested chats; only selected 100 retained or quality-inspected. No image bytes fetched.'))
    print('Realm complete: 100 distinct characters',flush=True)

def sotopia():
    repo='cmu-lti/sotopia-pi'
    meta=json.loads((PREV/'cmu-lti__sotopia-pi.json').read_text(encoding='utf-8'))
    revision=meta['revision']
    envs=json.loads((OUT/'upstream_data_generate__env_files__used_env.json').read_text())
    denied=set(envs['sotopia_env'])|set(envs['sotopia_hard_env'])|set(envs['dev-round-1'])
    allowed=set().union(*(set(v) for k,v in envs.items() if k=='SFT-round-1' or k.startswith('selftrain-')))-denied
    assert not allowed&denied
    deny_ids=set()
    for path in OUT.glob('upstream*eval*'):
        deny_ids.update(re.findall(r'\b[0-9A-HJKMNP-TV-Z]{26}\b',path.read_text(encoding='utf-8')))
    denied |= deny_ids
    allowed -= denied
    dump('sotopia_exclusions.json',dict(upstream_revision=json.loads((OUT/'upstream_tree.json').read_text())['sha'],
      training_environment_ids=sorted(allowed),excluded_environment_or_episode_ids=sorted(denied),
      excluded_tag_pattern='test|eval|baseline|dev|held.?out|validation',
      evidence_files=['upstream_data_generate__env_files__used_env.json','upstream_data_generate__README.md',
                      'upstream_data_process__utils__human_eval_episodes.py','upstream_llm_self_train__pipelines__submit_single_eval.sh'],
      note='Exclude documented benchmark/dev environments plus explicit IDs in evaluation scripts; require training environment allowlist and non-evaluation tag. Unknown environments are excluded.'))
    cap=8*1024*1024
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/sotopia_pi_episodes.jsonl'
    counts=Counter(); strata=Counter(); scanned_tags=Counter(); selected=[]; chosen_envs=Counter(); ids=set(); read=0; line_number=0; digest=hashlib.sha256()
    offset=0;prior=None
    if (OUT/'sotopia_collection.json').exists():
        prior=json.loads((OUT/'sotopia_collection.json').read_text())
        selected=[json.loads(x) for x in (OUT/'sotopia_samples.jsonl').read_text(encoding='utf-8').splitlines()]
        offset=prior['bytes_read'];line_number=prior['raw_lines_screened_for_eligibility']
        counts.update(prior['selection_exclusions']);scanned_tags.update(prior['scanned_tag_counts'])
        strata.update(x['stratum'] for x in selected);chosen_envs.update(x['environment_id'] for x in selected);ids.update(x['episode_id'] for x in selected)
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+cap-1}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=40) as response:
        if offset:
            assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {offset}-')
            remainder=response.readline(1048577);read+=len(remainder);digest.update(remainder);line_number+=1
            counts['boundary_row_skipped']+=1
        while read<cap and len(selected)<100:
            raw=response.readline(min(1048577,cap-read));read+=len(raw);digest.update(raw)
            if not raw:break
            if not raw.endswith(b'\n'):
                counts['bounded_partial_line']+=1
                break
            line_number+=1;r=json.loads(raw)
            tag=r.get('experiment_tag','');env=r.get('environment_id');eid=r.get('episode_id'); models=r.get('experiment_model_name_pairs',[])
            scanned_tags[tag]+=1
            if env in denied or eid in denied:counts['documented_evaluation_id']+=1;continue
            if re.search(r'test|eval|baseline|dev|held.?out|validation',tag,re.I):counts['evaluation_tag']+=1;continue
            if env not in allowed:counts['unknown_or_nontraining_environment']+=1;continue
            if len(models)!=3:counts['unknown_model_shape']+=1;continue
            actors=tuple(models[1:]);stratum='expert' if all(m=='gpt-4' for m in actors) else 'self' if all(m=='custom_model' for m in actors) else 'other'
            if stratum=='other':counts['other_actor_pair']+=1;continue
            # Probe found self-policy rows explicitly tagged test. Do not relabel
            # them as train merely to fill a stratum; allow expert-only fallback.
            if stratum!='expert':counts['nonexpert_unverified_training_tag']+=1;continue
            if chosen_envs[env]>=2:counts['environment_cap_two']+=1;continue
            if eid in ids:counts['duplicate_episode']+=1;continue
            # Exclude goals, evaluator content, rewards and raw prompts from retained rows.
            backgrounds={name:text.split(f"{name.split()[0]}'s secrets:")[0].split(f"{name}'s secrets:")[0].strip() for name,text in r['agents_background'].items()}
            # Defensive general secret marker, independent of first-name assumptions.
            backgrounds={name:re.split(r"\b[^.\n]{0,80}['’]s secrets:",text)[0].strip() for name,text in backgrounds.items()}
            item=dict(source='sotopia',sample_id=f'sotopia_{len(selected)+1:03d}',repo=repo,revision=revision,
              source_line=line_number,episode_id=eid,environment_id=env,agent_ids=r['agent_ids'],experiment_tag=tag,
              evaluator_model=models[0],actor_models=list(actors),stratum=stratum,scenario=r['scenario'],
              public_profiles=backgrounds,social_interactions=r['social_interactions'],
              raw_row_sha256=hashlib.sha256(raw).hexdigest(),human_review='pending',provenance_status='pending',training_ready=False)
            selected.append(item);strata[stratum]+=1;chosen_envs[env]+=1;ids.add(eid)
    (OUT/'sotopia_samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected),encoding='utf-8')
    dump('sotopia_collection.json',dict(rows=len(selected),bytes_read=offset+read,byte_cap=offset+cap,raw_lines_screened_for_eligibility=line_number,
      segment_sha256=digest.hexdigest(),segment_start=offset,segment_bytes=read,prior_collection=prior,revision=revision,url=url,selection_exclusions=dict(counts),model_strata=dict(strata),
      distinct_environments=len({x['environment_id'] for x in selected}),scanned_tag_counts=dict(scanned_tags),
      selected_tag_counts=dict(Counter(x['experiment_tag'] for x in selected)),
      sampling='100 eligible expert-model episodes, maximum two per training environment. Self-policy rows encountered had test tags, so excluded. Final metadata eligibility scan capped at 8 MiB.',
      prior_collection_attempt='Initial balanced-stratum eligibility scan capped at 32 MiB ended on a partial line before quotas could fill; no raw rows retained. A subsequent 1 MiB metadata probe identified test-tagged self-policy rows. No evaluation content admitted for quality review.',
      hidden_goals_retained=False,evaluator_reasoning_retained=False,rewards_retained=False,prompt_metadata_in_transcripts=False))
    print(f'SOTOPIA selected {len(selected)}; strata {dict(strata)}; bytes {read}',flush=True)
    assert len(selected)==100,'Not enough eligible episodes within bound; inspect collection audit before any further retrieval.'

if __name__=='__main__':
    import sys
    (realm if sys.argv[1]=='realm' else sotopia)()
