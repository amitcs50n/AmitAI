"""Apply frozen v2 to explicit text projections; no pipeline/config mutations."""
import json
import re
import sys
from collections import Counter
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from training.rp_data import normalize_conversation,assess,record_metrics,RowError
import yaml
OUT=Path(__file__).resolve().parent
CONFIG=yaml.safe_load((ROOT/'configs/rp_v1_sources.yaml').read_text())

def project(r):
    if r['source']=='realm':
        system='Character: '+r['character']+'\n'+r['context']+'\nGreeting: '+r['greeting']+'\nExample dialogue:\n'+'\n'.join(m['role']+': '+m['content'] for m in r['example_dialogue'])
        chat=[{'role':{'char':'assistant','user':'user'}[m['role']],'content':m['content']} for m in r['chat']]
        return [{'role':'system','content':system}]+chat
    names=list(r['public_profiles'])
    pat=re.compile(r'^('+ '|'.join(re.escape(x) for x in names)+r')( said: |: )',re.M)
    text=r['social_interactions'];matches=list(pat.finditer(text));chat=[]
    assert len(matches)>0 and not text[:matches[0].start()].strip()
    mapping={names[0]:'user',names[1]:'assistant'}
    for i,m in enumerate(matches):
        content=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        if m[2]==' said: ' and content.startswith('"') and content.endswith('"'):content=content[1:-1]
        chat.append({'role':mapping[m[1]],'content':content,'source_speaker':m[1],'event_kind':'speech' if m[2]==' said: ' else 'action'})
    system='Scenario: '+r['scenario']+'\nPublic participant profiles:\n'+'\n'.join(f'{n}: {r["public_profiles"][n]}' for n in names)
    return [{'role':'system','content':system}]+chat

def run(source):
    path=OUT/f'{source}_samples.jsonl'
    if not path.exists():return
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    results=[];sc=Counter();reasons=Counter();families=Counter()
    for r in rows:
        messages=project(r)
        item={'sample_id':r['sample_id'],'messages':messages,'human_review':'pending','provenance_status':'pending','training_ready':False}
        src={'adapter':'sharegpt','provenance_status':'pending'}
        raw={'conversations':[{'from':m['role'],'value':m['content']} for m in messages]}
        # Assess quality/safety diagnostics even when the source's event structure
        # is not compatible with the frozen conversation normalizer.
        direct={'messages':messages}
        decision=assess(direct,src,CONFIG['filters'])
        structural=None
        try:normalize_conversation(raw,src,CONFIG)
        except RowError as exc:structural=str(exc)
        item.update(status='rejected' if structural else decision.status,
            reasons=sorted(set(decision.reasons+((structural,) if structural else ()))),
            normalizer_error=structural,signals=[s.as_dict() for s in decision.signals],metrics=record_metrics(direct),
            final_role=messages[-1]['role'],adjacent_same_speaker=sum(a['role']==b['role'] for a,b in zip(messages[1:],messages[2:])),
            provenance_only=(not structural and decision.reasons==('source_provenance_review',)))
        sc[item['status']]+=1;reasons.update(item['reasons']);families.update(set(s.family for s in decision.signals))
        results.append(item)
    (OUT/f'{source}_diagnostics.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8')
    (OUT/f'{source}_metrics.json').write_text(json.dumps({'selected':len(rows),'status_counts':sc,'reason_counts':reasons,
        'signal_family_rows':families,'provenance_only':sum(x['provenance_only'] for x in results),
        'content_only_no_review_flags':sum(set(x['reasons'])<= {'source_provenance_review','broken_speaker_order'} for x in results),
        'context_character_sizes':sorted(len(x['messages'][0]['content']) for x in results),
        'approx_token_sizes':sorted(x['metrics']['approx_tokens'] for x in results),
        'projection_note':'No turn removal/merging or ending repair. Structural rejects remain rejects. Content-only diagnostics are reported separately and never acceptance.',
        'training_ready':False},indent=2)+'\n',encoding='utf-8')
    print(source,dict(sc),'provenance_only',sum(x['provenance_only'] for x in results),'reasons',dict(reasons))

if __name__=='__main__':
    for source in ['realm','sotopia']:run(source)
