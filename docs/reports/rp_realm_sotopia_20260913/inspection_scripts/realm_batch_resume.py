import hashlib,json,urllib.parse
from collections import Counter
from collect import OUT,get,dump
repo='IlyaGusev/gpt_roleplay_realm'
revision=json.loads(get('https://huggingface.co/api/datasets/'+repo,32768)[0])['sha']
indices=[round(i*215/99) for i in range(100)]
network=[]
for start in range(0,216,20):
    targets={index:i for i,index in enumerate(indices) if start<=index<start+20 and not (OUT/f'realm_selected_{i:03d}.json').exists()}
    if not targets:continue
    url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=repo,config='default',split='en',offset=start,length=min(20,216-start)))
    raw,headers=get(url,2*1024*1024)
    payload=json.loads(raw);assert payload['num_rows_total']==216
    network.append(dict(url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    for wrapper in payload['rows']:
        index=wrapper['row_idx']
        if index not in targets:continue
        assert not wrapper['truncated_cells'],wrapper['truncated_cells']
        i=targets[index];r=wrapper['row'];label='gpt-4' if i%2==0 else 'gpt-3.5'
        choices=[j for j,d in enumerate(r['dialogues']) if d['model_name'].startswith(label)];assert choices
        j=choices[int(hashlib.sha256(r['char_id'].encode()).hexdigest(),16)%len(choices)]
        d=r['dialogues'][j]
        result=dict(source='realm',repo=repo,parent_index=index,parent_character_id=r['char_id'],sample_id=f'realm_{i+1:03d}',
            character=r['name'],context=r['context'],greeting=r['greeting'],example_dialogue=r['example_dialogue'],
            nested_index=j,model=d['model_name'],topic=d['topic'],chat=d['chat'],retrieval_url=url,
            response_bytes=len(raw),response_sha256=hashlib.sha256(raw).hexdigest(),source_revision_observed=revision,
            revision_binding='Viewer cache not commit-pinned; live repository revision checked before/after.',human_review='pending',provenance_status='pending',training_ready=False)
        dump(f'realm_selected_{i:03d}.json',result)
    dump('realm_batch_network.json',network)
    print('batch',start,'selected cached',len(list(OUT.glob('realm_selected_*.json'))),flush=True)
rows=[json.loads((OUT/f'realm_selected_{i:03d}.json').read_text(encoding='utf-8')) for i in range(100)]
assert len({r['parent_character_id'] for r in rows})==100
assert revision==json.loads(get('https://huggingface.co/api/datasets/'+repo,32768)[0])['sha']
(OUT/'realm_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
responses={r['retrieval_url']:r['response_bytes'] for r in rows}
dump('realm_collection.json',dict(rows=100,distinct_characters=100,split='en',model_counts=dict(Counter(r['model'] for r in rows)),
    response_bytes=sum(responses.values()),revision=revision,image_requests=0,
    sampling='100 evenly spaced parent indices across 216 English characters; one chat per parent, 50 GPT-4 and 50 GPT-3.5, hashed topic selection within label.',
    retrieval_note='Parent envelopes contain unselected chats; only selected 100 retained or quality-inspected. Rate-limited single-row requests resumed using 20-parent batches. No image bytes fetched. Responses are cached, not commit-pinned.'))
print('Realm complete',flush=True)
