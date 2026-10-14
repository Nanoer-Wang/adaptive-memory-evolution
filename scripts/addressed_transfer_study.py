"""Frozen public qa6 transfer plus clearly marked repetition stress.

No checkpoint, threshold, ontology or policy is fitted using qa6.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,time,urllib.request,tarfile,io,re
import numpy as np
import torch
from src.memory.addressed_codec import AddressedCodec,GuardedWriter
from src.data.babi_signed import assertion,question
from src.data.babi_events import read_key
from src.provenance import snapshot_source,runtime_info

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/addressed-memory-v1'
OUT=ROOT/'runs/addressed-transfer-v2'
URL='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'
SHA='84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def download():
    previous=ROOT/'runs/addressed-transfer-v1/data_provenance.json'
    if previous.exists():
        manifest=json.loads(previous.read_text())
        if all(hashlib.sha256((ROOT/'data/babi'/f['file']).read_bytes()).hexdigest()==f['sha256'] for f in manifest['files']):
            save('data_provenance.json',manifest)
            return
    blob=urllib.request.urlopen(URL,timeout=120).read()
    if hashlib.sha256(blob).hexdigest()!=SHA:raise ValueError('Archive hash mismatch')
    paths=[]
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
        for split in ('train','test'):
            name=f'qa6_yes-no-questions_{split}.txt'
            member=f'tasks_1-20_v1-2/en/{name}'
            data=archive.extractfile(member).read();path=ROOT/'data/babi'/name
            if path.exists() and path.read_bytes()!=data:raise ValueError('Existing source differs')
            path.write_bytes(data);paths.append(dict(file=name,sha256=hashlib.sha256(data).hexdigest()))
    save('data_provenance.json',dict(url=URL,archive_sha256=SHA,files=paths,scope='Public bAbI en1k qa6; same generator family as qa1, not broad out-of-domain validation'))


def parse(path,entities,values,yesno):
    groups=[];story=[];sid=-1
    for raw in path.read_text().splitlines():
        n,text=raw.split(' ',1);n=int(n)
        if n==1:
            if story:groups.append(story)
            story=[];sid+=1
        if '\t' in text:
            q,label,support=text.split('\t')
            if yesno:owner,room=question(q);target=values.index(room)
            else:owner=read_key(q)[0];target=None
            story.append(dict(type='query',id=f'{path.stem}:{sid}:{n}',story=sid,owner=entities.index(owner),target=target,label=label))
        else:
            # This public task includes possession distractors. Recognize their
            # syntax explicitly using training-side schema; they do not change
            # a person's location. Never silently ignore arbitrary unknown text.
            if re.fullmatch(r'\w+ (?:got|took|grabbed|picked up|left|dropped|discarded|put down) the (?:apple|football|milk)(?: there)?\.',text):
                story.append(dict(type='nonlocation',owner=None))
                continue
            owner,relation,rooms=assertion(text)
            if relation!='positive':raise ValueError('Unsupported transfer assertion')
            story.append(dict(type='event',owner=entities.index(owner),value=values.index(rooms[0])))
    if story:groups.append(story)
    return groups


def run(model,groups,values,policy,repeats,name,dataset):
    owners=[torch.tensor([i]) for i in range(4)];vectors=[torch.eye(6)[i:i+1] for i in range(6)]
    predictions=[];costs=[]
    with torch.no_grad():
        # Warm up kernels outside measurement.
        warm=GuardedWriter(model,policy=='dedup')
        for i in range(30):warm.write(i%4,i%6,owners[i%4],vectors[i%6])
        for sid,story in enumerate(groups):
            writer=GuardedWriter(model,policy=='dedup');write_ns=0;read_ns=0;questions=0
            for event in story:
                owner=event['owner']
                if event['type']=='event':
                    value=event['value']
                    # Repeat insertion is a deterministic derived stress, not native data.
                    a=time.perf_counter_ns()
                    for _ in range(repeats):writer.write(owner,value,owners[owner],vectors[value])
                    write_ns+=time.perf_counter_ns()-a
                elif event['type']=='query':
                    # Serialize all persistent data; restore both hidden and guard.
                    payload=writer.pack();assert len(payload)==writer.payload_bytes
                    restored=GuardedWriter(model,policy=='dedup');restored.restore(payload)
                    a=time.perf_counter_ns();logits=model.read(restored.state,owners[owner]);read_ns+=time.perf_counter_ns()-a
                    value=int(logits.argmax(-1));questions+=1
                    pred=values[value] if event['target'] is None else ('yes' if value==event['target'] else 'no')
                    predictions.append(dict(id=event['id'],story=sid,method=name,dataset=dataset,policy=policy,repeats=repeats,
                                            label=event['label'],prediction=pred,correct=pred==event['label']))
            costs.append(dict(story=sid,observations=writer.observations,writes=writer.writes,write_ns=write_ns,read_ns=read_ns,questions=questions))
    observations=sum(c['observations'] for c in costs);writes=sum(c['writes'] for c in costs)
    return predictions,costs,dict(method=name,dataset=dataset,policy=policy,repeats=repeats,state_bytes=writer.payload_bytes,
        accuracy=float(np.mean([p['correct'] for p in predictions])),questions=len(predictions),observations=observations,writes=writes,
        skip_fraction=1-writes/observations,write_us_per_observation=sum(c['write_ns'] for c in costs)/observations/1000,
        read_us_per_question=sum(c['read_ns'] for c in costs)/len(predictions)/1000)


def main():
    if not (SOURCE/'complete.json').exists():raise SystemExit('Training must complete and freeze first')
    if OUT.exists():raise SystemExit('Frozen transfer run exists')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    original=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    selected=[r for r in json.loads((SOURCE/'summary.json').read_text()) if r['slot_width']==8]
    # All three methods and all three seeds: no best-test seed or model selection.
    for r in selected:
        if hashlib.sha256((SOURCE/f'{r["method"]}.pt').read_bytes()).hexdigest()!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    signature,archive=snapshot_source()
    save('protocol.json',dict(signature=signature,source_archive=archive,runtime=runtime_info(),checkpoints=selected,
        policy='always and exact semantic-ID dedup; cache costs 4 extra bytes; guard is NOT learned/liquid and itself suffices to answer this finite ontology task.',
        native_tasks=['qa1','qa6'],repeats=[1,4],checkpoints_frozen_before_qa6_download=True,
        adapter_audit='v1 stopped before scoring: unsupported possession sentence in qa6 test. v2 schema extended from qa6 TRAIN only; explicit possession no-op for location task. Not pristine schema-blind transfer. No labels used to change models.',
        conditions='Native first. Fourfold repeats duplicate every observed event before next query, preserve label; transformed stress is not official benchmark.',
        selection='128-byte models of all kinds and seeds predeclared; no further training or tuning.',
        evaluation='Full official test, qa6 new to this experiment but shares entities/rooms/generator with qa1. No guarantee of distinct raw stories; audit overlap separately.',
        efficiency='All gate and neural update overhead counted, input parsing/tensor encoding/serialization/generation excluded. Single-thread CPU batch1; per-story costs saved.',
        acceptance='No more than one percentage point accuracy loss AND at least 20% measured write-path reduction versus same always-write model; engineering gate benefit, not liquid superiority.',
        statistical='Exploratory story-paired bootstrap, all seeds averaged per story; previously seen qa1 is development.'))
    download()
    datasets={task:parse(ROOT/f'data/babi/{file}_test.txt',original['entities'],original['values'],task=='qa6') for task,file in [('qa1','qa1_single-supporting-fact'),('qa6','qa6_yes-no-questions')]}
    save('parser_audit.json',{task:{kind:sum(e['type']==kind for g in groups for e in g) for kind in ('event','query','nonlocation')} for task,groups in datasets.items()})
    # Exact event-sequence overlap with qa1 train; keep full primary benchmark.
    from scripts.bounded_codec_study import prepare
    prior,_,_,_=prepare()
    train_full={}
    for r in prior:
        if r['split'] in ('train','validation'):train_full[r['story']]=r['events']
    fingerprints={tuple(map(tuple,events)) for events in train_full.values()}
    matched=[i for i,g in enumerate(datasets['qa6']) if tuple((e['owner'],e['value']) for e in g if e['type']=='event') in fingerprints]
    save('overlap_audit.json',dict(qa6_test_stories=len(datasets['qa6']),qa1_train_full_event_sequence_matches=matched,
        caveat='Exact complete event-sequence overlap only; short prefixes and generator vocabulary are shared. Full score primary.'))
    results=[];all_predictions=[];timings=[]
    # Exact table establishes native and transformed correctness upper baseline.
    exact=[]
    for dataset,groups in datasets.items():
        for story in groups:
            table=bytearray([255]*4)
            for event in story:
                if event['type']=='event':table[event['owner']]=event['value']
                elif event['type']=='query':
                    value=table[event['owner']]
                    pred=original['values'][value] if event['target'] is None else ('yes' if value==event['target'] else 'no')
                    exact.append(dict(dataset=dataset,id=event['id'],prediction=pred,label=event['label']))
    save('exact_table.json',dict(payload_bytes=4,results=exact))
    for item in selected:
        model=AddressedCodec(item['kind'],8);model.load_state_dict(torch.load(SOURCE/f'{item["method"]}.pt',weights_only=True));model.eval()
        for dataset,groups in datasets.items():
            for repeats in (1,4):
                for policy in (('always','dedup') if item['seed']%2 else ('dedup','always')):
                    predictions,costs,result=run(model,groups,original['values'],policy,repeats,item['method'],dataset)
                    all_predictions.extend(predictions);results.append(result)
                    timings.append(dict(method=item['method'],dataset=dataset,repeats=repeats,policy=policy,stories=costs))
        save('summary.json',results);save('timing.json',timings)
        print(item['method'],'transfer conditions completed',flush=True)
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in all_predictions))
    save('complete.json',dict(configurations=len(results),predictions=len(all_predictions),api_requests=0))


if __name__=='__main__':main()
