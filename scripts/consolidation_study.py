"""Public bAbI multi-query consolidation study with causal external storage."""
import sys, json, time, hashlib, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from src.config import atomic_json
from src.provenance import snapshot_source
from src.data.babi_events import stories, read_key
from src.models.content_encoder import ContentEncoder
from src.memory.consolidation import ConsolidationController,ConsolidationMemory,PrefixFeatures,project,KINDS
OUT=ROOT/'runs/liquid-consolidation-v2'
SEEDS=[11,23,42];BUDGETS=[1024,2048,4096]
CONDITIONS=['regular','bursty','stretched']
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def time_at(line,condition):
    if condition=='regular':return float(line)
    if condition=='stretched':return float(line)*10
    return sum([.01,.01,.01,7.][i%4] for i in range(line))


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    source,archive=snapshot_source();provenance=read(ROOT/'data/babi/provenance.json')
    for name,digest in provenance['files'].items():assert hashlib.sha256((ROOT/'data/babi'/name).read_bytes()).hexdigest()==digest
    datasets={};texts=set()
    for split in ['train','test']:
        original=stories(ROOT/f'data/babi/qa1_single-supporting-fact_{split}.txt',split)
        for index,story in enumerate(original):
            target='test' if split=='test' else ('train' if index<120 else 'val')
            if target=='val' and index>=160:continue
            events=[]
            for event in story:
                if isinstance(event,dict):events.append(dict(type='query',**event,key=json.dumps(read_key(event['question']))))
                else:
                    texts.add(event.quote);events.append(dict(type='observation',line=event.turn,text=event.quote,key=json.dumps(event.key),value=event.value))
            datasets[f'{split}-{index}']=dict(split=target,events=events)
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),source=source,archive=archive,dataset=provenance,
        counts={s:sum(d['split']==s for d in datasets.values()) for s in ['train','val','test']},seeds=SEEDS,budgets=BUDGETS,conditions=CONDITIONS,
        hypothesis='Liquid dynamics control admission/consolidation/forgetting of external evidence; latent state is not factual storage.',
        training='Training120 complete stories, validation40 from official train, official test200 stories. Train regular and bursty schedules; test includes10x time stretch. Future support annotations supervise admission, importance and horizon retention ONLY on training stories. No labels/questions in controller inputs. Feature/state updates occur on every observation identically in training and execution; admission does not alter prefix inputs.',
        objective='Weighted admission BCE +0.5 importance Brier +0.5 horizon retention BCE; same future-window labels for all controllers;20epochs AdamW0.003, gradient clip1; validation objective chooses first lowest epoch. Hidden8, projected content32 plus8prefix features. beta=0 and w=1 retrained independently with their disabled loss terms removed.',
        feedback='At each actual query, reinforce returned evidence after successful exact task answer; success signal is simulated task feedback, available identically to every policy. Support IDs NEVER select runtime writes, reads or feedback.',
        baselines='FIFO,constant exponential decay, pooled-free static,linear,GRU; all get exact same trusted bAbI key parser and version watermarks. All share512bytes reserved persistent controller/prefix metadata, byte-accounted evidence/provenance, same retrieval. Shared model weights reported separately.',
        limitations='Public synthetic bAbI, not natural lifelong-agent success. Timestamps are controlled interventions, not dataset real dates. Story-level reset preserves user boundaries; multiple queries inside each story. This project already used bAbI previously, so no untouched-confirmation claim. Dynamic controller uses exact piecewise-linear flow, not canonical LTC/CfC. No semantic abstraction beyond exact duplicate consolidation.',
        storage='Packed payload byte accounting, not Python RSS/JSON snapshot size. beta0 disables continuous strength decay; hard capacity eviction and explicit version supersession remain. Similarity is never truth.'))
    protocol=read(OUT/'protocol.json');protocol['admission']='Competitive candidate admission, no arbitrary probability threshold when capacity is available. alpha controls initial strength, not a fixed0.5reject gate. Previous v1 threshold run preserved as exploratory diagnosis; v2 uses same training recipe and already-exposed test stories.';atomic_json(OUT/'protocol.json',protocol)
    encoder=ContentEncoder();ordered=sorted(texts);v=encoder.encode(ordered);vectors={t:project(x).tolist() for t,x in zip(ordered,v)};encoder.close()
    atomic_json(OUT/'stories.json',datasets);atomic_json(OUT/'vectors.json',vectors);atomic_json(OUT/'prepared.json',dict(stories=len(datasets),unique_texts=len(texts)))


def training_arrays(split):
    vectors=read(OUT/'vectors.json');samples=[]
    for sid,story in read(OUT/'stories.json').items():
        if story['split']!=split:continue
        for condition in ['regular','bursty']:
            prefix=PrefixFeatures();xs=[];ds=[];ys=[];horizons=[];weights=[]
            observations=[e for e in story['events'] if e['type']=='observation'];queries=[e for e in story['events'] if e['type']=='query']
            for event in observations:
                now=time_at(event['line'],condition);x,dt=prefix.observe(event['text'],vectors[event['text']],now)
                future=[q for q in queries if q['line']>event['line'] and event['line'] in q['support']]
                # Relevance within this observed task window, not universal importance.
                horizon=time_at(future[-1]['line'] if future else story['events'][-1]['line'],condition)-now
                xs.append(x);ds.append(dt);ys.append(float(bool(future)));horizons.append(max(horizon,.01));weights.append(min(len(future)/3,1.))
            samples.append((xs,ds,ys,horizons,weights))
    n=len(samples);length=max(len(s[0]) for s in samples);x=np.zeros((n,length,40),np.float32);dt=np.zeros((n,length),np.float32);y=dt.copy();h=dt.copy();w=dt.copy();mask=dt.copy()
    for i,(xx,dd,yy,hh,ww) in enumerate(samples):
        l=len(xx);x[i,:l]=xx;dt[i,:l]=dd;y[i,:l]=yy;h[i,:l]=hh;w[i,:l]=ww;mask[i,:l]=1
    return [torch.tensor(a) for a in (x,dt,y,h,w,mask)]


def loss(model,arrays):
    x,dt,y,h,w,mask=arrays;state=model.initial(len(x));losses=[]
    positive=((mask.sum()-(y*mask).sum())/(y*mask).sum().clamp_min(1)).clamp(1,10)
    for i in range(x.shape[1]):
        policy,state=model.step(x[:,i],dt[:,i],state)
        l=F.binary_cross_entropy_with_logits(policy['logits'][:,0],y[:,i],pos_weight=positive,reduction='none')
        if model.kind!='no_importance':l=l+.5*(policy['w']/2-w[:,i]).square()
        if model.kind!='no_forgetting':
            survival=torch.exp(-policy['beta']*h[:,i]).clamp(1e-5,1-1e-5)
            l=l+.5*F.binary_cross_entropy(survival,y[:,i],reduction='none')
        losses.append(l*mask[:,i])
    return torch.stack(losses).sum()/mask.sum()


def train():
    if (OUT/'training_complete.json').exists():return
    torch.set_num_threads(1);tr=training_arrays('train');va=training_arrays('val');logs=[]
    folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
    for kind in KINDS:
        for seed in SEEDS:
            torch.manual_seed(seed);model=ConsolidationController(kind);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);best=float('inf')
            for epoch in range(1,21):
                model.train();objective=loss(model,tr);opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();model.eval()
                with torch.no_grad():validation=float(loss(model,va))
                logs.append(dict(kind=kind,seed=seed,epoch=epoch,train_loss=float(objective.detach()),val_loss=validation))
                if validation<best:
                    best=validation;torch.save(dict(kind=kind,seed=seed,state_dict=model.state_dict(),epoch=epoch,parameters=sum(p.numel() for p in model.parameters())),folder/f'{kind}-{seed}.pt')
            print('Trained consolidation',kind,seed,flush=True)
    atomic_json(OUT/'training.json',logs);atomic_json(OUT/'training_complete.json',dict(models=len(KINDS)*3,completed_at=time.time()))


def load_model(kind,seed):
    ck=torch.load(OUT/'checkpoints'/f'{kind}-{seed}.pt',weights_only=False);model=ConsolidationController(kind);model.load_state_dict(ck['state_dict']);model.eval();return model


def evaluate():
    if (OUT/'evaluation_complete.json').exists():return
    assert (OUT/'training_complete.json').exists();torch.set_num_threads(1)
    stories=read(OUT/'stories.json');vectors=read(OUT/'vectors.json');records=[];streams=[]
    methods=[(kind,seed) for kind in KINDS for seed in SEEDS]+[('fifo',0),('decay',0)]
    for kind,seed in methods:
        model=None if seed==0 else load_model(kind,seed)
        for budget in BUDGETS:
            for condition in CONDITIONS:
                for sid,story in stories.items():
                    if story['split']!='test':continue
                    memory=ConsolidationMemory(model,budget,kind);peak=memory.payload_bytes;started=time.perf_counter();correct=[]
                    for event in story['events']:
                        now=time_at(event['line'],condition)
                        if event['type']=='observation':
                            memory.observe(f"{sid}:{event['line']}",event['text'],vectors[event['text']],now,event['key'],event['value']);peak=max(peak,memory.payload_bytes)
                        else:
                            retrieved=memory.retrieve(now,key=event['key']);answer=retrieved[0].value if retrieved else '';ok=answer==event['answer'];correct.append(ok)
                            memory.reinforce([r.identity for r in retrieved],now,float(ok))
                            records.append(dict(kind=kind,seed=seed,budget=budget,condition=condition,story=sid,line=event['line'],correct=ok,abstain=not retrieved,wrong=bool(retrieved) and not ok,answer=answer,reference=event['answer'],payload_bytes=memory.payload_bytes))
                    streams.append(dict(kind=kind,seed=seed,budget=budget,condition=condition,story=sid,accuracy=float(np.mean(correct)),peak_bytes=peak,elapsed_seconds=time.perf_counter()-started,**memory.stats))
        print('Evaluated',kind,seed,flush=True)
    atomic_json(OUT/'query_records.json',records);atomic_json(OUT/'stream_records.json',streams);atomic_json(OUT/'evaluation_complete.json',dict(queries=len(records),streams=len(streams)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','train','evaluate'],default='all');a=p.parse_args()
    if a.phase in ['all','prepare']:prepare()
    if a.phase in ['all','train']:train()
    if a.phase in ['all','evaluate']:evaluate()
