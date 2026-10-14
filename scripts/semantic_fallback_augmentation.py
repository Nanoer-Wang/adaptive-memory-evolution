"""Training-only source-withholding intervention; no test labels used in fitting."""
import sys,json,hashlib,time,re,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json
from src.provenance import snapshot_source
from src.models.content_encoder import ContentEncoder
from src.memory.semantic_reading import parse_units,render
from scripts import semantic_fallback as base
from scripts import context_utility as original
PARENT=base.OUT;OUT=ROOT/'runs/semantic-fallback-augmentation-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    assert not any(read(p)['split']=='test' for p in (PARENT/'qa').glob('*.json'))
    source,archive=snapshot_source()
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),source=source,archive=archive,parent=str(PARENT.relative_to(ROOT)),
        question_ids=read(PARENT/'protocol.json')['question_ids'],
        reason='Natural training compression produced only sparse quality contrasts. Add ONE source-withholding view for EVERY one of20 existing factual train questions; no sampling by correctness or reference answer.',
        intervention='Start from semantic1800 context; remove its included source having highest frozen MiniLM query-line similarity. Full context/answer unchanged. Recompute omission features and generate actual compact answer. No withheld-source interventions in val/test.',
        fitting='Same architectures,60epochs,optimizer,seeds,validation selection and utility as parent. Parent40 natural train action pairs plus20 intervention pairs; all methods receive identical data. No access to test answer files or outcomes during fitting.',
        timing='Registered before any parent test generation. Test requests may finish while supplemental development training runs; training reads only parent development outcomes. Supplemental weights are frozen before test analysis. This is not a claim all supplemental weights predate test API generation.',
        scope='Intervention is training augmentation, not naturally occurring loss or independent benchmark evidence. No answer annotations used to choose withheld content.'))
    features=read(PARENT/'features.json');tasks=[];observations={};encoder=ContentEncoder()
    for key,f in features.items():
        if f['split']!='train' or f['budget']!=1800:continue
        full=next(r for r in read(PARENT/'tasks.json') if r['question_id']==f['question_id'] and r['method']=='full')
        units=parse_units(full['context']);included={units[i][0] for i in f['selected_unit_indices']}
        candidates=[i for i,h in enumerate(f['source_headers']) if h in included];pos=max(candidates,key=lambda i:f['x'][i][1]);header=f['source_headers'][pos]
        selected=[i for i in f['selected_unit_indices'] if units[i][0]!=header];context=render(units,selected);v=encoder.encode([line for _,line in units]);q=encoder.encode([full['question']])[0];dense=np.maximum(v@q,0)
        coverage=np.maximum(0,(v@v[selected].T).max(1)) if selected else np.zeros(len(units));x=np.array(f['x'])
        terms=set(re.findall(r'[a-z0-9]+',full['question'].lower()))-{'the','a','i','my','what','how','was','did','in','of','to','and','is','do'}
        numbers=set(re.findall(r'\b\d+(?:[.:/-]\d+)*\b',context));words=set(re.findall(r'[a-z0-9]+',context.lower()))
        for j,h in enumerate(f['source_headers']):
            ix=np.array([i for i,(hh,_) in enumerate(units) if hh==h]);take=np.array([i in selected for i in ix]);text=' '.join(units[i][1] for i in ix)
            ns=set(re.findall(r'\b\d+(?:[.:/-]\d+)*\b',text));needed=terms&set(re.findall(r'[a-z0-9]+',text.lower()));risk=dense[ix]*(1-coverage[ix])
            x[j,4:10]=[take.mean(),np.average(coverage[ix],weights=dense[ix]+1e-6),risk.max(),risk.mean(),len(ns-numbers)/max(1,len(ns)),dense[ix][take].max() if take.any() else 0.]
            x[j,13]=len(context)/len(full['context']);x[j,14]=len(needed-words)/max(1,len(needed))
        task=dict(full,method='withheld1800',context=context,context_chars=len(context),context_sha256=hashlib.sha256(context.encode()).hexdigest(),trace=dict(withheld_source=header,selected_unit_indices=selected))
        tasks.append(task);observations[f['question_id']]=dict(question_id=f['question_id'],split='train',category=f['category'],budget=1800,x=x.tolist(),dt=f['dt'],withheld_source=header)
    encoder.close();assert len(tasks)==20
    atomic_json(OUT/'tasks.json',tasks);atomic_json(OUT/'observations.json',observations);atomic_json(OUT/'prepared.json',dict(training_interventions=20))

def qa():
    prev=original.OUT
    try:original.OUT=OUT;original.qa('development')
    finally:original.OUT=prev

def augmented_data(include_test=False):
    ds=base.data(include_test)
    for q,f in read(OUT/'observations.json').items():
        c=read(OUT/'qa'/f'{q}-withheld1800.json');full=read(PARENT/'qa'/f'{q}-full.json');assert c['correct'] in (0,1) and c['evaluation_status'] in ['graded','explicit_abstention']
        x=np.array(f['x']);x[:,16]=float(c['parsed']['abstain']);x[:,17]=min(len(c['hypothesis'])/1000,2)
        u0=c['correct']-base.LAMBDA*c['generation']['usage']['prompt_tokens']/4000
        u1=full['correct']-base.LAMBDA*(c['generation']['usage']['prompt_tokens']+full['generation']['usage']['prompt_tokens'])/4000
        ds[q+':withheld']=dict(**{k:f[k] for k in ['question_id','split','category','budget']},x=x,dt=np.array(f['dt']),target=u1-u0,utilities=[u0,u1],compact=c,full=full,intervention=True)
    return ds

def train():
    if (OUT/'training_complete.json').exists():return
    while not (PARENT/'development_qa_complete.json').exists():time.sleep(2)
    ds=augmented_data(False);prev_out,prev_data=base.OUT,base.data
    try:
        base.OUT=OUT;base.data=lambda include_test=False:ds
        base.train()
    finally:base.OUT=prev_out;base.data=prev_data
    meta=read(OUT/'training_complete.json');meta.update(train_action_pairs=60,training_only_interventions=20,natural_train_pairs=40,test_outcomes_read_during_training=False)
    atomic_json(OUT/'training_complete.json',meta)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','qa','train'],default='all');a=p.parse_args()
    if a.phase in ['all','prepare']:prepare()
    if a.phase in ['all','qa']:qa()
    if a.phase in ['all','train']:train()
