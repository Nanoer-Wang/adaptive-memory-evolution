"""Actual factual-task losses under nested reading budgets, then frozen fallback models."""
import sys,json,hashlib,time,argparse,shutil
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.provenance import snapshot_source,runtime_info
from src.data.longmemeval import prepared
from src.models.content_encoder import ContentEncoder
from src.memory.semantic_reading import semantic_views,FEATURES,OMISSION_COLUMNS
from src.memory.task_controller import TaskController
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.retrieval.preserved_context import pack_preserved_context
from scripts.content_experiment import Reader
from scripts.risk_trajectory import history
from scripts import context_utility as old
OUT=ROOT/'runs/semantic-fallback-v1';PARENT=old.OUT
BUDGETS=[600,1800];SEEDS=[11,23,42];KINDS=['linear_no_omission','linear','gru','liquid'];LAMBDA=.02
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    if (OUT/'protocol.json').exists():raise RuntimeError('Partial preparation requires explicit recovery')
    rows,manifest=prepared(load_config('configs/minute_precision.yaml'));by={r['question_id']:r for r in rows}
    parent=read(PARENT/'tasks.json');dev=[r for r in parent if r['method']=='full' and r['split']!='test' and r['category']!='single-session-preference']
    used={read(p).get('question_id') for p in (ROOT/'runs').glob('*/qa/*.json')};buckets=defaultdict(list)
    for r in rows:
        if r['question_id'] in manifest['splits']['test'] and r['question_id'] not in used and not r['question_id'].endswith('_abs') and r['question_type']!='single-session-preference':buckets[r['question_type']].append(r)
    test=[]
    for category in sorted(buckets):
        candidates=sorted(buckets[category],key=lambda r:hashlib.sha256(('semantic-fallback-v1:'+r['question_id']).encode()).hexdigest())[:3]
        assert len(candidates)==3;test+=candidates
    assert len(dev)==30 and len(test)==12
    source,archive=snapshot_source();ids={s:[r['question_id'] for r in dev if r['split']==s] for s in ['train','val']};ids['test']=[r['question_id'] for r in test]
    atomic_json(OUT/'protocol.json',dict(source=source,archive=archive,registered_at=time.time(),runtime=runtime_info(),question_ids=ids,prior_qa_excluded=sorted(q for q in used if q),
        design='20 factual train,10 factual val,12 new-to-QA factual test questions. Test has3questions in each of4remaining categories; assistant-category new QA pool exhausted, no reuse disguised as fresh data. Existing dev full answers reused explicitly. Generate600/1800char nested semantic excerpts for all42 and full for12 new test questions:96 new answer requests plus judges.',
        extraction='Frozen MiniLM line cosine75% + query word overlap25%; greedy complete attributed units, nested600->1800; original full context unchanged. No gold or category in extraction/features. Embedding similarity is not entailment.',
        supervision='Actual full correct minus compact correct minus0.02*full_prompt_tokens/4000. Predict whether replacing compact answer with a full answer is worth the EXTRA full call. Count both calls and tokens. Correctness is strict reference judging of factual tasks; refusals0.',
        observation='18per-source features of semantic omission, exact numeric/query-word coverage, chronology, and observed compact abstention/answer length. Chronological same-query retrieved-source stream; no gold feedback or unrelated questions in state.',
        models='Same8hidden TaskController architecture, pooled linear with/without omission features,GRU,liquid;3seeds60epochs AdamW0.01 wd0.01 MSE. Train-normalization only; val mean deployed utility selects first best epoch. No model size increase. Predict fallback iff gain>0.',
        baselines='Never fallback,always fallback,compact abstention fallback. Same-weight liquid reset,zero-time,reverse-source diagnostics. Query-averaged two-budget results and paired question bootstrap; no test-driven changes.',
        scope='Continuation within the dated evidence history of ONE query, not long-lived state across agent tasks. Public offline development split previously exposed; new QA IDs are not a new independent dataset. No claims of causal continuous-time benefit from chronological recurrence alone.',
        cost='Research executes counterfactual alternatives; deployed accounting charges compact+optional full. Fixed three-seed action replay shares measured outcomes; not three independent test sets.'))
    enc=ContentEncoder();tasks=[];features={};provenance=[];start=time.perf_counter();weights=np.load(ROOT/'runs/content-fallback-v1/students.npz')['liquid_23']
    entries=[(r['split'],by[r['question_id']],r) for r in dev]+[('test',r,None) for r in test]
    for split,row,existing in entries:
        q=row['question_id']
        if existing:full=dict(existing)
        else:
            h=history(row);v=enc.encode(list(h.texts));query=enc.encode([h.question])[0];scores=v@weights[:-1]+weights[-1]
            items=[BudgetItem(s,d,t,z,0,j) for j,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))];cost=np.array([i.nbytes for i in items]);kept=byte_retained_indices(scores,cost,64000,16)
            local,_=Reader(SimpleNamespace(items=[items[j] for j in kept])).query(h.question,query);selected=sorted(kept[j] for j in local);context,trace=pack_preserved_context(h,selected,16000)
            full=dict(question_id=q,split=split,category=row['question_type'],method='full',question=h.question,question_date=h.question_date,answer=str(row['answer']),context=context,trace=trace,payload_bytes=int(cost[kept].sum()+16))
        full.update(context_chars=len(full['context']),context_sha256=hashlib.sha256(full['context'].encode()).hexdigest());tasks.append(full)
        if existing:
            dest=OUT/'qa'/f'{q}-full.json';dest.parent.mkdir(exist_ok=True);shutil.copyfile(PARENT/'qa'/dest.name,dest)
            provenance.append(dict(question_id=q,record=str((PARENT/'qa'/dest.name).relative_to(ROOT)),sha256=sha(dest),reuse='Previously generated factual development full answer; no new independent generation.'))
        views=semantic_views(full['question'],full['context'],enc,BUDGETS)
        for budget,view in views.items():
            method=f'compact{budget}';t=dict(full,method=method,context=view['context'],context_chars=len(view['context']),context_sha256=hashlib.sha256(view['context'].encode()).hexdigest(),trace=dict(selected_unit_indices=view['selected_unit_indices'],budget=budget))
            tasks.append(t);features[f'{q}:{budget}']=dict(question_id=q,split=split,category=full['category'],budget=budget,**view)
        print('Prepared',split,q,flush=True)
    enc.close();atomic_json(OUT/'tasks.json',tasks);atomic_json(OUT/'features.json',features);atomic_json(OUT/'reused_records.json',provenance)
    atomic_json(OUT/'prepared.json',dict(questions=42,action_tasks=126,new_answers=96,preparation_seconds=time.perf_counter()-start,encoder_source_sha256=sha(ROOT/'artifacts/encoders/minilm/source.json')))

def qa(which):
    prev=old.OUT
    try:old.OUT=OUT;old.qa(which)
    finally:old.OUT=prev

def data(include_test=False):
    result={}
    for key,f in read(OUT/'features.json').items():
        if f['split']=='test' and not include_test:continue
        q=f['question_id'];c=read(OUT/'qa'/f"{q}-compact{f['budget']}.json");full=read(OUT/'qa'/f'{q}-full.json')
        assert all(r['correct'] in (0,1) and r['evaluation_status'] in ['graded','explicit_abstention'] for r in [c,full])
        x=np.array(f['x']);x[:,16]=float(c['parsed']['abstain']);x[:,17]=min(len(c['hypothesis'])/1000,2)
        u0=c['correct']-LAMBDA*c['generation']['usage']['prompt_tokens']/4000
        u1=full['correct']-LAMBDA*(c['generation']['usage']['prompt_tokens']+full['generation']['usage']['prompt_tokens'])/4000
        result[key]=dict(**{k:f[k] for k in ['question_id','split','category','budget']},x=x,dt=np.array(f['dt']),target=u1-u0,utilities=[u0,u1],compact=c,full=full)
    return result

def transform(d,normal,kind):
    x=torch.tensor((d['x']-normal['mean'])/normal['std'],dtype=torch.float32)
    if kind=='linear_no_omission':x[:,OMISSION_COLUMNS]=0
    return x,torch.tensor(d['dt'],dtype=torch.float32)

def train():
    if (OUT/'training_complete.json').exists():return
    torch.set_num_threads(1);ds=data();train=[d for d in ds.values() if d['split']=='train'];val=[d for d in ds.values() if d['split']=='val']
    xx=np.concatenate([d['x'] for d in train]);normal=dict(mean=xx.mean(0),std=np.maximum(xx.std(0),.05));np.savez(OUT/'normalization.npz',**normal)
    signal=[]
    for split in ['train','val']:
        for budget in BUDGETS:
            rr=[d for d in ds.values() if d['split']==split and d['budget']==budget]
            signal.append(dict(split=split,budget=budget,n=len(rr),full_better=sum(d['full']['correct']>d['compact']['correct'] for d in rr),compact_better=sum(d['full']['correct']<d['compact']['correct'] for d in rr),equal=sum(d['full']['correct']==d['compact']['correct'] for d in rr)))
    atomic_json(OUT/'development_signal.json',signal);print('Actual factual supervision:',json.dumps(signal),flush=True)
    logs=[]
    for kind in KINDS:
        for seed in SEEDS:
            torch.manual_seed(seed);m=TaskController('linear' if kind=='linear_no_omission' else kind,features=len(FEATURES));opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01);best=-float('inf');folder=OUT/'checkpoints'/f'{kind}-{seed}';folder.mkdir(parents=True,exist_ok=True)
            for epoch in range(1,61):
                m.train();loss=torch.stack([(m(*transform(d,normal,kind))-d['target'])**2 for d in train]).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();m.eval()
                with torch.no_grad():score=float(np.mean([d['utilities'][int(float(m(*transform(d,normal,kind)))>0)] for d in val]))
                logs.append(dict(kind=kind,seed=seed,epoch=epoch,train_mse=float(loss.detach()),val_utility=score))
                if score>best:
                    best=score;torch.save(dict(state_dict=m.state_dict(),kind=kind,seed=seed,epoch=epoch,train_ids=sorted({d['question_id'] for d in train}),val_ids=sorted({d['question_id'] for d in val})),folder/'best.pt')
            print('Trained',kind,seed,flush=True)
    atomic_json(OUT/'training.json',logs);atomic_json(OUT/'training_complete.json',dict(models=12,train_questions=20,val_questions=10,train_action_pairs=40,feature_names=FEATURES,signal=signal))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','dev','train','test'],default='all');a=p.parse_args()
    if a.phase in ['all','prepare']:prepare()
    if a.phase in ['all','dev']:qa('development')
    if a.phase in ['all','train']:train()
    if a.phase in ['all','test']:qa('test')
