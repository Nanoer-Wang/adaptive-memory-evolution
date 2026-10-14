"""Frozen hybrid mechanism attribution and expanded, label-blind QA selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,argparse
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.provenance import snapshot_source,runtime_info
from src.data.longmemeval import prepared
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.models.content_encoder import ContentEncoder
from src.retrieval.preserved_context import pack_preserved_context
from scripts.writer_efficiency import models,examples,read
from scripts.content_experiment import Reader,deltas
OUT=ROOT/'runs/hybrid-validation-v1'
CONDITIONS=['liquid_hybrid','liquid_linear','gru_hybrid','static_hybrid','liquid_reset','liquid_zero_time','liquid_shift','gru_shift','liquid_full']

def hybrid(model,v,dt,w,mode):
    if mode=='full':return model(v,dt).numpy(),len(v)
    out=v.numpy()@w[:-1]+w[-1]
    if mode=='linear':return out,0
    state=v.new_zeros(model.hidden);pending=0.;calls=0
    for i,(x,d) in enumerate(zip(v,dt)):
        pending+=float(d)
        if i%2==(1 if mode=='shift' else 0):
            if mode=='reset':state=torch.zeros_like(state)
            value,state=model.step(x,state,0. if mode=='zero_time' else min(pending,12.))
            out[i]=float(value);pending=0.;calls+=1
    return out,calls

def offline():
    if (OUT/'offline_complete.json').exists():return
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve partial run before retry')
    OUT.mkdir(parents=True,exist_ok=True);ee=examples();old=set(read(ROOT/'runs/language-readiness-v1/protocol.json')['qa_ids'])
    bycategory=defaultdict(list)
    for e in ee:
        if e['dataset']=='longmemeval' and e['eligible'] and e['history'].question_id not in old:bycategory[e['category']].append(e['history'].question_id)
    qa_ids=[]
    for category in sorted(bycategory):qa_ids+=sorted(bycategory[category],key=lambda q:hashlib.sha256(('hybrid-validation-v1:'+q).encode()).hexdigest())[:2]
    sig,archive=snapshot_source();weights_path=ROOT/'runs/content-fallback-v1/students.npz'
    atomic_json(OUT/'protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),conditions=CONDITIONS,seeds=[11,23,42],budgets=[64000,128000],
        checkpoints=read(ROOT/'runs/liquid-attribution-v1/protocol.json')['checkpoint_hashes'],student_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        primary='LME64k liquid_hybrid minus liquid_linear, gru_hybrid, liquid_reset, liquid_zero_time. Three seed scores averaged per question; paired history-cluster bootstrap5000,95% and98.75% simultaneous Bonferroni intervals. Mechanism signal requires all four corrected lower bounds >0. Noninferiority to liquid_full uses pre-existing -1pp recall margin.',
        secondary='LoCoMo and128k plus shifted schedule. Phase1 calls on odd indices instead of even, includes no forced first call; odd-length histories differ by one call. Record actual costs. Phase comparison is sensitivity, not best-phase selection.',
        semantics='Same input/compression/reader/budget. Same liquid weights AND same student for reset/time/phase ablations. Static/GRU use their own frozen train-only students as before. Every record still participates in storage; no query/label during writes.',
        qa='Two new answerable questions per available LME category, sorted by fixed salted hash, excluding prior six. Four conditions seed23/64k: liquid_hybrid,liquid_linear,gru_hybrid,liquid_full. Same-provider custom judge. No outcome-based exclusions. Budget48answers if six categories; actual selected IDs fixed below.',
        qa_ids=qa_ids,category_counts={k:len(v) for k,v in bycategory.items()},limitations='Previously exposed offline development data; new QA questions are not a new independent dataset. No tuning/retraining. No publication-level claim from small QA.'))
    ms=models();students=np.load(weights_path);enc=ContentEncoder();groups=defaultdict(list)
    for e in ee:
        h=e['history'];key=hashlib.sha256(json.dumps([h.session_ids,h.dates,h.texts],ensure_ascii=False).encode()).hexdigest();groups[key].append(e)
    raw={str(r['question_id']):r for r in prepared(load_config('configs/minute_precision.yaml'))[0]};qa=[];calls_log=[]
    with (OUT/'records.jsonl').open('w',encoding='utf-8') as f,torch.inference_mode():
        for gi,(key,group) in enumerate(groups.items()):
            h=group[0]['history'];v=enc.encode(list(h.texts));vt=torch.from_numpy(v);dt=torch.from_numpy(deltas(h));qs=enc.encode([e['history'].question for e in group]);cache={}
            costs=np.asarray([BudgetItem(s,d,t,z,0,i).nbytes for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))])
            for seed in [11,23,42]:
                for name in CONDITIONS:
                    kind,mode=name.split('_',1);m=ms[kind,seed];scores,calls=hybrid(m,vt,dt,students[f'{kind}_{seed}'],mode)
                    calls_log.append(dict(cluster=key,dataset=group[0]['dataset'],method=name,seed=seed,events=len(v),calls=calls))
                    for budget in [64000,128000]:
                        state=0 if mode=='linear' else m.hidden*4;kept=byte_retained_indices(scores,costs,budget,state+16);used=int(costs[kept].sum()+state+16);assert used<=budget
                        ids=[h.session_ids[i] for i in kept];ck=tuple(kept)
                        if ck not in cache:
                            items=[BudgetItem(h.session_ids[i],h.dates[i],h.texts[i],v[i],0,i) for i in kept];reader=Reader(SimpleNamespace(items=items)) if items else None
                            cache[ck]=[reader.query(e['history'].question,q)[0] if reader else [] for e,q in zip(group,qs)]
                        for e,selected in zip(group,cache[ck]):
                            gold=set(e['gold']);ok=e['eligible'] and bool(gold);rec=dict(question_id=e['history'].question_id,dataset=e['dataset'],cluster=key,category=e['category'],method=name,seed=seed,budget=budget,recall=len(gold&{ids[i] for i in selected})/len(gold) if ok else None,retention=len(gold&set(ids))/len(gold) if ok else None,payload_bytes=used)
                            f.write(json.dumps(rec)+'\n')
                            if e['history'].question_id in qa_ids and seed==23 and budget==64000 and name in ['liquid_hybrid','liquid_linear','gru_hybrid','liquid_full']:
                                hh=replace(e['history'],session_ids=tuple(ids),dates=tuple(h.dates[i] for i in kept),texts=tuple(h.texts[i] for i in kept));context,trace=pack_preserved_context(hh,selected,16000)
                                qa.append(dict(**rec,question=hh.question,question_date=hh.question_date,answer=str(raw[hh.question_id]['answer']),context=context,trace=trace))
            f.flush()
            if gi%10==0:print('Hybrid attribution histories',gi+1,'/',len(groups),flush=True)
    enc.close();atomic_json(OUT/'calls.json',calls_log);atomic_json(OUT/'qa_tasks.json',qa);atomic_json(OUT/'offline_complete.json',dict(questions=len(ee),histories=len(groups),qa_questions=len(qa_ids),qa_tasks=len(qa)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['offline','qa','all'],default='all');a=p.parse_args();torch.set_num_threads(1)
    if a.phase in ['offline','all']:offline()
    if a.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=OUT;runner.qa_run()

