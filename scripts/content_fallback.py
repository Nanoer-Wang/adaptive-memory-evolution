"""Post-diagnostic experiment: train-only linear teacher distillation for skipped scores."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.data.longmemeval import prepared,public_history,date_days
from src.memory.turn_compression import compress_turns
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.models.content_encoder import ContentEncoder
from src.retrieval.preserved_context import pack_preserved_context
from src.provenance import snapshot_source,runtime_info
from scripts.content_experiment import Reader,deltas
from scripts.writer_efficiency import models,examples,read
OUT=ROOT/'runs/content-fallback-v1'

def fallback_scores(model,v,dt,weight,hybrid):
    cheap=v.numpy()@weight[:-1]+weight[-1]
    if not hybrid:return cheap,0
    state=v.new_zeros(model.hidden);pending=0.;calls=0;out=cheap.copy()
    for i,(x,d) in enumerate(zip(v,dt)):
        pending+=float(d)
        if i%2==0:
            value,state=model.step(x,state,min(pending,12.));pending=0.;calls+=1;out[i]=float(value)
    return out,calls

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'offline_complete.json').exists():return
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve partial run before retry')
    rows,manifest=prepared(load_config('configs/minute_precision.yaml'));train_ids=set(manifest['splits']['train']);test_ids=set(manifest['splits']['test']);assert not train_ids&test_ids
    sig,archive=snapshot_source();atomic_json(OUT/'protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),
        reason='Post-v1 diagnostic extension: holding an unrelated previous content score damages LME. New protocol frozen before fitting or scoring; prior data exposure explicitly retained.',
        training='Only manifest train histories; targets are frozen teacher insertion logits, never test labels. Ridge alpha1,384-vector+intercept; unpenalized intercept. No validation tuning. Nine students, same recipe for liquid/GRU/static and three seeds.',
        conditions='linear_only: content student every event,zero recurrent teacher calls. hybrid2: student fallback odd events,teacher call even events with elapsed accumulation; every session still enters storage. Fair same recipe for all writers. No threshold search.',
        primary='LME64k hybrid2 paired against original full teacher, >=45% fewer teacher calls and cluster95%lower bound > -1pp evidence recall. All-linear baseline must be included to identify generic distillation gains. QA six fixed questions, liquid hybrid2/liquid linear_only/GRU hybrid2,seed23,64k.',
        limitations='Student computation and embedding still required; do not equate teacher calls with total FLOPs, writes, retrievals or API. Existing exposed development data. Distillation benefit alone is not liquid-specific.',
        qa_ids=read(ROOT/'runs/language-readiness-v1/protocol.json')['qa_ids'],train_ids=sorted(train_ids),test_ids=sorted(test_ids),data_signature=manifest['signature']))
    ms=models();enc=ContentEncoder();xx=[];yy=defaultdict(list)
    with torch.inference_mode():
        for idx,r in enumerate(r for r in rows if r['question_id'] in train_ids):
            h=public_history(r,'minute');ordered=sorted(zip(r['haystack_dates'],r['haystack_sessions']),key=lambda x:date_days(x[0],'minute'))
            h=replace(h,texts=tuple(compress_turns(t,2400,'sentence_user_first') for _,t in ordered));v=enc.encode(list(h.texts));dt=torch.from_numpy(deltas(h));vt=torch.from_numpy(v)
            xx.append(np.column_stack([v,np.ones(len(v))]))
            for key,m in ms.items():yy[key].append(m(vt,dt).numpy())
            if idx%50==0:print('Distillation train histories',idx+1,flush=True)
    x=np.concatenate(xx);gram=x.T@x;regularizer=np.eye(385);regularizer[-1,-1]=0;weights={};trainmetrics=[]
    allkeys=list(ms);targets=np.column_stack([np.concatenate(yy[key]) for key in allkeys]);w=np.linalg.solve(gram+regularizer,x.T@targets)
    for j,key in enumerate(allkeys):
        weights[key]=w[:,j].astype(np.float32);trainmetrics.append(dict(kind=key[0],seed=key[1],sessions=len(x),mse=float(np.mean((x@w[:,j]-targets[:,j])**2))))
    np.savez(OUT/'students.npz',**{f'{k[0]}_{k[1]}':v for k,v in weights.items()});atomic_json(OUT/'training.json',trainmetrics)
    groups=defaultdict(list)
    for e in examples():
        h=e['history'];key=hashlib.sha256(json.dumps([h.session_ids,h.dates,h.texts],ensure_ascii=False).encode()).hexdigest();groups[key].append(e)
    raw={str(r['question_id']):r for r in rows};qa_ids=read(OUT/'protocol.json')['qa_ids'];qa=[];timings=[]
    with (OUT/'records.jsonl').open('w',encoding='utf-8') as f,torch.inference_mode():
        for gi,(key,ee) in enumerate(groups.items()):
            h=ee[0]['history'];v=enc.encode(list(h.texts));vt=torch.from_numpy(v);dt=torch.from_numpy(deltas(h));qs=enc.encode([e['history'].question for e in ee]);costs=np.array([BudgetItem(s,d,t,z,0,i).nbytes for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))]);cache={}
            for (kind,seed),m in ms.items():
                for policy in ['linear_only','hybrid2']:
                    start=time.perf_counter();scores,calls=fallback_scores(m,vt,dt,weights[kind,seed],policy=='hybrid2');elapsed=time.perf_counter()-start
                    timings.append(dict(cluster=key,dataset=ee[0]['dataset'],kind=kind,seed=seed,policy=policy,events=len(v),teacher_calls=calls,seconds=elapsed))
                    for budget in [64000,128000]:
                        state=0 if policy=='linear_only' else m.hidden*4
                        kept=byte_retained_indices(scores,costs,budget,state+16);used=int(costs[kept].sum()+state+16);assert used<=budget;ids=[h.session_ids[i] for i in kept];ck=tuple(kept)
                        if ck not in cache:
                            items=[BudgetItem(h.session_ids[i],h.dates[i],h.texts[i],v[i],0,i) for i in kept];reader=Reader(SimpleNamespace(items=items)) if items else None
                            cache[ck]=[reader.query(e['history'].question,q)[0] if reader else [] for e,q in zip(ee,qs)]
                        for e,selected in zip(ee,cache[ck]):
                            gold=set(e['gold']);ok=e['eligible'] and bool(gold)
                            rec=dict(question_id=e['history'].question_id,dataset=e['dataset'],cluster=key,category=e['category'],kind=kind,seed=seed,policy=policy,budget=budget,recall=len(gold&{ids[i] for i in selected})/len(gold) if ok else None,payload_bytes=used,teacher_calls=calls,events=len(v))
                            f.write(json.dumps(rec)+'\n')
                            if e['history'].question_id in qa_ids and seed==23 and budget==64000 and (kind=='liquid' or (kind=='gru' and policy=='hybrid2')):
                                hh=replace(e['history'],session_ids=tuple(ids),dates=tuple(h.dates[i] for i in kept),texts=tuple(h.texts[i] for i in kept));ctx,trace=pack_preserved_context(hh,selected,16000)
                                qa.append(dict(**rec,retention=len(gold&set(ids))/len(gold),method=kind+'_'+policy,question=hh.question,question_date=hh.question_date,answer=str(raw[hh.question_id]['answer']),context=ctx,trace=trace))
            f.flush()
            if gi%10==0:print('Fallback histories',gi+1,'/',len(groups),flush=True)
    enc.close();atomic_json(OUT/'qa_tasks.json',qa);atomic_json(OUT/'timing.json',timings);atomic_json(OUT/'offline_complete.json',dict(histories=len(groups),train_histories=len(train_ids),train_sessions=len(x),qa_tasks=len(qa)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['offline','qa','all'],default='all');args=parser.parse_args()
    if args.phase in ['offline','all']:main()
    if args.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=OUT;runner.qa_run()
