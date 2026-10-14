"""Frozen protocol: implementation latency separate from scorer-budget quality."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,time,hashlib,argparse
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.provenance import snapshot_source,runtime_info
from src.memory.content_store import ContentWriter
from src.memory.fast_writer import frozen_writer
from src.memory.sparse_writer import sparse_scores
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.models.content_encoder import ContentEncoder
from src.retrieval.preserved_context import pack_preserved_context
from src.data.longmemeval import prepared
from scripts.content_experiment import Reader,deltas
from scripts.liquid_attribution import examples,read
OUT=ROOT/'runs/writer-efficiency-v1'

def models():
    result={}
    for kind in ['liquid_content','gru_content','static_content']:
        for seed in [11,23,42]:
            model=ContentWriter(kind)
            model.load_state_dict(torch.load(ROOT/'runs/preservation-study-v2/checkpoints'/f'{kind}-{seed}/best.pt',weights_only=True)['state_dict'])
            result[kind.split('_')[0],seed]=model.eval()
    return result

def init():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'protocol.json').exists():return
    sig,archive=snapshot_source()
    atomic_json(OUT/'protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),
        checkpoints=read(ROOT/'runs/liquid-attribution-v1/protocol.json')['checkpoint_hashes'],
        engineering='CPU torch threads=1. Unchanged frozen seed23 liquid/GRU/static. Batch sizes1/8/32,64 timesteps from public cached vectors. Original serial streams, fused eager, scripted fused. 3warmups+15repeats rotating order. Output max absolute error <=2e-6; same retention audited on real histories. Batch throughput is not single-agent latency. Script compilation excluded and reported separately.',
        research='All previously exposed LME test + eligible/abstention LoCoMo, frozen three seeds, budgets64k/128k, full/stride2/stride4/novelty(cosine<0.8 or4events). Same observation-only gate for liquid/GRU/static. Every session still stored; skipped priority holds last score. Delta accumulated between scorer calls and clipped12. No semantic equivalent claim, no skipped writes/embeddings/retrieval/API claim.',
        primary='LongMemEval64k stride2 vs full within each writer; require >=45% fewer scorer calls and paired history-cluster95%CI lower bound > -0.01 recall. This is a predeclared exploratory non-inferiority screen, not proof of QA non-inferiority. No parameter selection from test.',
        attribution='Liquid-specific savings require lower quality degradation than equally gated GRU/static at matched scorer count, evaluated by difference-in-differences, plus downstream QA; absolute call reduction alone is a generic scheduling benefit.',
        qa='The same six question IDs predetermined in language-readiness-v1; seed23,64k,liquid/gru x full/stride2 =24 answers plus separate same-provider custom judging. All results retained; smoke size cannot establish noninferiority. No picking questions by outcome.',
        qa_ids=read(ROOT/'runs/language-readiness-v1/protocol.json')['qa_ids']))

def engineering():
    if (OUT/'engineering.json').exists():return
    ms=models();enc=ContentEncoder();hh=examples()[:32]
    vv=[];dd=[]
    for e in hh:
        h=e['history'];v=enc.encode(list(h.texts));d=deltas(h)
        ix=np.arange(64)%len(v)
        vv.append(v[ix]);dd.append(d[ix])
    enc.close();records=[]
    with torch.inference_mode():
        for kind in ['liquid','gru','static']:
            m=ms[kind,23];eager=frozen_writer(m,False)
            t=time.perf_counter();script=frozen_writer(m,True);compile_seconds=time.perf_counter()-t
            for batch in [1,8,32]:
                v=torch.from_numpy(np.stack(vv[:batch],axis=1));d=torch.from_numpy(np.stack(dd[:batch],axis=1))
                methods=dict(original=lambda:torch.stack([m(v[:,b],d[:,b]) for b in range(batch)],dim=1),fused=lambda:eager(v,d),scripted=lambda:script(v,d))
                ref=methods['original']();errors={n:float((f()-ref).abs().max()) for n,f in methods.items()}
                assert max(errors.values())<=2e-6
                for _ in range(3):
                    for f in methods.values():f()
                times=defaultdict(list);names=list(methods)
                for repeat in range(15):
                    for name in names[repeat%3:]+names[:repeat%3]:
                        start=time.perf_counter();methods[name]();times[name].append(time.perf_counter()-start)
                for name,ts in times.items():
                    records.append(dict(kind=kind,batch=batch,method=name,seconds=ts,median_seconds=float(np.median(ts)),ms_per_session=float(np.median(ts)*1000/(64*batch)),max_abs_error=errors[name],script_creation_seconds=compile_seconds))
            # Strict online arrival path: no future input projection or across-time batching.
            v=torch.from_numpy(vv[0]);d=torch.from_numpy(dd[0]);times=defaultdict(list)
            for repeat in range(18):
                names=['original','fused','scripted'];names=names[repeat%3:]+names[:repeat%3]
                for name in names:
                    start=time.perf_counter()
                    if name=='original':out=m(v,d)
                    else:
                        adapter=eager if name=='fused' else script
                        state=v.new_zeros((1,m.hidden));out=[]
                        for x,dt in zip(v,d):
                            s,state=adapter.step(x[None],state,dt[None]);out.append(s[0])
                        out=torch.stack(out)
                    elapsed=time.perf_counter()-start
                    assert torch.allclose(out,m(v,d),atol=2e-6,rtol=2e-6)
                    if repeat>=3:times[name].append(elapsed)
            for name,ts in times.items():records.append(dict(kind=kind,batch=1,method='online_'+name,seconds=ts,median_seconds=float(np.median(ts)),ms_per_session=float(np.median(ts)*1000/64)))
    atomic_json(OUT/'engineering.json',records)
    print('Engineering benchmark saved',flush=True)

def research():
    if (OUT/'offline_complete.json').exists():return
    if (OUT/'records.jsonl').exists():raise RuntimeError('Partial run: preserve before retry')
    ms=models();fast={k:frozen_writer(m) for k,m in ms.items()};enc=ContentEncoder();gg=defaultdict(list)
    for e in examples():
        h=e['history'];key=hashlib.sha256(json.dumps([h.session_ids,h.dates,h.texts],ensure_ascii=False).encode()).hexdigest();gg[key].append(e)
    raw={str(r['question_id']):r for r in prepared(load_config('configs/minute_precision.yaml'))[0]}
    qa_ids=read(OUT/'protocol.json')['qa_ids'];qa=[];costlog=[];audit=[]
    with (OUT/'records.jsonl').open('w',encoding='utf-8') as f,torch.inference_mode():
        for gi,(key,ee) in enumerate(gg.items()):
            h=ee[0]['history'];v=enc.encode(list(h.texts));dt=deltas(h);vt=torch.from_numpy(v);dt=torch.from_numpy(dt)
            costs=np.array([BudgetItem(s,d,t,z,0,i).nbytes for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))]);qs=enc.encode([e['history'].question for e in ee]);cache={}
            for (kind,seed),m in ms.items():
                original=m(vt,dt);optimized=fast[kind,seed](vt[:,None],dt[:,None])[:,0]
                assert torch.allclose(original,optimized,atol=2e-6,rtol=2e-6)
                for budget in [64000,128000]:
                    assert byte_retained_indices(original,costs,budget,m.hidden*4+16)==byte_retained_indices(optimized,costs,budget,m.hidden*4+16)
                audit.append(dict(history=key,kind=kind,seed=seed,max_abs_error=float((original-optimized).abs().max()),retention_identical=True))
                for policy in ['full','stride2','stride4','novelty']:
                    start=time.perf_counter();scores,calls=sparse_scores(m,vt,dt,policy);elapsed=time.perf_counter()-start
                    if policy=='full':assert torch.allclose(original,scores,atol=2e-6,rtol=2e-6)
                    costlog.append(dict(history=key,dataset=ee[0]['dataset'],kind=kind,seed=seed,policy=policy,events=len(v),scorer_calls=len(calls),call_indices=calls.tolist(),seconds=elapsed))
                    for budget in [64000,128000]:
                        kept=byte_retained_indices(scores,costs,budget,m.hidden*4+16);used=int(costs[kept].sum()+m.hidden*4+16);assert used<=budget
                        ck=tuple(kept)
                        if ck not in cache:
                            items=[BudgetItem(h.session_ids[i],h.dates[i],h.texts[i],v[i],0,i) for i in kept];reader=Reader(SimpleNamespace(items=items)) if items else None
                            cache[ck]=[(reader.query(e['history'].question,q)[0] if reader else []) for e,q in zip(ee,qs)]
                        ids=[h.session_ids[i] for i in kept]
                        for e,selected in zip(ee,cache[ck]):
                            gold=set(e['gold']);ok=e['eligible'] and bool(gold)
                            rec=dict(question_id=e['history'].question_id,dataset=e['dataset'],cluster=key,category=e['category'],kind=kind,seed=seed,policy=policy,budget=budget,
                                recall=len(gold&{ids[i] for i in selected})/len(gold) if ok else None,retention=len(gold&set(ids))/len(gold) if ok else None,payload_bytes=used,scorer_calls=len(calls),events=len(v))
                            f.write(json.dumps(rec)+'\n')
                            if e['history'].question_id in qa_ids and kind in ['liquid','gru'] and seed==23 and budget==64000 and policy in ['full','stride2']:
                                hh=replace(e['history'],session_ids=tuple(ids),dates=tuple(h.dates[i] for i in kept),texts=tuple(h.texts[i] for i in kept))
                                ctx,trace=pack_preserved_context(hh,selected,16000)
                                qa.append(dict(**rec,method=kind+'_'+policy,question=hh.question,question_date=hh.question_date,answer=str(raw[hh.question_id]['answer']),context=ctx,trace=trace))
            f.flush()
            if gi%10==0:print('Budget research histories',gi+1,'/',len(gg),flush=True)
    enc.close();atomic_json(OUT/'qa_tasks.json',qa);atomic_json(OUT/'costs.json',costlog);atomic_json(OUT/'equivalence.json',audit);atomic_json(OUT/'offline_complete.json',dict(histories=len(gg),qa_tasks=len(qa)))

def qa_run():
    from src.api.agnes import AgnesClient
    from src.evaluation.runner import messages_for,custom_judge
    config=dict(load_config('configs/minute_precision.yaml')['api'],rpm=9)
    namespace='writer-efficiency-v1:'+hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    tasks=read(OUT/'qa_tasks.json');(OUT/'qa').mkdir(exist_ok=True)
    for i,task in enumerate(tasks):
        p=OUT/'qa'/f"{task['question_id']}-{task['method']}.json";result=read(p) if p.exists() else dict(task)
        if 'judge_response' in result:continue
        client=AgnesClient(config)
        try:
            if 'generation' not in result:
                result['generation']=client.complete(messages_for(SimpleNamespace(question=task['question'],question_date=task['question_date']),task['context']),namespace=namespace)
                result['hypothesis']=result['generation']['content'];atomic_json(p,result)
            grade,response=custom_judge(client,task,result['hypothesis'],namespace,max_tokens=512)
            result.update(custom_judge_correct=grade,judge_response=response);result.pop('error_type',None);atomic_json(p,result)
        except Exception as exc:
            result['error_type']=type(exc).__name__;atomic_json(p,result);raise RuntimeError('QA stopped; safe error type: '+type(exc).__name__) from None
        finally:client.close()
        print('QA saved',i+1,'/',len(tasks),flush=True)
    atomic_json(OUT/'qa_complete.json',dict(tasks=len(tasks)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['engineering','research','qa','all'],default='all');args=parser.parse_args()
    torch.set_num_threads(1);init()
    if args.phase in ['engineering','all']:engineering()
    if args.phase in ['research','all']:research()
    if args.phase in ['qa','all']:qa_run()
