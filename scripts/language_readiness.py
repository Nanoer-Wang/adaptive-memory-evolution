"""Finite language-memory readiness suite: offline public QA then resumable API smoke."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import load_config,atomic_json
from src.data.longmemeval import prepared,public_history,date_days
from src.data.content_benchmark import locomo_examples
from src.memory.turn_compression import compress_turns,sentences
from src.memory.content_store import ContentWriter
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.models.content_encoder import ContentEncoder
from src.retrieval.preserved_context import pack_preserved_context
from scripts.content_experiment import Reader,deltas
from src.evaluation.runner import messages_for,custom_judge
from src.evaluation.metrics import answer_metrics
from src.api.agnes import AgnesClient
from src.provenance import snapshot_source,runtime_info
OUT=ROOT/'runs/language-readiness-v1';PARENT=ROOT/'runs/preservation-study-v2'
def save(n,x):atomic_json(OUT/n,x)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def offline():
    if (OUT/'offline_complete.json').exists():return
    if (OUT/'protocol.json').exists():raise RuntimeError('Partial offline run exists; preserve artifacts and diagnose before rerun')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);config=load_config('configs/minute_precision.yaml');rows,manifest=prepared(config)
    examples=[];raw={str(r['question_id']):r for r in rows}
    for row in rows:
        if row['question_id'] not in manifest['splits']['test']:continue
        h=public_history(row,'minute');ordered=sorted(zip(row['haystack_session_ids'],row['haystack_dates'],row['haystack_sessions']),key=lambda t:date_days(t[1],'minute'))
        h=replace(h,texts=tuple(compress_turns(t,2400,'sentence_user_first') for _,_,t in ordered))
        examples.append(dict(history=h,dataset='longmemeval',group=h.question_id,answer=str(row['answer']),category=row['question_type'],gold=list(map(str,row.get('answer_session_ids',[]))),exclusion='unanswerable' if h.question_id.endswith('_abs') else None))
    external=locomo_examples(read(ROOT/'data/locomo_reserved/locomo10.json'))
    for e in external:
        if e['exclusion'] not in (None,'unanswerable'):continue
        h=e['history'];texts=tuple(compress_turns([dict(role='dialogue',content=t)],2400,'sentence_user_first') for t in h.texts);e=dict(e,history=replace(h,texts=texts));examples.append(e)
    qids=[]
    for category in ['knowledge-update','single-session-assistant','temporal-reasoning']:
        candidates=sorted([e for e in examples if e['dataset']=='longmemeval' and e['category']==category and not e['exclusion']],key=lambda e:hashlib.sha256(e['history'].question_id.encode()).hexdigest())
        qids.extend(e['history'].question_id for e in candidates[:2])
    models={};hashes={}
    for kind in ['static_content','gru_content','liquid_content','liquid_fixed_time']:
        for seed in [11,23,42]:
            p=PARENT/'checkpoints'/f'{kind}-{seed}'/'best.pt';m=ContentWriter('liquid_content' if kind=='liquid_fixed_time' else kind)
            if kind=='liquid_fixed_time':m.cell.fixed_time=True
            m.load_state_dict(torch.load(p,weights_only=True)['state_dict']);m.eval();name=f'{kind}-{seed}';models[name]=m;hashes[name]=sha(p)
    sig,archive=snapshot_source();save('protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),checkpoint_hashes=hashes,data_signature=manifest['signature'],external_sha256=sha(ROOT/'data/locomo_reserved/locomo10.json'),qa_ids=qids,
        scope='Ready-to-run language memory evaluation, not claim of liquid superiority. Existing frozen language-trained checkpoints, never numeric forecasting weights.',
        data='All custom LongMemEval test questions + eligible/abstention LoCoMo questions. Both public datasets were previously exposed; engineering/development benchmark, not fresh confirmatory evidence.',
        representation='2400 UTF8 bytes/session complete sentence extraction, dates and source retained. No semantic overwrite: all different dated records remain eligible until explicit capacity eviction.',
        controls='Unbounded hybrid; FIFO; static/GRU/liquid/independently-trained fixed-time liquid, all3seeds, budgets32000/128000 bytes. Same input texts and hybrid TFIDF+MiniLM reader, top5,16000 context chars. No query during writes.',
        accounting='Text+source+date+384float32 embedding+16 item metadata+writer state+16 global metadata. Offline score/trace files not part of searchable memory. Shared model weights and Python/TFIDF index overhead separately excluded.',
        qa='6 predetermined LME test questions x unbounded vs liquid seed23 at128k. Answer/judge Agnes flash, 9rpm shared persistent limiter, max512. Same-provider custom grading only; not official benchmark accuracy.',
        metric='Evidence session retention/retrieval, LME labeled sentence coverage in final context, storage/time, update/temporal categories. Evidence metrics only defined for labeled answerable examples.',
        limitations='No universal semantic extractor/truth resolver; conservative dated evidence. Current learned writers fixed insertion priorities, no dynamic revision truth claim. This is ready baseline infrastructure, not publication proof.'))
    encoder=ContentEncoder();groups=defaultdict(list)
    for e in examples:
        h=e['history'];key=hashlib.sha256(json.dumps([h.session_ids,h.dates,h.texts],ensure_ascii=False).encode()).hexdigest();groups[key].append(e)
    records=[];qa=[];timings=[]
    for gi,(key,ee) in enumerate(groups.items()):
        h=ee[0]['history'];t=time.perf_counter();v=encoder.encode(list(h.texts));dv=deltas(h);costs=np.asarray([BudgetItem(s,d,t,z,0,i).nbytes for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))]);scores={'fifo':np.arange(len(v),dtype=float)}
        with torch.no_grad():
            for name,m in models.items():scores[name]=m(torch.from_numpy(v),torch.from_numpy(dv)).numpy()
        qs=encoder.encode([e['history'].question for e in ee]);timings.append(dict(group=key,sessions=len(v),encode_and_write_seconds=time.perf_counter()-t))
        conditions=[('full_hybrid',None,list(range(len(v))),0)]
        for budget in [32000,128000]:
            for name,score in scores.items():
                state=0 if name=='fifo' else models[name].hidden*4
                kept=byte_retained_indices(score,costs,budget,state+16);conditions.append((name,budget,kept,state))
        for name,budget,kept,state in conditions:
            items=[BudgetItem(h.session_ids[i],h.dates[i],h.texts[i],v[i],float(i),i) for i in kept];reader=Reader(SimpleNamespace(items=items)) if items else None;used=int(costs[kept].sum()+state+16)
            if budget is not None and used>budget:raise AssertionError('Capacity violation')
            hh=replace(h,session_ids=tuple(i.session_id for i in items),dates=tuple(i.date for i in items),texts=tuple(i.text for i in items))
            for e,q in zip(ee,qs):
                t=time.perf_counter();sel,_=reader.query(e['history'].question,q) if reader else ([],[])
                ctx,trace=pack_preserved_context(replace(hh,question=e['history'].question),sel,16000)
                gold=set(e['gold']);selected={items[i].session_id for i in sel};retained={i.session_id for i in items};answerable=e['exclusion'] is None and bool(gold)
                coverage=None
                if e['dataset']=='longmemeval' and answerable:
                    r=raw[e['history'].question_id];ss=[s for turns in r['haystack_sessions'] for turn in turns if turn.get('has_answer') is True for s in sentences(turn['content'])];norm=' '.join(ctx.split());coverage=float(np.mean([' '.join(s.split()) in norm for s in ss])) if ss else None
                record=dict(question_id=e['history'].question_id,dataset=e['dataset'],group=e['group'],category=e['category'],method=name,budget=budget,payload_bytes=used,retained_sessions=len(items),input_sessions=len(v),evidence_retention=len(gold&retained)/len(gold) if answerable else None,retrieval_recall=len(gold&selected)/len(gold) if answerable else None,context_coverage=coverage,context_chars=len(ctx),retrieval_seconds=time.perf_counter()-t)
                records.append(record)
                if e['history'].question_id in qids and (name=='full_hybrid' or (name=='liquid_content-23' and budget==128000)):
                    qa.append(dict(**record,question=e['history'].question,question_date=e['history'].question_date,answer=e['answer'],context=ctx,trace=trace))
        if gi%10==0:print('language histories',gi+1,'/',len(groups),flush=True)
    encoder.close()
    with (OUT/'offline.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save('qa_tasks.json',qa);save('timing.json',timings);save('offline_complete.json',dict(records=len(records),questions=len(examples),histories=len(groups),qa_tasks=len(qa)))

def qa_run():
    tasks=read(OUT/'qa_tasks.json');config=load_config('configs/minute_precision.yaml')['api'];config=dict(config,rpm=9);namespace='language-readiness-v1:'+sha(OUT/'protocol.json')
    for i,task in enumerate(tasks):
        key=hashlib.sha256((task['question_id']+task['method']).encode()).hexdigest();path=OUT/'qa'/f'{key}.json';path.parent.mkdir(exist_ok=True);result=read(path) if path.exists() else dict(task)
        if 'judge_response' in result:continue
        client=AgnesClient(config)
        try:
            if 'generation' not in result:
                result['generation']=client.complete(messages_for(SimpleNamespace(question=task['question'],question_date=task['question_date']),task['context']),namespace=namespace)
                result['hypothesis']=result['generation']['content'];result.update(answer_metrics(result['hypothesis'],task['answer']));atomic_json(path,result)
            grade,response=custom_judge(client,task,result['hypothesis'],namespace,max_tokens=512);result.update(custom_judge_correct=grade,judge_response=response);result.pop('error_type',None);atomic_json(path,result)
        except Exception as exc:
            result['error_type']=type(exc).__name__;atomic_json(path,result);save('qa_status.json',dict(complete=False,error_type=type(exc).__name(),completed=sum('judge_response' in read(p) for p in (OUT/'qa').glob('*.json'))));raise
        finally:client.close()
        print('QA',i+1,'/',len(tasks),'saved',flush=True)
    save('qa_status.json',dict(complete=True,completed=len(tasks)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['offline','qa','all'],default='all');args=parser.parse_args()
    if args.phase in ['offline','all']:offline()
    if args.phase in ['qa','all']:qa_run()
