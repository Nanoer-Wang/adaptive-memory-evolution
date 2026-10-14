"""Task-first public QA and controlled add-back interventions, frozen writers."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,time,itertools
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.provenance import snapshot_source,runtime_info
from src.models.content_encoder import ContentEncoder
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.memory.risk_residual import RiskResidualWriter
from src.retrieval.preserved_context import pack_preserved_context
from src.data.longmemeval import prepared
from scripts.content_experiment import Reader,deltas,history_key
from scripts.writer_efficiency import models,examples,read
from scripts.risk_trajectory import apply_shared

OUT=ROOT/'runs/task-utility-v1'

def supplement_indices(kept,selected,ranking,texts):
    """No gold/answers: top two query-ranked evictions; length-matched retained control."""
    keptset=set(kept);selectedset=set(selected)
    added=[int(i) for i in ranking if i not in keptset][:2]
    candidates=[int(i) for i in ranking if i in keptset and i not in selectedset]
    n=min(len(added),len(candidates));added=added[:n]
    target=sum(len(texts[i]) for i in added)
    choices=itertools.combinations(candidates,n)
    control=list(min(choices,key=lambda ids:abs(sum(len(texts[i]) for i in ids)-target))) if n else []
    return added,control

def prepare():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'offline_complete.json').exists():return
    if (OUT/'protocol.json').exists():raise RuntimeError('Partial preparation exists; preserve before repair')
    all_e=examples();prior=set()
    for p in (ROOT/'runs').glob('*/qa/*.json'):
        prior.add(read(p).get('question_id'))
    buckets=defaultdict(list)
    for e in all_e:
        if e['dataset']=='longmemeval' and e['eligible'] and e['history'].question_id not in prior:buckets[e['category']].append(e)
    chosen=[]
    for cat in sorted(buckets):
        chosen+=sorted(buckets[cat],key=lambda e:hashlib.sha256(('task-utility-v1:'+e['history'].question_id).encode()).hexdigest())[:2]
    assert len(chosen)==12
    sig,archive=snapshot_source()
    atomic_json(OUT/'protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),
        objective='Task correctness primary; recall descriptive only and never a gate for answer generation.',
        selection='Two eligible LME test questions per category by fixed salted hash, excluding IDs in existing runs/*/qa/*.json. New QA subset of previously exposed offline development data, not an independent benchmark.',
        qa_ids=[e['history'].question_id for e in chosen],excluded_prior_qa_ids=sorted(q for q in prior if q),seed=23,budget_bytes=64000,
        conditions=['linear','liquid','gru','risk_liquid','full_reference','linear_addback','linear_retained_control','liquid_addback','liquid_retained_control'],
        controls='Same compressed observations, encoder, reader top5 and 16000-character base context. Writers never see question or gold. Full reference uses unlimited memory with same top5 reader, not a performance upper bound. Compact risk scorer has different training recipe and is a descriptive candidate, not a liquid-only ablation.',
        intervention='For linear and liquid, append up to two query-ranked evicted sessions from full archive. Matched retained control appends the same number of retained-but-unretrieved sessions minimizing raw-character length difference. Preserve base context byte-for-byte; separately pack supplement within6000chars. No gold in selection. Extra archive access/storage and tokens are diagnostic privileges, not deployable same-budget baselines; matching is characters, not exact tokens.',
        primary='Paired question accuracy liquid minus strong linear, and liquid minus GRU, with95% paired bootstrap intervals and exact two-sided discordant-pair tests. Do not infer beneficial forgetting from recall alone. Addback harm must be inspected against retained control and raw answers.',
        secondary='Risk liquid, full reference, category, recall versus correctness, addback benefit/harm, prompt tokens, cached versus actual usage, provider latency excluding rate wait, context size. No parameter updates, no outcome-selected subset.',
        limits='12 questions,1 seed,1 generation per unique prompt,temperature0,same-provider custom judge. Cached identical prompts are not independent trials. Single retrieve-answer agent component, not multi-step tool-use or long-term agent success. No automatic obsolete-fact attribution without textual evidence.'))
    raw={str(r['question_id']):r for r in prepared(load_config('configs/minute_precision.yaml'))[0]}
    ms=models();students=np.load(ROOT/'runs/content-fallback-v1/students.npz')
    riskroot=ROOT/'runs/risk-trajectory-compact-v1';shared=np.load(riskroot/'shared.npz')
    riskmodel=RiskResidualWriter('liquid');riskmodel.load_state_dict(torch.load(riskroot/'checkpoints/liquid-risk-23/best.pt',weights_only=True)['state_dict']);riskmodel.eval()
    enc=ContentEncoder();tasks=[];writes=[]
    with torch.inference_mode():
        for e in chosen:
            h=e['history'];v=enc.encode(list(h.texts));vt=torch.from_numpy(v);dt=deltas(h);dtt=torch.from_numpy(dt);qv=enc.encode([h.question])[0]
            items=[BudgetItem(s,d,t,z,0,i) for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))];costs=np.array([i.nbytes for i in items]);w=students['liquid_23']
            scores={'linear':v@w[:-1]+w[-1]}
            for kind in ['liquid','gru']:scores[kind]=ms[kind,23](vt,dtt).numpy()
            base,risk,masks=apply_shared(v,dt,shared)
            scores['risk_liquid']=riskmodel(vt,dtt,torch.from_numpy(base),torch.from_numpy(masks['risk'])).numpy()
            fullreader=Reader(SimpleNamespace(items=items));fullsel,fullscores=fullreader.query(h.question,qv);ranking=sorted(range(len(v)),key=lambda i:(-float(fullscores[i]),i))
            for name in ['linear','liquid','gru','risk_liquid','full_reference']:
                start=time.perf_counter();state={'linear':16,'liquid':80,'gru':68,'risk_liquid':96,'full_reference':16}[name]
                kept=list(range(len(v))) if name=='full_reference' else byte_retained_indices(scores[name],costs,64000,state)
                reader=Reader(SimpleNamespace(items=[items[i] for i in kept]));selected_local,_=reader.query(h.question,qv);selected=[kept[i] for i in selected_local]
                context,trace=pack_preserved_context(h,selected,16000);retrieval_seconds=time.perf_counter()-start
                gold=set(e['gold']);base_rec=dict(question_id=h.question_id,dataset=e['dataset'],cluster=history_key(h),category=e['category'],seed=23,budget=64000 if name!='full_reference' else None,
                    method=name,question=h.question,question_date=h.question_date,answer=str(raw[h.question_id]['answer']),context=context,trace=trace,
                    recall=len(gold&{h.session_ids[i] for i in selected})/len(gold),retention=len(gold&{h.session_ids[i] for i in kept})/len(gold),payload_bytes=int(costs[kept].sum()+state),
                    events=len(v),kept_ids=[h.session_ids[i] for i in kept],selected_ids=[h.session_ids[i] for i in selected],context_chars=len(context),context_sha256=hashlib.sha256(context.encode()).hexdigest(),retrieval_seconds=retrieval_seconds,
                    base_method=name,intervention='none',added_ids=[],writer_calls=int(masks['risk'].sum()) if name=='risk_liquid' else (0 if name in ['linear','full_reference'] else len(v)))
                tasks.append(base_rec)
                writes.append(dict(question_id=h.question_id,method=name,kept=kept,scores=scores[name].tolist() if name in scores else None,events=len(v)))
                if name in ['linear','liquid']:
                    added,control=supplement_indices(kept,selected,ranking,h.texts)
                    for suffix,extra in [('addback',added),('retained_control',control)]:
                        supplement,extratrace=pack_preserved_context(h,extra,6000)
                        ctx=context+('\n\n'+supplement if supplement else '')
                        tasks.append(dict(base_rec,method=name+'_'+suffix,intervention=suffix,context=ctx,trace=trace+extratrace,
                            selected_ids=[h.session_ids[i] for i in selected+extra],added_ids=[h.session_ids[i] for i in extra],
                            recall=len(gold&{h.session_ids[i] for i in selected+extra})/len(gold),context_chars=len(ctx),context_sha256=hashlib.sha256(ctx.encode()).hexdigest(),
                            supplement_chars=len(supplement),diagnostic_archive_access=suffix=='addback'))
            print('Prepared task',len(writes)//5,'/',len(chosen),flush=True)
    enc.close();atomic_json(OUT/'qa_tasks.json',tasks);atomic_json(OUT/'write_traces.json',writes)
    atomic_json(OUT/'offline_complete.json',dict(questions=len(chosen),tasks=len(tasks),unique_context_question_pairs=len({(t['question_id'],t['context_sha256']) for t in tasks})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','qa','all'],default='all');a=p.parse_args()
    if a.phase in ['prepare','all']:prepare()
    if a.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=OUT;runner.qa_run()
