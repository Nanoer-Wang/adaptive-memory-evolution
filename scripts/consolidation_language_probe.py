"""Frozen structured-trained controllers on public, new-to-QA language histories."""
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.data.longmemeval import prepared,public_history,date_days
from src.models.content_encoder import ContentEncoder
from src.memory.consolidation_agent import ConsolidatingAgentMemory
from scripts import consolidation_study as study
from scripts import context_utility as generator
OUT=ROOT/'runs/consolidation-language-v1'
read=study.read


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    assert (study.OUT/'training_complete.json').exists()
    rows,manifest=prepared(load_config('configs/minute_precision.yaml'));used=set()
    for p in (ROOT/'runs').glob('*/qa/*.json'):used.add(read(p).get('question_id'))
    candidates=[r for r in rows if r['question_id'] in manifest['splits']['test'] and r['question_id'] not in used and not r['question_id'].endswith('_abs')]
    wanted={'knowledge-update':2,'temporal-reasoning':1,'multi-session':2,'single-session-user':1};selected=[]
    for category,n in wanted.items():
        group=sorted([r for r in candidates if r['question_type']==category],key=lambda r:hashlib.sha256(('consolidation-language:'+r['question_id']).encode()).hexdigest())
        assert len(group)>=n;selected+=group[:n]
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),ids=[r['question_id'] for r in selected],prior_qa_excluded=sorted(q for q in used if q),
        methods=['decay','linear','gru','liquid'],training='All learned models frozen seed23 from bAbI consolidation-v2. Zero-shot cross-domain probe, NOT language-trained policy efficacy.',
        memory='32000byte defined payload, complete original user/assistant text partitioned into <=1200char attributed spans, no pre-truncation or label-conditioned extraction. All methods use same32dim projected frozen MiniLM and8retrieved records,12000char cap. Controller state continuous within full dated history, reset between user questions; no unrelated users spliced.',
        answers='24new measured Agnes answers (6questions x4methods),same generator/grader; invalid-format repair only. No hyperparameter/seed changes after outcomes.',
        scope='Public existing offline split, new-to-QA IDs, not independent new dataset or multi-step tool-agent benchmark. Structured-trained policy transfer is diagnostic. Similarity does not imply fact revision; no privileged keys passed for natural text.',
        checkpoints={k:hashlib.sha256((study.OUT/'checkpoints'/f'{k}-23.pt').read_bytes()).hexdigest() for k in ['linear','gru','liquid']}))
    torch.set_num_threads(1);encoder=ContentEncoder();tasks=[];metrics=[]
    for row in selected:
        h=public_history(row,'minute');qdate=date_days(h.question_date,'minute');source_turns={}
        ordered=sorted(zip(row['haystack_session_ids'],row['haystack_dates'],row['haystack_sessions']),key=lambda t:date_days(t[1],'minute'))
        for kind in ['decay','linear','gru','liquid']:
            model=None if kind=='decay' else study.load_model(kind,23);agent=ConsolidatingAgentMemory(encoder,model,32000,kind,retrieval='projected');started=time.perf_counter();peak=agent.memory.payload_bytes;future=0
            for sid,date,turns in ordered:
                if date_days(date,'minute')>qdate:future+=1;continue
                agent.observe(str(sid),date,turns);peak=max(peak,agent.memory.payload_bytes)
            result=agent.query(h.question,h.question_date);context=result['context'];gold=set(map(str,row['answer_session_ids']));retrieved={s['source_id'] for s in result['sources']}
            metrics.append(dict(question_id=h.question_id,method=kind,source_recall=len(retrieved&gold)/max(len(gold),1),peak_bytes=peak,elapsed_seconds=time.perf_counter()-started,future_sessions_excluded=future,**agent.memory.stats))
            tasks.append(dict(question_id=h.question_id,split='test',category=row['question_type'],method=kind,question=h.question,question_date=h.question_date,answer=str(row['answer']),context=context,trace=result['sources'],payload_bytes=result['payload_bytes'],context_chars=len(context),context_sha256=hashlib.sha256(context.encode()).hexdigest()))
            agent.checkpoint(OUT/'memory_snapshots'/f'{h.question_id}-{kind}.json');print('Prepared language',h.question_id,kind,flush=True)
    encoder.close();atomic_json(OUT/'tasks.json',tasks);atomic_json(OUT/'memory_metrics.json',metrics);atomic_json(OUT/'prepared.json',dict(questions=6,tasks=24));atomic_json(OUT/'training_complete.json',dict(frozen_model_source=str(study.OUT.relative_to(ROOT))))


def qa():
    prev=generator.OUT
    try:generator.OUT=OUT;generator.qa('test')
    finally:generator.OUT=prev
    from scripts.run_semantic_series import repair_invalid
    repair_invalid(OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','qa'],default='all');a=p.parse_args()
    if a.phase in ['all','prepare']:prepare()
    if a.phase in ['all','qa']:qa()
