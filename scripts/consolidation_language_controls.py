"""Post-hoc unlimited-storage diagnostic; never counted as fair-budget competitor."""
import sys,time,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.data.longmemeval import prepared,date_days
from src.models.content_encoder import ContentEncoder
from src.memory.consolidation_agent import ConsolidatingAgentMemory
from scripts import consolidation_language_probe as probe
from scripts import context_utility as generator
OUT=ROOT/'runs/consolidation-language-controls-v1'


def main(hybrid=False):
    global OUT
    if hybrid:OUT=ROOT/'runs/consolidation-language-hybrid-v1'
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    if not (OUT/'prepared.json').exists():
        ids=probe.read(probe.OUT/'protocol.json')['ids'];rows,_=prepared(load_config('configs/minute_precision.yaml'));by={r['question_id']:r for r in rows}
        atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),question_ids=ids,retrieval='full384 dense plus lexical' if hybrid else 'projected32 dense',scope='Post-hoc diagnosis after bounded policies all refused. Unlimited raw text store; not a capacity-matched competitor. No policy retraining or changes to main outcomes. Hybrid is an additional six-question fixed re-run, includes generation variability.'))
        encoder=ContentEncoder();tasks=[];audit=[]
        for q in ids:
            row=by[q];gold=set(map(str,row['answer_session_ids']));now=date_days(row['question_date'],'minute')
            for kind in ['decay','linear','gru','liquid']:
                snap=probe.read(probe.OUT/'memory_snapshots'/f'{q}-{kind}.json');kept={json.loads(origin[0])[0] for item in snap['items'] for origin in item['origins']}
                audit.append(dict(question_id=q,method=kind,gold_sources=len(gold),gold_sources_in_store=len(kept&gold),stored_records=len(snap['items'])))
            agent=ConsolidatingAgentMemory(encoder,None,None,'fifo',retrieval='hybrid' if hybrid else 'projected')
            ordered=sorted(zip(row['haystack_session_ids'],row['haystack_dates'],row['haystack_sessions']),key=lambda e:date_days(e[1],'minute'))
            for sid,date,turns in ordered:
                if date_days(date,'minute')<=now:agent.observe(str(sid),date,turns)
            result=agent.query(row['question'],row['question_date']);tasks.append(dict(question_id=q,split='test',category=row['question_type'],method='unlimited',question=row['question'],question_date=row['question_date'],answer=str(row['answer']),context=result['context'],trace=result['sources'],payload_bytes=result['payload_bytes']))
        encoder.close();atomic_json(OUT/'tasks.json',tasks);atomic_json(OUT/'storage_audit.json',audit);atomic_json(OUT/'prepared.json',dict(questions=6));atomic_json(OUT/'training_complete.json',dict(no_training=True))
    prev=generator.OUT
    try:generator.OUT=OUT;generator.qa('test')
    finally:generator.OUT=prev
    from scripts.run_semantic_series import repair_invalid
    repair_invalid(OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hybrid',action='store_true');a=p.parse_args();main(a.hybrid)
