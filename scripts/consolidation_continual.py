"""Persistent scoped-task replay: no mixing facts between independent bAbI worlds."""
import sys,time,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.memory.consolidation import ConsolidationMemory,KINDS
from scripts import consolidation_study as study
OUT=ROOT/'runs/consolidation-continual-v1'


def main(frozen_controls=False):
    global OUT
    if frozen_controls:OUT=ROOT/'runs/consolidation-mechanism-v2'
    OUT.mkdir(exist_ok=True)
    if (OUT/'complete.json').exists():return
    torch.set_num_threads(1)
    stories=[(sid,r) for sid,r in study.read(study.OUT/'stories.json').items() if r['split']=='test'][:100]
    kinds=('liquid_beta0_frozen','liquid_w1_frozen','liquid_fixed_beta_frozen') if frozen_controls else KINDS
    vectors=study.read(study.OUT/'vectors.json');methods=[(k,s) for k in kinds for s in study.SEEDS]+[('fifo',0),('decay',0)]
    protocol=dict(registered_at=time.time(),stories=[sid for sid,r in stories],budgets=[4096,8192],revisit_lag_episodes=5,
        scope='Synthetic lifelong-agent stress replay of100 PUBLIC bAbI worlds. Persistent controller/store across tasks, explicit episode namespace prevents unrelated worlds overwriting each other. Not a natural same-user longitudinal dataset. Controllers remain frozen from episodic training, so this also tests length/distribution shift.',
        queries='All original queries at original within-episode order. After each completed world, revisit the final original query of the world5episodes earlier; it remains valid because worlds stop changing after completion. Tail pending revisits executed once. Same exact-answer feedback for every policy. No gold annotations in write features.',
        audit='Every record/control/index counted under same packed budget; label data only determines task success after action. Report immediate and delayed quality separately. No outcome-driven schedules or budgets.')
    if frozen_controls:protocol['posthoc_diagnostic']='After primary results, freeze liquid weights and override ONLY beta=0, w=1, or beta=1/30 perday during inference. No refit; distinguish adaptive forgetting from any forgetting. Not independent confirmation.'
    atomic_json(OUT/'protocol.json',protocol);rows=[];streams=[]
    for kind,seed in methods:
        model=None if seed==0 else study.load_model('liquid' if frozen_controls else kind,seed)
        if model is not None and frozen_controls:
            if kind=='liquid_fixed_beta_frozen':model.beta_override=1/30
            else:model.kind='no_forgetting' if kind=='liquid_beta0_frozen' else 'no_importance'
        for budget in protocol['budgets']:
            memory=ConsolidationMemory(model,budget,kind);now=0.;pending=[];peak=memory.payload_bytes;started=time.perf_counter()
            def ask(sid,e,phase):
                nonlocal now
                now+=1;result=memory.retrieve(now,key=sid+'::'+e['key']);answer=result[0].value if result else '';ok=answer==e['answer'];memory.reinforce([r.identity for r in result],now,float(ok))
                rows.append(dict(kind=kind,seed=seed,budget=budget,story=sid,phase=phase,correct=ok,answer=answer,reference=e['answer'],payload_bytes=memory.payload_bytes))
            for sid,story in stories:
                for e in story['events']:
                    if e['type']=='observation':
                        now+=1;memory.observe(f"{sid}:{e['line']}",e['text'],vectors[e['text']],now,sid+'::'+e['key'],e['value']);peak=max(peak,memory.payload_bytes)
                    else:ask(sid,e,'immediate')
                pending.append((sid,[e for e in story['events'] if e['type']=='query'][-1]))
                if len(pending)>5:old,q=pending.pop(0);ask(old,q,'delayed')
            for sid,q in pending:ask(sid,q,'delayed')
            streams.append(dict(kind=kind,seed=seed,budget=budget,peak_bytes=peak,seconds=time.perf_counter()-started,**memory.stats))
        print('Continual replay',kind,seed,flush=True)
    atomic_json(OUT/'query_records.json',rows);atomic_json(OUT/'stream_records.json',streams)
    summary=[]
    for kind in list(kinds)+['fifo','decay']:
        for budget in protocol['budgets']:
            for phase in ['immediate','delayed']:
                rr=[r for r in rows if r['kind']==kind and r['budget']==budget and r['phase']==phase]
                summary.append(dict(kind=kind,budget=budget,phase=phase,accuracy=float(np.mean([r['correct'] for r in rr]))))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'complete.json',dict(queries=len(rows),streams=len(streams)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen-controls',action='store_true');a=p.parse_args();main(a.frozen_controls)
