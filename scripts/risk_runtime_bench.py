"""Controller-inclusive online latency, including actual per-event risk features."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import time
import numpy as np
import torch
from src.memory.fast_writer import OnlineFrozenWriter
from src.memory.risk_residual import RiskResidualWriter,risk_features,StreamingRiskGate
from src.models.content_encoder import ContentEncoder
from src.config import atomic_json
from scripts.risk_trajectory import OUT,examples,apply_shared,deltas


def replay(model,v,dt,shared,policy,fast,optimized=False):
    state=torch.zeros(model.hidden) if model else None;decay=torch.nn.functional.softplus(model.decay).detach() if model else None
    previous=np.zeros(384,dtype=np.float32);credit=1.;scores=[];calls=[];w=shared['cheap'];gate=StreamingRiskGate(shared) if optimized and policy=='risk' else None
    for i,(x,d) in enumerate(zip(v,dt)):
        base=float(x@w[:-1]+w[-1]);take=policy=='full' or (policy=='periodic' and i%2==0)
        if policy=='risk' and gate is not None:
            base,_,take=gate.observe(x,d)
        elif policy=='risk':
            z=risk_features(np.stack([previous,x]),np.array([0,d]),np.array([0,base]))[-1:]
            z[:,:-1]=(z[:,:-1]-shared['mean'])/shared['std'];risk=float((z@shared['risk'])[0])
            credit=min(2.,credit+.5);take=risk>=float(shared['threshold']) and credit>=1.
            if take:credit-=1.
            previous=x.copy()
        if model is None:scores.append(base);continue
        state=state*torch.exp(-decay*float(d))
        if take:
            _,state=fast.step(torch.from_numpy(x),state,0.);calls.append(i)
        scores.append(base+float(model.core.head(state).squeeze(-1)))
    return np.array(scores),calls


def main():
    torch.set_num_threads(1);shared=np.load(OUT/'shared.npz');enc=ContentEncoder();data=[];gate_audit=[]
    seen=set()
    for e in examples():
        h=e['history'];key=tuple(h.session_ids)
        if key in seen:continue
        seen.add(key);v=enc.encode(list(h.texts));dt=deltas(h);base,risks,masks=apply_shared(v,dt,shared);gate=StreamingRiskGate(shared);actual=[gate.observe(x,d) for x,d in zip(v,dt)]
        errors=float(np.max(np.abs(np.array([a[1] for a in actual])-risks)));disagreements=int(np.sum(np.array([a[2] for a in actual])!=masks['risk']))
        assert disagreements==0
        gate_audit.append(dict(histories_index=len(gate_audit),events=len(v),risk_max_abs_error=errors,call_disagreements=disagreements))
    atomic_json(OUT/'gate_optimization_audit.json',gate_audit)
    for e in examples()[:8]:
        h=e['history'];data.append((enc.encode(list(h.texts)),deltas(h)))
    enc.close();records=[]
    with torch.inference_mode():
        for kind in ['liquid','gru','static']:
            models={};fast={}
            for regime in ['full','periodic','risk']:
                m=RiskResidualWriter(kind);m.load_state_dict(torch.load(OUT/'checkpoints'/f'{kind}-{regime}-23/best.pt',weights_only=True)['state_dict']);m.eval();models[regime]=m;fast[regime]=OnlineFrozenWriter(m.core)
            for hi,(v,dt) in enumerate(data):
                base,_,masks=apply_shared(v,dt,shared)
                errors={}
                for regime,m in models.items():
                    ref=m(torch.from_numpy(v),torch.from_numpy(dt),torch.from_numpy(base),masks[regime]).numpy();actual,calls=replay(m,v,dt,shared,regime,fast[regime],True);errors[regime]=float(np.max(np.abs(ref-actual)))
                    assert np.array_equal(calls,np.flatnonzero(masks[regime]));assert errors[regime]<5e-5
                names=['full','periodic','risk','cheap'];times={n:[] for n in names}
                for repeat in range(10):
                    for name in names[repeat%4:]+names[:repeat%4]:
                        start=time.perf_counter();replay(models.get(name),v,dt,shared,name,fast.get(name),True);elapsed=time.perf_counter()-start
                        if repeat>=3:times[name].append(elapsed)
                for name,ts in times.items():records.append(dict(kind=kind,policy=name,history=hi,events=len(v),times=ts,median_ms_per_event=float(np.median(ts)*1000/len(v)),max_abs_error=errors.get(name)))
    atomic_json(OUT/'online_latency.json',dict(protocol='CPU1thread; first8 public histories;seed23;3warmups7rotatingrepeats;scripted core shared;includes live cheap+folded gate+drift+head,excludes embedding/storage/retrieval/API;gate algebra validated all unique test histories',records=records))
    summary=[dict(kind=k,policy=p,ms_per_event=float(np.median([r['median_ms_per_event'] for r in records if (r['kind'],r['policy'])==(k,p)]))) for k in ['liquid','gru','static'] for p in ['full','periodic','risk','cheap']]
    atomic_json(OUT/'online_latency_summary.json',summary);print(summary)

if __name__=='__main__':main()
