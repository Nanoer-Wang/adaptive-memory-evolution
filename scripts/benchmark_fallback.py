"""Strict online controller latency, including every cheap current-content score."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import time
import numpy as np
import torch
from scripts.writer_efficiency import models,examples
from scripts.content_experiment import deltas
from src.models.content_encoder import ContentEncoder
from src.memory.fast_writer import OnlineFrozenWriter
from src.config import atomic_json
OUT=ROOT/'runs/content-fallback-v1'

def online(model,v,dt,w,policy):
    state=v.new_zeros(model.hidden);pending=0.;scores=[]
    for i,(x,d) in enumerate(zip(v,dt)):
        pending+=float(d)
        if policy=='full' or (policy=='hybrid2' and i%2==0):
            s,state=model.step(x,state,min(pending,12.));pending=0.
        else:s=torch.dot(x,w[:-1])+w[-1]
        scores.append(s)
    return torch.stack(scores)

def main():
    torch.set_num_threads(1);ms=models();weights=np.load(OUT/'students.npz');enc=ContentEncoder();histories=examples()[:8];data=[]
    for e in histories:
        h=e['history'];data.append((torch.from_numpy(enc.encode(list(h.texts))),torch.from_numpy(deltas(h))))
    enc.close();records=[]
    with torch.inference_mode():
        for kind in ['liquid','gru','static']:
            model=OnlineFrozenWriter(ms[kind,23]);w=torch.from_numpy(weights[f'{kind}_23'])
            for index,(v,dt) in enumerate(data):
                names=['full','hybrid2','linear_only'];timing={n:[] for n in names}
                for repeat in range(18):
                    for name in names[repeat%3:]+names[:repeat%3]:
                        start=time.perf_counter();online(model,v,dt,w,name);elapsed=time.perf_counter()-start
                        if repeat>=3:timing[name].append(elapsed)
                for name,ts in timing.items():records.append(dict(kind=kind,history=index,policy=name,events=len(v),repeats=ts,ms_per_event=float(np.median(ts)*1000/len(v))))
    atomic_json(OUT/'online_timing.json',dict(protocol='CPU1thread,first8publicLMEhistories,seed23,3warmups15rotatingrepeats;all methods scripted teacher;current-content dot product included only when used;excludes embeddings,eviction,retrieval and API',records=records))
    summary=[]
    for kind in ['liquid','gru','static']:
        for name in ['full','hybrid2','linear_only']:
            ts=[r['ms_per_event'] for r in records if (r['kind'],r['policy'])==(kind,name)]
            summary.append(dict(kind=kind,policy=name,median_ms_per_event=float(np.median(ts))))
    atomic_json(OUT/'online_timing_summary.json',summary);print(summary)

if __name__=='__main__':main()
