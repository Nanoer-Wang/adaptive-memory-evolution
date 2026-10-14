"""Post-result mechanism diagnostic; no fitting, tuning or API calls."""
import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import spearmanr
from collections import defaultdict
from scripts.writer_efficiency import models, examples
from scripts.content_experiment import deltas, history_key
from src.models.content_encoder import ContentEncoder
from src.config import atomic_json

def main():
    torch.set_num_threads(1)
    out=ROOT/'runs/linear-mechanism-diagnostic-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        print((out/'summary.json').read_text());return
    atomic_json(out/'protocol.json',dict(status='Post-result exploratory mechanism diagnostic, exposed evaluation histories',
        recipe='All nine frozen teachers and train-only distilled students; per-history rank correlations. Reset hidden state for each item, preserving its observed delta. No fitting or threshold selection.',
        limitations='Teacher approximation is not label accuracy; reset sensitivity is not useful history information; histories equally weighted; no confidence intervals.'))
    ms=models(); weights=np.load(ROOT/'runs/content-fallback-v1/students.npz');enc=ContentEncoder();seen=set();records=[]
    with torch.inference_mode():
        for e in examples():
            h=e['history'];key=history_key(h)
            if key in seen:continue
            seen.add(key);v=enc.encode(list(h.texts));vt=torch.from_numpy(v);dt=torch.from_numpy(deltas(h))
            for (kind,seed),m in ms.items():
                full=m(vt,dt).numpy();w=weights[f'{kind}_{seed}'];student=v@w[:-1]+w[-1]
                reset=np.array([float(m.step(x,vt.new_zeros(m.hidden),d)[0]) for x,d in zip(vt,dt)])
                var=float(np.var(full));mse=float(np.mean((student-full)**2))
                records.append(dict(dataset=e['dataset'],cluster=key,kind=kind,seed=seed,events=len(v),
                    student_teacher_spearman=float(spearmanr(student,full).statistic),
                    reset_teacher_spearman=float(spearmanr(reset,full).statistic),
                    student_teacher_r2=1-mse/var if var>1e-12 else None,
                    reset_rmse=float(np.sqrt(np.mean((reset-full)**2))),teacher_std=float(np.std(full))))
    enc.close();groups=defaultdict(list)
    for r in records:groups[r['dataset'],r['kind']].append(r)
    summary=[]
    for (dataset,kind),rr in groups.items():
        summary.append(dict(dataset=dataset,kind=kind,histories=len({r['cluster'] for r in rr}),
            **{k:float(np.nanmean([r[k] for r in rr])) for k in ['student_teacher_spearman','reset_teacher_spearman','student_teacher_r2','reset_rmse','teacher_std']}))
    atomic_json(out/'records.json',records);atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
