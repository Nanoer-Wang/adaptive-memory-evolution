"""Post-primary audits; no checkpoint selection or hyperparameter changes."""
import sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.memory.consolidation import ConsolidationController
from scripts.continual_utility_study import OUT,read,SEEDS,BUDGETS,METHODS,ENVIRONMENTS,evaluate_set


def main():
    assert (OUT/'evaluation_complete.json').exists()
    if (OUT/'diagnostics_complete.json').exists():return
    data=read(OUT/'data.json');learning=[];ablations=[]
    for kind in METHODS[1:]:
        for seed in SEEDS:
            torch.manual_seed(seed);model=ConsolidationController(kind)
            initial=float(np.mean([np.mean(evaluate_set(data['val'],c,model,seed=700)['metrics']['utility']) for c in BUDGETS]))
            checkpoint=torch.load(OUT/'checkpoints'/f'{kind}-{seed}.pt',weights_only=True)
            history=read(OUT/f'training-{kind}-{seed}.json')
            learning.append(dict(method=kind,seed=seed,initial_validation_utility=initial,
                                 selected_validation_utility=checkpoint['validation_utility'],
                                 gain=checkpoint['validation_utility']-initial,selected_epoch=checkpoint['epoch'],
                                 last_validation_utility=history['history'][-1]['validation_utility']))
            if kind!='liquid':continue
            model.load_state_dict(checkpoint['state_dict']);model.kind='no_time';model.eval()
            for env in ENVIRONMENTS:
                rows=[r for r in data['test'] if r['environment']==env]
                for capacity in BUDGETS:
                    output=evaluate_set(rows,capacity,model,seed=seed*100+9000)
                    for i,row in enumerate(rows):
                        ablations.append(dict(method='liquid_no_time_frozen',seed=seed,environment=env,capacity=capacity,
                                              trajectory_seed=row['seed'],**{k:v[i] for k,v in output['metrics'].items()}))
    atomic_json(OUT/'learning_audit.json',learning)
    atomic_json(OUT/'no_time_posthoc.json',dict(note='Post-primary, same trained weights, disable ONLY continuous hidden-state flow. Not retrained or used for checkpoint selection.',results=ablations))
    rows=read(OUT/'results.json');interactions=[]
    def mean(method,env,capacity,trajectory,metric):
        return np.mean([r[metric] for r in rows if r['method']==method and r['environment']==env and r['capacity']==capacity and r['trajectory_seed']==trajectory])
    for capacity in BUDGETS:
        for env in ['slow','rapid']:
            for baseline in METHODS[:-1]:
                for metric in ['accuracy','utility']:
                    diff=[]
                    for trajectory in sorted({r['trajectory_seed'] for r in rows}):
                        diff.append((mean('liquid',env,capacity,trajectory,metric)-mean(baseline,env,capacity,trajectory,metric))-
                                    (mean('liquid','stationary',capacity,trajectory,metric)-mean(baseline,'stationary',capacity,trajectory,metric)))
                    rng=np.random.default_rng(456);samples=np.mean(rng.choice(diff,size=(3000,len(diff)),replace=True),axis=1)
                    interactions.append(dict(capacity=capacity,environment=env,baseline=baseline,metric=metric,
                                             difference_of_differences=float(np.mean(diff)),ci95=np.quantile(samples,[.025,.975]).tolist()))
    atomic_json(OUT/'environment_interactions.json',interactions)
    peaks=[]
    for path in (OUT/'traces').glob('*.json'):
        capacity=int(path.stem.split('-')[-1]);trace=read(path)
        peak=max(max(r['slots']) for r in trace['events']);assert peak<=capacity
        peaks.append(dict(file=path.name,capacity=capacity,peak_slots=peak))
    atomic_json(OUT/'capacity_audit.json',peaks)
    atomic_json(OUT/'diagnostics_complete.json',dict(at=time.time(),learning_rows=len(learning),
                note='Validation-initialization and frozen time-flow audit added after primary evaluation; exploratory, no refit.',
                checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'checkpoints').glob('*.pt')}))
    print('Diagnostics complete; no models modified',flush=True)


if __name__=='__main__':torch.set_num_threads(1);main()
