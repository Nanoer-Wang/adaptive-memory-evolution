"""Replay saved evidence without training or changing primary experimental results."""
import sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import evolving_memory_study as study
from src.config import atomic_json


def main():
    torch.set_num_threads(1)
    out=study.OUT;data=study.read(out/'data.json');saved=study.read(out/'results.json')
    evidence=[out/'data.json',out/'protocol.json',out/'rates.json',out/'results.json']+sorted((out/'checkpoints').glob('*.pt'))
    digest=lambda:{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    before=digest();checked=0;max_error=0.;batch_count=0
    for method in study.METHODS:
        for seed in study.SEEDS:
            model,rate=study.load(method,seed)
            for condition in study.CONDITIONS:
                rows=[r for r in data['test'] if r['environment']==condition]
                for capacity in study.CAPACITIES:
                    replay=study.evaluate_rows(rows,capacity,method,rate,model,seed*100+77,record=True)
                    trace=study.read(out/'traces'/f'{method}-{seed}-{condition}-{capacity}.json')
                    assert trace['trace']==replay['trace'],'Trace replay mismatch'
                    for i,row in enumerate(rows):
                        old=next(r for r in saved if r['method']==method and r['seed']==seed and r['condition']==condition and r['capacity']==capacity and r['trajectory_seed']==row['seed'])
                        for metric,values in replay['metrics'].items():
                            error=abs(old[metric]-values[i]);max_error=max(max_error,error)
                            assert error<1e-6,(method,seed,condition,capacity,metric,error)
                        assert old['episodes']==replay['episodes'][i]
                        assert replay['metrics']['peak_slots'][i]<=capacity
                        checked+=1
                    batch_count+=1
    # A cached repeat must preserve original training, test evidence and checkpoints.
    study.prepare();study.train();study.evaluate()
    assert before==digest(),'Cached run mutated experiment evidence'
    atomic_json(out/'replay_audit.json',dict(passed=True,metric_rows=checked,trace_batches=batch_count,
        maximum_metric_absolute_difference=max_error,capacity_verified=True,episodes_verified=True,
        cached_evidence_unchanged=True,evidence_sha256=before,
        scope='Deterministic replay of saved models and all event traces; not independent replication of training or independent dataset validation. Wall-clock timing is excluded.'))
    print(f'Replayed {checked} metric rows and {batch_count} complete event traces; max error {max_error}; cached evidence unchanged.')


if __name__=='__main__':main()
