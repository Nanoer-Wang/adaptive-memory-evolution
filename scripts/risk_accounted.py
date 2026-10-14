"""Resource accounting correction; retain original run, weights and decisions."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse
import numpy as np
import torch
from scripts.risk_trajectory import OUT,read,evaluate
from src.config import atomic_json
from src.provenance import snapshot_source,utc_now

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['offline','qa','all'],default='all');args=p.parse_args();torch.set_num_threads(1)
    dest=OUT/'controller-accounted';dest.mkdir(exist_ok=True)
    if not (dest/'amendment.json').exists():
        sig,archive=snapshot_source();atomic_json(dest/'amendment.json',dict(time=utc_now(),source=sig,archive=archive,
            correction='Original evaluation charged evidence and model hidden state but omitted risk controller previous384float32 vector and8byte credit. Add1544bytes only to risk execution under same total budgets. Original results retained. No retraining, model selection, gates, thresholds, labels or primary comparisons changed.',
            validation_limit='Checkpoint selection still uses original validation payload+core-state proxy; it was not reselected after this accounting correction. This limits claims of optimally tuned resource frontiers.',
            cache='Exact copy of original protocol retained for original API namespace and identical prompt-cache reuse; this separate dated amendment documents changed capacity accounting, not a new preregistration.'))
        atomic_json(dest/'protocol.json',read(OUT/'protocol.json'))
    if args.phase in ['offline','all']:evaluate(dest,True)
    if args.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=dest;runner.qa_run()
