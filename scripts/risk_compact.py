"""Staged controller memory repair; sparse models retrained on the changed gate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import shutil,argparse
import numpy as np
import torch
from src.config import atomic_json
from src.provenance import snapshot_source,utc_now
import scripts.risk_trajectory as exp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','offline','qa','all'],default='all');a=p.parse_args();torch.set_num_threads(1)
    original=exp.OUT;dest=ROOT/'runs/risk-trajectory-compact-v1';dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'protocol.json').exists():
        protocol=exp.read(original/'protocol.json');sig,archive=snapshot_source()
        protocol.update(source=sig,archive=archive,staged_extension=dict(time=utc_now(),reason='Controller previous-vector state consumed1544bytes. Replace cosine novelty with abs(current cheap score-previous cheap score), retain previous score+credit as16bytes total. Refit gate on train, recalibrate48% onval, retrain9risk models3epochs. Reuse18full/periodic models because those trajectories do not use gate. Same cheap training recipe must reproduce original weights exactly. Post-diagnostic extension, not a new independent test.',controller_bytes=16),qa='Same12development questions and four conditions; new experiment namespace, standard within-run caching. No new independent QA cohort.')
        atomic_json(dest/'protocol.json',protocol);atomic_json(dest/'compact_mode.json',dict(enabled=True))
        for kind in exp.KINDS:
            for regime in ['full','periodic']:
                for seed in exp.SEEDS:
                    source=original/'checkpoints'/f'{kind}-{regime}-{seed}';target=dest/'checkpoints'/source.name
                    shutil.copytree(source,target,dirs_exist_ok=True)
    exp.OUT=dest
    if a.phase in ['train','all']:
        exp.train();assert np.array_equal(np.load(dest/'shared.npz')['cheap'],np.load(original/'shared.npz')['cheap'])
    if a.phase in ['offline','all']:exp.evaluate(account_controller=True)
    if a.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=dest;runner.qa_run()

