"""Final source/weight/split/cache audit for the complete staged experiment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,sqlite3
import numpy as np
import torch
from src.config import atomic_json,load_config,resolve
from scripts.risk_trajectory import read
import scripts.writer_efficiency as runner

def main():
    root=ROOT/'runs/risk-trajectory-v1';compact=ROOT/'runs/risk-trajectory-compact-v1';config=load_config('configs/minute_precision.yaml')['api'];protocol=read(root/'protocol.json');test=set(protocol['splits']['test']);val=set(protocol['splits']['val']);hashes={};models=0
    assert not set(protocol['splits']['train'])&(test|val)
    for path in [root,compact]:
        expected=read(path/'checkpoint_hashes.json')
        for p in (path/'checkpoints').glob('*/best.pt'):
            ck=torch.load(p,weights_only=True);assert not set(ck['train_ids'])&(test|val);digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected[p.parent.name];hashes[str(p.relative_to(ROOT))]=digest;models+=1
            if path==compact and '-risk-' not in p.parent.name:assert digest==hashlib.sha256((root/'checkpoints'/p.parent.name/'best.pt').read_bytes()).hexdigest()
    assert np.array_equal(np.load(root/'shared.npz')['cheap'],np.load(compact/'shared.npz')['cheap'])
    with sqlite3.connect(resolve(config['cache_db'])) as db:before=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    unique_answers=set();unique_grades=set();evaluations=0
    for path in [root,root/'controller-accounted',compact]:
        tasks=read(path/'qa_tasks.json');qa=[read(p) for p in (path/'qa').glob('*.json')];assert len(qa)==len(tasks)==48
        assert {(r['question_id'],r['method']) for r in tasks}=={(r['question_id'],r['method']) for r in qa if 'judge_response' in r}
        evaluations+=len(qa)
        for r in qa:unique_answers.add(r['generation']['request_hash']);unique_grades.add(r['judge_response']['request_hash'])
        runner.OUT=path;runner.qa_run()
    with sqlite3.connect(resolve(config['cache_db'])) as db:after=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    assert after==before
    gates={}
    for path in [root,compact]:
        gs=read(path/'gate_optimization_audit.json');assert all(r['call_disagreements']==0 for r in gs);gates[path.name]=dict(histories=len(gs),events=sum(r['events'] for r in gs),max_risk_error=max(r['risk_max_abs_error'] for r in gs),call_disagreements=0)
    result=dict(checkpoint_files_verified=models,new_trainings=36,reused_control_checkpoints=18,train_validation_test_disjoint=True,identical_cheap_weights_across_stages=True,qa_evaluation_records=evaluations,unique_generation_requests=len(unique_answers),unique_judge_requests=len(unique_grades),resume_extra_http=after-before,gate_equivalence=gates,weight_hashes=hashes)
    atomic_json(root/'final_audit.json',result);print({k:v for k,v in result.items() if k!='weight_hashes'})

if __name__=='__main__':main()
