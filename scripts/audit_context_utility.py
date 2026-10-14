"""Check data separation, real action costs and train-before-test execution order."""
import sys,json,hashlib,sqlite3,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config,resolve
from scripts.context_utility import OUT,read,SEEDS,KINDS
from src.memory.task_controller import parse_task_answer

def main():
    protocol=read(OUT/'protocol.json');features=read(OUT/'features.json');tasks=read(OUT/'tasks.json');rows=[read(p) for p in (OUT/'qa').glob('*.json')]
    assert len(tasks)==len(rows)==96
    sets={s:set(ids) for s,ids in protocol['question_ids'].items()}
    assert len(sets['train'])==24 and len(sets['val'])==len(sets['test'])==12
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    assert not sets['test']&set(protocol['prior_qa_excluded_from_test'])
    by={(r['question_id'],r['method']):r for r in rows}
    for t in tasks:
        r=by[t['question_id'],t['method']];assert r['context']==t['context']
        assert r['correct'] in (0,1) and r['evaluation_status'] in ('graded','explicit_abstention')
        assert parse_task_answer(r['generation']['content'])==r['parsed']
        if r['parsed']['abstain']:assert r['correct']==0 and 'judge_response' not in r
        assert t['payload_bytes']<=64000
        if t['method']=='compact':assert len(t['context'])<=4200
    norm=np.load(OUT/'normalization.npz');x=np.concatenate([d['x'] for q,d in features.items() if q in sets['train']])
    assert np.allclose(norm['mean'],x.mean(0)) and np.allclose(norm['std'],np.maximum(x.std(0),.05))
    # Executed writer/controller code must still match the source archived at preparation.
    with zipfile.ZipFile(ROOT/protocol['archive']) as archive:
        for path in ['src/memory/task_controller.py','scripts/context_utility.py']:
            assert hashlib.sha256(archive.read(path)).hexdigest()==hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    hashes={};latest=0
    for kind in KINDS:
        for seed in SEEDS:
            p=OUT/'checkpoints'/f'{kind}-{seed}'/'best.pt';ck=torch.load(p,weights_only=True)
            assert set(ck['train_ids'])==sets['train'] and set(ck['val_ids'])==sets['val']
            hashes[f'{kind}-{seed}']=hashlib.sha256(p.read_bytes()).hexdigest();latest=max(latest,p.stat().st_mtime)
    hashes['normalization']=hashlib.sha256((OUT/'normalization.npz').read_bytes()).hexdigest()
    keys=[r['generation']['request_hash'] for r in rows if r['split']=='test'];starts=[]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        for key in keys:starts.extend(row[0] for row in db.execute('SELECT started FROM requests_log WHERE request_hash=?',(key,)))
    assert starts and latest<=min(starts)
    assert (OUT/'training_complete.json').stat().st_mtime<=min(starts)
    assert (OUT/'cascade_amendment.json').stat().st_mtime<=min(starts)
    atomic_json(OUT/'audit.json',dict(tasks=96,train_questions=24,val_questions=12,test_questions=12,checkpoints=12,checkpoint_hashes=hashes,
        no_split_overlap=True,test_qa_ids_previously_unanswered=True,normalization_train_only=True,controller_source_matches_frozen_archive=True,
        checkpoints_saved_before_first_test_request=True,last_checkpoint_timestamp=latest,first_test_request=min(starts),all_contexts_preserved=True,all_contracts_valid=True,
        limitations='Filesystem timestamp order corroborates execution order, not tamper-proof registration. Public offline split previously exposed; this is not a new independent dataset.'))
    print('Audit passed:96answers,12checkpoints,train-before-test,separate IDs,valid contracts')

if __name__=='__main__':main()
