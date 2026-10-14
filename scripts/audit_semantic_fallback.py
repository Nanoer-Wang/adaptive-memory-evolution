"""Audit fixed observations, causal input scope, budgets, data splits and reuse."""
import sys,json,hashlib,zipfile,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config,resolve
from src.memory.semantic_reading import parse_units
from scripts import semantic_fallback as base
from scripts import semantic_fallback_augmentation as extra
P=base.OUT;E=extra.OUT
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=read(P/'protocol.json');ids={s:set(q) for s,q in protocol['question_ids'].items()}
    assert [len(ids[s]) for s in ['train','val','test']]==[20,10,12]
    assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    assert not ids['test']&set(protocol['prior_qa_excluded'])
    tasks=read(P/'tasks.json');by={(t['question_id'],t['method']):t for t in tasks};assert len(tasks)==126
    features=read(P/'features.json')
    for q in set.union(*ids.values()):
        full=by[q,'full'];a,b=by[q,'compact600'],by[q,'compact1800'];units=set(parse_units(full['context']))
        assert set(parse_units(a['context']))<=set(parse_units(b['context']))<=units
        assert len(a['context'])<=600 and len(b['context'])<=1800 and full['payload_bytes']<=64000
        for t in [full,a,b]:
            r=read(P/'qa'/f"{q}-{t['method']}.json");assert r['context']==t['context'] and r['correct'] in (0,1)
            assert r['evaluation_status'] in ['graded','explicit_abstention']
            if r['parsed']['abstain']:assert r['correct']==0
    with zipfile.ZipFile(ROOT/protocol['archive']) as z:
        for file in ['src/memory/semantic_reading.py','src/memory/task_controller.py','scripts/semantic_fallback.py','scripts/context_utility.py']:
            assert hashlib.sha256(z.read(file)).hexdigest()==sha(ROOT/file)
    for r in read(P/'reused_records.json'):assert sha(ROOT/r['record'])==r['sha256']==sha(P/'qa'/f"{r['question_id']}-full.json")
    interventions=read(E/'tasks.json');assert len(interventions)==20 and {r['question_id'] for r in interventions}==ids['train']
    for t in interventions:
        q=t['question_id'];original=by[q,'compact1800']['context'];h=t['trace']['withheld_source'];units=parse_units(original)
        assert h in {a for a,b in units}
        assert set(parse_units(t['context']))=={(a,b) for a,b in units if a!=h}
    checkpoints={};latest_parent=0;latest_extra=0
    for out,ds in [(P,base.data(False)),(E,extra.augmented_data(False))]:
        xx=np.concatenate([d['x'] for d in ds.values() if d['split']=='train']);norm=np.load(out/'normalization.npz')
        assert np.allclose(norm['mean'],xx.mean(0)) and np.allclose(norm['std'],np.maximum(xx.std(0),.05))
        for p in (out/'checkpoints').glob('*/best.pt'):
            c=torch.load(p,weights_only=True);assert set(c['train_ids'])==ids['train'] and set(c['val_ids'])==ids['val'];checkpoints[str(p.relative_to(ROOT))]=sha(p)
            if out==P:latest_parent=max(latest_parent,p.stat().st_mtime)
            else:latest_extra=max(latest_extra,p.stat().st_mtime)
    assert len(checkpoints)==24
    keys=[read(P/'qa'/f'{q}-{method}.json')['generation']['request_hash'] for q in ids['test'] for method in ['full','compact600','compact1800']]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        starts=[s for key in keys for (s,) in db.execute('SELECT started FROM requests_log WHERE request_hash=?',(key,))]
    assert latest_parent<min(starts) and read(E/'protocol.json')['registered_at']<min(starts)
    assert (P/'baseline_amendment.json').stat().st_mtime<min(starts)
    for out in [P,E]:assert out.joinpath('policy_records.json').stat().st_mtime>latest_extra
    audit=dict(parent_action_records=126,extra_training_interventions=20,new_test_questions=12,previously_used_test_ids_excluded=True,
        nested_attributed_compact_contexts=True,full_store_payload_within_64000=True,original_code_unchanged_since_preparation=True,
        no_split_overlap=True,normalization_train_only=True,training_interventions_cover_all20_train_questions=True,
        primary_weights_before_test_generation=True,augmentation_registered_before_test_generation=True,
        supplemental_weights_before_test_analysis=True,supplemental_weights_before_first_test_request=latest_extra<min(starts),
        training_data_access='Supplemental fitting calls augmented_data(False); base.data skips test BEFORE reading answer files. File timestamp order is corroboration, not tamper-proof blinding.',
        checkpoint_sha256=checkpoints)
    atomic_json(P/'audit.json',audit);print('Audit passed:126 base action records,20 train-only interventions,24 frozen checkpoints; supplemental timing disclosed.')

if __name__=='__main__':main()
