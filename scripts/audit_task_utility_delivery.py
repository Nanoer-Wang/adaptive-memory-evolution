"""Artifact, provenance and actual-request accounting for the task utility series."""
import sys,json,hashlib,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
OUT=ROOT/'runs/task-utility-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    tasks=read(OUT/'qa_tasks.json');rows=[read(p) for p in (OUT/'qa').glob('*.json')]
    assert len(tasks)==len(rows)==108 and all('judge_response' in r for r in rows)
    assert all(r['custom_judge_correct'] in (0,1) for r in rows)
    by={(r['question_id'],r['method']):r for r in rows}
    for t in tasks:
        r=by[t['question_id'],t['method']];assert r['context']==t['context']
        assert r['context'].startswith(by[t['question_id'],t['base_method']]['context'])
        if r['budget'] is not None:assert r['payload_bytes']<=r['budget']
    for path,digest in read(OUT/'input_hashes.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    reviewed=read(OUT/'reference_audit.json')['records'];assert len(reviewed)==108
    for r in reviewed:assert r['answer_sha256']==hashlib.sha256(by[r['question_id'],r['method']]['hypothesis'].encode()).hexdigest()
    dest=OUT/'length_sensitivity';selection=read(dest/'complete.json')
    extra=[read(p) for p in dest.glob('*.json') if p.name not in ('selection.json','complete.json','summary.json')]
    assert len(extra)==selection['tasks']==9*len(selection['question_ids'])
    expected={r['question_id'] for r in rows if r['generation']['finish_reason']=='length'}
    assert expected==set(selection['question_ids'])
    for r in extra:
        assert r['context']==by[r['question_id'],r['method']]['context']
        assert 'judge_response' in r
    reviewed_by={(r['question_id'],r['method']):r for r in reviewed}
    expanded_review={(r['question_id'],r['method']):r for r in read(OUT/'length_reference_audit.json')['records']}
    assert len(expanded_review)==len(extra)
    for r in extra:assert expanded_review[r['question_id'],r['method']]['answer_sha256']==hashlib.sha256(r['hypothesis'].encode()).hexdigest()
    length_changes=[dict(question_id=r['question_id'],method=r['method'],primary_correct=by[r['question_id'],r['method']]['custom_judge_correct'],expanded_correct=r['custom_judge_correct'],
        primary_review_correct=reviewed_by[r['question_id'],r['method']]['review_correct'],expanded_review_correct=expanded_review[r['question_id'],r['method']]['review_correct'],
        primary_finish_reason=by[r['question_id'],r['method']]['generation']['finish_reason'],expanded_finish_reason=r['generation']['finish_reason'],
        hypothesis=r['hypothesis']) for r in extra]
    atomic_json(dest/'summary.json',dict(question_ids=selection['question_ids'],scope='Post-diagnostic output-budget sensitivity with raw provider judge and assistant reference audit; no replacement of primary outcomes. New generations conflate budget effects and generation variation, including changes in originally untruncated answers.',rows=length_changes))
    keys={r[k]['request_hash'] for r in rows+extra for k in ['generation','judge_response']};attempts=[]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        for key in keys:
            attempts += [dict(request_hash=key,started=s,status=status,elapsed=elapsed,usage=json.loads(u)) for s,status,elapsed,u in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,))]
    attempts.sort(key=lambda a:a['started']);left=0;peak=0
    for right,a in enumerate(attempts):
        while attempts[left]['started']<=a['started']-60:left+=1
        peak=max(peak,right-left+1)
    assert peak<=9
    api=dict(evaluation_assignments=len(rows)+len(extra),unique_generations=len({r['generation']['request_hash'] for r in rows+extra}),unique_judgments=len({r['judge_response']['request_hash'] for r in rows+extra}),
        http_attempts=len(attempts),statuses=dict(Counter(a['status'] for a in attempts)),tokens_reported=sum(a['usage'].get('total_tokens',0) or 0 for a in attempts),rolling60_peak=peak)
    atomic_json(OUT/'combined_api_attempts.json',attempts);atomic_json(OUT/'combined_api.json',api)
    sig,archive=snapshot_source();atomic_json(OUT/'final_audit.json',dict(primary_tasks=108,reviewed_tasks=108,length_sensitivity_tasks=len(extra),questions=12,
        frozen_contexts=True,unchanged_model_hashes=True,within_memory_budgets=True,api=api,source=sig,source_archive=archive,
        limitations='Observed development cases; not independent human review or multi-step agent benchmark. Same prompts cached, not repeated trials. Archive interventions use extra resources.'))
    print(json.dumps(api,indent=2))

if __name__=='__main__':main()
