"""Repair online evaluation with opaque IDs and a resumable shared-limit worker pool."""
import argparse
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from types import SimpleNamespace
import numpy as np  # Before torch via runner/retriever.
from filelock import FileLock
from src.config import resolve, load_config, atomic_json
from src.data.download import file_sha256
from src.provenance import snapshot_source, runtime_info, utc_now
from src.retrieval.retriever import opaque_session_id
from src.evaluation.runner import messages_for, custom_judge
from src.evaluation.metrics import answer_metrics
from src.api.agnes import AgnesClient

PARENT = resolve('runs/content-memory-v1')
ROOT = resolve('runs/content-memory-qa-v2')


def anonymous_context(row):
    seen = []
    def replace(match):
        sid = match[1]
        if sid not in row['selected_ids']:
            raise ValueError('Context header not found among saved retrieval IDs')
        seen.append(sid)
        return '[session=' + opaque_session_id(sid) + ' date='
    context = re.sub(r'^\[session=(\S+) date=', replace, row['context'], flags=re.MULTILINE)
    if seen != row['selected_ids'] or len(context) > 16000:
        raise ValueError('Unexpected context structure or budget')
    return context


def database():
    db = sqlite3.connect(ROOT / 'results.sqlite3', timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS qa (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    return db


def export():
    with database() as db:
        rows = [json.loads(s) for (s,) in db.execute('SELECT payload FROM qa ORDER BY key')]
    with (ROOT / 'qa.jsonl.tmp').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    (ROOT / 'qa.jsonl.tmp').replace(ROOT / 'qa.jsonl')
    metrics = []
    for method in sorted({r['method'] for r in rows}):
        for dataset in ('all', 'locomo', 'longmemeval'):
            for answerability in ('all', 'answerable', 'unanswerable'):
                selected = [r for r in rows if r['method']==method and (dataset=='all' or r['dataset']==dataset)
                            and (answerability=='all' or r['question_id'].endswith('_abs')==(answerability=='unanswerable'))]
                if not selected:
                    continue
                grades = [r['custom_judge_correct'] for r in selected if r.get('custom_judge_correct') is not None]
                completed = [r for r in selected if 'judge_response' in r]
                metrics.append(dict(method=method,dataset=dataset,answerability=answerability,generated=len(selected),completed=len(completed),
                                    graded=len(grades),parse_failures=len(completed)-len(grades),
                                    accuracy=float(np.mean(grades)) if grades else None,
                                    accuracy_missing_as_wrong=sum(grades)/len(completed) if completed else None,
                                    accuracy_missing_as_correct=(sum(grades)+len(completed)-len(grades))/len(completed) if completed else None))
    atomic_json(ROOT / 'summary.json',dict(expected=150,completed=sum('judge_response' in r for r in rows),metrics=metrics))


def evaluate_one(row, config, namespace):
    with database() as db:
        prior = db.execute('SELECT payload FROM qa WHERE key=?',(row['key'],)).fetchone()
    result = json.loads(prior[0]) if prior else dict(row,context=anonymous_context(row),qa_protocol='anonymous-ids-v2')
    if 'judge_response' in result:
        return row['key'], True
    client = AgnesClient(config)
    start = time.perf_counter()
    def save():
        with database() as db:
            db.execute('INSERT OR REPLACE INTO qa VALUES (?,?)',(row['key'],json.dumps(result,ensure_ascii=False)))
    try:
        if 'generation' not in result:
            messages = messages_for(SimpleNamespace(question_date=row['question_date'],question=row['question']),result['context'])
            result['messages'] = messages
            result['generation'] = client.complete(messages,namespace=namespace)
            result['hypothesis'] = result['generation']['content']
            result.update(answer_metrics(result['hypothesis'],row['answer']))
            save()
        grade, response = custom_judge(client,row,result['hypothesis'],namespace,max_tokens=512)
        result.update(custom_judge_correct=grade,judge_response=response,qa_seconds=time.perf_counter()-start)
        result.pop('error_type',None);save()
        return row['key'], False
    except Exception as exc:
        result['error_type'] = type(exc).__name__;save();raise
    finally:
        client.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if not 1<=args.workers<=3:
        raise ValueError('Use 1 to 3 workers; all share a single nine-per-minute limiter')
    ROOT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'runner.lock'),timeout=0):
        parent=sqlite3.connect(f'file:{(PARENT/"results.sqlite3").as_posix()}?mode=ro',uri=True)
        rows=[json.loads(s) for (s,) in parent.execute('SELECT payload FROM retrieval ORDER BY key')]
        parent.close();rows=[r for r in rows if 'context' in r]
        if len(rows)!=150 or len({r['question_id'] for r in rows})!=30:
            raise ValueError('Incomplete original fixed QA sample')
        payload_hash=hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        config=load_config('configs/minute_precision.yaml')['api'].copy();config['rpm']=9
        signature,archive=snapshot_source()
        specification=dict(source_signature=signature,source_archive=archive,parent_execution_sha256=file_sha256(PARENT/'execution_manifest.json'),
                           selected_payloads_sha256=payload_hash,questions=[r['question_id'] for r in rows],api=config,
                           changes=['Opaque hashed session headers; identical selected sessions and excerpt bodies.',
                                    'Judge max output 512 instead of 128; accept one complete JSON boolean verdict, never truncated/no verdict.',
                                    'Three I/O workers share persistent rate limiter, 9 requests/minute with margin below user maximum 10.',
                                    'All old online results are diagnostic only. Offline results/checkpoints remain unchanged.'])
        manifest_path=ROOT/'execution_manifest.json'
        if manifest_path.exists():
            saved=json.loads(manifest_path.read_text(encoding='utf-8'))
            if any(saved[k]!=v for k,v in specification.items()):
                raise RuntimeError('Frozen repaired evaluation changed')
        else:
            atomic_json(manifest_path,dict(**specification,frozen_at=utc_now(),environment=runtime_info()))
        namespace='content-memory-qa-v2:'+file_sha256(manifest_path)
        completed=0;errors=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(evaluate_one,row,config,namespace):row['key'] for row in rows}
            for future in as_completed(futures):
                try:
                    key,cached=future.result();completed+=1
                    print(f'{utc_now()} anonymous QA {completed}/150 {key}'+(' resumed' if cached else ''),flush=True)
                    export()
                except Exception as exc:
                    errors.append(dict(key=futures[future],error_type=type(exc).__name__))
                    print(f'{utc_now()} saved failure: {type(exc).__name__}',flush=True)
        export();atomic_json(ROOT/'execution_status.json',dict(completed=completed,errors=errors,finished_at=utc_now()))
        if errors:
            raise RuntimeError('Some requests failed; saved records can be resumed with the same command')


if __name__=='__main__':
    main()
