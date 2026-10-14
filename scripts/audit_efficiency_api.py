"""Metadata-only API accounting and no-repeat resume audit for both experiments."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sqlite3,json
from collections import Counter
from src.config import load_config,resolve,atomic_json
import scripts.writer_efficiency as runner

def main():
    paths=[ROOT/'runs/writer-efficiency-v1',ROOT/'runs/content-fallback-v1'];hashes=set()
    for p in paths:
        for q in (p/'qa').glob('*.json'):
            row=json.loads(q.read_text(encoding='utf-8'))
            for field in ['generation','judge_response']:
                if field in row:hashes.add(row[field]['request_hash'])
    config=load_config('configs/minute_precision.yaml')['api']
    with sqlite3.connect(resolve(config['cache_db'])) as db:
        before=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    for p in paths:runner.OUT=p;runner.qa_run()
    with sqlite3.connect(resolve(config['cache_db'])) as db:
        after=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
        rows=[dict(zip(['request_hash','started','status','elapsed','usage'],r)) for r in db.execute('SELECT request_hash,started,status,elapsed,usage_json FROM requests_log') if r[0] in hashes]
    assert after==before
    for r in rows:r['usage']=json.loads(r['usage'])
    times=sorted(r['started'] for r in rows);left=0;maximum=0
    for i,t in enumerate(times):
        while times[left]<=t-60:left+=1
        maximum=max(maximum,i-left+1)
    audit=dict(http_attempts=len(rows),statuses=dict(Counter(r['status'] for r in rows)),observed_max_attempts_per_rolling60s=maximum,
        reported_tokens=sum(r['usage'].get('total_tokens',0) for r in rows),resume_extra_http_requests=after-before,
        limit_configured_rpm=9,scope='Both efficiency and content fallback request hashes; no credentials or response bodies')
    atomic_json(paths[0]/'api_attempts.json',rows);atomic_json(paths[0]/'combined_api_audit.json',audit);print(audit)

if __name__=='__main__':main()
