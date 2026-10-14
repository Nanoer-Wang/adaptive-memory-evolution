"""Require complete paired QA and verify rerun makes no HTTP requests."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sqlite3
from src.config import load_config,resolve,atomic_json
from scripts.hybrid_validation import OUT,read
import scripts.writer_efficiency as runner

def main():
    tasks=read(OUT/'qa_tasks.json');rows=[read(p) for p in (OUT/'qa').glob('*.json')]
    expected={(r['question_id'],r['method']) for r in tasks}
    completed={(r['question_id'],r['method']) for r in rows if 'judge_response' in r}
    assert expected==completed
    config=load_config('configs/minute_precision.yaml')['api']
    with sqlite3.connect(resolve(config['cache_db'])) as db:before=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    runner.OUT=OUT;runner.qa_run()
    with sqlite3.connect(resolve(config['cache_db'])) as db:after=db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    assert before==after
    atomic_json(OUT/'resume_audit.json',dict(expected=len(expected),complete=len(completed),extra_http=after-before))
    print('All paired QA complete; rerun added zero HTTP requests')

if __name__=='__main__':main()
