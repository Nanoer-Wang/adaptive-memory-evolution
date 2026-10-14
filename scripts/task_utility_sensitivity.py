"""Registered diagnostic: re-answer every condition of any output-truncated question."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from types import SimpleNamespace
from src.config import atomic_json,load_config
from src.api.agnes import AgnesClient
from src.evaluation.runner import messages_for,custom_judge

OUT=ROOT/'runs/task-utility-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def register():
    p=OUT/'length_sensitivity_protocol.json'
    if p.exists():return
    atomic_json(p,dict(timing='After observing first truncated answer while primary experiment is running',
        selector='All question IDs with ANY primary generation finish_reason=length; no correctness filtering',
        conditions='All nine fixed contexts per selected question, unchanged prompts/models/temperature; max_tokens1536 instead of original512. Same-provider judge512.',
        scope='Post-diagnostic output-budget sensitivity; retain original primary results; never replace primary answers with better ones.',
        max_tokens=1536))
def run():
    protocol=read(OUT/'length_sensitivity_protocol.json')
    assert (OUT/'qa_complete.json').exists()
    rows=[read(p) for p in (OUT/'qa').glob('*.json')];ids=sorted({r['question_id'] for r in rows if r['generation'].get('finish_reason')=='length'})
    tasks=[t for t in read(OUT/'qa_tasks.json') if t['question_id'] in ids]
    dest=OUT/'length_sensitivity';dest.mkdir(exist_ok=True)
    atomic_json(dest/'selection.json',dict(question_ids=ids,tasks=len(tasks),selection='finish_reason only'))
    namespace='task-utility-length-v1:'+hashlib.sha256((OUT/'length_sensitivity_protocol.json').read_bytes()).hexdigest()
    config=dict(load_config('configs/minute_precision.yaml')['api'],rpm=9)
    for i,t in enumerate(tasks):
        p=dest/f"{t['question_id']}-{t['method']}.json";r=read(p) if p.exists() else dict(t)
        if 'judge_response' in r:continue
        client=AgnesClient(config)
        try:
            if 'generation' not in r:
                r['generation']=client.complete(messages_for(SimpleNamespace(question=t['question'],question_date=t['question_date']),t['context']),namespace=namespace,max_tokens=protocol['max_tokens'])
                r['hypothesis']=r['generation']['content'];atomic_json(p,r)
            grade,response=custom_judge(client,t,r['hypothesis'],namespace,max_tokens=512)
            r.update(custom_judge_correct=grade,judge_response=response);atomic_json(p,r)
        except Exception as exc:
            r['error_type']=type(exc).__name__;atomic_json(p,r);raise RuntimeError('Sensitivity paused; safe error type:'+type(exc).__name__) from None
        finally:client.close()
        print('Length sensitivity',i+1,'/',len(tasks),flush=True)
    atomic_json(dest/'complete.json',dict(tasks=len(tasks),question_ids=ids,max_tokens=protocol['max_tokens']))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--register-only',action='store_true');a=p.parse_args();register()
    if not a.register_only:run()
