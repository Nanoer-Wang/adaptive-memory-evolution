"""Persistent multi-event agent memory runner; API calls require --answer."""
import sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.models.content_encoder import ContentEncoder
from src.memory.consolidation_agent import ConsolidatingAgentMemory
from src.memory.consolidation import KINDS
from scripts.consolidation_study import load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--state',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=list(KINDS)+['fifo','decay'],default='liquid');p.add_argument('--seed',type=int,default=23);p.add_argument('--budget',type=int,default=32000);p.add_argument('--answer',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);raw=Path(a.input).read_bytes();doc=json.loads(raw);fingerprint=hashlib.sha256(raw+str(a.answer).encode()).hexdigest()
    encoder=ContentEncoder();model=None if a.method in ['fifo','decay'] else load_model(a.method,a.seed);agent=ConsolidatingAgentMemory(encoder,model,a.budget,a.method)
    state=Path(a.state);results=[];start=0
    if state.exists():
        saved=json.loads(state.read_text(encoding='utf-8'))
        if saved.get('runner',{}).get('input_sha256')!=fingerprint:raise ValueError('Input or answer mode changed; use a fresh state file')
        agent.restore(state);start=saved['runner']['next_event'];results=saved['runner']['results']
    try:
        for index,event in enumerate(doc['events'][start:],start):
            if event['type']=='observe':agent.observe(event['source_id'],event['date'],event['turns'])
            elif event['type']=='query':
                result=agent.query(event['question'],event['date']);result.update(event_id=event['id'],question=event['question'],date=event['date'])
                if a.answer:
                    from src.api.agnes import AgnesClient
                    cfg=dict(load_config('configs/minute_precision.yaml')['api'],rpm=9);client=AgnesClient(cfg)
                    try:result['generation']=client.complete([dict(role='system',content='Answer from supplied dated evidence only. If facts are absent, abstain. Return JSON with answer and abstain.'),dict(role='user',content=json.dumps(dict(question=event['question'],date=event['date'],context=result['context']),ensure_ascii=False))],namespace='consolidation-agent:'+fingerprint+':'+str(index)+':'+agent.memory.signature(),max_tokens=1024)
                    finally:client.close()
                results.append(result)
            elif event['type']=='feedback':
                prior=next(r for r in results if r['event_id']==event['query_id']);agent.feedback(prior,event['date'],event['success'])
            else:raise ValueError('Unknown event type')
            agent.checkpoint(state);saved=json.loads(state.read_text(encoding='utf-8'));saved['runner']=dict(input_sha256=fingerprint,next_event=index+1,results=results);atomic_json(state,saved)
        atomic_json(Path(a.output),dict(method=a.method,answer_calls_enabled=a.answer,completed_events=len(doc['events']),results=results,payload_bytes=agent.memory.payload_bytes,stats=agent.memory.stats))
        print('Completed persistent memory run; output saved.')
    finally:encoder.close()


if __name__=='__main__':main()
