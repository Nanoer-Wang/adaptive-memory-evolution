"""One reproducible agent-memory replay. Dry run by default; --answer calls Agnes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib
from types import SimpleNamespace
import numpy as np
import torch
from src.memory.agent_memory import AgentMemory
from src.memory.content_store import ContentWriter
from src.memory.fast_writer import OnlineFrozenWriter
from src.models.content_encoder import ContentEncoder
from src.config import load_config,atomic_json
from src.evaluation.runner import messages_for
from src.api.agnes import AgnesClient
from filelock import FileLock

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--state',required=True);p.add_argument('--output',required=True);p.add_argument('--method',choices=['fifo','static','gru','liquid'],default='fifo');p.add_argument('--budget',type=int,default=128000);p.add_argument('--answer',action='store_true');p.add_argument('--writer-backend',choices=['original','fused','scripted'],default='original');a=p.parse_args()
    if a.method=='fifo' and a.writer_backend!='original':p.error('FIFO has no learned writer to optimize')
    source=Path(a.input);doc=json.loads(source.read_text(encoding='utf-8'));digest=hashlib.sha256(source.read_bytes()).hexdigest();writer=None;signature='fifo'
    events_digest=hashlib.sha256(json.dumps(doc['events'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if a.method!='fifo':
        kind=a.method+'_content';checkpoint=ROOT/'runs/preservation-study-v2/checkpoints'/f'{kind}-23/best.pt';signature=hashlib.sha256(checkpoint.read_bytes()).hexdigest();writer=ContentWriter(kind);writer.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);writer.eval()
        if a.writer_backend!='original':
            writer=OnlineFrozenWriter(writer,scripted=a.writer_backend=='scripted');signature+=':'+a.writer_backend+'-v1'
    encoder=ContentEncoder();memory=AgentMemory(encoder,a.budget,writer,signature);state=Path(a.state);state.parent.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(state)+'.lock',timeout=0):
            cursor=0
            if state.exists():
                metadata=json.loads(state.read_text(encoding='utf-8')).get('runner',{})
                prior=metadata.get('events_sha256')
                if (prior is not None and prior!=events_digest) or (prior is None and metadata.get('input_sha256')!=digest):raise ValueError('Events changed; choose a new state file')
                memory.restore(state);cursor=metadata['cursor']
            for index,event in enumerate(doc['events'][cursor:],cursor):
                memory.observe(event['source_id'],event['date'],event['turns'])
                memory.checkpoint(state,dict(input_sha256=digest,events_sha256=events_digest,cursor=index+1))
            # Upgrade old checkpoints and allow new questions over identical history.
            memory.checkpoint(state,dict(input_sha256=digest,events_sha256=events_digest,cursor=len(doc['events'])))
            result=memory.query(doc['question'],doc['question_date']);result.update(method=a.method,writer_backend=a.writer_backend,events=len(doc['events']),evictions=memory.evictions,input_sha256=digest,mode='answer' if a.answer else 'dry-run')
            if a.answer:
                config=dict(load_config('configs/minute_precision.yaml')['api'],rpm=9);client=AgnesClient(config)
                try:result['generation']=client.complete(messages_for(SimpleNamespace(question=doc['question'],question_date=doc['question_date']),result['context']),namespace='agent-memory-run-v1:'+digest)
                finally:client.close()
            atomic_json(Path(a.output),result)
    finally:encoder.close()
    print('Saved agent memory output; mode='+result['mode'])
if __name__=='__main__':main()
