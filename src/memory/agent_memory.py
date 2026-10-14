"""Agent-facing dated evidence store. No gold labels or semantic overwrite API."""
import hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from src.memory.budget_store import BudgetStore,BudgetItem
from src.memory.turn_compression import compress_turns
from src.data.longmemeval import History,date_days
from src.retrieval.preserved_context import pack_preserved_context
from src.retrieval.retriever import lexical_scores,rank

class AgentMemory:
    def __init__(self,encoder,budget_bytes=128000,writer=None,writer_signature='fifo'):
        self.encoder=encoder;self.budget_bytes=budget_bytes;self.writer_signature=writer_signature
        self.store=BudgetStore(None if budget_bytes is None else budget_bytes-16,writer)
        self.last_time=None;self.evictions=0

    @property
    def payload_bytes(self):return self.store.payload_bytes+16

    def observe(self,source_id,date,turns):
        now=date_days(date,'minute')
        if self.last_time is not None and now<self.last_time:raise ValueError('Out-of-order evidence')
        # Allowlist compressor ignores has_answer and all other annotation keys.
        text=compress_turns(turns,2400,'sentence_user_first')
        identity='s_'+hashlib.sha256((str(source_id)+'\0'+date).encode()).hexdigest()[:24]
        for item in self.store.items:
            if item.session_id==identity:
                if item.text!=text:raise ValueError('Source/date reused with different payload')
                return False
        delta=0. if self.last_time is None else min((now-self.last_time)/30,12.)
        vector=self.encoder.encode([text])[0]
        self.store.insert(identity,date,text,vector,delta);self.last_time=now
        self.evictions+=len(self.store.evicted);self.store.evicted.clear()
        return True

    def query(self,question,question_date,context_chars=16000):
        items=self.store.items
        if not items:return dict(context='',sources=[],payload_bytes=self.payload_bytes)
        texts=[i.text for i in items];q=self.encoder.encode([question])[0]
        lexical=lexical_scores(texts,question);dense=np.maximum(0,np.stack([i.vector for i in items])@q)
        scores=.5*lexical/max(float(lexical.max()),1e-8)+.5*dense/max(float(dense.max()),1e-8)
        selected=rank(scores,5);h=History('runtime',question,question_date,tuple(i.session_id for i in items),tuple(i.date for i in items),tuple(texts))
        context,trace=pack_preserved_context(h,selected,context_chars)
        return dict(context=context,sources=trace,payload_bytes=self.payload_bytes)

    def checkpoint(self,path,runner=None):
        value=dict(version=1,budget_bytes=self.budget_bytes,writer_signature=self.writer_signature,encoder_signature=self.encoder.signature,last_time=self.last_time,evictions=self.evictions,
            sequence=self.store.sequence,state=None if self.store.state is None else self.store.state.tolist(),
            items=[dict(**{k:v for k,v in asdict(i).items() if k!='vector'},vector=i.vector.tolist()) for i in self.store.items])
        if runner is not None:value['runner']=runner
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temp.replace(path)

    def restore(self,path):
        v=json.loads(Path(path).read_text(encoding='utf-8'))
        if v['version']!=1 or v['budget_bytes']!=self.budget_bytes or v['writer_signature']!=self.writer_signature or v['encoder_signature']!=self.encoder.signature:raise ValueError('Incompatible memory checkpoint')
        items=[BudgetItem(**dict(i,vector=np.asarray(i['vector'],dtype=np.float32))) for i in v['items']]
        state=None if v['state'] is None else torch.tensor(v['state'],dtype=torch.float32)
        if (state is None)!=(self.store.state is None) or (state is not None and state.shape!=self.store.state.shape):raise ValueError('State mismatch')
        size=sum(i.nbytes for i in items)+(0 if state is None else state.numel()*4)+16
        if self.budget_bytes is not None and size>self.budget_bytes:raise ValueError('Snapshot exceeds budget')
        self.store.items=items;self.store.state=state;self.store.sequence=v['sequence'];self.last_time=v['last_time'];self.evictions=v['evictions']
