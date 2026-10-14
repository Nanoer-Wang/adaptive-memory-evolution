"""Causal natural-language interface to the frozen B1 retention core.

No inferred version equivalence, no gold-answer feedback, no parameter updates.
The coarse topic is a fixed projection sign, not a trained semantic taxonomy.
"""
import copy,hashlib,time
import numpy as np
from src.memory.validation_matrix import MatrixPolicy
from src.memory.consolidation import project
from src.retrieval.retriever import lexical_scores

class LanguageMemory:
    def __init__(self,method,seed,head,capacity=64,admission=.75,minimum_age=0.):
        needs=method not in ('fixed_decay','frequency','lru','replacement_matched','adaptive_rule')
        self.policy=MatrixPolicy(method,capacity,seed,head if needs else None,admission,minimum_age)
        self.vectors={};self.origins={};self.trace=[];self.seconds=0.;self.retrieval_seconds=0.;self.retrieval_calls=0
    def state(self):
        return [dict(item_id=i.item_id,key=i.key,version=i.version,birth=i.birth,observed_at=i.observed_at,
                     I=i.I,C=i.C,U=i.U,T=i.T,count=i.count) for i in self.policy.items.values()]
    def event(self,kind,now,vector,**fields):
        v=project(vector).tolist()
        return dict(kind=kind,time=float(now),vector=v,topic=int(v[0]>=0),**fields)
    def record(self,event,action=None,answer=None):
        self.trace.append(dict(event=event,resident=self.state(),action=action,answer=answer,counts=dict(self.policy.counts)))
        assert len(self.policy.items)<=self.policy.capacity
        assert set(self.policy.items)==set(self.vectors)==set(self.origins)
    def rank(self,text,vector,limit):
        start=time.perf_counter();ids=list(self.policy.items)
        if not ids:return []
        texts=[self.policy.items[i].content for i in ids]
        dense=np.maximum(np.stack([self.vectors[i] for i in ids])@vector,0.)
        lexical=lexical_scores(texts,text)
        score=.5*dense/max(float(dense.max()),1e-8)+.5*lexical/max(float(lexical.max()),1e-8)
        result=[ids[j] for j in np.argsort(-score,kind='stable')[:limit]]
        self.retrieval_seconds+=time.perf_counter()-start;self.retrieval_calls+=1
        return result
    def observe(self,record,vector):
        p=self.policy;now=record['time'];text=record['text'];start=time.perf_counter()
        # Historical user questions are actual observed demand, never heldout QA.
        if record['role']=='user':
            e=self.event('context',now,vector);p.update_state(e);self.record(e)
            if '?' in text:
                candidates=self.rank(text,vector,1);key=p.items[candidates[0]].key if candidates else -1
                e=self.event('query',now,vector,key=key);p.update_state(e);a=p.retrieve(e);self.record(e,answer=a)
        e=self.event('observe',now,vector,key=record['key'],item_id=record['item_id'],version=1,content=text)
        p.update_state(e);p.observe(e);self.vectors[e['item_id']]=vector;self.origins[e['item_id']]=record['origin']
        action=p.evict()
        if action['victim'] is not None:
            self.vectors.pop(action['victim']);self.origins.pop(action['victim'])
        self.record(e,action=action)
        if record.get('feedback') is not None:
            e=self.event('feedback',now,vector,key=record['key'],value=record['feedback'],provenance='MemoryBench pre-recorded implicit feedback')
            p.update_state(e);p.feedback(record['item_id'] if record['item_id'] in p.items else None,record['feedback']);self.record(e)
        self.seconds+=time.perf_counter()-start
    def retrieve(self,question,vector,now,top_k=5,context_chars=12000):
        p=self.policy;start=time.perf_counter();e=self.event('query',max(float(now),p.now),vector,key=-1)
        p.update_state(e);ids=self.rank(question,vector,top_k);sources=[];chunks=[];used=0
        for item_id in ids:
            item=p.items[item_id];origin=self.origins[item_id]
            chunk=f"[source={origin['source']} date={origin['date']}]\n{item.content}"
            if used+len(chunk)+2>context_chars:continue
            a=p.retrieve(dict(e,key=item.key));assert a['item_id']==item_id
            chunks.append(chunk);used+=len(chunk)+2;sources.append(dict(origin,item_id=item_id))
        self.record(e,answer=dict(selected=sources));self.seconds+=time.perf_counter()-start
        return dict(context='\n\n'.join(chunks),sources=sources,resident_count=len(p.items),
                    resident_payload_bytes=sum(len(i.content.encode())+self.vectors[k].nbytes for k,i in p.items.items()),
                    policy_seconds=self.seconds,retrieval_seconds=self.retrieval_seconds,retrieval_calls=self.retrieval_calls,
                    counts=copy.deepcopy(p.counts),ledger_entries=len(p.ledger))
