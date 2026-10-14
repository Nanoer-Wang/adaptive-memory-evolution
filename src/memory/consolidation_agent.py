"""Agent adapter preserving complete attributed spans, without gold annotations."""
import hashlib,json
import numpy as np
from pathlib import Path
from src.config import atomic_json
from src.memory.consolidation import PROJECTION
from src.memory.consolidation import ConsolidationMemory, project
from src.memory.semantic_evidence import extract_spans
from src.data.longmemeval import date_days
from src.retrieval.retriever import lexical_scores


class ConsolidatingAgentMemory:
    def __init__(self,encoder,controller=None,budget_bytes=32000,rule='fifo',retrieval='hybrid'):
        if retrieval not in ('hybrid','projected'):raise ValueError('Unknown retrieval mode')
        self.retrieval=retrieval
        self.encoder=encoder;self.memory=ConsolidationMemory(controller,budget_bytes,rule)

    def observe(self,source_id,date,turns):
        now=date_days(date,'minute')
        spans=extract_spans(turns,str(source_id),now,max_chars=1200)
        texts=[s['role']+': '+s['quote'] for s in spans]
        if not texts:return []
        vectors=self.encoder.encode(texts);events=[]
        for span,text,vector in zip(spans,texts,vectors):
            identity=json.dumps([str(source_id),span['turn'],span['start'],date],ensure_ascii=False)
            events.append(self.memory.observe(identity,text,project(vector),now))
        return events

    def query(self,question,question_date,context_chars=12000,top_k=8):
        now=date_days(question_date,'minute');query=self.encoder.encode([question])[0];vector=project(query)
        items=self.memory.retrieve(now,vector=vector,limit=top_k if self.retrieval=='projected' else len(self.memory.items))
        if self.retrieval=='hybrid' and items:
            # Re-encode ONLY surviving texts. These are transient query workspaces,
            # not an archive path to evicted memory. Charge retrieval runtime separately.
            texts=[r.text for r in items];dense=np.maximum(self.encoder.encode(texts)@query,0.);lexical=lexical_scores(texts,question)
            score=.5*dense/max(float(dense.max()),1e-8)+.5*lexical/max(float(lexical.max()),1e-8)
            items=[items[i] for i in np.argsort(-score,kind='stable')[:top_k]]
        chunks=[];sources=[];used=0
        for item in sorted(items,key=lambda r:r.updated_at):
            sid,t=item.origins[-1];source,turn,start,date=json.loads(sid)
            opaque=hashlib.sha256(source.encode()).hexdigest()[:24]
            chunk=f'[session={opaque} date={date}]\n{item.text}'
            if used+len(chunk)+2>context_chars:continue
            chunks.append(chunk);used+=len(chunk)+2
            sources.append(dict(identity=item.identity,source_id=source,turn=int(turn),start=int(start),origins=item.origins))
        return dict(context='\n\n'.join(chunks),sources=sources,payload_bytes=self.memory.payload_bytes)

    def feedback(self,result,date,success):
        self.memory.reinforce([s['identity'] for s in result['sources']],date_days(date,'minute'),float(success))

    def checkpoint(self,path):
        self.memory.checkpoint(path);p=Path(path);v=json.loads(p.read_text(encoding='utf-8'))
        v['encoder_signature']=self.encoder.signature;v['projection_sha256']=hashlib.sha256(PROJECTION.tobytes()).hexdigest();v['retrieval']=self.retrieval;atomic_json(p,v)

    def restore(self,path):
        v=json.loads(Path(path).read_text(encoding='utf-8'))
        if v.get('encoder_signature')!=self.encoder.signature or v.get('projection_sha256')!=hashlib.sha256(PROJECTION.tobytes()).hexdigest() or v.get('retrieval','projected')!=self.retrieval:raise ValueError('Incompatible encoder, projection or retrieval')
        self.memory.restore(path)
