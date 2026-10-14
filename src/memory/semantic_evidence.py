"""Lossless extractive evidence with semantic association, not asserted facts.

Semantic similarity triggers retention updates, NEVER fact supersession.
All vector storage is charged; writers cannot access questions or gold labels.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from src.memory.version_ledger import Evidence, VersionLedger

PROJECTION=np.random.default_rng(20261011).normal(size=(384,32)).astype(np.float32)/np.sqrt(32)
SEMANTIC_FEATURES=50


@dataclass(frozen=True)
class SemanticEvidence(Evidence):
    vector: tuple = ()
    projected: tuple = ()
    role: str = 'user'

    def __post_init__(self):
        super().__post_init__()
        if len(self.vector)!=384 or len(self.projected)!=32:
            raise ValueError('Invalid semantic representation')
        if not np.isfinite(self.vector).all() or not np.isfinite(self.projected).all():
            raise ValueError('Nonfinite semantic representation')
        if self.role not in ('user','assistant'):raise ValueError('Unsupported role')

    @property
    def payload_bytes(self):
        return super().payload_bytes+(384+32)*4+len(self.role.encode())


def extract_spans(turns, source_id, observed_at, max_chars=1000):
    """Return exhaustive, disjoint original spans. No question/label input."""
    if max_chars<1:raise ValueError('Invalid chunk limit')
    spans=[]
    for index,turn in enumerate(turns):
        role=turn.get('role')
        if role not in ('user','assistant'):continue
        text=turn.get('content','')
        if not isinstance(text,str):raise ValueError('Nontext turn')
        for start in range(0,len(text),max_chars):
            end=min(start+max_chars,len(text))
            spans.append(dict(source_id=source_id,turn=index,start=start,end=end,
                              quote=text[start:end],role=role,observed_at=observed_at))
    return spans


def materialize(span,vector):
    vector=np.asarray(vector,dtype=np.float32)
    projected=vector@PROJECTION
    # Every passage is an observation with a unique scope, NOT a mutable fact.
    scope=hashlib.sha256(f"{span['source_id']}:{span['turn']}:{span['start']}".encode()).hexdigest()[:24]
    return SemanticEvidence(owner=span['role'],predicate='quoted_passage',scope=scope,
        value='original passage',observed_at=span['observed_at'],source_id=span['source_id'],
        turn=span['turn'],start=span['start'],end=span['end'],quote=span['quote'],mode='past',
        vector=tuple(vector.tolist()),projected=tuple(projected.tolist()),role=span['role'])


class SemanticLedger(VersionLedger):
    def features(self,version,event=None,now=None):
        base=super().features(version,event,now)
        # Do not encode the unique source-derived scope in controller features.
        base[12:16]=0
        e=version.evidence
        similarity=float(np.dot(e.vector,event.vector)) if event is not None else 0.
        return torch.cat((base,torch.tensor(e.projected,dtype=torch.float32),
                          torch.tensor([similarity,float(e.role=='assistant')])))

    def affected(self,evidence):
        versions=list(self.items.values())
        if not versions:return []
        similarities=np.asarray([v.evidence.vector for v in versions],dtype=np.float32)@np.asarray(evidence.vector,dtype=np.float32)
        # Fixed protocol, no threshold fitting on validation; stable tie order.
        indices=np.argsort(-similarities,kind='stable')[:4]
        selected=[versions[i] for i in indices if similarities[i]>=.55]
        # For quoted passages this counter means related-event exposure,
        # not corroboration or truth confidence. No value is superseded.
        for version in selected:version.support+=1
        return selected

    def write(self,evidence):
        if not isinstance(evidence,SemanticEvidence):raise ValueError('Semantic evidence required')
        return super().write(evidence)

    def retrieve(self,query_vector,limit=8):
        items=list(self.items.values())
        if not items:return []
        similarity=np.asarray([v.evidence.vector for v in items])@query_vector
        order=np.argsort(-similarity,kind='stable')[:limit]
        return [items[i].evidence for i in order]
