"""Query-time context-budget decision; no answers/labels in inference inputs."""
import json,re
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from src.memory.liquid_memory import LiquidMemory
from src.retrieval.preserved_context import pack_preserved_context

def compact_context(history,selected,budget=4200):
    """Extract query-relevant lines with adjacent context; preserve roles and dates."""
    selected=sorted(selected);units=[]
    for index in selected:
        for position,line in enumerate(history.texts[index].splitlines()):
            if line.strip():units.append((index,position,line))
    if not units:return '',[]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english')
    try:
        mat=vectorizer.fit_transform([u[2] for u in units]);scores=(mat@vectorizer.transform([history.question]).T).toarray().ravel()
    except ValueError:scores=np.zeros(len(units))
    ranking=sorted(range(len(units)),key=lambda i:(-float(scores[i]),i))
    chosen=set()
    def render(indices):
        from dataclasses import replace
        texts=list(history.texts)
        for index in selected:texts[index]='\n'.join(units[j][2] for j in sorted(indices) if units[j][0]==index)
        included=[i for i in selected if texts[i]]
        return pack_preserved_context(replace(history,texts=tuple(texts)),included,100000)
    for i in ranking:
        adjacent=[j for j in (i-1,i,i+1) if 0<=j<len(units) and units[j][0]==units[i][0]]
        # First preserve local context if it fits; fall back to the actual line.
        for proposal in [chosen|set(adjacent),chosen|{i}]:
            context,_=render(proposal)
            if len(context)<=budget:
                chosen=proposal;break
    context,trace=render(chosen)
    assert len(context)<=budget
    return context,trace

def parse_task_answer(text):
    decoder=json.JSONDecoder();found=[]
    for i,c in enumerate(text):
        if c!='{':continue
        try:value,_=decoder.raw_decode(text[i:])
        except ValueError:continue
        if isinstance(value,dict) and set(value)=={'answer','abstain'} and isinstance(value['answer'],str) and type(value['abstain']) is bool:found.append(value)
    return found[0] if len(found)==1 and found[0]['answer'].strip() else None

def controller_summary(x):
    return torch.cat([x.mean(0),x.max(0).values,x.min(0).values,x[-1]])

class TaskController(nn.Module):
    def __init__(self,kind,features=14,hidden=8):
        super().__init__();self.kind=kind;self.hidden=hidden
        if kind=='linear':self.head=nn.Linear(features*4,1)
        elif kind=='mlp':self.head=nn.Sequential(nn.Linear(features*4,8),nn.Tanh(),nn.Linear(8,1))
        else:
            self.encode=nn.Sequential(nn.Linear(features,hidden),nn.Tanh())
            self.cell=nn.GRUCell(hidden,hidden) if kind=='gru' else LiquidMemory(hidden)
            self.head=nn.Linear(hidden,1)
    def forward(self,x,dt,reset=False,zero_time=False):
        if self.kind in ('linear','mlp'):return self.head(controller_summary(x)).squeeze()
        h=x.new_zeros(self.hidden)
        for v,d in zip(self.encode(x),dt):
            if reset:h=torch.zeros_like(h)
            h=self.cell(v,h) if self.kind=='gru' else self.cell(h,v,delta=0. if zero_time else d)
        return self.head(h).squeeze()
