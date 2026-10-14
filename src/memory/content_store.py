"""Causal content writes; queries can only read surviving payloads."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from src.memory.liquid_memory import LiquidMemory


def bounded_text(text, limit=2400):
    if len(text) <= limit:
        return text
    width=(limit-14)//3
    starts=[0,(len(text)-width)//2,len(text)-width]
    return '\n[...]\n'.join(text[s:s+width] for s in starts)[:limit]


class ContentWriter(nn.Module):
    def __init__(self,kind='liquid_content'):
        super().__init__()
        if kind not in ('liquid_content','static_content','gru_content'):
            raise ValueError('Unknown writer')
        self.kind=kind
        self.encoder=nn.Sequential(nn.Linear(384,16),nn.Tanh())
        self.hidden=13 if kind=='gru_content' else 16
        self.cell=nn.GRUCell(16,13) if kind=='gru_content' else LiquidMemory(16)
        self.head=nn.Linear(self.hidden,1)

    def step(self,vector,state,delta):
        encoded=self.encoder(vector)
        if self.kind=='static_content':
            state=torch.zeros_like(state)
        state=self.cell(encoded,state) if self.kind=='gru_content' else self.cell(state,encoded,delta=delta)
        return self.head(state).squeeze(-1),state

    def forward(self,vectors,deltas):
        state=vectors.new_zeros(self.hidden)
        outputs=[]
        for vector,delta in zip(vectors,deltas):
            score,state=self.step(vector,state,delta)
            outputs.append(score)
        return torch.stack(outputs)


@dataclass
class Item:
    session_id:str
    date:str
    text:str
    vector:np.ndarray
    priority:float
    sequence:int


class ContentStore:
    def __init__(self,capacity=None,writer=None):
        if capacity is not None and capacity<1:
            raise ValueError('capacity must be positive')
        self.capacity,self.writer=capacity,writer
        self.items=[]
        self.sequence=0
        self.state=torch.zeros(writer.hidden) if writer else None

    def insert(self,session_id,date,text,vector,delta):
        # There is intentionally no question, label, category or evidence argument.
        priority=float(self.sequence)
        if self.writer is not None:
            with torch.no_grad():
                score,self.state=self.writer.step(torch.as_tensor(vector),self.state,delta)
                priority=float(score)
        self.items.append(Item(session_id,date,bounded_text(text),np.asarray(vector,dtype=np.float32).copy(),priority,self.sequence))
        self.sequence+=1
        if self.capacity is not None and len(self.items)>self.capacity:
            worst=min(range(len(self.items)),key=lambda i:(self.items[i].priority,self.items[i].sequence))
            self.items.pop(worst)

    @property
    def payload_bytes(self):
        return sum(len(i.text.encode('utf-8'))+len(i.session_id.encode('utf-8'))+len(i.date.encode('utf-8'))+i.vector.nbytes+16 for i in self.items)+(self.state.numel()*4 if self.state is not None else 0)
