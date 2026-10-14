import copy,math
import numpy as np
from src.memory.characterization import StrengthPolicy,environment
from src.memory.validation_matrix import MatrixPolicy
from scripts.ame_validation_batches import read
from src.config import ROOT

def head():return {'weight':np.zeros((3,40),dtype=np.float32),'bias':np.zeros(3,dtype=np.float32)}
def event(kind,time,**kw):return dict(kind=kind,time=time,key=1,topic=0,vector=[0.]*32,**kw)
def item():return event('observe',0,item_id='1:1',version=1,content='public fact')

def test_zero_flow_preserves_icu_but_keeps_age_and_feedback():
    p=StrengthPolicy('full_0',8,11,head());p.bootstrap([item()]);p.update_state(event('context',0));i=p.items['1:1'];i.U=3
    before=(i.I,i.C,i.U);p.update_state(event('query',10))
    assert (i.I,i.C,i.U)==before and i.T==10
    p.feedback('1:1',1);assert i.U==4 and i.T==0

def test_rate_solution_matches_closed_form():
    p=StrengthPolicy('I_2',8,11,head());p.bootstrap([item()]);p.focus=0;p.held_k=.2;p.held_w=1.5
    p.update_state(event('query',3));assert abs(p.items['1:1'].I-(2-math.exp(-1.2)))<1e-12
    assert p.items['1:1'].C==1 and p.items['1:1'].U==0

def test_unit_rates_equal_original_causal_trace():
    a=StrengthPolicy('full_1',8,11,head());b=MatrixPolicy('adaptive_evolution',8,11,head())
    for p in (a,b):p.bootstrap([item()])
    for e in [event('context',0),event('query',2),event('feedback',3),event('observe',7,item_id='1:2',version=2,content='updated fact'),event('query',20)]:
        for p in (a,b):
            p.update_state(e)
            if e['kind']=='observe':p.observe(e);p.evict()
            if e['kind']=='query':p.retrieve(e)
            if e['kind']=='feedback':p.feedback('1:1',1)
        assert a.snapshot()==b.snapshot() and a.counts==b.counts

def test_stress_truth_only_uses_retained_observations():
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    for n in (500,1000,5000):
        row=environment(stories,vectors,94001,8,'rapid_high',n);assert len(row['events'])==n
        assert {w['evaluation']['phase'] for w in row['events']}=={-1,0,1,2}
        ledger={e['key']:e['item_id'] for e in row['initial_memory']};queries=set()
        for w in row['events']:
            e=w['input']
            if e['kind']=='observe':ledger[e['key']]=e['item_id']
            if e['kind']=='query':assert w['evaluation']['target_id']==ledger[e['key']];queries.add(e['event_id'])
            if e['kind']=='feedback':assert e['query_event_id'] in queries
