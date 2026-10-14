"""Scope-sensitive dependency features; no label or source-ID access.

These are fallible parse features, not logical rules or overwrite permission.
Negation is bound to its syntactic head; modifiers to their noun/event heads.
"""
import numpy as np

def scope_bags(doc):
    bags=[set() for _ in range(7)]
    for t in doc:
        lemma=t.lemma_.lower();head=t.head.lemma_.lower()
        if t.dep_ in ('amod','compound','nummod'):
            bags[0].add((head,t.dep_,lemma))
        if t.dep_=='advmod':bags[1].add((head,lemma))
        if t.dep_=='neg' or (t.dep_=='det' and lemma in ('no','neither')):
            bags[2].add(head)
        if t.dep_=='det' and lemma in ('all','every','each','some','any','no','neither'):
            bags[3].add((head,lemma))
        if t.dep_=='pobj':bags[4].add((t.head.head.lemma_.lower(),head,lemma))
        if t.dep_ in ('nsubj','nsubjpass','dobj','obj'):
            role='patient' if t.dep_ in ('nsubjpass','dobj','obj') else 'agent'
            bags[5].update((head,role,lemma,c.lemma_.lower()) for c in t.children if c.dep_ in ('amod','nummod','compound'))
        if t.dep_ in ('conj','ccomp','xcomp','advcl','relcl'):
            bags[6].add((head,t.dep_,lemma))
    return bags

def scope_features(a,b):
    out=[]
    for x,y in zip(a,b):
        out.extend([len(x&y)/max(len(x|y),1),len(x-y)/max(len(x),1),len(y-x)/max(len(y),1),float(bool(x)),float(bool(y))])
    return np.asarray(out,dtype=np.float64)
