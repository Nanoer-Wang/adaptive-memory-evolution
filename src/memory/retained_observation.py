"""Fixed retained-view observation for training-only evidence interventions.

No gold answers, task categories or evidence annotations are accepted here.
Similarity measures coverage, not entailment or factual correctness.
"""
import re
import numpy as np
from src.data.longmemeval import date_days

FEATURES=['log_units','query_max','query_mean','lexical_mean','selected_fraction',
          'semantic_coverage','omitted_risk_max','omitted_risk_mean','missing_number_fraction',
          'selected_query_max','user_fraction','log_time_gap','source_position','char_ratio',
          'missing_query_word_fraction','source_novelty','compact_abstain','answer_length']
OMISSION_COLUMNS=[4,5,6,7,8,9,14]

def parse_units(context):
    units=[];header=None
    for line in context.splitlines():
        if line.startswith('[session='):
            if not re.fullmatch(r'\[session=\S+ date=.+\]',line):raise ValueError('Invalid source header')
            header=line
        elif line.strip():
            if header is None:raise ValueError('Missing source header')
            units.append((header,line))
    return units

def render(units,selected):
    chunks=[];last=None
    for i in sorted(selected):
        header,line=units[i]
        if header!=last:chunks.append(('\n' if chunks else '')+header);last=header
        chunks.append(line)
    return '\n'.join(chunks)

def retained_observation(question,context,encoder,retained):
    units=parse_units(context)
    if not units:raise ValueError('Empty full context')
    vectors=encoder.encode([line for _,line in units]);q=encoder.encode([question])[0]
    dense=np.maximum(vectors@q,0.)
    terms=set(re.findall(r'[a-z0-9]+',question.lower()))-{'the','a','i','my','what','how','was','did','in','of','to','and','is','do'}
    lexical=np.array([len(terms&set(re.findall(r'[a-z0-9]+',line.lower())))/max(1,len(terms)) for _,line in units])
    score=.75*dense+.25*lexical;order=sorted(range(len(units)),key=lambda i:(-float(score[i]),i))
    chosen=set();out={};headers=list(dict.fromkeys(h for h,_ in units));dates=[date_days(h.split(' date=',1)[1][:-1],'minute') for h in headers]
    dt=np.clip(np.diff(dates,prepend=dates[0])/30,0,12)
    for budget in [len(retained)]:
        retained_units=set(parse_units(retained))
        chosen={i for i,u in enumerate(units) if u in retained_units}
        assert retained_units <= set(units), "Retained evidence must be a subset"
        compact=render(units,chosen)
        # Semantically similar retained wording can cover a dropped line; exact
        # number loss is tracked separately to avoid treating changed values as equal.
        coverage=np.maximum(0,(vectors@vectors[sorted(chosen)].T).max(1)) if chosen else np.zeros(len(units))
        numbers=set(re.findall(r'\b\d+(?:[.:/-]\d+)*\b',compact))
        words=set(re.findall(r'[a-z0-9]+',compact.lower()));x=[];previous=None
        for pos,header in enumerate(headers):
            ix=np.array([i for i,(h,_) in enumerate(units) if h==header]);take=np.array([i in chosen for i in ix])
            text=' '.join(units[i][1] for i in ix);ns=set(re.findall(r'\b\d+(?:[.:/-]\d+)*\b',text));needed=terms&set(re.findall(r'[a-z0-9]+',text.lower()))
            risk=dense[ix]*(1-coverage[ix]);centroid=vectors[ix].mean(0);centroid/=max(np.linalg.norm(centroid),1e-8)
            x.append([np.log1p(len(ix))/6,float(dense[ix].max()),float(dense[ix].mean()),float(lexical[ix].mean()),float(take.mean()),
                float(np.average(coverage[ix],weights=dense[ix]+1e-6)),float(risk.max()),float(risk.mean()),len(ns-numbers)/max(1,len(ns)),
                float(dense[ix][take].max()) if take.any() else 0.,sum(units[i][1].startswith('user:') for i in ix)/len(ix),
                float(np.log1p(dt[pos])),pos/max(1,len(headers)-1),len(compact)/len(context),len(needed-words)/max(1,len(needed)),
                1-float(centroid@previous) if previous is not None else 1.,0.,0.])
            previous=centroid
        out[budget]=dict(context=compact,selected_unit_indices=sorted(chosen),source_headers=headers,x=x,dt=dt.tolist(),
            full_units=len(units),selected_units=len(chosen),feature_names=FEATURES,units=[dict(header=h,text=t) for h,t in units])
    return out[budget]
