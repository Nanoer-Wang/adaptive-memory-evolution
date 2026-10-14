"""Directional relation supervision with automatically parsed role bindings.

Parser frames are fallible input features, never gold semantic roles. Swapping
arguments is not assumed to imply contradiction (both events may be true).
"""
import numpy as np

CLASSES=('equivalent','new_addition','new_weaker','contradiction','unresolved')

def relation_label(ab,ba):
    """A is old, B is new. Inconsistent directional annotations stay unaudited."""
    a=ab.split('_')[1];b=ba.split('_')[1]
    return {('entails','entails'):0,('neutral','entails'):1,
            ('entails','neutral'):2,('contradicts','contradicts'):3,
            ('neutral','neutral'):4}.get((a,b),-1)

def source_groups(rows):
    """Bind both sentences AND their original source captions across splits."""
    import hashlib
    from src.data.semantic_gate import normalize
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in rows:
        keys=[normalize(r[k]) for k in ('sentence_A','sentence_B','sentence_A_original','sentence_B_original') if r[k].strip()]
        roots=[find(k) for k in keys];root=min(roots)
        for v in roots:parent[v]=root
    return [hashlib.sha256(find(normalize(r['sentence_A'])).encode()).hexdigest() for r in rows]

def frames(doc):
    result=[]
    for v in doc:
        if v.pos_!='VERB':continue
        agent=[];patient=[]
        for c in v.children:
            if c.dep_ in ('nsubj','csubj'):agent.append(c.lemma_.lower())
            if c.dep_ in ('dobj','obj','nsubjpass','csubjpass'):patient.append(c.lemma_.lower())
            if c.dep_=='agent':
                agent.extend(t.lemma_.lower() for t in c.children if t.dep_=='pobj')
        result.append(dict(predicate=v.lemma_.lower(),agent=sorted(set(agent)),patient=sorted(set(patient)),
                           negated=any(c.dep_=='neg' for c in v.children)))
    return result

def role_features(a,b):
    def bags(fs):
        p=set();s=set();o=set();triples=set();negative=set()
        for f in fs:
            v=f['predicate'];p.add(v)
            s.update((v,x) for x in f['agent']);o.update((v,x) for x in f['patient'])
            triples.update((v,x,y) for x in f['agent'] for y in f['patient'])
            if f['negated']:negative.add(v)
        return [p,s,o,triples,negative]
    aa,bb=bags(a),bags(b);out=[]
    for x,y in zip(aa,bb):
        out.extend([len(x&y)/max(len(x|y),1),len(x-y)/max(len(x),1),len(y-x)/max(len(y),1)])
    out.extend([len(aa[1]&bb[2])/max(len(aa[1]),1),len(aa[2]&bb[1])/max(len(aa[2]),1),
                float(bool(aa[3])),float(bool(bb[3])),min(len(a),5)/5,min(len(b),5)/5])
    return np.asarray(out,dtype=np.float64)

def nli_features(f,b):
    f=np.asarray(f);b=np.asarray(b)
    return np.r_[f,b,np.log(np.clip(f,1e-7,1)),np.log(np.clip(b,1e-7,1)),
                 min(f[1],b[1]),f[1]-b[1]]
