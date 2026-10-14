"""Additional frozen PAWS risk audit; does not train or change SICK models."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from src.data.semantic_gate import sentence_groups,normalize,lexical_features
from src.models.role_relation import frames,role_features,nli_features
from src.models.relation_verifier import PairNLI,choose_candidate,certify
OUT=ROOT/'runs/role-relation-v1/external'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    if OUT.exists():raise SystemExit('Existing external audit preserved')
    OUT.mkdir(parents=True)
    previous=read(ROOT/'runs/semantic-gate-v2/examples.json')+read(ROOT/'runs/relation-verifier-v1/examples.json')
    seen={normalize(r[k]) for r in previous for k in ('text1','text2')}
    raw=pq.read_table(ROOT/'data/paws/train.parquet').to_pylist()
    g=sentence_groups([dict(text1=r['sentence1'],text2=r['sentence2']) for r in raw]);blocked={gg for r,gg in zip(raw,g) if normalize(r['sentence1']) in seen or normalize(r['sentence2']) in seen}
    available=defaultdict(list)
    for r,gg in zip(raw,g):
        if gg not in blocked:available[gg].append(r)
    groups=sorted(available,key=lambda x:hashlib.sha256(('role-external:'+x).encode()).hexdigest())[:2000]
    if len(groups)!=2000:raise ValueError('Insufficient previously unused components')
    rows=[dict(r) for r in read(ROOT/'runs/semantic-gate-v2/examples.json') if r['split']=='selection']
    for i,gg in enumerate(groups):
        r=min(available[gg],key=lambda r:hashlib.sha256(str(r['id']).encode()).hexdigest())
        rows.append(dict(id='paws:train:'+str(r['id']),split='certificate' if i<1000 else 'test',group=gg,text1=r['sentence1'],text2=r['sentence2'],label=int(r['label'])))
    save('examples.json',rows)
    save('protocol.json',dict(primary='nli_lexical_roles',note='Prospective external binary equivalence audit after SICK weights frozen. Reuses old selection solely for threshold fitting; 1000+1000 previously unused PAWS components, one pair each. No multiclass labels invented from PAWS negatives. Pretraining/article dependence remains.',
        methods=['frozen_mutual_nli','nli_calibrated','nli_lexical','nli_lexical_roles'],
        rule='Same >=100 selection skips with empirical errors<=0.5%, fixed grid. Certificate CP95 upper<=1%, >=100 skips. Test observes fixed threshold. SICK multiclass admission remains separate and underpowered.',
        remaining_components=len(available),models=read(ROOT/'runs/role-relation-v1/frozen_models.json')))
    import spacy
    parser=spacy.load('en_core_web_sm',disable=['ner']);nli=PairNLI(ROOT/'artifacts/encoders/nli-deberta-small',ROOT/'artifacts/nli_pair_cache.sqlite3')
    models={k:read(ROOT/('runs/role-relation-v1/'+k+'.json')) for k in ('nli_calibrated','nli_lexical','nli_lexical_roles')}
    results=[];timing=[];candidates=None;certs=None
    for split in ('selection','certificate','test'):
        rs=[r for r in rows if r['split']==split];records=[]
        for start in range(0,len(rs),128):
            t=time.perf_counter();block=rs[start:start+128];pairs=[(r['text1'],r['text2']) for r in block]
            f,mf=nli.predict(pairs);b,mb=nli.predict([(y,x) for x,y in pairs]);docs=list(parser.pipe([s for pair in pairs for s in pair]));parsed=[frames(d) for d in docs]
            x=np.stack([np.r_[nli_features(f[i],b[i]),lexical_features(*pair),role_features(parsed[2*i],parsed[2*i+1])] for i,pair in enumerate(pairs)])
            probs={'frozen_mutual_nli':np.minimum(f[:,1],b[:,1])}
            for name,m in models.items():
                z=((x[:,:m['features']]-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coef']).T+np.asarray(m['intercept'])
                z=np.exp(z-z.max(1,keepdims=True));probs[name]=(z/z.sum(1,keepdims=True))[:,m['classes'].index(0)]
            valid=np.asarray(mf['valid'])&np.asarray(mb['valid'])
            for i,r in enumerate(block):records.append(dict(id=r['id'],split=split,label=r['label'],scores={k:float(v[i]) if valid[i] else 0. for k,v in probs.items()}))
            timing.append(dict(split=split,n=len(block),seconds=time.perf_counter()-t))
        with (OUT/'scores.jsonl').open('a',encoding='utf-8') as out:
            for r in records:out.write(json.dumps(r)+'\n')
        if split=='selection':
            candidates={m:choose_candidate([r['scores'][m] for r in records],[r['label'] for r in records]) for m in records[0]['scores']};save('frozen_candidates.json',candidates)
        if split=='certificate':
            certs={m:certify([r['scores'][m] for r in records],[r['label'] for r in records],c['threshold']) for m,c in candidates.items()};save('certificates.json',certs)
        for m,c in candidates.items():
            s=np.asarray([r['scores'][m] for r in records]);y=np.asarray([r['label'] for r in records]);sk=s>=c['threshold'];n=int(sk.sum());e=int((sk&(y==0)).sum())
            results.append(dict(split=split,method=m,n=len(y),auc=roc_auc_score(y,s),skips=n,errors=e,risk=e/n if n else None))
        print('external',split,'complete',flush=True)
    save('metrics.json',results);save('timing.json',timing);save('complete.json',dict(api_requests=0,primary_certified=certs['nli_lexical_roles']['approved']));nli.close()
if __name__=='__main__':main()
