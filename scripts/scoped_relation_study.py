"""Factorial scope-feature / class-balancing experiment with fresh external tests."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,csv,hashlib,time
from collections import defaultdict,Counter
import numpy as np
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,f1_score,confusion_matrix,accuracy_score
from src.models.role_relation import CLASSES,frames,role_features,nli_features
from src.models.scoped_relation import scope_bags,scope_features
from src.models.relation_verifier import PairNLI
from src.data.semantic_gate import normalize,lexical_features
from src.provenance import snapshot_source,runtime_info

OUT=ROOT/'runs/scoped-relation-v1';OLD=ROOT/'runs/role-relation-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def key(s):return hashlib.sha256(s.encode()).hexdigest()
def measure(y,p,labels):
    pred=p.argmax(1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=range(len(labels)),average='macro')),
        report=classification_report(y,pred,labels=range(len(labels)),target_names=labels,output_dict=True,zero_division=0),confusion=confusion_matrix(y,pred,labels=range(len(labels))).tolist())

def main():
    if OUT.exists():raise SystemExit('Preserve existing run')
    OUT.mkdir(parents=True)
    rows=read(OLD/'examples.json');old=np.load(OLD/'features.npz');y=old['y'];oldx=old['x']
    # New external rows selected by text hash and public subcase BEFORE fitting.
    used={normalize(r[k]) for r in rows for k in ('text1','text2')}
    snli=pq.read_table(ROOT/'data/snli/test.parquet').to_pylist();premises={};excluded=0
    for r in snli:
        if r['label']<0 or any(normalize(r[k]) in used for k in ('premise','hypothesis')):excluded+=1;continue
        g=normalize(r['premise']);h=key(r['hypothesis'])
        if g not in premises or h<premises[g][0]:premises[g]=(h,r)
    ext=[]
    for g in sorted(premises,key=key)[:2000]:
        r=premises[g][1];ext.append(dict(id='snli:'+key(g),dataset='snli',group=key(g),text1=r['premise'],text2=r['hypothesis'],label=int(r['label'])))
    hans=list(csv.DictReader((ROOT/'data/hans/evaluation.txt').open(encoding='utf-8'),delimiter='\t'));cases=defaultdict(list)
    for r in hans:
        if any(normalize(r[k]) in used for k in ('sentence1','sentence2')):continue
        cases[r['subcase']].append(r)
    for case,rr in sorted(cases.items()):
        for r in sorted(rr,key=lambda r:key(r['pairID']))[:50]:
            ext.append(dict(id='hans:'+r['pairID'],dataset='hans',group=case,text1=r['sentence1'],text2=r['sentence2'],label=0 if r['gold_label']=='entailment' else 1))
    save('external_examples.json',ext)
    sig,archive=snapshot_source()
    save('protocol.json',dict(primary='scoped_balanced',source_signature=sig,source_archive=archive,runtime=runtime_info(),
        source={d:read(ROOT/f'data/{d}/provenance.json') for d in ('snli','hans')},
        feature_basis='Frozen prior 42 features +35 dependency modifier/negation/quantifier/preposition/role-modifier/clause features. No gold parses or original source text as features.',
        training='Same prior SICK train/selection only; all four models same C=[.1,1,10] search by selection macroF1. Factorial old/scoped features x ordinary/balanced class loss. Primary fixed in advance.',
        reused_audit='Prior SICK test reused for developmental comparison ONLY; not fresh independent evidence. Prior certificate is not reused for certification.',
        external='2000 label-independent SNLI unique-premise representatives; HANS 50 per subcase, 1500 total. Filter exact normalized SICK sentence overlaps. Scores generated only after freezing weights. No external fitting.',
        mapping='SICK five classes -> directional SNLI [E,N,C] = [equivalent+new_weaker,new_addition+unresolved,contradiction]. HANS [E,notE]. No reverse-label inference from one-way gold.',
        admissions='Diagnostic only: external one-way labels cannot certify five-way equivalence/addition or deletion. History remains blocked pending valid five-way and skip-risk evidence.',
        limitations='SNLI was used in NLI encoder pretraining; new to this project evaluation, not a novel pretraining distribution. HANS is template-generated with correlated cases. No certainty from lexical negation.',
        snli_excluded_invalid_or_overlap=excluded,counts=dict(Counter(r['dataset'] for r in ext))))
    import spacy
    parser=spacy.load('en_core_web_sm',disable=['ner']);t=time.perf_counter()
    texts=sorted({r[k] for r in rows for k in ('text1','text2')});bags={s:scope_bags(d) for s,d in zip(texts,parser.pipe(texts,batch_size=128))}
    x=np.c_[oldx,np.stack([scope_features(bags[r['text1']],bags[r['text2']]) for r in rows])]
    save('scope_parse_timing.json',dict(unique_sentences=len(texts),seconds=time.perf_counter()-t));np.savez_compressed(OUT/'development_features.npz',x=x,y=y)
    ix={s:np.asarray([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','selection','test')}
    models={};search=[]
    for name,d,balance in [('roles',42,None),('roles_balanced',42,'balanced'),('scoped',77,None),('scoped_balanced',77,'balanced')]:
        choices=[]
        for c in (.1,1,10):
            m=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight=balance,max_iter=2500,random_state=42));m.fit(x[ix['train'],:d],y[ix['train']]);f1=f1_score(y[ix['selection']],m.predict(x[ix['selection'],:d]),average='macro')
            search.append(dict(method=name,C=c,f1=f1));choices.append((f1,-c,m))
        _,_,m=max(choices,key=lambda t:t[:2]);models[name]=(m,d)
        sc,lr=m.steps[0][1],m.steps[1][1];save(name+'.json',dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),classes=lr.classes_.tolist(),C=lr.C,balance=balance,features=d))
    save('selection.json',search);save('frozen_models.json',{m:key((OUT/(m+'.json')).read_text(encoding='utf-8')) for m in models})
    metrics=[];predictions=[]
    for name,(m,d) in models.items():
        j=ix['test'];p=m.predict_proba(x[j,:d]);metrics.append(dict(dataset='sick_reused_development',method=name,**measure(y[j],p,CLASSES)))
    nli=PairNLI(ROOT/'artifacts/encoders/nli-deberta-small',ROOT/'artifacts/nli_pair_cache.sqlite3');features=[];forward=[];timings=[]
    for start in range(0,len(ext),128):
        t=time.perf_counter();block=ext[start:start+128];pairs=[(r['text1'],r['text2']) for r in block]
        f,mf=nli.predict(pairs);b,mb=nli.predict([(z,a) for a,z in pairs]);docs=list(parser.pipe([s for pair in pairs for s in pair],batch_size=128))
        if not all(mf['valid']) or not all(mb['valid']):raise ValueError('Unscored long text')
        for i,pair in enumerate(pairs):
            a,z=docs[2*i:2*i+2];features.append(np.r_[nli_features(f[i],b[i]),lexical_features(*pair),role_features(frames(a),frames(z)),scope_features(scope_bags(a),scope_bags(z))])
        forward.extend(f);timings.append(dict(start=start,n=len(block),seconds=time.perf_counter()-t))
        if start%512==0:print('external features',start+len(block),'/',len(ext),flush=True)
    xx=np.asarray(features);ff=np.asarray(forward);np.savez_compressed(OUT/'external_features.npz',x=xx,forward=ff)
    probabilities={'frozen_nli':ff[:,[1,2,0]]}
    for name,(m,d) in models.items():
        p=m.predict_proba(xx[:,:d]);probabilities[name]=np.stack((p[:,0]+p[:,2],p[:,1]+p[:,4],p[:,3]),axis=1)
    for dataset in ('snli','hans'):
        ids=np.asarray([i for i,r in enumerate(ext) if r['dataset']==dataset]);yy=np.asarray([ext[i]['label'] for i in ids]);labels=['entailment','neutral','contradiction'] if dataset=='snli' else ['entailment','non_entailment']
        for name,p in probabilities.items():
            pp=p[ids] if dataset=='snli' else np.c_[p[ids,0],p[ids,1]+p[ids,2]]
            metrics.append(dict(dataset=dataset,method=name,**measure(yy,pp,labels)))
            for i,pr in zip(ids,pp):predictions.append(dict(id=ext[i]['id'],dataset=dataset,group=ext[i]['group'],method=name,label=ext[i]['label'],probabilities=pr.tolist(),prediction=int(pr.argmax())))
    save('metrics.json',metrics);save('timing.json',dict(blocks=timings,nli_batches=nli.batches));save('predictions.json',predictions)
    save('complete.json',dict(api_requests=0,history_admitted=False,external_pairs=len(ext)));nli.close();print('Complete',flush=True)
if __name__=='__main__':main()
