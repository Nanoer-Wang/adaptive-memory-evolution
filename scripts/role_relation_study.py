"""Frozen source-grouped SICK study: relation supervision and role ablation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import csv,json,time,hashlib
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,confusion_matrix,classification_report,accuracy_score
from src.models.role_relation import CLASSES,relation_label,source_groups,frames,role_features,nli_features
from src.models.relation_verifier import PairNLI,choose_candidate,certify
from src.data.semantic_gate import lexical_features
from src.provenance import snapshot_source,runtime_info

OUT=ROOT/'runs/role-relation-v1'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    if OUT.exists():raise SystemExit('Existing run preserved')
    OUT.mkdir(parents=True)
    raw=list(csv.DictReader((ROOT/'data/sick/SICK.txt').open(encoding='utf-8'),delimiter='\t'))
    groups=source_groups(raw);rows=[];conflicts=[]
    for r,g in zip(raw,groups):
        q=int(hashlib.sha256(('role-v1:'+g).encode()).hexdigest()[:8],16)%100
        split='train' if q<60 else 'selection' if q<75 else 'certificate' if q<85 else 'test'
        label=relation_label(r['entailment_AB'],r['entailment_BA'])
        if label<0:conflicts.append(dict(**r,group=g,split=split));continue
        for reverse in (False,True):
            rows.append(dict(id=r['pair_ID']+(':BA' if reverse else ':AB'),group=g,split=split,
                text1=r['sentence_B' if reverse else 'sentence_A'],text2=r['sentence_A' if reverse else 'sentence_B'],
                label=({1:2,2:1}.get(label,label) if reverse else label),official_split=r['SemEval_set']))
    save('examples.json',rows);save('annotation_conflicts.json',conflicts)
    signature,archive=snapshot_source()
    import spacy
    parser=spacy.load('en_core_web_sm',disable=['ner'])
    model_hash=hashlib.sha256()
    for f in sorted(Path(parser.path).rglob('*')):
        if f.is_file():model_hash.update(f.relative_to(parser.path).as_posix().encode());model_hash.update(f.read_bytes())
    save('protocol.json',dict(primary='nli_lexical_roles',classes=CLASSES,source_signature=signature,source_archive=archive,runtime=runtime_info(),
        data=json.loads((ROOT/'data/sick/provenance.json').read_text(encoding='utf-8')),
        nli=json.loads((ROOT/'artifacts/encoders/nli-deberta-small/source.json').read_text(encoding='utf-8')),
        parser=dict(spacy=spacy.__version__,model=parser.meta['version'],sha256=model_hash.hexdigest()),
        split='Custom 60/15/10/15 hash groups linking both surface sentences and original captions; not official SICK split. Both orientations stay together.',
        supervision='Human directional entailment labels; role features from frozen dependency parser, NOT manual role gold. Inconsistent directional judgments excluded with full records; results conditional on consistent subset.',
        models='Frozen NLI argmax baseline; calibrated NLI; NLI+lexical; primary NLI+lexical+roles. Logistic C in [.1,1,10] selected macro F1; all shared NLI frozen.',
        admission='Primary certificate macroF1>=.80, per-class recall>=.75, contradiction recall>=.90 AND equivalent skip CP95 upper<=.01 with >=100 accepted independent source representatives. Frozen test must meet same descriptive macro/recall and observed skip error<=.01 with >=100 skips.',
        skip='Selection: one label-independent directed representative per source group, >=100 equivalent skips and empirical errors<=.005; calibration same independent representatives, fixed threshold. No candidate => retain all.',
        history='Only if primary admission passes; otherwise no liquid/GRU history comparison or benefit claim.',
        limitations='English caption domain only; parser errors and pretrained contamination possible. Public subset agreement is not real-world revision supervision. No invented contradiction labels from swapped roles.'))
    save('split_audit.json',dict(rows=dict(Counter(r['split'] for r in rows)),groups={s:len({r['group'] for r in rows if r['split']==s}) for s in ('train','selection','certificate','test')},
        conflicts=len(conflicts),class_counts={s:dict(Counter(CLASSES[r['label']] for r in rows if r['split']==s)) for s in ('train','selection','certificate','test')}))
    nli=PairNLI(ROOT/'artifacts/encoders/nli-deberta-small',ROOT/'artifacts/nli_pair_cache.sqlite3')
    features=[];probabilities=[];timings=[]
    # Features contain no dataset IDs, original captions, split names or labels.
    for start in range(0,len(rows),128):
        block=rows[start:start+128];t=time.perf_counter();pairs=[(r['text1'],r['text2']) for r in block]
        f,mf=nli.predict(pairs);b,mb=nli.predict([(y,x) for x,y in pairs])
        if not all(mf['valid']) or not all(mb['valid']):raise ValueError('Long text needs explicit retention handling')
        docs=list(parser.pipe([x for pair in pairs for x in pair],batch_size=128));parsed=[frames(d) for d in docs]
        for i,r in enumerate(block):
            a,z=parsed[2*i:2*i+2]
            features.append(np.r_[nli_features(f[i],b[i]),lexical_features(*pairs[i]),role_features(a,z)])
            probabilities.append((f[i],b[i]))
            with (OUT/'parsed_roles.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps(dict(id=r['id'],old=a,new=z))+'\n')
        timings.append(dict(start=start,n=len(block),seconds=time.perf_counter()-t,forward=mf,backward=mb))
        if start%1024==0:print('features',start+len(block),'/',len(rows),flush=True)
    x=np.asarray(features);y=np.asarray([r['label'] for r in rows]);np.savez_compressed(OUT/'features.npz',x=x,y=y,nli=np.asarray(probabilities))
    save('timing.json',dict(blocks=timings,nli_batches=nli.batches));nli.close()
    ix={s:np.asarray([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','selection','certificate','test')}
    models={};scores={};search=[]
    for name,d in [('nli_calibrated',14),('nli_lexical',21),('nli_lexical_roles',42)]:
        choices=[]
        for c in (.1,1,10):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
            model.fit(x[ix['train'],:d],y[ix['train']]);f1=f1_score(y[ix['selection']],model.predict(x[ix['selection'],:d]),average='macro')
            search.append(dict(method=name,C=c,selection_macro_f1=f1));choices.append((f1,-c,model))
        _,_,m=max(choices,key=lambda z:z[:2]);models[name]=(m,d)
        sc,lr=m.steps[0][1],m.steps[1][1]
        save(name+'.json',dict(classes=lr.classes_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),C=lr.C,features=d))
    save('selection.json',search)
    # Weights fixed before certificate or test predictions/metrics.
    save('frozen_models.json',{k:hashlib.sha256((OUT/(k+'.json')).read_bytes()).hexdigest() for k in models})
    for name,(m,d) in models.items():scores[name]=m.predict_proba(x[:,:d])
    # Convert independently predicted directional relations without learning.
    base=[]
    for f,b in probabilities:
        a,c=int(np.argmax(f)),int(np.argmax(b));label={(1,1):0,(2,1):1,(1,2):2,(0,0):3}.get((a,c),4)
        base.append(label)
    def representatives(indices):
        selected={}
        for i in indices:
            g=rows[i]['group'];key=hashlib.sha256(rows[i]['id'].encode()).hexdigest()
            if g not in selected or key<selected[g][0]:selected[g]=(key,i)
        return np.asarray([v[1] for v in selected.values()])
    ri={s:representatives(ix[s]) for s in ix};results=[];certs={};candidates={};preds=[]
    for name,p in scores.items():
        si=ri['selection'];candidate=choose_candidate(p[si,0],(y[si]==0).astype(int));candidates[name]=candidate
        ci=ri['certificate'];certs[name]=certify(p[ci,0],(y[ci]==0).astype(int),candidate['threshold'])
    save('frozen_candidates.json',candidates);save('certificates.json',certs)
    for name in ['frozen_nli']+list(scores):
        pred=np.asarray(base) if name=='frozen_nli' else scores[name].argmax(1)
        for split in ('selection','certificate','test'):
            j=ix[split];report=classification_report(y[j],pred[j],labels=range(5),target_names=CLASSES,output_dict=True,zero_division=0)
            row=dict(method=name,split=split,n=len(j),accuracy=float(accuracy_score(y[j],pred[j])),macro_f1=float(f1_score(y[j],pred[j],labels=range(5),average='macro')),report=report,confusion=confusion_matrix(y[j],pred[j],labels=range(5)).tolist())
            if name!='frozen_nli':
                ids=ri[split];sk=scores[name][ids,0]>=candidates[name]['threshold'];n=int(sk.sum());e=int(((y[ids]!=0)&sk).sum())
                row['skip']=dict(representatives=len(ids),accepted=n,errors=e,risk=e/n if n else None)
            results.append(row)
        for i,r in enumerate(rows):
            if r['split']!='train':preds.append(dict(id=r['id'],group=r['group'],split=r['split'],label=int(y[i]),method=name,prediction=int(pred[i]),probabilities=None if name=='frozen_nli' else scores[name][i].tolist()))
    save('metrics.json',results)
    with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for p in preds:f.write(json.dumps(p)+'\n')
    def quality(r):return r['macro_f1']>=.8 and min(r['report'][c]['recall'] for c in CLASSES)>=.75 and r['report']['contradiction']['recall']>=.9
    primary=[r for r in results if r['method']=='nli_lexical_roles' and r['split'] in ('certificate','test')]
    test=next(r for r in primary if r['split']=='test')
    admitted=all(quality(r) for r in primary) and certs['nli_lexical_roles']['approved'] and test['skip']['accepted']>=100 and test['skip']['risk']<=.01
    save('complete.json',dict(rows=len(rows),api_requests=0,history_admitted=admitted,reason='All predefined criteria required; no history run if false'))
    print('FINISHED',json.dumps(dict(history_admitted=admitted,test_macro_f1=test['macro_f1'],certificate=certs['nli_lexical_roles'])),flush=True)
if __name__=='__main__':main()
