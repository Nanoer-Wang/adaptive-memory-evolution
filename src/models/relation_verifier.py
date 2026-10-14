"""Pinned CPU NLI cross-encoder, both directions, conservative skip proposal.

No LLM/API, remote code, truncation, or implicit state mutation. Contradiction
does not itself authorize overwriting a fact. Unknown/long inputs are retained.
"""
import hashlib,json,sqlite3,time,re
from pathlib import Path
import numpy as np


def changed_literals(a,b):
    """Cheap veto, not a semantic proof. Dates/numbers and explicit negation."""
    numbers=lambda s:set(re.findall(r'\d+(?:[.,]\d+)*',s))
    negative=lambda s:bool(re.search(r"\b(?:no|not|never|neither|nor|without)\b|n't\b",s.lower()))
    return numbers(a)!=numbers(b) or negative(a)!=negative(b)


def relation_scores(forward,backward,texts):
    # Label order is asserted by the pinned configuration: C, E, N.
    one=np.asarray(forward)[:,1]
    mutual=np.minimum(one,np.asarray(backward)[:,1])
    veto=np.asarray([changed_literals(a,b) for a,b in texts])
    return {'nli_oneway':one,'nli_bidirectional':mutual,
            'nli_bidirectional_guarded':np.where(veto,0.,mutual)}


class PairNLI:
    def __init__(self,root,cache,threads=4,max_tokens=512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root=Path(root);self.source=json.loads((root/'source.json').read_text())
        for file in self.source['files']:
            if hashlib.sha256((root/file['name']).read_bytes()).hexdigest()!=file['sha256']:
                raise ValueError('NLI artifact hash mismatch')
        config=json.loads((root/'config.json').read_text())
        if config['id2label']!={'0':'contradiction','1':'entailment','2':'neutral'}:raise ValueError('Unexpected NLI label order')
        self.max_tokens=max_tokens;self.tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'));self.tokenizer.no_truncation();self.tokenizer.no_padding()
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(root/'model_quint8_avx2.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.inputs={x.name for x in self.session.get_inputs()}
        self.signature=hashlib.sha256(json.dumps(self.source,sort_keys=True).encode()).hexdigest()+f':pair-untruncated-{max_tokens}-v1'
        self.db=sqlite3.connect(cache);self.db.execute('CREATE TABLE IF NOT EXISTS logits (key TEXT PRIMARY KEY, data BLOB NOT NULL)')
        self.batches=[]

    def predict(self,pairs,batch_size=8):
        start=time.perf_counter();encoded=self.tokenizer.encode_batch(pairs,add_special_tokens=True)
        tokenize_seconds=time.perf_counter()-start
        logits=np.zeros((len(pairs),3),dtype=np.float32);valid=np.ones(len(pairs),dtype=bool)
        missing={};hits=0
        for i,((a,b),tokens) in enumerate(zip(pairs,encoded)):
            if len(tokens.ids)>self.max_tokens:
                valid[i]=False;logits[i]=[-100,-100,0];continue
            key=hashlib.sha256((self.signature+'\0'+a+'\0'+b).encode()).hexdigest()
            found=self.db.execute('SELECT data FROM logits WHERE key=?',(key,)).fetchone()
            if found:logits[i]=np.frombuffer(found[0],dtype=np.float32);hits+=1
            else:missing.setdefault(key,dict(tokens=tokens,indices=[]))['indices'].append(i)
        items=sorted(missing.items(),key=lambda x:len(x[1]['tokens'].ids))
        for begin in range(0,len(items),batch_size):
            group=items[begin:begin+batch_size];length=max(len(r['tokens'].ids) for _,r in group)
            ids=np.zeros((len(group),length),np.int64);mask=np.zeros_like(ids);types=np.zeros_like(ids)
            for j,(_,item) in enumerate(group):
                tokens=item['tokens'];n=len(tokens.ids);ids[j,:n]=tokens.ids;mask[j,:n]=1;types[j,:n]=tokens.type_ids
            feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':types};t=time.perf_counter()
            raw=self.session.run(None,{k:v for k,v in feed.items() if k in self.inputs})[0]
            self.batches.append(dict(pairs=len(group),padded_tokens=length,seconds=time.perf_counter()-t))
            if raw.shape!=(len(group),3) or not np.isfinite(raw).all():raise ValueError('Bad NLI logits')
            for (key,item),value in zip(group,raw):
                self.db.execute('INSERT OR REPLACE INTO logits VALUES (?,?)',(key,np.asarray(value,dtype=np.float32).tobytes()))
                for i in item['indices']:logits[i]=value
            self.db.commit()
        p=np.exp(logits.astype(np.float64)-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        return p,dict(valid=valid.tolist(),token_lengths=[len(e.ids) for e in encoded],cache_hits=hits,
                      tokenize_seconds=tokenize_seconds,total_seconds=time.perf_counter()-start)

    def close(self):self.db.close()


def choose_candidate(scores,labels):
    """Frozen grid; empirical selection-set wrong-among-skips <=0.5%."""
    scores=np.asarray(scores);labels=np.asarray(labels)
    candidates=[]
    for t in (.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,.999,.9995,.9999):
        skip=scores>=t;n=int(skip.sum());errors=int(((labels==0)&skip).sum())
        if n>=100 and errors/n<=.005:candidates.append((n,t,errors))
    if not candidates:return dict(threshold=1.000001,selection_skips=0,selection_errors=0,eligible=False)
    n,t,e=max(candidates,key=lambda x:(x[0],x[1]))
    return dict(threshold=t,selection_skips=n,selection_errors=e,eligible=True)


def certify(scores,labels,threshold,alpha=.05,risk=.01):
    """One preselected primary policy; exact binomial upper bound, IID caveat."""
    from scipy.stats import beta
    skip=np.asarray(scores)>=threshold;labels=np.asarray(labels);n=int(skip.sum());e=int((skip&(labels==0)).sum())
    upper=1. if n==0 or e==n else float(beta.ppf(1-alpha,e+1,n-e))
    return dict(accepted=n,errors=e,wrong_among_skips=e/n if n else None,one_sided_cp95_upper=upper,
                approved=n>=100 and upper<=risk,
                caveat='Single primary-policy IID-pair binomial diagnostic; not guaranteed under shared articles, correlated pairs or domain shift.')
