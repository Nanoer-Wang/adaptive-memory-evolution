"""Shared protected evidence, source-separated constructed public-data streams."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,time,copy
from collections import defaultdict
import numpy as np
import torch
from src.models.evidence_history import HistoryGate
from src.provenance import snapshot_source,runtime_info
CONDITIONAL='--eligible-only' in sys.argv
OUT=ROOT/('runs/evidence-history-v2' if CONDITIONAL else 'runs/evidence-history-v1');OLD=ROOT/'runs/role-relation-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def summarize(skip,y,seen):
    semantic=skip&~seen;n=int(skip.sum());e=int((skip&~y).sum());sn=int(semantic.sum());se=int((semantic&~y).sum())
    return dict(events=int(y.size),writes=int((~skip).sum()),skips=n,errors=e,skip_fraction=float(skip.mean()),wrong_among_skips=e/n if n else None,
        semantic_skips=sn,semantic_errors=se,semantic_risk=se/sn if sn else None,exact_repeat_skips=int((skip&seen).sum()),
        missed_redundancy=int((~skip&y).sum()))

def main():
    if OUT.exists():raise SystemExit('Run exists')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    rows=read(OLD/'examples.json');f=np.load(OLD/'features.npz');nli=f['nli'];rawx=f['x'];role=read(OLD/'nli_lexical_roles.json')
    logits=((rawx-np.asarray(role['mean']))/np.asarray(role['scale']))@np.asarray(role['coef']).T+np.asarray(role['intercept']);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    anchors=defaultdict(list)
    for i,r in enumerate(rows):anchors[(r['split'],r['text1'])].append(i)
    streams=[]
    for split,source_split,limit in [('train','train',800),('selection','selection',200),('test','certificate' if CONDITIONAL else 'test',159 if CONDITIONAL else 277)]:
        keys=sorted([k for k,v in anchors.items() if k[0]==source_split and len(v)>=3],key=lambda k:digest(k[1]))[:limit]
        for k in keys:
            rng=np.random.default_rng(int(digest(k[1])[:8],16));available=anchors[k];ids=rng.choice(available,20,replace=True)
            seen=set();events=[]
            for i in ids:
                r=rows[i];repeat=r['text2'] in seen;seen.add(r['text2']);dt=float(rng.choice([.25,1.,4.]))
                a,b=nli[i];eq=float(p[i,0]);x=np.r_[a,b,eq,dt].tolist()
                eligible=min(a[1],b[1])>=.9 and max(a[0],b[0])<.1 and eq>=.5
                events.append(dict(id=r['id'],x=x,seen=repeat,eligible=bool(eligible),label=bool(repeat or r['label']==0),dt=dt))
            streams.append(dict(id=digest(k[1]),group=rows[available[0]]['group'],split=split,events=events))
    save('streams.json',streams);sig,archive=snapshot_source()
    save('protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),primary='liquid',seeds=[11,23,42],
        task='20-event anchored text replay. Skip target: exact text seen before OR human equivalence to fixed anchor. Other novel content retained. No state replacement or QA.',
        conditional_objective=CONDITIONAL,
        correction='v2 trains only nonliteral events eligible under fixed guard; v1 trained all nonliteral events. Same inputs and architecture. v2 uses previously unused history anchors from old SICK certificate split for history audit, not a fresh semantic dataset. This correction follows v1 diagnostics; no v1 test retuning claimed.',
        provenance='Public SICK previous source-isolated splits. These rows were already evaluated; developmental mechanism comparison, not fresh semantic validation. Streams and intervals synthetic. Both directions kept in original source group.',
        evidence='Same frozen six directional NLI probabilities, role-equality probability and dt for all models. Role classifier unchanged. No target labels/IDs/source group as model input.',
        protection='All nonliteral skips require both entailments>=.9, both contradictions<.1 and role equality>=.5. History can veto, cannot waive guard. Exact string repeats handled identically by common observed-text ledger.',
        methods='Write all; exact-only; protected single observation; EMA (.25,1,4 time constants); static neural; GRU; four-substep state-coupled liquid. Same width16, training epochs12, Adam lr.003, batch64; params reported, not exactly matched.',
        selection='Best selection BCE epoch per neural model. Skip threshold grid .5,.6,.7,.8,.9,.95,.99,1.000001 maximizing semantic skips with >=20 semantic skips and <=5% semantic error on selection. Exploratory 5% empirical operating point, NOT prior 1% safety certification. Literal repeats excluded from risk denominator.',
        ablations='Neural state reset every event (same weights and threshold). Liquid dt replaced by1 in transition only; input dt unchanged. No claim of real temporal advantage: targets do not depend on synthetic time.',
        costs='Offline decisions, not actual irreversible deletion. Writes are count proxy; time excludes common frozen encoders. Exact ledger up to20 SHA256 keys =640 payload bytes + fixed anchor, Python overhead excluded; neural state16float32=64 additional bytes plus weights. Full content still retained in evaluation records.',
        limitations='SICK-derived role adapter has known cross-domain failures. This experiment only explores guarded streams and cannot establish general reliable compression or paper acceptance.'))
    sets={}
    for split in ['train','selection','test']:
        ss=[s for s in streams if s['split']==split]
        sets[split]=dict(x=np.asarray([[e['x'] for e in s['events']] for s in ss],dtype=np.float32),y=np.asarray([[e['label'] for e in s['events']] for s in ss],dtype=bool),seen=np.asarray([[e['seen'] for e in s['events']] for s in ss],dtype=bool),eligible=np.asarray([[e['eligible'] for e in s['events']] for s in ss],dtype=bool))
    def choose(score):
        s=sets['selection'];choices=[]
        for t in [.5,.6,.7,.8,.9,.95,.99,1.000001]:
            q=s['eligible']&~s['seen']&(score>=t);n=int(q.sum());e=int((q&~s['y']).sum())
            if n>=20 and e/n<=.05:choices.append((n,t))
        return max(choices)[1] if choices else 1.000001
    scores={};models={};thresholds={};training=[];params={}
    for split,s in sets.items():
        scores[split]={'exact_only':np.zeros_like(s['y'],dtype=float),'protected_single':np.minimum(np.minimum(s['x'][:,:,1],s['x'][:,:,4]),s['x'][:,:,6])}
        for tau in [.25,1.,4.]:
            h=np.zeros(len(s['x']));out=[]
            for j in range(20):
                decay=np.exp(-s['x'][:,j,7]/tau);h=h*decay+(1-decay)*scores[split]['protected_single'][:,j];out.append(h.copy())
            scores[split]['ema_'+str(tau)]=np.stack(out,1)
    for k in scores['selection']:thresholds[k]=choose(scores['selection'][k])
    tx=torch.from_numpy(sets['train']['x']);ty=torch.from_numpy(sets['train']['y'].astype(np.float32));sx=torch.from_numpy(sets['selection']['x']);sy=torch.from_numpy(sets['selection']['y'].astype(np.float32))
    # Labels on repeats excluded from learning: exact ledger already solves them.
    mask=torch.from_numpy((~sets['train']['seen']).astype(np.float32));smask=torch.from_numpy((~sets['selection']['seen']).astype(np.float32))
    if CONDITIONAL:
        mask*=torch.from_numpy(sets['train']['eligible']);smask*=torch.from_numpy(sets['selection']['eligible'])
    for kind in ['static','gru','liquid']:
        for seed in [11,23,42]:
            torch.manual_seed(seed);m=HistoryGate(kind);opt=torch.optim.Adam(m.parameters(),lr=.003);best=float('inf');state=None;start=time.perf_counter()
            for epoch in range(12):
                m.train()
                for ids in torch.randperm(len(tx)).split(64):
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(tx[ids]),ty[ids],reduction='none')*mask[ids]).sum()/mask[ids].sum().clamp_min(1);opt.zero_grad();loss.backward();opt.step()
                m.eval()
                with torch.no_grad():val=float((torch.nn.functional.binary_cross_entropy_with_logits(m(sx),sy,reduction='none')*smask).sum()/smask.sum())
                training.append(dict(kind=kind,seed=seed,epoch=epoch,selection_bce=val))
                if val<best:best=val;state=copy.deepcopy(m.state_dict())
            m.load_state_dict(state);name=f'{kind}-{seed}';models[name]=m;params[name]=dict(parameters=sum(p.numel() for p in m.parameters()),state_bytes=0 if kind=='static' else 64,training_seconds=time.perf_counter()-start)
            torch.save(state,OUT/(name+'.pt'))
            with torch.no_grad():scores['selection'][name]=m(sx).sigmoid().numpy()
            thresholds[name]=choose(scores['selection'][name]);print(name,'trained',flush=True)
    save('training.json',training);save('parameters.json',params);save('frozen_thresholds.json',thresholds);save('frozen_weights.json',{k:digest((OUT/(k+'.pt')).read_bytes().hex()) for k in models})
    results=[];predictions=[];timing=[];s=sets['test'];testx=torch.from_numpy(s['x'])
    for name,m in models.items():
        with torch.no_grad():
            scores['test'][name]=m(testx).sigmoid().numpy()
            if m.kind!='static':scores['test'][name+'_reset']=m(testx,reset=True).sigmoid().numpy();thresholds[name+'_reset']=thresholds[name]
            if m.kind=='liquid':scores['test'][name+'_unitdt']=m(testx,unit_time=True).sigmoid().numpy();thresholds[name+'_unitdt']=thresholds[name]
            m(testx[:1]);start=time.perf_counter()
            for i in range(min(50,len(testx))):m(testx[i:i+1])
            timing.append(dict(method=name,streams=min(50,len(testx)),seconds=time.perf_counter()-start,scope='neural controller only, batch1, 20events; excludes encoder/ledger/guard'))
    for name,score in scores['test'].items():
        skip=s['seen']|(s['eligible']&(score>=thresholds[name]));results.append(dict(method=name,threshold=thresholds[name],**summarize(skip,s['y'],s['seen'])))
        for i,ss in enumerate([t for t in streams if t['split']=='test']):predictions.append(dict(stream=ss['id'],group=ss['group'],method=name,scores=score[i].tolist(),skip=skip[i].tolist(),labels=s['y'][i].tolist(),seen=s['seen'][i].tolist()))
    results.append(dict(method='write_all',**summarize(np.zeros_like(s['y']),s['y'],s['seen'])))
    save('metrics.json',results);save('predictions.json',predictions);save('timing.json',timing);save('complete.json',dict(api_requests=0,test_streams=len(testx),test_events=int(s['y'].size),history_experiment_completed=True));print('Complete',flush=True)
if __name__=='__main__':main()
