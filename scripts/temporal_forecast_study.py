"""Chronological UCI PM2.5 mechanism diagnostic, not a language-memory result."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,time,copy
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from src.models.temporal_forecast import TemporalForecast
from src.provenance import snapshot_source,runtime_info
OUT=ROOT/'runs/temporal-forecast-v1'
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def metrics(p,y):return dict(mae=float(np.abs(p-y).mean()),rmse=float(np.sqrt(((p-y)**2).mean())))
def main():
    if OUT.exists():raise SystemExit('Existing run preserved')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    df=pd.read_csv(ROOT/'data/beijing_pm25/data.csv');dates=pd.to_datetime(df[['year','month','day','hour']]);assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
    cols=['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir'];raw=df[cols].to_numpy(float);winds=np.stack([(df.cbwd==c).to_numpy(float) for c in ['NE','NW','SE','cv']],1);raw=np.c_[raw,winds]
    trainmask=df.year.to_numpy()<=2012;mean=np.nanmean(raw[trainmask],0);scale=np.nanstd(raw[trainmask],0);scale=np.maximum(scale,1e-6);data=((raw-mean)/scale).astype(np.float32);validraw=np.isfinite(raw).all(1)
    save('normalization.json',dict(mean=mean.tolist(),scale=scale.tolist(),features=cols+['NE','NW','SE','cv'],fit_years=[2010,2011,2012]))
    pools={s:[] for s in ['train','validation','test']};records=[]
    for end in range(23,len(df)-6,6):
        target=end+6;year=int(df.year.iloc[target]);split='train' if year<=2012 else 'validation' if year==2013 else 'test'
        # Complete input and target interval contained in a split; no boundary leakage.
        sy=int(df.year.iloc[end-23]);source_split='train' if sy<=2012 else 'validation' if sy==2013 else 'test'
        if source_split!=split or not np.isfinite(raw[target,0]):continue
        for condition,keep in [('observed',1.),('half_missing',.5)]:
            ids=np.arange(end-23,end+1);mask=validraw[ids].copy()
            if keep<1:
                rng=np.random.default_rng(int(hashlib.sha256(f'pm25:{end}'.encode()).hexdigest()[:8],16));mask&=rng.random(24)<keep
            if mask.sum()<2:continue
            x=np.where(mask[:,None],data[ids],0);delta=np.zeros(24,np.float32);last=end-24
            for j,i in enumerate(ids):
                if mask[j]:delta[j]=i-last;last=i
            row=dict(index=len(records),end=end,target=target,split=split,condition=condition,target_time=str(dates.iloc[target]),gap=float(target-last),observations=int(mask.sum()))
            pools[split].append((x,mask,delta,float(target-last),float(data[target,0]),len(records)));records.append(row)
    sets={s:dict(x=np.stack([r[0] for r in rr]),valid=np.stack([r[1] for r in rr]),dt=np.stack([r[2] for r in rr]),gap=np.asarray([r[3] for r in rr],np.float32),y=np.asarray([r[4] for r in rr],np.float32),ids=np.asarray([r[5] for r in rr])) for s,rr in pools.items()}
    save('examples.json',records);save('split_audit.json',{s:dict(n=len(d['y']),first=records[int(d['ids'][0])]['target_time'],last=records[int(d['ids'][-1])]['target_time']) for s,d in sets.items()})
    sig,archive=snapshot_source();save('protocol.json',dict(source=sig,source_archive=archive,runtime=runtime_info(),data=json.loads((ROOT/'data/beijing_pm25/provenance.json').read_text()),primary='liquid',
        task='24h past window predicts point PM2.5 six hours after window end; stride6h. Real timestamps and measurements. Additional 50% deterministic random observation thinning is synthetic. No future weather inputs.',
        split='2010-12 train; 2013 validation; 2014 test. All input timestamps and targets contained in same split. Normalization train years only. Naturally incomplete rows unobserved, never future-filled.',
        models='Last-value persistence; time EMA tau1,3,6,12 selected validation; full 24x11 masked history Ridge alpha .1,1,10,100; last-MLP, GRU16, input-only flow16, coupled liquid16. Same input features/dt/gap/residual readout. Parameter differences reported.',
        learning='3 seeds11,23,42; Adam .003,12 epochs,batch256, squared error; lowest pooled validation MSE checkpoint. Fixed training budget, not extensive hyperparameter search.',
        diagnostics='State reset and unit-time ablations on frozen liquid; both observed and thinned conditions reported. Whole-week paired bootstrap across seed-mean test predictions; overlapping windows and one site limit generalization.',
        limit='Numeric recurrent-state diagnostic only. Does not prove language memory advantage, low-risk semantic deletion, efficient storage or CCF-A novelty. No mechanism selected on test. CPU controller timing excludes data loading; causal input processing included.'))
    def last(d):return np.asarray([r[np.flatnonzero(v)[-1]] for r,v in zip(d['x'],d['valid'])])
    def design(d):return np.c_[d['x'].reshape(len(d['x']),-1),d['valid'].astype(float),d['gap']/6]
    train,val,test=[sets[s] for s in ['train','validation','test']];baseline={};frozen={}
    baseline['persistence']=last(test)[:,0]
    def ema(d,tau):
        h=np.zeros(len(d['x']));initialized=np.zeros(len(h),bool)
        for i in range(24):
            a=np.exp(-d['dt'][:,i]/tau);z=np.where(initialized,a*h+(1-a)*d['x'][:,i,0],d['x'][:,i,0]);h=np.where(d['valid'][:,i],z,h);initialized|=d['valid'][:,i]
        return h
    tau=min([1,3,6,12],key=lambda t:np.mean((ema(val,t)-val['y'])**2));baseline['ema']=ema(test,tau);frozen['ema_tau']=tau
    choices=[]
    for alpha in [.1,1,10,100]:
        m=Ridge(alpha=alpha).fit(design(train),train['y']);choices.append((np.mean((m.predict(design(val))-val['y'])**2),alpha,m))
    _,alpha,ridge=min(choices,key=lambda r:r[0]);baseline['ridge_history']=ridge.predict(design(test));frozen['ridge_alpha']=alpha;save('ridge.json',dict(coef=ridge.coef_.tolist(),intercept=float(ridge.intercept_),alpha=alpha))
    tensors={s:{k:torch.from_numpy(d[k]) for k in ['x','valid','dt','gap','y']} for s,d in sets.items()}
    def predict(m,s,**kwargs):
        d=tensors[s];outputs=[]
        with torch.no_grad():
            for start in range(0,len(d['x']),512):outputs.append(m(**{k:d[k][start:start+512] for k in ['x','valid','dt','gap']},**kwargs))
        return torch.cat(outputs)
    models={};logs=[];cost=[]
    for kind in ['last_mlp','gru','input_flow','liquid']:
        for seed in [11,23,42]:
            torch.manual_seed(seed);m=TemporalForecast(kind);opt=torch.optim.Adam(m.parameters(),lr=.003);best=float('inf');state=None;start=time.perf_counter();tr=tensors['train']
            for epoch in range(12):
                m.train()
                for ids in torch.randperm(len(tr['x'])).split(256):
                    pr=m(**{k:tr[k][ids] for k in ['x','valid','dt','gap']});loss=((pr-tr['y'][ids])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                m.eval();vl=float(((predict(m,'validation')-tensors['validation']['y'])**2).mean());logs.append(dict(kind=kind,seed=seed,epoch=epoch,validation_mse=vl))
                if vl<best:best=vl;state=copy.deepcopy(m.state_dict())
            m.load_state_dict(state);name=f'{kind}-{seed}';torch.save(state,OUT/(name+'.pt'));models[name]=m;cost.append(dict(method=name,parameters=sum(p.numel() for p in m.parameters()),hidden_state_bytes=0 if kind=='last_mlp' else 64,training_seconds=time.perf_counter()-start));print(name,'frozen',flush=True)
    save('training.json',logs);save('frozen.json',dict(baselines=frozen,weights={k:hashlib.sha256((OUT/(k+'.pt')).read_bytes()).hexdigest() for k in models}));save('parameters.json',cost)
    # No test evaluation before all checkpoints and hyperparameters frozen.
    predictions={k:np.asarray(v)*scale[0]+mean[0] for k,v in baseline.items()};timing=[]
    for name,m in models.items():
        predictions[name]=predict(m,'test').numpy()*scale[0]+mean[0]
        if name.startswith('liquid'):
            predictions[name+'_reset']=predict(m,'test',reset=True).numpy()*scale[0]+mean[0]
            predictions[name+'_unitdt']=predict(m,'test',unit_time=True).numpy()*scale[0]+mean[0]
        sample={k:v[:1] for k,v in tensors['test'].items() if k!='y'}
        with torch.no_grad():
            m(**sample);t=time.perf_counter()
            for _ in range(100):m(**sample)
        timing.append(dict(method=name,microseconds_per_24h_window=(time.perf_counter()-t)*1e4))
    truth=test['y']*scale[0]+mean[0];conditions=np.asarray([records[int(i)]['condition'] for i in test['ids']]);results=[]
    for name,p in predictions.items():
        for condition in ['observed','half_missing']:
            ix=conditions==condition;results.append(dict(method=name,condition=condition,n=int(ix.sum()),**metrics(p[ix],truth[ix])))
    np.savez_compressed(OUT/'predictions.npz',truth=truth,ids=test['ids'],**predictions);save('metrics.json',results);save('timing.json',timing);save('complete.json',dict(api_requests=0,models=len(models),test_samples=len(truth)));print('Complete',flush=True)
if __name__=='__main__':main()
