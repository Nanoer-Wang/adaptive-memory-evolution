"""Adaptive Memory Evolution: controller-agnostic, per-item dynamical states.

m_i=(content_i,z_i), z_i=(I_i,C_i,U_i,T_i). Content is immutable within a version;
only an observed revision creates a new content version. Continuous held-input
flow and discrete observation/feedback jumps are separate causal operations.
"""
import numpy as np
import torch
from torch import nn
from src.memory.adaptive_evolution import evolve_items
from src.memory.continual_utility import batch_trajectories,KEYS

CONDITIONS=('mixture','feedback_noise','feedback_misleading','cyclic')
METHODS=('fixed_decay','frequency','adaptive_rule','adaptive_learned')


class EvolutionPolicy(nn.Module):
    """123 parameters, no recurrence or hidden network; complexity is reduced."""
    def __init__(self,rate):
        super().__init__();self.policy=nn.Linear(40,3)
        if not np.isfinite(rate) or rate<=1e-4/30:raise ValueError('Invalid initial rate')
        with torch.no_grad():
            self.policy.weight.zero_()
            self.policy.bias.copy_(torch.tensor([np.log(2),np.log(3),np.log(np.expm1(rate*30-1e-4))],dtype=torch.float32))

    def forward(self,x):
        a,w,k=self.policy(x).unbind(-1)
        return torch.sigmoid(a),2*torch.sigmoid(w),(torch.nn.functional.softplus(k)+1e-4)/30


def make_trajectory(stories,vectors,split,condition,seed):
    """Public facts plus registered schedules. Noise copies share world/query tape."""
    if condition not in CONDITIONS:raise ValueError(condition)
    rng=np.random.default_rng(seed);feedback_rng=np.random.default_rng(seed+99991)
    pools={}
    for story in stories.values():
        if story['split']==split:
            for e in story['events']:
                if e['type']=='observation':pools.setdefault(e['key'],{})[e['value']]=e['text']
    subjects=sorted(pools);n=len(subjects);groups=list(map(int,rng.permutation(3)))
    stable_group,ga,gb=groups
    stable=(stable_group*8+rng.choice(8,4,replace=False)).tolist()
    first=(ga*8+rng.choice(8,4,replace=False)).tolist()
    second=(gb*8+rng.choice(8,4,replace=False)).tolist() if condition=='cyclic' else []
    core=stable+first+second;noise=[k for k in range(KEYS) if k not in core]
    scope=np.random.default_rng(871).normal(size=((KEYS+n-1)//n,32));scope/=np.linalg.norm(scope,axis=1,keepdims=True)
    events=[];current={};versions=np.zeros(KEYS,dtype=int);seen_at={};now=0.;phase=-1
    def vector(key,text=None):
        v=np.asarray(vectors[text]) if text else np.mean([vectors[t] for t in pools[subjects[key%n]].values()],axis=0)
        v=v+.35*scope[key//n];return (v/max(np.linalg.norm(v),1e-8)).tolist()
    def observe(key,change=False):
        nonlocal now
        now+=float(rng.uniform(.5,2.));old=current.get(key);values=sorted(pools[subjects[key%n]])
        value=old['value'] if old else str(rng.choice(values))
        if old and change:value=values[(values.index(value)+1)%len(values)]
        if old is None or change:versions[key]+=1
        identity=key*10000+int(versions[key]);text=pools[subjects[key%n]][value]
        current[key]=dict(identity=identity,value=value);seen_at[key]=now
        events.append(dict(type=1,key=key,identity=identity,time=now,text=text,vector=vector(key,text),
                           conflict=int(old is not None and change),regime=phase,delay=0.,stable=key in stable,
                           target_query=False,return_probe=False,flip=False,round=-1))
    def cue(group):
        nonlocal now
        now+=float(rng.uniform(1,3))
        events.append(dict(type=2,key=group*8,identity=-1,time=now,text=f'Current work topic group {group}',
                           vector=vector(group*8),conflict=0,regime=phase,delay=0.,stable=False,
                           target_query=False,return_probe=False,flip=False,round=-1))
    def ask(key,round_index,returning=False):
        nonlocal now
        now+=float(rng.uniform(.5,1.5));u=feedback_rng.random()
        flip=bool(u<.25) if condition=='feedback_noise' else bool(phase%2==1) if condition=='feedback_misleading' else False
        events.append(dict(type=0,key=key,identity=current[key]['identity'],time=now,
                           text=f'Where is scope {key//n}, subject {subjects[key%n]}?',vector=vector(key),
                           conflict=0,regime=phase,delay=now-seen_at[key],stable=key in stable,
                           target_query=key not in stable,return_probe=returning and round_index==0,
                           flip=flip,round=round_index))
    for key in rng.permutation(core):observe(int(key))
    for phase in range(6):
        active=second if condition=='cyclic' and phase%2 else first
        returning=condition=='cyclic' and phase>=2
        cue(active[0]//8)
        if condition!='cyclic':
            # Revisions are explicitly observed. Stable records never change.
            for key in rng.choice(active,2,replace=False):observe(int(key),True)
        for key in rng.choice(noise,min(6,len(noise)),replace=False):observe(int(key),True)
        refresh=list(map(int,rng.permutation(active)))
        for round_index in range(3):
            if round_index:
                # Reminder/relearning opportunities are distinct from pre-reminder recall.
                for key in refresh[(round_index-1)*2:round_index*2]:observe(key)
                for key in rng.choice(noise,2,replace=False):observe(int(key),True)
            for key in rng.permutation(active):ask(int(key),round_index,returning)
            ask(stable[(phase+round_index)%4],round_index)
    last=events[0]['time'];demand=np.zeros(KEYS);last_query=np.full(KEYS,-1000.);conflicts=0.;focus=-1
    for e in events:
        dt=e['time']-last;key=e['key'];demand*=np.exp(-dt/30.)
        if e['type']==1:conflicts=.95*conflicts+.05*e['conflict']
        # Feedback slots are overwritten online, identically in train/evaluation.
        extra=[float(e['type']==1)-float(e['type']==2),np.log1p(dt)/5,demand[key],
               min((e['time']-last_query[key])/100,4),conflicts,.5,.5,float(focus==key//8)]
        e['features']=e['vector']+list(map(float,extra));e['dt']=float(dt)
        if e['type']==2:focus=key//8
        if e['type']==0:demand[key]+=1.;last_query[key]=e['time']
        last=e['time']
    return dict(split=split,environment=condition,seed=seed,events=events,stable_keys=stable,
                active_keys=first+second,content_source='Public split-specific bAbI facts; synthetic schedules, exact topic groups.')


def batch(rows):
    data=batch_trajectories(rows);t,b=data['kind'].shape
    for name in ['stable','target_query','return_probe','flip']:
        data[name]=torch.zeros(t,b,dtype=torch.bool)
        for j,row in enumerate(rows):
            for i,e in enumerate(row['events']):data[name][i,j]=e[name]
    return data


def rollout(data,capacity,method,rate=.03,model=None,seed=0,greedy=False,record=False):
    if method not in METHODS or capacity<1:raise ValueError('Invalid policy/capacity')
    adaptive=method.startswith('adaptive');t,b=data['kind'].shape;s=capacity+1;idx=torch.arange(b)
    ids=torch.full((b,s),-1,dtype=torch.long);keys=torch.zeros(b,s,dtype=torch.long)
    ledger=torch.full((b,KEYS),-1,dtype=torch.long)
    strength=torch.zeros(b,s);decay=torch.zeros(b,s);written=torch.zeros(b,s);frequency=torch.zeros(b,s)
    z=torch.zeros(b,s,4);focus=torch.full((b,),-1,dtype=torch.long);previous_time=data['now'][0]
    held_rate=torch.full((b,),float(rate));held_weight=torch.full((b,),1.5)
    mean_feedback=torch.full((b,),.5);last_feedback=torch.full((b,),.5)
    rng=torch.Generator().manual_seed(seed);uniforms=torch.rand(t,b,generator=rng)
    querymask=data['kind']==0;stablemask=querymask&data['stable'];nq=querymask.sum(0).clamp_min(1);ns=stablemask.sum(0).clamp_min(1)
    logps=[];rewards=[];corrects=[];stalelist=[];answeredlist=[];guardedlist=[];trace=[];retainedlist=[]
    peak=torch.zeros(b);flows=torch.zeros(b);evictions=torch.zeros(b)
    for step in range(t):
        kind=data['kind'][step];active=kind>=0;obs=kind==1;query=kind==0;cue=kind==2
        now=data['now'][step];key=data['key'][step];target=data['identity'][step]
        live=ids>=0;valid=ids==ledger.gather(1,keys)
        elapsed=torch.where(active,now-previous_time,0.)
        if adaptive:
            relevance=(keys//8==focus[:,None]).float()
            evolved=evolve_items(z,elapsed,held_rate,.5+held_weight[:,None]*relevance,valid.float())
            z=torch.where(live[:,:,None]&active[:,None,None],evolved,z);flows+=live.sum(1)*active
        previous_time=torch.where(active,now,previous_time)
        if model:
            x=data['x'][step].clone();x[:,37]=mean_feedback;x[:,38]=last_feedback
            a,w,k=model(x)
        else:a=torch.full((b,),2/3);w=torch.full((b,),1.5);k=torch.full((b,),float(rate))
        # Common retrieval: newest actually retained version for the queried key.
        # No truth/target lookup; stale fallback is graded separately from abstention.
        candidates=(keys==key[:,None])&live
        selected=ids.masked_fill(~candidates,-1).argmax(1)
        answered=candidates.any(1)&query;prediction=ids[idx,selected]
        correct=answered&(prediction==target);stale=answered&(prediction!=target)
        guarded=answered&(prediction==ledger[idx,key])
        returned=(torch.arange(s)[None,:]==selected[:,None])&answered[:,None]
        observed=torch.logical_xor(correct,data['flip'][step]).float()*answered
        oldscore=strength-decay*(now[:,None]-written).clamp_min(0)
        strength=torch.where(returned&(observed>0)[:,None],torch.logaddexp(oldscore,torch.zeros_like(oldscore)),strength)
        written=torch.where(returned&(observed>0)[:,None],now[:,None],written)
        frequency=frequency+returned.float()
        if adaptive:
            importance,confidence,usage,age=z.unbind(-1)
            confidence=torch.where(returned,.8*confidence+.2*observed[:,None],confidence)
            usage=usage+returned.float()*observed[:,None]
            age=torch.where(returned,0.,age);z=torch.stack((importance,confidence,usage,age),-1)
        # Only observed feedback, never true correctness, reaches online policy state.
        mean_feedback=torch.where(query,.9*mean_feedback+.1*observed,mean_feedback)
        last_feedback=torch.where(query,observed,last_feedback)
        same=(ids==target[:,None])&(keys==key[:,None])&live
        duplicate=same.any(1)&obs
        ledger=ledger.scatter(1,key[:,None],torch.where(obs,target,ledger[idx,key])[:,None])
        slot=torch.where(duplicate,same.long().argmax(1),(ids<0).long().argmax(1))
        mask=(torch.arange(s)[None,:]==slot[:,None])&obs[:,None]
        initial=(a*w).clamp_min(1e-8)
        inserted=torch.where(duplicate,torch.logaddexp(oldscore[idx,slot],torch.log(initial)),torch.log(initial))
        strength=torch.where(mask,inserted[:,None],strength);decay=torch.where(mask,k[:,None],decay);written=torch.where(mask,now[:,None],written)
        frequency=torch.where(mask,torch.where(duplicate,frequency[idx,slot],torch.ones(b))[:,None],frequency)
        if adaptive:
            old=z[idx,slot]
            new=torch.stack((torch.where(duplicate,(old[:,0]+initial).clamp_max(20),initial),torch.ones(b),
                             torch.where(duplicate,old[:,2],0.),torch.zeros(b)),-1)
            z=torch.where(mask[:,:,None],new[:,None,:],z)
        ids=torch.where(mask,target[:,None],ids);keys=torch.where(mask,key[:,None],keys)
        live=ids>=0;valid=ids==ledger.gather(1,keys)
        if adaptive:
            i,c,u,age=z.unbind(-1);score=torch.log(i.clamp_min(1e-8))+torch.log(c.clamp_min(.05))+torch.log1p(u)-.05*torch.log1p(age)
        elif method=='frequency':score=torch.log(frequency.clamp_min(1))
        else:score=strength-decay*(now[:,None]-written).clamp_min(0)
        score=score+torch.where(valid,0.,float(np.log(.1)))
        overflow=live.sum(1)>capacity;logits=(-score/.5).masked_fill(~live,-1e9)
        probabilities=torch.softmax(logits,-1)
        victim=logits.argmax(1) if greedy else (probabilities.detach().cumsum(1)<uniforms[step,:,None]).sum(1).clamp_max(s-1)
        logp=torch.log_softmax(logits,-1)[idx,victim]*overflow
        removed=(torch.arange(s)[None,:]==victim[:,None])&overflow[:,None]
        ids=torch.where(removed,-1,ids);z=torch.where(removed[:,:,None],0.,z)
        reward=correct.float()/nq+.25*(correct&stablemask[step]).float()/ns-.25*stale.float()/nq
        logps.append(logp);rewards.append(reward);corrects.append(correct);stalelist.append(stale);answeredlist.append(answered);guardedlist.append(guarded)
        peak=torch.maximum(peak,(ids>=0).sum(1).float());evictions+=overflow
        # Query-time target survival is an evaluation measure only.
        retainedlist.append(((ids==target[:,None])&(ids>=0)).any(1)&stablemask[step])
        focus=torch.where(cue,key//8,focus);held_rate=torch.where(active,k,held_rate);held_weight=torch.where(active,w,held_weight)
        if record:
            trace.append(dict(step=step,kind=kind.tolist(),phase=data['regime'][step].tolist(),key=key.tolist(),
                              correct=correct.tolist(),stale=stale.tolist(),answered=answered.tolist(),feedback=observed.tolist(),
                              slots=(ids>=0).sum(1).tolist(),ids=ids.tolist(),states=z.detach().tolist() if adaptive else None))
    correct=torch.stack(corrects);stale=torch.stack(stalelist);answered=torch.stack(answeredlist)
    rewards=torch.stack(rewards);returns=querymask&data['return_probe']
    metrics=dict(accuracy=(correct.sum(0)/nq).tolist(),stale_error=(stale.sum(0)/nq).tolist(),
                 stale_given_answer=(stale.sum(0)/answered.sum(0).clamp_min(1)).tolist(),answer_coverage=(answered.sum(0)/nq).tolist(),
                 guarded_coverage=(torch.stack(guardedlist).sum(0)/nq).tolist(),
                 stable_retention=((correct&stablemask).sum(0)/ns).tolist(),
                 pre_reminder_recovery=((correct&returns).sum(0)/returns.sum(0).clamp_min(1)).tolist(),
                 return_queries=returns.sum(0).tolist(),utility=rewards.sum(0).tolist(),peak_slots=peak.tolist(),
                 item_updates=flows.tolist(),evictions=evictions.tolist(),queries=nq.tolist(),stable_queries=ns.tolist())
    # Adaptation: first TWO consecutive 4-query windows >=75%, censored at12.
    # Report restricted mean lag AND success rate, never omit failed recoveries.
    lags=[];success_rates=[];end_scores=[];episode_rows=[]
    for col in range(b):
        episodes=[]
        for phase in range(6):
            mask=(data['regime'][:,col]==phase)&data['target_query'][:,col]&querymask[:,col]
            vals=correct[:,col][mask].float().tolist()
            lag=len(vals);recovered=False
            for j in range(4,len(vals)):
                if sum(vals[j-4:j])>=3 and sum(vals[j-3:j+1])>=3:
                    lag=j+1;recovered=True;break
            episodes.append(dict(phase=phase,lag_queries=lag,recovered=recovered,final_window_accuracy=float(np.mean(vals[-4:])) if vals else 0.))
        lags.append(float(np.mean([e['lag_queries'] for e in episodes])))
        success_rates.append(float(np.mean([e['recovered'] for e in episodes])))
        end_scores.append(float(np.mean([e['final_window_accuracy'] for e in episodes])))
        episode_rows.append(episodes)
    metrics.update(adaptation_lag=lags,adaptation_success=success_rates,post_shift_recovery=end_scores)
    return dict(rewards=rewards,log_probs=torch.stack(logps),metrics=metrics,episodes=episode_rows,trace=trace)
