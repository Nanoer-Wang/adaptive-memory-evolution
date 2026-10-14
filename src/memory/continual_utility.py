"""Controlled public-fact trajectories and hard-memory policy-gradient utility.

No future-use labels enter the controller. Slots hold versioned external evidence;
the evaluator's target ledger is never read by the retention policy.
"""
import heapq
import json
from pathlib import Path
import numpy as np  # Must precede torch on this project's Windows runtime.
import torch
from src.memory.consolidation import ConsolidationController

ENVIRONMENTS = ('stationary', 'slow', 'rapid')
METHODS = ('decay', 'linear', 'gru', 'liquid')
KEYS = 24
LAMBDA_RETENTION = .25
LAMBDA_FORGETTING = .05


def make_trajectory(stories, vectors, split, environment, seed, arrivals=360):
    """Same regime counts, different temporal ordering; fresh query schedule seeds.

    Stationary = shuffled balanced mixture, slow = 60-arrival dwell, rapid =15.
    Synthetic scoped identities combine public bAbI subject/value observations.
    """
    if environment not in ENVIRONMENTS or arrivals % 180:
        raise ValueError('Unknown environment or horizon not divisible by180')
    pools = {}
    for story in stories.values():
        if story['split'] != split: continue
        for event in story['events']:
            if event['type'] == 'observation':
                pools.setdefault(event['key'], {})[event['value']] = event['text']
    subjects = sorted(pools)
    if not subjects: raise ValueError('Need public subjects')
    nsubjects=len(subjects)
    rng = np.random.default_rng(seed)
    order = rng.permutation(3)
    dwell = 60 if environment == 'slow' else 15
    regimes = np.tile(np.repeat(order, dwell), arrivals // (3*dwell))
    if environment == 'stationary': rng.shuffle(regimes)
    # A fixed common random tape prevents environment-specific RNG consumption.
    tape = np.random.default_rng(seed+100000).random((arrivals, 10))
    scope_vectors = np.random.default_rng(871).normal(size=((KEYS+nsubjects-1)//nsubjects,32))
    scope_vectors /= np.linalg.norm(scope_vectors,axis=1,keepdims=True)
    current = {}; versions = np.zeros(KEYS,dtype=int); pending=[]; events=[]
    now=0.; query_serial=0

    def vector(key, text=None):
        subject=subjects[key%nsubjects]
        base=np.asarray(vectors[text]) if text else np.mean([vectors[t] for t in pools[subject].values()],axis=0)
        result=base+.35*scope_vectors[key//nsubjects]
        return (result/max(np.linalg.norm(result),1e-8)).tolist()

    def query(tick):
        nonlocal query_serial
        if not pending or pending[0][0]>tick:return
        due, serial, key, requested, requested_time=heapq.heappop(pending)
        # Evaluation-only target is the latest OBSERVED version at query time.
        target=current.get(key,{}).get('identity',-1)
        events.append(dict(type=0,key=key,identity=target,time=now+.25,
                           delay=now+.25-requested_time,scheduled_delay=due-requested,
                           text=f'What is the current location for scope {key//nsubjects}, {subjects[key%nsubjects]}?',
                           vector=vector(key),conflict=0,regime=int(regimes[min(tick,arrivals-1)])))
        query_serial+=1

    for tick,regime in enumerate(regimes):
        u=tape[tick];regime=int(regime)
        now+=float(-np.log(max(u[0],1e-6))*[.5,1.,2.][regime]+.1)
        key=regime*8+min(7,int(u[2]*8)) if u[1]<.65 else min(KEYS-1,int(u[2]*KEYS))
        subject=subjects[key%nsubjects];values=sorted(pools[subject]);old=current.get(key)
        changed=old is None or u[3]<[.08,.25,.5][regime]
        value=values[min(len(values)-1,int(u[4]*len(values)))]
        if old and changed and value==old['value']:value=values[(values.index(value)+1)%len(values)]
        if old and not changed:value=old['value']
        if changed:versions[key]+=1
        identity=key*10000+int(versions[key]);text=pools[subject][value]
        current[key]=dict(identity=identity,value=value)
        events.append(dict(type=1,key=key,identity=identity,time=now,delay=0.,
                           text=text,value=value,vector=vector(key,text),conflict=int(old is not None and changed),regime=regime))
        if u[5]<.7:
            qkey=regime*8+min(7,int(u[7]*8)) if u[6]<.8 else min(KEYS-1,int(u[7]*KEYS))
            if qkey not in current:qkey=key
            lag=int([12,40,100][regime]*(.5+u[8]))
            heapq.heappush(pending,(tick+lag,tick,qkey,tick,now))
        query(tick)
        now=max(now,events[-1]['time'])
    tick=arrivals
    while pending:
        now+=1.;query(tick);now=max(now,events[-1]['time']);tick+=1
    # Causal, policy-independent history; future targets/delays are not features.
    demand=np.zeros(KEYS);last_query=np.full(KEYS,-1000.);last_observation=np.full(KEYS,-1000.)
    last=events[0]['time'];conflict_rate=0.;gap_mean=0.
    for index,event in enumerate(events):
        dt=event['time']-last;key=event['key'];demand*=np.exp(-dt/30.)
        gap_mean+=(dt-gap_mean)/(index+1)
        conflict_rate=.95*conflict_rate+.05*event['conflict'] if event['type'] else conflict_rate
        extra=[event['type'],np.log1p(dt)/5,demand[key],min((event['time']-last_query[key])/100,4),
               conflict_rate,np.log1p(gap_mean)/5,event['conflict'],min((event['time']-last_observation[key])/100,4)]
        event['features']=event['vector']+list(map(float,extra));event['dt']=float(dt)
        if event['type']:last_observation[key]=event['time']
        else:demand[key]+=1.;last_query[key]=event['time']
        last=event['time']
    return dict(split=split,environment=environment,seed=seed,arrivals=arrivals,events=events,
                regime_counts=np.bincount(regimes,minlength=3).tolist())


def batch_trajectories(trajectories):
    b=len(trajectories);t=max(len(r['events']) for r in trajectories)
    arrays=dict(x=np.zeros((t,b,40),np.float32),dt=np.zeros((t,b),np.float32),
                now=np.zeros((t,b),np.float32),kind=np.full((t,b),-1,np.int64),
                key=np.zeros((t,b),np.int64),identity=np.full((t,b),-1,np.int64),
                delay=np.zeros((t,b),np.float32),regime=np.zeros((t,b),np.int64))
    for col,trajectory in enumerate(trajectories):
        for row,event in enumerate(trajectory['events']):
            for target,source in [('x','features'),('dt','dt'),('now','time'),('kind','type'),('key','key'),('identity','identity'),('delay','delay'),('regime','regime')]:
                arrays[target][row,col]=event[source]
        arrays['now'][len(trajectory['events']):,col]=trajectory['events'][-1]['time']
    return {k:torch.from_numpy(v) for k,v in arrays.items()}


def rollout(data, capacity, model=None, decay=1/30, seed=0, greedy=False, record=False):
    """Actual capacity-constrained discrete decisions, identical train/eval kernel.

    Stochastic eviction is the primary policy in both training and evaluation.
    REINFORCE differentiates later realized utility, not a soft-memory surrogate.
    Metadata ledger keys are bounded by24; capacity denotes evidence slots, not bytes.
    """
    if capacity<1:raise ValueError('Positive capacity required')
    t,b=data['kind'].shape;s=capacity+1;idx=torch.arange(b)
    identities=torch.full((b,s),-1,dtype=torch.long);keys=torch.zeros((b,s),dtype=torch.long)
    latest=torch.full((b,KEYS),-1,dtype=torch.long)
    strength=torch.zeros(b,s);beta=torch.zeros(b,s);written=torch.zeros(b,s)
    state=model.initial(b) if model is not None else None
    generator=torch.Generator().manual_seed(seed)
    uniforms=torch.rand(t,b,generator=generator)
    nq=(data['kind']==0).sum(0).clamp_min(1)
    long_mask=(data['kind']==0)&(data['delay']>=30)
    nl=long_mask.sum(0).clamp_min(1);ne=(data['kind']>=0).sum(0).clamp_min(1)
    rewards=[];logps=[];corrects=[];stales=[];decisions=[];betas=[];trace=[]
    for step in range(t):
        kind=data['kind'][step];active=kind>=0;obs=kind==1;query=kind==0
        now=data['now'][step];key=data['key'][step];identity=data['identity'][step]
        if model is None:
            a=torch.ones(b);w=torch.ones(b);rate=torch.full((b,),float(decay))
        else:
            policy,next_state=model.step(data['x'][step],data['dt'][step],state)
            state=tuple(torch.where(active[:,None],new,old) for new,old in zip(next_state,state))
            a,w,rate=(policy[k] for k in ('alpha','w','beta'))
        # Equal successful-task reinforcement for every policy, including decay.
        lookup_identity=torch.where(obs,identity,latest[idx,key])
        matches=(identities==lookup_identity[:,None])&(keys==key[:,None])&(identities>=0)
        success=matches.any(1)&query&(identity>=0)&(lookup_identity==identity)
        long_success=success&long_mask[step]
        current_score=strength-beta*(now[:,None]-written).clamp_min(0)
        reinforcement=torch.logaddexp(current_score,torch.zeros_like(current_score))
        strength=torch.where(matches&success[:,None],reinforcement,strength)
        written=torch.where(matches&success[:,None],now[:,None],written)
        latest=latest.scatter(1,key[:,None],torch.where(obs,identity,latest[idx,key])[:,None])
        duplicate=matches.any(1)&obs
        slot=torch.where(duplicate,matches.long().argmax(1),(identities<0).long().argmax(1))
        mask=(torch.arange(s)[None,:]==slot[:,None])&obs[:,None]
        old_score=strength[idx,slot]-beta[idx,slot]*(now-written[idx,slot]).clamp_min(0)
        new_score=torch.log((a*w).clamp_min(1e-8))
        new_score=torch.where(duplicate,torch.logaddexp(old_score,new_score),new_score)
        strength=torch.where(mask,new_score[:,None],strength)
        beta=torch.where(mask,rate[:,None],beta)
        written=torch.where(mask,now[:,None],written)
        identities=torch.where(mask,identity[:,None],identities);keys=torch.where(mask,key[:,None],keys)
        live=identities>=0;valid=identities==latest.gather(1,keys)
        score=strength-beta*(now[:,None]-written).clamp_min(0)
        # Shared observed-version protection; no future task labels are consulted.
        score=score+torch.where(valid,0.,float(np.log(.1)))
        overflow=live.sum(1)>capacity
        logits=(-score/.5).masked_fill(~live,-1e9)
        probabilities=torch.softmax(logits,dim=1)
        if greedy:victim=logits.argmax(1)
        else:victim=(probabilities.detach().cumsum(1)<uniforms[step,:,None]).sum(1).clamp_max(s-1)
        lp=torch.log_softmax(logits,dim=1)[idx,victim]*overflow.float()
        remove=(torch.arange(s)[None,:]==victim[:,None])&overflow[:,None]
        identities=torch.where(remove,-1,identities)
        stale=((identities>=0)&(identities!=latest.gather(1,keys))).sum(1).float()/capacity
        reward=success.float()/nq+LAMBDA_RETENTION*long_success.float()/nl-LAMBDA_FORGETTING*stale*active/ne
        rewards.append(reward);logps.append(lp);corrects.append(success);stales.append(stale)
        decisions.append(overflow);betas.append(rate.detach())
        if record:
            trace.append(dict(step=step,kind=kind.tolist(),key=key.tolist(),correct=success.tolist(),
                              delay=data['delay'][step].tolist(),stale_fraction=stale.tolist(),
                              slots=(identities>=0).sum(1).tolist(),eviction=overflow.tolist(),beta=rate.detach().tolist()))
    reward=torch.stack(rewards);correct=torch.stack(corrects);stale=torch.stack(stales)
    active=data['kind']>=0;queries=data['kind']==0;observation=data['kind']==1
    metrics=dict(accuracy=(correct.sum(0)/nq).tolist(),delayed_accuracy=((correct&long_mask).sum(0)/nl).tolist(),
                 stale_occupancy=((stale*active).sum(0)/ne).tolist(),utility=reward.sum(0).tolist(),
                 queries=nq.tolist(),long_queries=nl.tolist(),evictions=torch.stack(decisions).sum(0).tolist(),
                 mean_beta=((torch.stack(betas)*observation).sum(0)/observation.sum(0).clamp_min(1)).tolist())
    for regime in range(3):
        mask=queries&(data['regime']==regime)
        metrics[f'regime{regime}_accuracy']=((correct&mask).sum(0)/mask.sum(0).clamp_min(1)).tolist()
    return dict(rewards=reward,log_probs=torch.stack(logps),metrics=metrics,trace=trace)


def policy_loss(roll, baseline):
    """Undiscounted future utility minus an action-independent rule baseline."""
    future=lambda r:torch.flip(torch.cumsum(torch.flip(r,[0]),0),[0])
    advantage=(future(roll['rewards'])-future(baseline['rewards'])).detach()
    # Constant rescaling stabilizes small per-query reward units, no learned critic.
    return -(roll['log_probs']*advantage).sum(0).mean()
