"""Four-state evidence dynamics without additional neural parameters.

Facts remain external. Continuous forcing is held from PREVIOUS observed events;
the next query is never applied retroactively to the preceding interval.
"""
import numpy as np
import torch
from src.memory.consolidation import ConsolidationController
from src.memory.continual_utility import batch_trajectories,KEYS

BENCHMARKS=('inversion','reactivation','query_conditioned')
METHODS=('decay','linear','gru','liquid','adaptive','adaptive_rule','adaptive_linear','adaptive_gru')
LEARNED=('linear','gru','liquid','adaptive','adaptive_linear','adaptive_gru')


def evolve_items(states, elapsed, rate, importance_target, validity):
    """Exact held-input flow for m=(importance, confidence, usage, elapsed time).

    I'=k(target-I), C'=4k(valid-C), U'=-kU/20, T'=1.
    Usage receives successful-retrieval impulses; evidence resets C and T.
    """
    if bool((elapsed<0).any()) or not bool(torch.isfinite(elapsed).all()):raise ValueError('Invalid time')
    if bool((rate<0).any()) or not bool(torch.isfinite(rate).all()):raise ValueError('Invalid rate')
    dt=elapsed[:,None];k=rate[:,None]
    i,c,u,t=states.unbind(-1)
    i=importance_target+(i-importance_target)*torch.exp(-k*dt)
    c=validity+(c-validity)*torch.exp(-4*k*dt)
    u=u*torch.exp(-k*dt/20)
    return torch.stack((i,c,u,t+dt),-1)


def build_trajectory(stories,vectors,split,benchmark,seed):
    if benchmark not in BENCHMARKS:raise ValueError(benchmark)
    rng=np.random.default_rng(seed);pools={}
    for story in stories.values():
        if story['split']!=split:continue
        for e in story['events']:
            if e['type']=='observation':pools.setdefault(e['key'],{})[e['value']]=e['text']
    subjects=sorted(pools);n=len(subjects)
    groups=rng.permutation(3);ga,gb,gnoise=map(int,groups)
    core={g:(g*8+rng.choice(8,4,replace=False)).tolist() for g in [ga,gb]}
    noise=(gnoise*8+rng.permutation(8)).tolist()
    scope=np.random.default_rng(871).normal(size=((KEYS+n-1)//n,32));scope/=np.linalg.norm(scope,axis=1,keepdims=True)
    events=[];current={};versions=np.zeros(KEYS,dtype=int);now=0.;phase=0;observed_times={}
    def vec(key,text=None):
        base=np.asarray(vectors[text]) if text else np.mean([vectors[t] for t in pools[subjects[key%n]].values()],axis=0)
        v=base+.35*scope[key//n];return (v/max(np.linalg.norm(v),1e-8)).tolist()
    def observe(key,change=False):
        nonlocal now
        now+=float(rng.uniform(2,5));values=sorted(pools[subjects[key%n]])
        old=current.get(key);value=old['value'] if old else str(rng.choice(values))
        if old and change:value=values[(values.index(value)+1)%len(values)]
        if not old or change:versions[key]+=1
        identity=key*10000+int(versions[key]);text=pools[subjects[key%n]][value]
        current[key]=dict(identity=identity,value=value);observed_times[key]=now
        events.append(dict(type=1,key=key,identity=identity,time=now,text=text,vector=vec(key,text),
                           conflict=int(old is not None and change),delay=0.,regime=phase,probe=False,reactivation=False))
    def cue(group,returning=False):
        nonlocal now
        now+=1.
        events.append(dict(type=2,key=group*8,identity=-1,time=now,
                           text=f'Current work agenda now concerns observed scope group {group}.',
                           vector=np.mean([vec(k) for k in core[group]],axis=0).tolist(),conflict=0,delay=0.,regime=phase,
                           probe=False,reactivation=returning))
    def ask(key,probe=False,returning=False):
        nonlocal now
        now+=float(rng.uniform(1,3))
        events.append(dict(type=0,key=key,identity=current[key]['identity'],time=now,
                           text=f'Where is observed scope {key//n}, {subjects[key%n]}?',vector=vec(key),conflict=0,
                           delay=now-observed_times[key],regime=phase,probe=probe,reactivation=returning))
    # All facts are public; scope and scheduling form a synthetic mechanism test.
    for key in rng.permutation(core[ga]+core[gb]+noise[:4]):observe(int(key))
    cue(ga)
    for _ in range(4):
        for key in rng.permutation(core[ga]):ask(int(key))
    if benchmark=='inversion':targets=[gb,gb,gb]
    elif benchmark=='reactivation':targets=[gb,ga,gb,ga]
    else:targets=[int(rng.choice([ga,gb])) for _ in range(5)]
    for phase,group in enumerate(targets,1):
        returning=benchmark=='reactivation' and group==ga
        cue(group,returning)
        # Cue precedes capacity competition: only observed agenda, no future answers.
        # Current query forcing is held across unrelated experience arrivals.
        for key in rng.permutation(noise):observe(int(key),change=True)
        if benchmark=='inversion':
            # Obsolete factual versions are distinct from merely irrelevant facts.
            observe(core[ga][(phase-1)%4],change=True)
        for repeat in range(3):
            for key in rng.permutation(core[group]):ask(int(key),probe=repeat==0,returning=returning)
    # Common causal input builder. Agenda is an observed event, not a future label.
    last=events[0]['time'];focus=-1;demand=np.zeros(KEYS);last_query=np.full(KEYS,-1000.)
    last_observation=np.full(KEYS,-1000.);conflict_rate=0.;gap_mean=0.
    for index,e in enumerate(events):
        dt=e['time']-last;key=e['key'];demand*=np.exp(-dt/30.)
        gap_mean+=(dt-gap_mean)/(index+1)
        if e['type']==1:conflict_rate=.95*conflict_rate+.05*e['conflict']
        extra=[float(e['type']==1)-float(e['type']==2),np.log1p(dt)/5,demand[key],
               min((e['time']-last_query[key])/100,4),conflict_rate,np.log1p(gap_mean)/5,
               e['conflict'],float(focus==key//8)]
        e['features']=e['vector']+list(map(float,extra));e['dt']=float(dt)
        if e['type'] in [0,2]:focus=key//8
        if e['type']==0:demand[key]+=1.;last_query[key]=e['time']
        if e['type']==1:last_observation[key]=e['time']
        last=e['time']
    return dict(split=split,environment=benchmark,seed=seed,events=events,groups=dict(first=ga,second=gb,noise=gnoise),
                scope='Public facts, synthetic agenda and pressure. Agenda is reliable observed task context; not oracle future queries.')


def batch(rows):
    result=batch_trajectories(rows);t,b=result['kind'].shape
    for field in ['probe','reactivation']:
        result[field]=torch.zeros(t,b,dtype=torch.bool)
        for j,row in enumerate(rows):
            for i,e in enumerate(row['events']):result[field][i,j]=e[field]
    return result


def make_model(method,seed,rate):
    kind=method.replace('adaptive_','') if method.startswith('adaptive_') else method
    if method=='adaptive':kind='liquid'
    torch.manual_seed(seed);model=ConsolidationController(kind)
    # Same constant-policy starting point, with validation-selected initial time scale.
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.copy_(torch.tensor([np.log(2),np.log(3),np.log(np.expm1(rate*30-1e-4))],dtype=torch.float32))
    return model


def rollout(data,capacity,method,model=None,rate=.03,seed=0,greedy=False,record=False,conditioning=True):
    adaptive=method.startswith('adaptive');t,b=data['kind'].shape;s=capacity+1;idx=torch.arange(b)
    if capacity<1:raise ValueError('Positive capacity required')
    identities=torch.full((b,s),-1,dtype=torch.long);keys=torch.zeros((b,s),dtype=torch.long)
    ledger=torch.full((b,KEYS),-1,dtype=torch.long);strength=torch.zeros(b,s);decay=torch.zeros(b,s);written=torch.zeros(b,s)
    memory=torch.zeros(b,s,4);state=model.initial(b) if model else None
    focus=torch.full((b,),-1,dtype=torch.long);previous_time=data['now'][0]
    held_rate=torch.full((b,),float(rate));held_weight=torch.full((b,),1.5)
    generator=torch.Generator().manual_seed(seed);uniforms=torch.rand(t,b,generator=generator)
    query_mask=data['kind']==0;nq=query_mask.sum(0).clamp_min(1)
    probe_mask=data['probe']&query_mask;npb=probe_mask.sum(0).clamp_min(1)
    ne=(data['kind']>=0).sum(0).clamp_min(1)
    rewards=[];logps=[];corrects=[];stales=[];trace=[];flows=torch.zeros(b);evictions=torch.zeros(b);peak=torch.zeros(b)
    # Diagnostic only: whether a reactivation target survived at the preceding cue.
    cue_survivors=torch.zeros(b,KEYS,dtype=torch.bool);eligible_queries=[]
    for step in range(t):
        kind=data['kind'][step];active=kind>=0;obs=kind==1;query=kind==0;cue=kind==2
        now=data['now'][step];key=data['key'][step];identity=data['identity'][step]
        elapsed=torch.where(active,now-previous_time,0.)
        live=identities>=0;valid=identities==ledger.gather(1,keys)
        if adaptive:
            relevance=(keys//8==focus[:,None]).float() if conditioning else torch.zeros(b,s)
            evolved=evolve_items(memory,elapsed,held_rate,.5+held_weight[:,None]*relevance,valid.float())
            memory=torch.where(live[:,:,None]&active[:,None,None],evolved,memory)
            flows+=live.sum(1)*active
        previous_time=torch.where(active,now,previous_time)
        if model:
            policy,next_state=model.step(data['x'][step],data['dt'][step],state)
            state=tuple(torch.where(active[:,None],new,old) for new,old in zip(next_state,state))
            a,w,k=(policy[n] for n in ['alpha','w','beta'])
        else:a=torch.full((b,),2/3);w=torch.full((b,),1.5);k=torch.full((b,),float(rate))
        # Retrieval is determined by observed keys/versions; reference only grades it.
        lookup=torch.where(obs,identity,ledger[idx,key])
        matches=(identities==lookup[:,None])&(keys==key[:,None])&live
        success=query&matches.any(1)&(lookup==identity)&(identity>=0)
        score_now=strength-decay*(now[:,None]-written).clamp_min(0)
        strength=torch.where(matches&success[:,None],torch.logaddexp(score_now,torch.zeros_like(score_now)),strength)
        written=torch.where(matches&success[:,None],now[:,None],written)
        if adaptive:
            importance,confidence,usage,age=memory.unbind(-1)
            usage=usage+(matches&success[:,None]).float()
            age=torch.where(matches&success[:,None],0.,age)
            memory=torch.stack((importance,confidence,usage,age),-1)
        ledger=ledger.scatter(1,key[:,None],torch.where(obs,identity,ledger[idx,key])[:,None])
        duplicate=matches.any(1)&obs
        slot=torch.where(duplicate,matches.long().argmax(1),(identities<0).long().argmax(1))
        mask=(torch.arange(s)[None,:]==slot[:,None])&obs[:,None]
        initial=(a*w).clamp_min(1e-8)
        new_strength=torch.where(duplicate,torch.logaddexp(score_now[idx,slot],torch.log(initial)),torch.log(initial))
        strength=torch.where(mask,new_strength[:,None],strength);decay=torch.where(mask,k[:,None],decay)
        written=torch.where(mask,now[:,None],written)
        if adaptive:
            old=memory[idx,slot]
            new_i=torch.where(duplicate,(old[:,0]+initial).clamp_max(20),initial)
            new_u=torch.where(duplicate,old[:,2],0.)
            new=torch.stack((new_i,torch.ones_like(a),new_u,torch.zeros_like(a)),-1)
            memory=torch.where(mask[:,:,None],new[:,None,:],memory)
        identities=torch.where(mask,identity[:,None],identities);keys=torch.where(mask,key[:,None],keys)
        live=identities>=0;valid=identities==ledger.gather(1,keys)
        if adaptive:
            i,c,u,age=memory.unbind(-1)
            scores=torch.log(i.clamp_min(1e-8))+torch.log(c.clamp_min(.05))+torch.log1p(u)-.05*torch.log1p(age)
        else:scores=strength-decay*(now[:,None]-written).clamp_min(0)
        scores=scores+torch.where(valid,0.,float(np.log(.1)))
        overflow=live.sum(1)>capacity;logits=(-scores/.5).masked_fill(~live,-1e9)
        prob=torch.softmax(logits,-1)
        victim=logits.argmax(1) if greedy else (prob.detach().cumsum(1)<uniforms[step,:,None]).sum(1).clamp_max(s-1)
        lp=torch.log_softmax(logits,-1)[idx,victim]*overflow
        remove=(torch.arange(s)[None,:]==victim[:,None])&overflow[:,None]
        identities=torch.where(remove,-1,identities);memory=torch.where(remove[:,:,None],0.,memory)
        stale=((identities>=0)&(identities!=ledger.gather(1,keys))).sum(1).float()/capacity
        reward=success.float()/nq+.25*(success&probe_mask[step]).float()/npb-.05*stale*active/ne
        rewards.append(reward);logps.append(lp);corrects.append(success);stales.append(stale)
        evictions+=overflow;peak=torch.maximum(peak,(identities>=0).sum(1).float())
        if bool(cue.any()):
            present=((identities[:,:,None]==ledger[:,None,:])&(identities[:,:,None]>=0)).any(1)
            cue_survivors=torch.where(cue[:,None],present,cue_survivors)
        eligible_queries.append(cue_survivors[idx,key]&query&data['reactivation'][step])
        focus=torch.where(query|cue,key//8,focus)
        held_rate=torch.where(active,k,held_rate);held_weight=torch.where(active,w,held_weight)
        if record:
            trace.append(dict(step=step,kind=kind.tolist(),key=key.tolist(),identity=identities.tolist(),
                              correct=success.tolist(),slots=(identities>=0).sum(1).tolist(),
                              states=memory.detach().tolist() if adaptive else None,focus=focus.tolist(),beta=k.detach().tolist()))
    correct=torch.stack(corrects);rewards=torch.stack(rewards);stale=torch.stack(stales)
    returning=query_mask&data['reactivation'];eligible=torch.stack(eligible_queries)
    metrics=dict(accuracy=(correct.sum(0)/nq).tolist(),probe_accuracy=((correct&probe_mask).sum(0)/npb).tolist(),
                 utility=rewards.sum(0).tolist(),stale_occupancy=((stale*(data['kind']>=0)).sum(0)/ne).tolist(),
                 reactivation_accuracy=((correct&returning).sum(0)/returning.sum(0).clamp_min(1)).tolist(),
                 reactivation_conditional=((correct&eligible).sum(0)/eligible.sum(0).clamp_min(1)).tolist(),
                 reactivation_eligible=eligible.sum(0).tolist(),queries=nq.tolist(),probes=npb.tolist(),
                 evictions=evictions.tolist(),peak_slots=peak.tolist(),item_flow_updates=flows.tolist())
    return dict(rewards=rewards,log_probs=torch.stack(logps),metrics=metrics,trace=trace)
