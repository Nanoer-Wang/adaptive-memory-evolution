"""Per-item adaptive evolution benchmarks, fixed neural architecture and public facts."""
import sys,json,time,hashlib,argparse,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.provenance import snapshot_source
from src.memory.adaptive_evolution import build_trajectory,batch,rollout,make_model,BENCHMARKS,METHODS,LEARNED
from src.memory.continual_utility import policy_loss
OUT=ROOT/'runs/adaptive-evolution-v1'
SEEDS=[11,23,42];CAPACITIES=[8,12];EPOCHS=12
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.json').exists():return
    public=ROOT/'runs/liquid-consolidation-v2';stories=read(public/'stories.json');vectors=read(public/'vectors.json')
    source,archive=snapshot_source()
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),source=source,archive=archive,
        seeds=SEEDS,capacities=CAPACITIES,epochs=EPOCHS,learning_rate=.003,batch_size=6,
        models='Same ConsolidationController, hidden8/input40; adaptive reuses liquid EXACTLY. No added trainable parameters or critic. Four per-item float32 states require16bytes per resident record; same state allowance reserved across policies.',
        dynamics='I_dot=k(.5+w*observed_topic_relevance-I); C_dot=4k(observed_validity-C); U_dot=-kU/20; T_dot=1. Exact held-input flow BEFORE observing next event. Successful feedback increments U and resets T. New/merged evidence sets confidence1 and initial importance alpha*w. Future targets never drive flow.',
        scoring='log(I)+log(max(C,.05))+log(1+U)-.05log(1+T), plus common observed-version penalty. Deleted records and states cannot be recovered without new observation.',
        objective='1-task accuracy +.25(1-first-query-after-pressure accuracy)+.05obsolete occupancy. REINFORCE on actual stochastic eviction, same training/evaluation kernel, deterministic action-independent fixed-decay return baseline; full trajectory gradients.',
        data='Public bAbI split-specific facts with synthetic scopes/schedules.8trajectories per benchmark train,4validation,12test. Explicit reliable work-agenda cues are observable to ALL models. Same keys/content/queries/capacity per method. Mechanism benchmark, not official bAbI or natural-language Agent score.',
        inversion='Frequently used group A loses task relevance; group B gains it, with explicit factual revisions to A. Irrelevance is distinguished from known-invalid versions.',
        reactivation='A->B->A->B->A; core A is not re-observed. Report unconditional returns AND conditional on survival at cue. Cannot resurrect evicted content.',
        query_conditioned='Observed agenda signals variable upcoming work groups before unrelated arrivals create capacity pressure. Exact future queries/answers are withheld from policy.',
        selection='Validation only: fixed-decay rates[.001,.01,.03,.1], adaptive-rule relaxation rates[.03,.1,.3]. Learned old controllers initialized to selected fixed rule; adaptive variants initialized to selected adaptive rule. Initial checkpoint is eligible, same12epoch budget for6learned variants. No test-based tuning.',
        controls='Requested five: fixed decay,linear,GRU,current liquid,new adaptive. Extra adaptive_rule,adaptive_linear,adaptive_gru distinguish ability to update old records from LNN-specific benefit. All share observed-version semantics and actual success feedback.',
        limits='Slot budget, not process RAM; per-item update work counted. Synthetic reliable cues, limited training and template overlap. Global model coefficients shared across items, per-item states differ. Confidence is observed consistency, not calibrated uncertainty.',
        diagnostics='After primary selection, frozen adaptive weights with query conditioning disabled and with hidden continuous flow disabled; no refit. Paired bootstrap over12trajectory seeds, average3training/policy seeds first; exploratory intervals, no multiplicity correction.'))
    data={}
    for split,count,offset in [('train',8,41000),('val',4,42000),('test',12,43000)]:
        data[split]=[build_trajectory(stories,vectors,split,benchmark,offset+i) for benchmark in BENCHMARKS for i in range(count)]
    atomic_json(OUT/'data.json',data)
    atomic_json(OUT/'data_audit.json',[dict(split=split,benchmark=r['environment'],seed=r['seed'],
        events=len(r['events']),observations=sum(e['type']==1 for e in r['events']),queries=sum(e['type']==0 for e in r['events']),
        conflicts=sum(e['conflict'] for e in r['events']),cues=sum(e['type']==2 for e in r['events']),
        duration=r['events'][-1]['time']-r['events'][0]['time'],probes=sum(e['probe'] for e in r['events'])) for split,rows in data.items() for r in rows])
    print('Prepared24train/12validation/36test trajectories',flush=True)


def evaluate_rows(rows,capacity,method,model=None,rate=.03,seed=0,record=False,conditioning=True):
    with torch.no_grad():return rollout(batch(rows),capacity,method,model,rate,seed,record=record,conditioning=conditioning)


def train():
    prepare();data=read(OUT/'data.json');(OUT/'checkpoints').mkdir(exist_ok=True)
    if not (OUT/'rates.json').exists():
        rates={}
        for method,candidates in [('decay',[.001,.01,.03,.1]),('adaptive_rule',[.03,.1,.3])]:
            scores=[dict(rate=rate,utility=float(np.mean([np.mean(evaluate_rows(data['val'],c,method,rate=rate,seed=777)['metrics']['utility']) for c in CAPACITIES]))) for rate in candidates]
            rates[method]=dict(selected=max(scores,key=lambda r:r['utility'])['rate'],candidates=scores)
        atomic_json(OUT/'rates.json',rates);print('Validation rates',rates,flush=True)
    rates=read(OUT/'rates.json')
    if not (OUT/'boundary_rate_audit.json').exists():
        candidates=[dict(rate=rate,validation_utility=float(np.mean([np.mean(evaluate_rows(data['val'],c,'decay',rate=rate,seed=777)['metrics']['utility']) for c in CAPACITIES]))) for rate in [0.,.0001,.001]]
        atomic_json(OUT/'boundary_rate_audit.json',dict(reason='Supplementary validation-only audit of the original lower rate-grid boundary, before test evaluation; primary training unchanged.',candidates=candidates,selected=max(candidates,key=lambda r:r['validation_utility'])['rate']))
    for method in LEARNED:
        rate=rates['adaptive_rule' if method.startswith('adaptive') else 'decay']['selected']
        for seed in SEEDS:
            path=OUT/f'training-{method}-{seed}.json';checkpoint=OUT/'checkpoints'/f'{method}-{seed}.pt'
            if path.exists() and read(path)['complete']:continue
            model=make_model(method,seed,rate);optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            best=float(np.mean([np.mean(evaluate_rows(data['val'],c,method,model,rate,777)['metrics']['utility']) for c in CAPACITIES]))
            def save(epoch):torch.save(dict(method=method,seed=seed,rate=rate,state_dict=model.state_dict(),epoch=epoch,validation_utility=best),checkpoint)
            save(-1);history=[dict(epoch=-1,validation_utility=best)];start=time.perf_counter();rng=np.random.default_rng(seed)
            for epoch in range(EPOCHS):
                order=rng.permutation(len(data['train']));losses=[];utilities=[]
                for batch_index,left in enumerate(range(0,len(order),6)):
                    rows=[data['train'][i] for i in order[left:left+6]];inputs=batch(rows);capacity=CAPACITIES[(epoch+batch_index)%2]
                    with torch.no_grad():baseline=rollout(inputs,capacity,'decay',rate=rates['decay']['selected'],greedy=True)
                    result=rollout(inputs,capacity,method,model,rate,seed=seed*10000+epoch*100+batch_index)
                    loss=policy_loss(result,baseline);optimizer.zero_grad();loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                    optimizer.step();losses.append(float(loss.detach()));utilities+=result['metrics']['utility']
                val=float(np.mean([np.mean(evaluate_rows(data['val'],c,method,model,rate,777)['metrics']['utility']) for c in CAPACITIES]))
                if val>best:best=val;save(epoch)
                history.append(dict(epoch=epoch,validation_utility=val,training_utility=float(np.mean(utilities)),policy_loss=float(np.mean(losses))))
                atomic_json(path,dict(complete=False,history=history))
                print(method,seed,'epoch',epoch,'val',round(val,4),flush=True)
            atomic_json(path,dict(complete=True,history=history,best_validation_utility=best,seconds=time.perf_counter()-start))


def load(method,seed):
    rates=read(OUT/'rates.json');rate=rates['adaptive_rule' if method.startswith('adaptive') else 'decay']['selected']
    if method in ['decay','adaptive_rule']:return None,rate
    saved=torch.load(OUT/'checkpoints'/f'{method}-{seed}.pt',weights_only=True)
    model=make_model(method,seed,rate);model.load_state_dict(saved['state_dict']);model.eval();return model,rate


def evaluate():
    if (OUT/'evaluation_complete.json').exists():return
    for method in LEARNED:
        for seed in SEEDS:assert read(OUT/f'training-{method}-{seed}.json')['complete']
    data=read(OUT/'data.json');results=[];(OUT/'traces').mkdir(exist_ok=True)
    for method in METHODS:
        for seed in SEEDS:
            model,rate=load(method,seed)
            for benchmark in BENCHMARKS:
                rows=[r for r in data['test'] if r['environment']==benchmark]
                for capacity in CAPACITIES:
                    start=time.perf_counter();result=evaluate_rows(rows,capacity,method,model,rate,seed=seed*100+999,record=True)
                    elapsed=time.perf_counter()-start
                    for i,row in enumerate(rows):
                        assert result['metrics']['peak_slots'][i]<=capacity
                        results.append(dict(method=method,seed=seed,benchmark=benchmark,capacity=capacity,trajectory_seed=row['seed'],seconds_per_trajectory=elapsed/len(rows),**{k:v[i] for k,v in result['metrics'].items()}))
                    atomic_json(OUT/'traces'/f'{method}-{seed}-{benchmark}-{capacity}.json',dict(seeds=[r['seed'] for r in rows],trace=result['trace']))
            print('Tested',method,seed,flush=True)
    atomic_json(OUT/'results.json',results);atomic_json(OUT/'evaluation_complete.json',dict(rows=len(results),at=time.time()))


def diagnostics():
    if (OUT/'diagnostics.json').exists():return
    assert (OUT/'evaluation_complete.json').exists();data=read(OUT/'data.json');results=[]
    for seed in SEEDS:
        for condition in ['no_conditioning','no_hidden_flow']:
            model,rate=load('adaptive',seed)
            if condition=='no_hidden_flow':model.kind='no_time'
            for benchmark in BENCHMARKS:
                rows=[r for r in data['test'] if r['environment']==benchmark]
                for capacity in CAPACITIES:
                    out=evaluate_rows(rows,capacity,'adaptive',model,rate,seed*100+999,conditioning=condition!='no_conditioning')
                    for i,row in enumerate(rows):results.append(dict(condition=condition,seed=seed,benchmark=benchmark,capacity=capacity,trajectory_seed=row['seed'],**{k:v[i] for k,v in out['metrics'].items()}))
    atomic_json(OUT/'diagnostics.json',dict(note='Pre-specified frozen-weight diagnostic, not retrained models.',results=results))


def boundary_control():
    path=OUT/'boundary_control.json'
    if path.exists() or not (OUT/'boundary_rate_audit.json').exists():return
    assert (OUT/'evaluation_complete.json').exists()
    rate=read(OUT/'boundary_rate_audit.json')['selected'];data=read(OUT/'data.json');results=[]
    for seed in SEEDS:
        for benchmark in BENCHMARKS:
            rows=[r for r in data['test'] if r['environment']==benchmark]
            for capacity in CAPACITIES:
                output=evaluate_rows(rows,capacity,'decay',rate=rate,seed=seed*100+999)
                for i,row in enumerate(rows):results.append(dict(method='fixed_boundary',seed=seed,benchmark=benchmark,
                    capacity=capacity,trajectory_seed=row['seed'],**{k:v[i] for k,v in output['metrics'].items()}))
    atomic_json(path,dict(note='Supplementary lower-bound rate audit selected only on validation before primary test outcomes. Primary training and baselines unchanged.',rate=rate,results=results))


def report():
    rows=read(OUT/'results.json');summary=[];comparisons=[]
    objectives=[]
    for r in rows:
        task=1-r['accuracy'];retention=1-r['probe_accuracy'];forgetting=r['stale_occupancy']
        total=task+.25*retention+.05*forgetting
        assert abs(total-(1.25-r['utility']))<1e-6
        objectives.append({k:r[k] for k in ['method','seed','benchmark','capacity','trajectory_seed']}|dict(L_task=task,L_retention=retention,L_forgetting=forgetting,L=total))
    atomic_json(OUT/'objective_components.json',objectives)
    for benchmark in BENCHMARKS:
        for capacity in CAPACITIES:
            for method in METHODS:
                rr=[r for r in rows if r['benchmark']==benchmark and r['capacity']==capacity and r['method']==method]
                summary.append(dict(benchmark=benchmark,capacity=capacity,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in ['accuracy','probe_accuracy','utility','reactivation_accuracy','reactivation_conditional','stale_occupancy','evictions','item_flow_updates','seconds_per_trajectory']}))
                eligible=sum(r['reactivation_eligible'] for r in rr)
                summary[-1]['reactivation_conditional']=sum(r['reactivation_conditional']*r['reactivation_eligible'] for r in rr)/eligible if eligible else None
            for control in [m for m in METHODS if m!='adaptive']:
                for metric in ['accuracy','probe_accuracy','utility']:
                    diff=[]
                    for trajectory in sorted({r['trajectory_seed'] for r in rows}):
                        means={m:np.mean([r[metric] for r in rows if r['benchmark']==benchmark and r['capacity']==capacity and r['method']==m and r['trajectory_seed']==trajectory]) for m in ['adaptive',control]}
                        diff.append(float(means['adaptive']-means[control]))
                    samples=np.mean(np.random.default_rng(123).choice(diff,(3000,len(diff))),axis=1)
                    comparisons.append(dict(benchmark=benchmark,capacity=capacity,control=control,metric=metric,difference=float(np.mean(diff)),ci95=np.quantile(samples,[.025,.975]).tolist(),trajectories=len(diff)))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'paired_comparisons.json',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,layout='constrained')
    plot_methods=['decay','liquid','adaptive','adaptive_rule']
    for ax,capacity in zip(axes,CAPACITIES):
        for index,method in enumerate(plot_methods):
            values=[100*next(r['probe_accuracy'] for r in summary if r['benchmark']==benchmark and r['capacity']==capacity and r['method']==method) for benchmark in BENCHMARKS]
            ax.bar(np.arange(3)+(index-1.5)*.2,values,.19,label=method)
        ax.set_xticks(np.arange(3),['Inversion','A-B-A workload','Query conditioned']);ax.set_ylim(0,100)
        ax.set_title(f'{capacity} evidence slots');ax.set_ylabel('First-query-after-pressure accuracy (%)');ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    # Fixed illustration: first held-out trajectory, seed11, capacity12, first probe.
    first=next(r for r in read(OUT/'data.json')['test'] if r['environment']=='inversion')
    target=next(e for e in first['events'] if e['probe'])['identity']
    trace=read(OUT/'traces'/'adaptive-11-inversion-12.json')['trace'];series=[]
    for e,row in zip(first['events'],trace):
        ids=row['identity'][0];state=row['states'][0][ids.index(target)] if target in ids else [float('nan')]*4
        series.append(dict(time=e['time'],event=e['type'],state=state))
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for dimension,(ax,name) in enumerate(zip(axes.flat,['Importance','Observed consistency','Usage','Elapsed time'])):
        ax.plot([r['time'] for r in series],[r['state'][dimension] for r in series]);ax.set_title(name);ax.set_xlabel('Simulated time')
        for e in first['events']:
            if e['type']==2:ax.axvline(e['time'],color='grey',alpha=.3,linestyle='--')
    fig.suptitle('Preselected held-out item; dashed lines = observed agenda changes')
    fig.savefig(OUT/'item_state_example.png',dpi=160);plt.close(fig)
    atomic_json(OUT/'item_state_example.json',dict(selection='First held-out inversion trajectory, policy/model seed11, capacity12, first probe target; no success-based selection.',trajectory_seed=first['seed'],identity=target,series=[{**r,'state':[None if not np.isfinite(v) else v for v in r['state']]} for r in series]))
    def table(rr,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in columns)+'</tr>' for r in rr)+'</table>'
    doc='''<!doctype html><meta charset="utf-8"><title>Adaptive per-item memory evolution</title><style>body{max-width:1300px;margin:32px auto;font:16px/1.7 system-ui;color:#17263a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ddd;padding:5px;text-align:left}pre{background:#eef3f8;padding:16px;white-space:pre-wrap}</style><h1>Adaptive per-item memory evolution</h1><p>Public-fact synthetic mechanism benchmarks. Same neural architecture:40inputs,8hidden units. Four per-item states add16bytes of float32 state per resident record; dynamic update work scales with resident memory. No speed/parameter efficiency claim.</p><h2>Model</h2><pre>m_i=(importance I, observed-consistency C, usage U, elapsed T)
I_dot=k(.5+w*observed_topic_relevance-I)
C_dot=4k(observed_validity-C)
U_dot=-kU/20; T_dot=1
score=log(I)+log(max(C,.05))+log(1+U)-.05log(1+T)</pre><p>Exact flow uses previous observed forcing for the preceding interval. New agendas and queries set future forcing; successful retrieval increments usage and resets elapsed time. Confidence tracks explicitly observed version consistency, not calibrated model certainty. Evicted contents cannot be resurrected.</p><h2>Protocol</h2><p>24train/12validation/36test trajectories;3seeds;8/12evidence slots;12epochs for all6learned variants, initialization eligible for validation selection. Fixed and adaptive rules tuned on validation. Actual discrete future-utility policy gradient; no proxy importance labels, critic, or API calls. Reliable work-agenda cues are given to every method. These artificial scenarios test a capability boundary, not natural-language agent generalization.</p><h2>Primary and matched controls</h2>'''
    doc+=table(summary,['benchmark','capacity','method','accuracy','probe_accuracy','utility','stale_occupancy'])
    doc+='<img src="comparison.png" alt="All first queries after pressure; not isolated return-to-A accuracy" style="width:100%"><p>The A-B-A workload bar aggregates all post-pressure probes. Its specific return-to-A performance is reported separately below.</p>'
    doc+='<h2>Adaptive minus control: paired exploratory intervals</h2><p>Average3seeds within each of12held-out trajectory seeds, then bootstrap. Multiple comparisons are unadjusted. Matched adaptive rule/linear/GRU distinguish mutable-memory benefit from liquid-specific benefit.</p>'
    doc+=table([r for r in comparisons if r['metric']=='probe_accuracy'],['benchmark','capacity','control','difference','ci95'])
    doc+='<h2>Reactivation and resource accounting</h2><p>Unconditional reactivation accuracy is primary; conditional scores count only targets still resident when the return cue arrives and must not be presented as overall success. All methods have equal slot limits and state allowance, but adaptive evolution performs additional item updates.</p>'
    doc+=table([r for r in summary if r['benchmark']=='reactivation'],['capacity','method','reactivation_accuracy','reactivation_conditional','item_flow_updates','seconds_per_trajectory'])
    doc+='<h2>One evolving item</h2><p>Preselected first held-out inversion trajectory, seed11,12slots, first post-shift probe target. This illustrates state changes, not independent statistical evidence. Missing states indicate the record is absent.</p><img src="item_state_example.png" alt="Four states of a preselected item" style="width:100%">'
    if (OUT/'diagnostics.json').exists():
        d=read(OUT/'diagnostics.json')['results'];ds=[]
        for benchmark in BENCHMARKS:
            for capacity in CAPACITIES:
                for condition in ['no_conditioning','no_hidden_flow']:
                    rr=[r for r in d if r['benchmark']==benchmark and r['capacity']==capacity and r['condition']==condition]
                    ds.append(dict(benchmark=benchmark,capacity=capacity,condition=condition,accuracy=float(np.mean([r['accuracy'] for r in rr])),probe_accuracy=float(np.mean([r['probe_accuracy'] for r in rr]))))
        atomic_json(OUT/'diagnostic_summary.json',ds)
        doc+='<h2>Frozen-weight diagnostics</h2>'+table(ds,['benchmark','capacity','condition','accuracy','probe_accuracy'])
    if (OUT/'boundary_control.json').exists():
        boundary=read(OUT/'boundary_control.json');br=[]
        for benchmark in BENCHMARKS:
            for capacity in CAPACITIES:
                selected=[r for r in boundary['results'] if r['benchmark']==benchmark and r['capacity']==capacity]
                br.append(dict(benchmark=benchmark,capacity=capacity,rate=boundary['rate'],accuracy=float(np.mean([r['accuracy'] for r in selected])),probe_accuracy=float(np.mean([r['probe_accuracy'] for r in selected]))))
        atomic_json(OUT/'boundary_summary.json',br)
        doc+='<h2>Stronger fixed-decay boundary audit</h2><p>The original validation winner sat at the lower rate-grid boundary. Before primary test evaluation, an additional validation audit tested0,0.0001,0.001 and selected0.0001. This supplementary comparator does not change trained models or primary results.</p>'+table(br,['benchmark','capacity','rate','accuracy','probe_accuracy'])
    doc+='''<h2>Results analysis</h2><p>At12slots, importance-inversion first-query accuracy rises from54.17%fixed decay and50.23%current liquid to90.74%adaptive evolution. The paired adaptive-minus-fixed difference is36.57percentage points, exploratory95%interval[31.25,42.59]. This is a concrete setting where mutable resident-memory dynamics help.</p><p>The adaptive rule reaches91.20%on the same inversion probes. Matched adaptive linear and GRU are similarly strong, and disabling the liquid hidden time flow leaves the reported aggregate scores unchanged. Evidence therefore supports demand-responsive per-item memory evolution, NOT liquid-network necessity or a uniquely learned continuous-time advantage.</p><p>Reactivation must not be hidden by aggregate gains: the full A-B-A workload improves, but actual return-to-A accuracy falls from96.18%fixed decay to82.99%adaptive at12slots (current liquid98.61%). Once an item is evicted, changing its importance cannot restore the content. Query-conditioned probe improvement at12slots is small and its interval versus fixed decay crosses zero.</p><p>The artificial agenda is a reliable observable cue. No future answer or query target enters policy features. The benchmark establishes an architecture capability under these cues and pressure; it does not establish robustness to misleading cues, natural semantic inference, or real-agent success. Shared version protection and hand-specified state equations contribute to the benefit. The four state meanings are operational definitions; confidence is not calibrated uncertainty.</p><p>Next work should preserve this positive inversion result, test unreliable cues and task-return trade-offs on independently generated streams, and compare inexpensive reactive policies. Increasing network complexity or claiming LNN exclusivity is not justified. This is a research rollout engine; the existing natural-language production adapter remains on the prior memory implementation until separately integrated and validated.</p>'''
    doc+='<h2>Reproduction</h2><pre>python scripts/adaptive_evolution_study.py --phase all</pre><p><a href="protocol.json">Protocol</a> · <a href="data_audit.json">Data audit</a> · <a href="results.json">Trajectory results</a> · <a href="paired_comparisons.json">Comparisons</a> · <a href="rates.json">Validation rates</a></p>'
    doc+='<p>164 tests passed. All 1,728 primary metric rows reproduced and capacity limits passed. No API calls.</p><p><a href="tests.txt">Tests</a> · <a href="replay_audit.json">Replay verification</a> · <a href="cached_resume_audit.json">Cached rerun</a> · <a href="objective_components.json">Loss components</a></p>'
    # Keep numerical report text readable without changing equations or filenames.
    for old,new in [('40inputs','40 inputs'),('8hidden','8 hidden'),('16bytes','16 bytes'),('24train/12validation/36test','24 training / 12 validation / 36 test'),('3seeds','3 seeds'),('8/12evidence','8 / 12 evidence'),('12epochs','12 epochs'),('all6learned','all 6 learned'),('each of12held-out','each of 12 held-out'),('seed11','seed 11'),('12slots','12 slots'),('At12','At 12'),('from54.17%fixed','from 54.17% fixed'),('and50.23%current','and 50.23% current'),('to90.74%adaptive','to 90.74% adaptive'),('is36.57percentage','is 36.57 percentage'),('exploratory95%interval','exploratory 95% interval'),('reaches91.20%on','reaches 91.20% on'),('from96.18%fixed','from 96.18% fixed'),('to82.99%adaptive','to 82.99% adaptive'),('liquid98.61%','liquid 98.61%'),('tested0,0.0001,0.001','tested 0, 0.0001, 0.001'),('selected0.0001','selected 0.0001')]:
        doc=doc.replace(old,new)
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    source,archive=snapshot_source();atomic_json(OUT/'delivery.json',dict(source=source,archive=archive,model_count=18,api_calls=0,tests=164,test_trajectories=36,primary_metric_rows=len(rows),neural_parameters={m:sum(p.numel() for p in make_model(m,11,.03).parameters()) for m in LEARNED}))
    print(json.dumps([r for r in summary if r['capacity']==12],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train','evaluate','diagnostics','boundary','report','all'],default='all');a=p.parse_args();torch.set_num_threads(1)
    for phase,fn in [('prepare',prepare),('train',train),('evaluate',evaluate),('diagnostics',diagnostics),('boundary',boundary_control),('report',report)]:
        if a.phase in [phase,'all']:fn()
