"""Controller-agnostic lifelong-memory state benchmarks and full result archive."""
import sys,json,time,hashlib,argparse,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.provenance import snapshot_source,runtime_info
from src.memory.evolving_memory import EvolutionPolicy,make_trajectory,batch,rollout,CONDITIONS,METHODS
from src.memory.continual_utility import policy_loss
OUT=ROOT/'runs/evolving-memory-v1';SEEDS=[11,23,42];CAPACITIES=[8,12];EPOCHS=16
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
METRICS=['accuracy','stale_error','stable_retention','adaptation_lag','adaptation_success','post_shift_recovery','pre_reminder_recovery','utility','answer_coverage','guarded_coverage','item_updates','seconds_per_trajectory']


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'data.json').exists():return
    source,archive=snapshot_source();public=ROOT/'runs/liquid-consolidation-v2'
    stories=read(public/'stories.json');vectors=read(public/'vectors.json')
    atomic_json(OUT/'protocol.json',dict(title='Adaptive Memory Evolution for Lifelong Agents',registered_at=time.time(),source=source,archive=archive,
        hypothesis='Memory items are evolving dynamical states; neural architecture is optional. No LNN superiority or established novelty claim.',
        formalism='m_i=(content_i,z_i), z_i=(I,C,U,T); dz_i/dt=f(z_i,held observed context,held policy coefficients). I_dot=k(.5+wq-I), C_dot=4k(v-C), U_dot=-kU/20, T_dot=1. Observation and noisy-feedback jumps are separate. Exact flow uses past held forcing.',
        content='Content is immutable per version; public observed revisions create new versions. Confidence means observed consistency, not calibrated truth probability. Deleted records cannot be recovered without new observations.',
        data='Public split-specific bAbI facts plus synthetic schedules: stable/volatile mixture,25%feedback flips, adversarial feedback on alternate phases, cyclic importance. Noise variants share world/query tapes with clean mixture.8train/4validation/12test seeds per condition:32/16/48condition trajectories,16/8/24underlying worlds.',
        noise='Only feedback channels are corrupted. Reliable topic cues, content observations and evaluation references remain unchanged. Feedback=correctness XOR flip for answered queries; abstentions receive0. Gold outcomes used only for offline training/evaluation, not online confidence or importance.',
        methods=METHODS,seeds=SEEDS,capacities_slots=CAPACITIES,epochs=EPOCHS,batch_size=8,
        neural_policy='Single40-to3linear head,123parameters; no recurrent/global hidden network. Controls: fixed decay, retrieval-frequency/LFU, adaptive rule, adaptive learned coefficients of the same item ODE.',
        objective='L=(1-query accuracy)+.25(1-stable query accuracy)+.25stale-answer rate. REINFORCE actual hard-capacity eviction; fixed deterministic baseline independent of learned actions; full-trajectory gradient, AdamW.003,clip1. Initialization eligible for validation selection.',
        selection='All rate/epoch choices use validation utility pooled across conditions and capacities. Fixed decay rates[0,.0001,.001,.01,.1]; adaptive rule rates[.03,.1,.3]. Learned head initialized to selected adaptive rule;16epochs each seed. Test once after selection.',
        retrieval='Same newest-retained-version lookup for all methods, without reference access. Missing newest versions can expose stale answers. Also report common observed-version watermark filtering on identical retained contents; filtering is not a new policy rollout.',
        metrics='Stale error=wrong-version answers/allqueries; stable retention=correct designated-stable probes; adaptation lag=first two consecutive4target-query windows >=75%, censored at12query opportunities; report recovery success separately. Post-shift recovery=final4target-query accuracy. Cyclic pre-reminder recovery=old-topic first-round queries before refresh.',
        budget='Evidence slots8/12, shared bounded24-key metadata and per-item state allowance. Adaptive state adds4float32 values per item and O(resident items) update work. Not RSS/byte-budget or speed advantage.',
        inference='Reliable observed topic cues, structured key extraction, template overlap and limited optimization restrict generalization. Pair bootstrap12trajectory seeds per condition after averaging3model/policy seeds. Noise contrasts paired to same underlying clean-world seed. No multiplicity correction.'))
    data={}
    for split,count,offset in [('train',8,51000),('val',4,52000),('test',12,53000)]:
        data[split]=[make_trajectory(stories,vectors,split,condition,offset+i) for condition in CONDITIONS for i in range(count)]
    atomic_json(OUT/'data.json',data)
    atomic_json(OUT/'data_audit.json',[dict(split=split,condition=r['environment'],seed=r['seed'],events=len(r['events']),queries=sum(e['type']==0 for e in r['events']),stable_queries=sum(e['type']==0 and e['stable'] for e in r['events']),conflicts=sum(e['conflict'] for e in r['events']),flips=sum(e['flip'] for e in r['events']),duration=r['events'][-1]['time']) for split,rows in data.items() for r in rows])
    print('Prepared32train/16validation/48test condition trajectories',flush=True)


def evaluate_rows(rows,capacity,method,rate,model=None,seed=0,record=False):
    with torch.no_grad():return rollout(batch(rows),capacity,method,rate,model,seed,record=record)


def train():
    prepare();data=read(OUT/'data.json');(OUT/'checkpoints').mkdir(exist_ok=True)
    if not (OUT/'rates.json').exists():
        rates={}
        for method,candidates in [('fixed_decay',[0.,.0001,.001,.01,.1]),('adaptive_rule',[.03,.1,.3])]:
            scores=[dict(rate=rate,utility=float(np.mean([np.mean(evaluate_rows(data['val'],c,method,rate,seed=707)['metrics']['utility']) for c in CAPACITIES]))) for rate in candidates]
            rates[method]=dict(selected=max(scores,key=lambda r:r['utility'])['rate'],candidates=scores)
        atomic_json(OUT/'rates.json',rates);print('Validation rates',rates,flush=True)
    rates=read(OUT/'rates.json');rate=rates['adaptive_rule']['selected']
    for seed in SEEDS:
        path=OUT/f'training-{seed}.json';checkpoint=OUT/'checkpoints'/f'learned-{seed}.pt'
        if path.exists() and read(path)['complete']:continue
        torch.manual_seed(seed);model=EvolutionPolicy(rate);optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        best=float(np.mean([np.mean(evaluate_rows(data['val'],c,'adaptive_learned',rate,model,707)['metrics']['utility']) for c in CAPACITIES]))
        def save(epoch):torch.save(dict(seed=seed,rate=rate,state_dict=model.state_dict(),epoch=epoch,validation_utility=best),checkpoint)
        save(-1);history=[dict(epoch=-1,validation_utility=best)];rng=np.random.default_rng(seed);started=time.perf_counter()
        for epoch in range(EPOCHS):
            order=rng.permutation(len(data['train']));losses=[];utilities=[]
            for j,left in enumerate(range(0,len(order),8)):
                inputs=batch([data['train'][i] for i in order[left:left+8]]);capacity=CAPACITIES[(epoch+j)%2]
                with torch.no_grad():base=rollout(inputs,capacity,'fixed_decay',rates['fixed_decay']['selected'],greedy=True)
                result=rollout(inputs,capacity,'adaptive_learned',rate,model,seed*10000+epoch*100+j)
                loss=policy_loss(result,base);optimizer.zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                optimizer.step();losses.append(float(loss.detach()));utilities+=result['metrics']['utility']
            val=float(np.mean([np.mean(evaluate_rows(data['val'],c,'adaptive_learned',rate,model,707)['metrics']['utility']) for c in CAPACITIES]))
            if val>best:best=val;save(epoch)
            history.append(dict(epoch=epoch,validation_utility=val,training_utility=float(np.mean(utilities)),policy_loss=float(np.mean(losses))))
            atomic_json(path,dict(complete=False,history=history));print('Learned',seed,'epoch',epoch,'val',round(val,4),flush=True)
        atomic_json(path,dict(complete=True,history=history,seconds=time.perf_counter()-started))


def load(method,seed):
    rates=read(OUT/'rates.json');rate=rates['adaptive_rule' if method.startswith('adaptive') else 'fixed_decay']['selected']
    if method!='adaptive_learned':return None,rate
    saved=torch.load(OUT/'checkpoints'/f'learned-{seed}.pt',weights_only=True)
    model=EvolutionPolicy(rate);model.load_state_dict(saved['state_dict']);return model.eval(),rate


def evaluate():
    if (OUT/'evaluation_complete.json').exists():return
    for seed in SEEDS:assert read(OUT/f'training-{seed}.json')['complete']
    data=read(OUT/'data.json');results=[];(OUT/'traces').mkdir(exist_ok=True)
    for method in METHODS:
        for seed in SEEDS:
            model,rate=load(method,seed)
            for condition in CONDITIONS:
                rows=[r for r in data['test'] if r['environment']==condition]
                for capacity in CAPACITIES:
                    started=time.perf_counter();result=evaluate_rows(rows,capacity,method,rate,model,seed*100+77,record=True)
                    elapsed=time.perf_counter()-started
                    for i,row in enumerate(rows):
                        assert result['metrics']['peak_slots'][i]<=capacity
                        results.append(dict(method=method,seed=seed,condition=condition,capacity=capacity,trajectory_seed=row['seed'],seconds_per_trajectory=elapsed/len(rows),**{k:v[i] for k,v in result['metrics'].items()},episodes=result['episodes'][i]))
                    atomic_json(OUT/'traces'/f'{method}-{seed}-{condition}-{capacity}.json',dict(seeds=[r['seed'] for r in rows],trace=result['trace']))
            print('Evaluated',method,seed,flush=True)
    atomic_json(OUT/'results.json',results);atomic_json(OUT/'evaluation_complete.json',dict(rows=len(results),at=time.time()))


def report():
    rows=read(OUT/'results.json');summary=[];comparisons=[];noise=[]
    for condition in CONDITIONS:
        for capacity in CAPACITIES:
            for method in METHODS:
                rr=[r for r in rows if r['condition']==condition and r['capacity']==capacity and r['method']==method]
                summary.append(dict(condition=condition,capacity=capacity,method=method,**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
            for baseline in METHODS[:-1]:
                for metric in ['stale_error','stable_retention','adaptation_lag','post_shift_recovery','pre_reminder_recovery','accuracy']:
                    if metric=='pre_reminder_recovery' and condition!='cyclic':continue
                    differences=[]
                    for trajectory in sorted({r['trajectory_seed'] for r in rows}):
                        means={m:np.mean([r[metric] for r in rows if r['condition']==condition and r['capacity']==capacity and r['method']==m and r['trajectory_seed']==trajectory]) for m in ['adaptive_learned',baseline]}
                        differences.append(float(means['adaptive_learned']-means[baseline]))
                    samples=np.mean(np.random.default_rng(123).choice(differences,(3000,len(differences))),axis=1)
                    comparisons.append(dict(condition=condition,capacity=capacity,baseline=baseline,metric=metric,difference=float(np.mean(differences)),ci95=np.quantile(samples,[.025,.975]).tolist()))
    for capacity in CAPACITIES:
        for method in METHODS:
            clean=next(r for r in summary if r['condition']=='mixture' and r['capacity']==capacity and r['method']==method)
            for condition in ['feedback_noise','feedback_misleading']:
                altered=next(r for r in summary if r['condition']==condition and r['capacity']==capacity and r['method']==method)
                noise.append(dict(method=method,capacity=capacity,condition=condition,accuracy_change=altered['accuracy']-clean['accuracy'],stable_change=altered['stable_retention']-clean['stable_retention'],stale_change=altered['stale_error']-clean['stale_error']))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'paired_comparisons.json',comparisons);atomic_json(OUT/'noise_effects.json',noise)
    objectives=[]
    for r in rows:
        loss=(1-r['accuracy'])+.25*(1-r['stable_retention'])+.25*r['stale_error']
        assert abs(loss-(1.25-r['utility']))<1e-6
        objectives.append({k:r[k] for k in ['method','seed','condition','capacity','trajectory_seed']}|dict(L_task=1-r['accuracy'],L_retention=1-r['stable_retention'],L_stale=r['stale_error'],L=loss))
    atomic_json(OUT/'objective_components.json',objectives)
    plot_results(rows)
    def table(rr,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in columns)+'</tr>' for r in rr)+'</table>'
    doc='''<!doctype html><meta charset="utf-8"><title>Adaptive Memory Evolution</title><style>body{max-width:1320px;margin:32px auto;font:16px/1.7 system-ui;color:#183047}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}pre{background:#eef3f7;padding:18px;white-space:pre-wrap}</style><h1>Adaptive Memory Evolution for Lifelong Agents</h1><p>Research hypothesis: memory items are evolving dynamical states. Neural architecture is optional; no LNN superiority claim. This is a controlled public-fact experiment, not proof of novelty or natural-language agent effectiveness.</p><h2>Formalization</h2><pre>m_i(t)=(content_i,z_i(t)), z_i=(I_i,C_i,U_i,T_i)
dz_i/dt=f(z_i,observed context,held feedback-dependent state)
I_dot=k(.5+wq_i-I); C_dot=4k(v_i-C)
U_dot=-kU/20; T_dot=1
At feedback events: C_plus=.8C_minus+.2*observed_feedback
U_plus=U_minus+observed_feedback; T_plus=0
At observed content revisions: create a new immutable content version.</pre><p>Held inputs are from previous observations. No future queries or gold correctness enter online updates; noisy observed feedback enters both rules and learned coefficients. C is observed consistency, not calibrated certainty. An evicted fact cannot be recovered without a new observation. Learned coefficients use only a 40-to-3 linear head: 123 parameters, no recurrent/global hidden network. The item transition equations are hand-specified, not learned wholesale.</p><h2>Protocol</h2><p>32 training / 16 validation / 48 test condition trajectories; noise variants share underlying world/query tapes. There are 24 underlying held-out worlds, with 12 seeds per condition. Three model/policy seeds; 8/12 evidence slots; 16 training epochs with initialization eligible for selection. Same content, retrieval, metadata allowance and eviction sampler across methods. Training optimizes actual task correctness, stable retention and stale-answer cost. Reliable topic cues remain available in all conditions; the noise experiment corrupts feedback only.</p><h2>Metric definitions</h2><p>Stale error: wrong-version answers / all queries (lower is better). Stable retention: correct probes of unchanged anchor facts. Adaptation lag: query count until two consecutive 4-query windows achieve at least 75%; failures are censored at 12 and reported separately. End-window recovery includes reminder opportunities. Cyclic pre-reminder recovery measures old-topic recall before any refresh and must not be conflated with relearning.</p><h2>All conditions</h2>'''
    findings='''<h2>Findings and limits</h2><p><strong>Adaptive evolution improves freshness and recovery, but sacrifices stable knowledge. Learning is not the main source of improvement.</strong> In the clean mixture at 12 slots, fixed decay / frequency / adaptive rule / learned evolution achieve 74.35 / 75.37 / 81.02 / 81.57% task accuracy. Stable-knowledge retention is 37.65 / 42.75 / 14.04 / 17.13%. The learned-minus-rule accuracy difference is only +0.56 percentage points, with exploratory paired 95% interval [-0.37, +1.42].</p>
<p>Relative to fixed decay, learned evolution reduces stale errors from 2.87% to 0%, shortens the censored adaptation lag from 5.97 to 5.08 query opportunities, and improves final-window recovery from 93.06% to 99.65%. Stable retention drops by 20.52 percentage points, with paired interval [-28.24, -12.19]. These are competing objectives, not universal dominance.</p>
<p>Under cyclic demand at 12 slots, final-window recovery rises from 86.00% to 96.41%, but recall <em>before reminders</em> is essentially unchanged: 68.23% versus 68.58%, difference interval [-6.08, +6.42] percentage points. At 8 slots, pre-reminder recall falls from 47.40% to 29.86%. Thus this experiment supports faster readaptation after new evidence, not improved preservation of dormant knowledge.</p>
<p>With 25% random feedback flips, learned task accuracy changes by -0.96 points at 8 slots and +0.15 at 12. Alternating misleading feedback changes it by -1.48 and +0.09 points. These are descriptive paired changes, not evidence that misinformation helps. Frequency ignores feedback and is exactly invariant. Reliable topic cues and observed-version consistency remain strong signals in all conditions; robustness to corrupted context or content is untested.</p>
<p><strong>Mechanistic interpretation:</strong> the rule explicitly pulls off-topic importance toward 0.5 and on-topic importance toward 2.0. This protects the current workload under capacity pressure, while unchanged anchor facts remain off-topic and receive fewer queries. Moreover, between events C is pulled toward observed version consistency, which can overwrite negative feedback. These equations explain a plausible source of the freshness/retention and feedback-insensitivity patterns; this run does not isolate each term causally.</p>
<p>The registered utility is accuracy + 0.25 stable retention - 0.25 stale error. In the 12-slot clean mixture, learned / frequency scores are 0.8586 / 0.8570; in cyclic demand they are 0.6745 / 0.6970. Aggregate task accuracy alone would conceal this tradeoff. There is no neural, LNN, computational-efficiency, or continual-agent superiority claim. The next justified question is how to preserve dormant stable evidence under a retention constraint, rather than adding neural complexity.</p>
<figure><img src="metrics.png" alt="Four requested evaluation metrics across conditions at twelve memory slots" style="width:100%"><figcaption>Means and exploratory 95% intervals over 12 trajectory clusters after averaging three policy/model seeds. Conditions with feedback corruption reuse clean worlds. Cyclic facts do not change version, so all cyclic stale-error rates are zero by construction.</figcaption></figure>'''
    doc=doc.replace('<h2>All conditions</h2>',findings+'<h2>All conditions</h2>')
    doc+=table(summary,['condition','capacity','method','accuracy','stale_error','stable_retention','adaptation_lag','adaptation_success','post_shift_recovery'])
    doc+='<h2>Cyclic retention before reminders</h2>'+table([r for r in summary if r['condition']=='cyclic'],['capacity','method','pre_reminder_recovery','post_shift_recovery'])
    doc+='<h2>Noisy-feedback sensitivity: paired world/query tapes</h2>'+table(noise,['condition','capacity','method','accuracy_change','stable_change','stale_change'])
    doc+='<h2>Stale-answer safeguards and resources</h2><p>Primary retrieval returns the newest version still stored, so stale fallback is exposed. A common observed-version filter would reject known-stale answers; guarded coverage below applies that filter to the same retained contents, not a separate rollout. Zero error produced by that filter is a correctness rule, not a learned-memory advantage. Adaptive updates add four float32 states per item and work proportional to resident items; no speed or byte-efficiency claim.</p>'
    doc+=table(summary,['condition','capacity','method','answer_coverage','guarded_coverage','item_updates','seconds_per_trajectory'])
    doc+='<h2>Paired exploratory differences: learned minus baseline</h2><p>Average three model/policy seeds within each trajectory, then resample 12 paired seeds. Negative stale-error and lag differences are favorable; positive retention and recovery differences are favorable. Intervals are unadjusted for multiple comparisons.</p>'
    doc+=table([r for r in comparisons if r['capacity']==12],['condition','baseline','metric','difference','ci95'])
    doc+='<h2>Reproduce</h2><pre>python scripts/evolving_memory_study.py --phase all</pre><p><a href="protocol.json">Protocol</a> · <a href="results.json">Trajectory records</a> · <a href="rates.json">Validation choices</a> · <a href="objective_components.json">Loss terms</a> · <a href="data_audit.json">Data audit</a></p>'
    doc+='<p>CLI: <code>python -m src.cli evolving-memory --phase all</code>. Existing completed artifacts are reused; use a new output directory constant for an independent rerun. Code checks: <a href="tests.txt">170 passing tests</a>. Independent deterministic replay: <a href="replay_audit.json">audit</a>. <a href="delivery.json">Source snapshot and delivery metadata</a>. Old language-agent adapters remain unchanged; this experiment does not invoke Agnes.</p>'
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    source,archive=snapshot_source();atomic_json(OUT/'delivery.json',dict(source=source,archive=archive,runtime=runtime_info(),neural_parameters=123,trained_models=3,api_calls=0,primary_metric_rows=len(rows),test_condition_trajectories=48,underlying_test_worlds=24))
    print(json.dumps([r for r in summary if r['capacity']==12],indent=2))


def plot_results(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names=['Clean mixture','25% noisy feedback','Misleading feedback','Cyclic demand']
    colors=['#718096','#cb8c2c','#2f8e80','#635bb6']
    labels=['Fixed decay','Retrieval frequency','Adaptive rule','Adaptive learned']
    for ax,(metric,title,scale) in zip(axes.flat,[('stale_error','Stale answers (%) — lower is better',100),('stable_retention','Stable knowledge retained (%) — higher is better',100),('adaptation_lag','Adaptation lag (queries) — lower is better',1),('post_shift_recovery','Recovery after reminders (%) — higher is better',100)]):
        for j,method in enumerate(METHODS):
            means=[];intervals=[]
            for condition in CONDITIONS:
                rr=[r for r in rows if r['condition']==condition and r['capacity']==12 and r['method']==method]
                clusters=np.array([np.mean([r[metric] for r in rr if r['trajectory_seed']==seed]) for seed in sorted({r['trajectory_seed'] for r in rr})])*scale
                means.append(clusters.mean())
                draws=np.random.default_rng(123).choice(clusters,(3000,len(clusters))).mean(1)
                lo,hi=np.quantile(draws,[.025,.975]);intervals.append([max(0,clusters.mean()-lo),max(0,hi-clusters.mean())])
            ax.errorbar(np.arange(4)+(j-1.5)*.15,means,yerr=np.array(intervals).T,fmt='o',capsize=3,color=colors[j],label=labels[j],markersize=5)
        ax.set_title(title,fontsize=11);ax.set_xticks(range(4),names,rotation=16,ha='right',fontsize=9)
        ax.grid(axis='y',alpha=.2)
        if scale==100:ax.set_ylim(bottom=-1)
    fig.suptitle('Adaptive Memory Evolution: freshness–retention tradeoff (12 slots)',fontsize=15)
    axes[0,1].legend(fontsize=9,loc='upper right')
    fig.savefig(OUT/'metrics.png',dpi=170);fig.savefig(OUT/'metrics.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train','evaluate','report','all'],default='all');a=p.parse_args();torch.set_num_threads(1)
    for phase,fn in [('prepare',prepare),('train',train),('evaluate',evaluate),('report',report)]:
        if a.phase in [phase,'all']:fn()
