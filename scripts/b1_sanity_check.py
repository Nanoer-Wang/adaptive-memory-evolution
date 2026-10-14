"""Run ONLY the registered B1 mini: eight condition tapes, two seeds, capacity8.

No training/optimization/Agnes entry point is provided. Existing completed output
is verified, never overwritten. Use --output for a separately registered rerun.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import html
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import snapshot_source, runtime_info
from src.memory.b1_environment import generate, audit_environment, digest
from src.memory.b1_policy import METHODS, Policy, frozen_head
from src.evaluation.b1_sanity import run_trajectory, audit_run
from scripts.ame_validation_batches import sha, read


def write_gzip(path, value):
    with gzip.open(path,'wt',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,allow_nan=False)


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f: return json.load(f)


def verify(out):
    manifest=read(out/'completion.json')
    for relative, expected in manifest['files'].items():
        assert sha(out/relative)==expected, relative
    assert manifest['training_steps']==0 and manifest['api_calls']==0
    print('Saved mini artifacts verify; no rerun or training performed.')


def probe(out, coefficients):
    def event(kind, now, key=0, topic=0, version=1):
        e=dict(kind=kind,time=now,key=key,topic=topic,vector=[0.]*32)
        if kind=='observe': e.update(version=version,item_id=f'{key}:{version}',content=f'probe fact version{version}')
        return e
    initial=[event('observe',0.)]
    tape=[event('context',0.),event('query',5.),event('context',10.,topic=1),
          event('feedback_probe',20.),event('context',30.),event('context',40.),
          event('observe',42.,version=2),event('context',48.)]
    records={}
    for method in ('adaptive_rule','adaptive_evolution'):
        p=Policy(method,8,11,coefficients if method=='adaptive_evolution' else None);p.bootstrap(initial)
        frames=[dict(time=0.,event='initial',state=p.snapshot()[0])]
        for e in tape:
            p.update_state(e)
            if e['kind']=='observe':p.observe(e);p.evict()
            if e['kind']=='query':p.retrieve(e);p.feedback('0:1',1.)
            if e['kind']=='feedback_probe':p.feedback('0:1',0.)
            frames.append(dict(time=e['time'],event=e['kind'],state=next(i for i in p.snapshot() if i['item_id']=='0:1')))
        records[method]=frames
    atomic_json(out/'controlled_probe.json',dict(purpose='Prescribed mechanics check, not a benchmark or a trained result. Negative feedback at20 is injected. No capacity pressure in this probe.',initial=initial,events=tape,results=records))
    # Closed-form qualitative checks without selecting successful random items.
    rule=records['adaptive_rule']; bytime={r['time']:r['state'] for r in rule}
    assert bytime[10.]['I']>bytime[20.]['I'] and bytime[40.]['I']>bytime[30.]['I']
    assert bytime[20.]['C']==.8 and bytime[48.]['C']<.01
    assert bytime[5.]['U']==1. and 0<bytime[40.]['U']<bytime[5.]['U']
    return records


def figures(out, probe_records, sample_result, sample_row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,field in zip(axes.flat,('I','C','U','T')):
        for method,color in [('adaptive_rule','#138b78'),('adaptive_evolution','#6253ae')]:
            rr=probe_records[method]
            ax.plot([r['time'] for r in rr],[r['state'][field] for r in rr],'o-',color=color,label=method)
        for t,label in [(10,'topic B'),(20,'negative feedback'),(30,'topic A'),(42,'new version')]:
            ax.axvline(t,color='#8c8c8c',alpha=.3,ls='--')
        ax.set_title(field);ax.set_xlabel('Observed time');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Controlled mechanics probe: topic A -> B -> A; new version at t=42\nNegative feedback at t=20 is prescribed, not a benchmark outcome',fontsize=13)
    fig.savefig(out/'memory_evolution.png',dpi=150);fig.savefig(out/'memory_evolution.svg');plt.close(fig)
    ids=[i['item_id'] for i in sample_row['initial_memory'][:3]]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for item_id in ids:
        x=[0.];y=[1.]
        for frame in sample_result['trace']:
            state=next((i for i in frame['resident'] if i['item_id']==item_id),None)
            x.append(frame['time']);y.append(state['I'] if state else np.nan)
        ax.plot(x,y,'o-',markersize=2,label=item_id)
    ax.set(title='First three initial records, selected before evaluation: gaps mean evicted',xlabel='Observed time',ylabel='Importance I')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'sample_residency.png',dpi=150);plt.close(fig)


def report(out, rows, summary, fairness, audits):
    def table(rr,columns):
        return '<table><tr>'+''.join(f'<th>{html.escape(c)}</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]))+'</td>' for c in columns)+'</tr>' for r in rr)+'</table>'
    text='''<!doctype html><meta charset="utf-8"><title>B1 sanity check</title>
<style>body{max-width:1180px;margin:30px auto;padding:0 20px;color:#243b4c;font:16px/1.7 system-ui}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px;border-bottom:1px solid #ddd}img{width:100%}code{background:#eef3f5}h2{margin-top:35px}</style>
<h1>B1 实验框架与 sanity check</h1><p><strong>仅实现检查，无训练，无API；不启动正式B1。</strong>8条条件轨迹=2个世界种子×(3种混合比例的快变条件+1个50%稳定比例的平稳条件)，2个策略种子，5种方法，共80次回放。只有2个世界种子，不能用于显著性检验或方法优越性结论。</p>
<h2>框架与策略</h2><p>独立生成器：80键，40核心+40干扰，稳定事实在4个主题内等比例分配。种子、初始记忆、事件、查询顺序和仅评估器可见的标签分开保存。统一接口为observe / update_state / retrieve / evict / feedback，查询仅返回一个驻留条目。</p>
<p>固定衰减：纯年龄指数排序，查询不强化；频率：初始计数1+返回次数，不看反馈；当前自适应：原123参数头的冻结权重；规则：相同I/C/U/T方程、常数系数；即时上下文规则：当前主题、已观察版本一致性和访问次数，无演化隐状态。种子11/23仅对应旧冻结权重及平分决策，不是新增训练种子。</p>
<p>相对旧v1的明确变化：采用确定性淘汰及共同哈希平分；不对全部方法额外叠加旧版ln(0.1)过期版本惩罚，以保留纯固定衰减/频率的定义。自适应原状态方程和读取的神经头未增加复杂度。新旧分数不可直接拼表。所有方法仍有同样的已观察版本元数据；另报告共同版本过滤后的覆盖率。</p>
<h2>审计</h2><p>相同初始快照、输入哈希、查询序列、容量8、top1检索预算和每次观察后的淘汰机会。所有mini观察版本唯一，因此每次观察都构成同一溢出机会。候选拒收和驻留替换分开；两者之和必须等于溢出次数。淘汰内容不能通过只保存ID的版本索引恢复。</p>'''
    text+=table(fairness,['trajectory_seed','stable_fraction','regime','passed','observation_opportunities','queries','initial_equal','opportunities_equal'])
    text+='<h2>描述性指标：不可作为正式研究结果</h2>'
    text+=table(summary,['method','accuracy','stable_retention','stale_retrieval','eviction_count','replacement_rate','rejected_write_count','retrieval_hit_rate','policy_ms','item_state_updates'])
    text+='<p>耗时是逐条Python实现的策略调用耗时，排除轨迹序列化/绘图；方法执行顺序轮换，资源计数单列。不是吞吐基准，也不是公平优化后的速度比较。命中率包含错误旧版本，不能替代准确率。缺少实际变化的平稳阶段适应指标为null，未恢复阶段按20个易变查询截尾并单列成功率。</p>'
    text+='<h2>逐项状态检查</h2><img src="memory_evolution.png"><p>上图是预先指定、不受容量压力的机制探针，用来检查重要性随上下文下降/回升、反馈跳变、版本变旧后C下降和U衰减，不是新增数据集效果。下图为真实样例轨迹的前三个初始条目，选择不依赖结果；淘汰后留空，不把遗忘后的状态延续画出来。</p><img src="sample_residency.png">'
    text+='<h2>已知限制与正式B1前需要解决的问题</h2><ul><li>固定50%稳定查询意味着稳定条目越少，单条稳定事实访问越频繁。主题角色偏差已消除，但访问频率效应仍需按预定查询比例敏感性测试分离。</li><li>稳定事实仅初始观察；易变事实会得到修订观察。因此这轮有信息到达频率差异，不能把收益全部归因记忆机制。</li><li>自适应与即时规则使用版本一致性；纯年龄/频率没有该评分项。零旧版本错误可能来自版本处理，正式机制归因必须包含一致性消融与共同过滤对照。</li><li>可靠提前主题提示是一种强信号；还没有验证自然查询推断需求或错误提示。</li><li>冻结旧权重在80键新环境是迁移检查，没有正式B1训练或验证选模；不能用本轮成绩选择胜者。</li><li>所有方法会优先保留/拒绝不同条目，真实替换次数不同；共享机会不等于匹配实际替换强度。需要后续写入预算曲线来完成归因。</li><li>寿命记录包含仍存活条目的右截尾标记；报告的分位数是已观察时长，不能当作无偏的完整寿命分布。</li></ul>'
    text+='<h2>交付</h2><p><a href="protocol.json">预先登记协议</a> · <a href="sample_trajectory.json">样例轨迹</a> · <a href="metrics.json">全部指标</a> · <a href="fairness_audit.json">公平性审计</a> · <a href="environment_audit.json">环境审计</a> · <a href="replay_audit.json">完整重放审计</a> · <a href="controlled_probe.json">机制探针</a> · <a href="source_manifest.json">源码与模型哈希</a></p><p>运行：<code>python scripts/b1_sanity_check.py</code>。已完成目录只做完整性验证；独立重跑使用<code>--output 新目录</code>。正式B1未开放启动入口。</p>'
    (out/'index.html').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/ame-validation-v2/B1-sanity-v1')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Output must stay in workspace')
    if (out/'completion.json').exists(): verify(out);return
    if out.exists() and any(out.iterdir()): raise FileExistsError('Incomplete output exists; preserve it and use a new output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir()
    source,archive=snapshot_source()
    inputs=[ROOT/'runs/liquid-consolidation-v2'/n for n in ('stories.json','vectors.json')]
    checkpoints={s:ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt' for s in (11,23)}
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs+list(checkpoints.values())}
    protocol=dict(stage='B1_sanity_only',registered_at=time.time(),world_seeds=[60001,60002],policy_seeds=[11,23],
                  capacity=8,trajectories=8,methods=METHODS,phases=6,core_keys=40,distractor_keys=40,
                  conditions=[dict(stable_fraction=r,regime='rapid') for r in (.2,.5,.8)]+[dict(stable_fraction=.5,regime='stationary')],
                  coefficient_source='Existing frozen v1 checkpoints11/23; no training. Rule rate0.3; fixed decay0.001.',
                  scoring_change='v2 canonical baselines: remove legacy shared explicit version penalty for EVERY policy. Adaptive internal C and immediate consistency remain; report as a mechanism confound, not dynamics evidence.',
                  evaluation='Newest retained version, budget1. Deterministic score eviction with shared seeded hash tie breaking. Unique candidate versions in mini make overflow opportunities identical.',
                  initial='8 common uniformly selected core records at time0, I=C=1,U=T=0. No per-method warmup selection.',
                  resources='Policy-only timing excludes trace serialization; rotated method order. Nominal numeric state bytes are not Python RSS.',
                  analysis='Descriptive only. Full event/state/answer/lifetime trace, exact replay, causal unit tests. No significance tests and no automatic full-B1 continuation.',
                  training_steps=0,api_calls=0,source=source,archive=archive,input_hashes=hashes)
    atomic_json(out/'protocol.json',protocol)
    atomic_json(out/'source_manifest.json',dict(source=source,archive=archive,inputs=hashes,runtime=runtime_info()))
    stories,vectors=[read(p) for p in inputs]
    heads={s:frozen_head(p) for s,p in checkpoints.items()}
    rows=[]
    for world_seed in protocol['world_seeds']:
        for condition in protocol['conditions']:
            rows.append(generate(stories,vectors,world_seed,world_seed+100000,**condition))
    audits=[audit_environment(r) for r in rows]
    write_gzip(out/'trajectories.json.gz',rows)
    atomic_json(out/'sample_trajectory.json',rows[0]);atomic_json(out/'environment_audit.json',audits)
    metrics=[];checks=[];replays=[];sample=None
    for n,row in enumerate(rows):
        for seed in protocol['policy_seeds']:
            order=list(METHODS);order=order[n%len(order):]+order[:n%len(order)]
            for method in order:
                coefficients=heads[seed] if method=='adaptive_evolution' else None
                result=run_trajectory(row,method,seed,coefficients)
                check=audit_run(row,result)
                filename=f'{n}-{seed}-{method}.json.gz';write_gzip(out/'traces'/filename,result)
                result['metrics']['trace_file']='traces/'+filename
                metrics.append(result['metrics']);checks.append(dict(trajectory=n,method=method,seed=seed,**check))
                # Independent rerun, excluding only real wall-clock timings.
                again=run_trajectory(copy.deepcopy(row),method,seed,coefficients)
                assert result['trace']==again['trace'] and result['lifetimes']==again['lifetimes']
                assert result['queries']==again['queries'] and result['adaptation_episodes']==again['adaptation_episodes']
                for key,value in again['metrics'].items():
                    if key not in ('policy_seconds','timing_seconds'): assert value==result['metrics'][key],key
                replays.append(dict(trajectory=n,method=method,seed=seed,passed=True))
                if n==0 and method=='adaptive_evolution' and seed==11:sample=result
    fairness=[]
    for row,audit in zip(rows,audits):
        selected=[m for m in metrics if m['environment_sha256']==row['environment_sha256']]
        for key in ('initial_memory_sha256','input_sha256','query_sha256','capacity','observation_opportunities','overflow_opportunities','queries'):
            assert len({m[key] for m in selected})==1,key
        assert all(m['overflow_opportunities']==audit['observation_opportunities'] for m in selected)
        assert len({m['policy_calls']['retrieve'] for m in selected})==1
        fairness.append(dict(trajectory_seed=row['trajectory_seed'],stable_fraction=row['stable_fraction'],regime=row['regime'],passed=True,initial_equal=True,opportunities_equal=True,queries=audit['queries'],observation_opportunities=audit['observation_opportunities']))
    summary=[]
    for method in METHODS:
        group=[m for m in metrics if m['method']==method]
        fields=['accuracy','stable_retention','stale_retrieval','eviction_count','replacement_rate','rejected_write_count','retrieval_hit_rate','item_state_updates']
        summary.append(dict(method=method,**{k:float(np.mean([m[k] for m in group])) for k in fields},policy_ms=1000*float(np.mean([m['policy_seconds'] for m in group]))))
    atomic_json(out/'metrics.json',metrics);atomic_json(out/'summary.json',summary)
    atomic_json(out/'fairness_audit.json',fairness);atomic_json(out/'trace_audit.json',checks)
    atomic_json(out/'replay_audit.json',dict(passed=True,runs=replays,training_steps=0,api_calls=0))
    probe_records=probe(out,heads[11]);figures(out,probe_records,sample,rows[0])
    report(out,rows,summary,fairness,audits)
    assert hashes=={p.relative_to(ROOT).as_posix():sha(p) for p in inputs+list(checkpoints.values())}
    atomic_json(out/'completion.json',dict(training_steps=0,api_calls=0,trajectory_count=8,policy_runs=len(metrics),
        source=source,files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(summary,indent=2));print('Mini complete. Stopped before formal B1.')


if __name__=='__main__': main()
