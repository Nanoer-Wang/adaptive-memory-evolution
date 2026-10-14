"""Ten tapes, two frozen policy seeds, capacity8; no training or formal B1."""
import argparse,copy,html,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import snapshot_source,runtime_info
from src.memory.b1_interleaved import generate,audit_environment,PRESSURES
from src.memory.b1_policy import METHODS,frozen_head
from src.evaluation.b1_interleaved import run,audit
from scripts.b1_sanity_check import write_gzip,read_gzip
from scripts.ame_validation_batches import read,sha


def verify(out):
    complete=read(out/'completion.json')
    for name,digest in complete['files'].items():assert sha(out/name)==digest,name
    assert complete['training_steps']==complete['api_calls']==0
    print('Interleaved mini verified; no training or formal evaluation started.')


def plots(out,rows,sample,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Sample fixed BEFORE results: first world, medium pressure, seed11,
    # adaptive evolution; first stable/A/B content versions, all episodes shown.
    row=rows[1];ids=[f'{row[k][0]}:1' for k in ('stable_keys','a_keys','b_keys')]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,field in zip(axes.flat,('I','C','U','T')):
        for item_id,label,color in zip(ids,('stable item','topic A item','topic B item'),('#288574','#bc8c24','#6b59ae')):
            times=[];values=[]
            initial=next((i for i in row['initial_memory'] if i['item_id']==item_id),None)
            times.append(0.);values.append((1. if field in ('I','C') else 0.) if initial else np.nan)
            for frame in sample['trace']:
                for states in (frame['flow_state'],frame['resident']):
                    state=next((i for i in (states or []) if i['item_id']==item_id),None)
                    times.append(frame['event']['time']);values.append(state[field] if state else np.nan)
            ax.plot(times,values,label=label,color=color,lw=1.)
        for frame in sample['trace']:
            if frame['event']['kind']=='context':
                ax.axvline(frame['event']['time'],color='grey',ls='--',alpha=.35)
        ax.set(title=field,xlabel='Observed time');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Actual interleaved stream: preselected version1 records, event samples\nGaps = evicted; reappearance requires a new observation. Dashed lines = context events.',fontsize=12)
    fig.savefig(out/'memory_evolution.png',dpi=160);fig.savefig(out/'memory_evolution.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in METHODS:
        rr=[next(r for r in summary if r['method']==method and r['pressure']==p) for p in PRESSURES]
        for ax,metric in zip(axes,('accuracy','eviction_rate_per_observation')):
            ax.plot([.25,1,2],[r[metric]*100 for r in rr],'o-',label=method)
    for ax,title in zip(axes,('Task accuracy (%)','Resident evictions / observations (%)')):
        ax.set(xlabel='Distinct noise keys /8active keys',title=title,xticks=[.25,1,2]);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Descriptive mini only: 3 paired worlds x2policy seeds; capacity8')
    fig.savefig(out/'pressure.png',dpi=160);plt.close(fig)


def full_trace_page(out,sample):
    body='''<!doctype html><meta charset="utf-8"><title>Complete interleaved event trace</title><style>body{font:14px/1.5 system-ui;margin:25px}table{border-collapse:collapse;width:100%}td,th{padding:5px;border:1px solid #ddd;vertical-align:top}summary{cursor:pointer}</style><h1>完整event trace：预先指定的medium压力、seed11、adaptive evolution</h1><p>每行一个输入事件；feedback值是回答后实际产生的信号。展开状态可查看全部驻留条目，数值顺序I/C/U/T。所有事件均保留。</p><table><tr><th>事件</th><th>时间</th><th>阶段</th><th>输入</th><th>回答/反馈</th><th>写入/淘汰</th><th>驻留状态</th></tr>'''
    for f in sample['trace']:
        e=f['event'];inp=f"{e['kind']} key={e.get('key','-')} version={e.get('version','-')} topic={e.get('topic','-')}"
        answer=f['answer']['item_id'] if f['answer'] else '-'
        if e['kind']=='feedback':answer=f"q{e['query_event_id']}: {f['feedback_value']}"
        action=f['action'] or {};states='\n'.join(f"{i['item_id']} episode={i['episode']}: "+'/'.join(f'{i[k]:.4f}' for k in ('I','C','U','T')) for i in f['resident'])
        body+=f"<tr><td>{e['event_id']}</td><td>{e['time']:.2f}</td><td>{f['evaluation']['phase']}</td><td>{html.escape(inp)}</td><td>{html.escape(str(answer))}</td><td>{html.escape(str(action.get('action','-')))} victim={html.escape(str(action.get('victim','-')))}</td><td><details><summary>{len(f['resident'])}条，active={f['active_residents']} noise={f['noise_residents']}</summary><pre>{html.escape(states)}</pre></details></td></tr>"
    (out/'event_trace.html').write_text(body+'</table>',encoding='utf-8')


def report(out,summary,diagnostics):
    def table(rows,cols):
        return '<table><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(format(r[k],".4f") if isinstance(r[k],float) else str(r[k]))}</td>' for k in cols)+'</tr>' for r in rows)+'</table>'
    page='''<!doctype html><meta charset="utf-8"><title>B1 interleaved sanity</title><style>body{max-width:1220px;margin:30px auto;padding:0 20px;font:16px/1.75 system-ui;color:#243b4c}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:15px;background:#edf4f7}</style><h1>B1：controlled pressure与interleaved stream sanity check</h1><p class="note">只运行mini：10条条件轨迹、2个冻结策略种子、5方法，容量8，共100次评测和100次确定性重放。没有训练或API调用，正式B1未启动。容量16仅进行单元检查，没有扩展正式评测。</p>
<h2>压力与日程定义</h2><p>每阶段活跃事实8条：4个稳定事实+4个当前主题事实；另有4条休眠主题事实。noise ratio定义为不同噪声键数/活跃键数，所以low/medium/high为2/8、8/8、16/8。这不是噪声事件百分比。所有压力条件每轮同样有8次核心观察和8次噪声观察，查询及时间轴完全一致，只扩大噪声池；每阶段实际覆盖全部配置噪声键。</p>
<p>先交错介绍知识，再执行A阶段、A→B切换/适应、B→A返回/恢复。每次观察后接一个查询和一个独立反馈事件，查询不问刚观察的同一键。稳定与当前主题事实拥有相同刷新次数；参考答案与稳定标签不进入策略。真实修订只在观察到达时发生。</p>
<p>三个世界各配对三种压力，共9条；第10条是另一个世界的medium无切换阴性对照。压力表仅汇总前9条，不把阴性对照混入medium平均。不同压力是同世界副本，不能当独立样本。</p>
<h2>检查与指标</h2><p>候选写入机会、初始记忆、输入、查询预算一致；有重复观察后，实际溢出机会会因驻留内容不同而不同，真实替换、拒收和原位刷新分别保存。寿命含右截尾，驻留年龄逐事件保存；分位数是已观察时长，非无偏寿命估计。</p>
<p>适应lag以切换后的当前主题查询为单位：连续两个4查询窗口≥75%，最低5，未成功截尾于32次；同时报告恢复成功率、时间单位lag及首次窗口已经良好的情况。返回A的恢复同样测量，但有新观察，不能冒称无提醒回忆。新版本获取另记录观察后第一次正确查询、失败后恢复及未恢复，未被查询的修订不计失败。</p>'''
    page+=table(diagnostics,['method','runs','state_change_without_write_events','new_content_correct_answers','recoveries_after_miss','episodes','episodes_at_lag_floor'])
    page+='<h2>描述性结果，不用于选模型或宣布优势</h2>'+table(summary,['pressure','method','accuracy','stable_retention','stale_error','adaptation_lag','adaptation_success','eviction_rate_per_observation','noise_memory_count'])
    page+='<img src="pressure.png"><h2>真实轨迹中的状态变化</h2><p>固定选择首个世界medium压力、seed11的三个version1条目。I/C/U/T在驻留期间连续更新和反馈跳变；淘汰处留空，再出现必须来自重新观察。曲线是事件采样，不是对删除后的隐状态进行插值。</p><img src="memory_evolution.png">'
    page+='''<h2>解释边界</h2><ul><li>新日程已提供实际的新观察与再次获取机会，不再把查询块开始时的存活内容当成整个适应过程。</li><li>缓存替换成分仍然存在：当前I/C/U/T最终只影响保留评分，检索仍取驻留的最新版本。状态在无写入时变化是机制检查，并不能证明效果独立于动态替换；后续仍需同替换预算与关闭演化的对照。</li><li>可靠主题提示、精确键/版本和干净反馈仍是受控假设；没有真实语言Agent效果证据。旧123参数头只做冻结迁移检查，不调参、不训练。</li><li>压力是不同键的工作集扩大；事件流量、重复频率、旧版本占槽、休眠知识也是实际压力来源，全部分别记录，不声称三个名义等级等于相同的真实替换率。</li><li>lag最低5的检测分辨率可能使早已恢复的配置饱和；成功率、修订获取和失败后恢复必须一起看，不仅比较平均lag。</li></ul><p><strong>停止在mini汇报阶段；不会自动启动正式B1。</strong></p>
<p><a href="event_trace.html">完整逐事件可读trace</a> · <a href="sample_full_trace.json">完整trace JSON</a> · <a href="sample_trajectory.json">样例环境</a> · <a href="metrics.json">全部metric JSON</a> · <a href="protocol.json">预登记协议</a> · <a href="fairness_audit.json">公平性审计</a> · <a href="diagnostics.json">适应机会与状态诊断</a> · <a href="replay_audit.json">重放记录</a></p>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='runs/ame-validation-v2/B1-interleaved-sanity-v1')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output outside workspace')
    if (out/'completion.json').exists():verify(out);return
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve incomplete evidence; use new output directory')
    out.mkdir(parents=True);(out/'traces').mkdir()
    source,archive=snapshot_source();paths=[ROOT/'runs/liquid-consolidation-v2'/n for n in ('stories.json','vectors.json')]
    checkpoints={s:ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt' for s in (11,23)}
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths+list(checkpoints.values())}
    cases=[dict(world_seed=w,trajectory_seed=w+200000,pressure=p,capacity=8,shift=True) for w in (60401,60402,60403) for p in PRESSURES]
    cases.append(dict(world_seed=60404,trajectory_seed=260404,pressure='medium',capacity=8,shift=False))
    protocol=dict(registered_at=time.time(),cases=cases,policy_seeds=[11,23],methods=METHODS,capacity=8,
        supported_capacities=[8,16],capacity16='unit tests only; not a mini result',training_steps=0,api_calls=0,
        source=source,archive=archive,input_hashes=hashes,runtime=runtime_info(),
        pressure='distinct noise keys /8active keys:2,8,16; same32noise and32core observations per phase, same query/time tapes',
        stream='12knowledge-introduction triples; A->B->A with4rounds x16observe/query/feedback triples per phase. No immediate same-key readback.',
        metrics='Scored phases0..2 exclude knowledge introduction;32current-topic queries per phase,lag threshold two4query windows >=75%,right-censor32; revision exposure clock separate.',
        control='one no-shift tape; excluded from pressure means; no tuning or statistical significance tests',
        visualization='Preselected first world/medium/seed11/adaptive evolution, first stable/A/B version1 items; show eviction gaps.',
        stop='Do not continue to formal B1. Frozen head11/23 only. All baseline parameters unchanged.')
    atomic_json(out/'protocol.json',protocol)
    stories,vectors=[read(p) for p in paths];heads={s:frozen_head(p) for s,p in checkpoints.items()}
    rows=[generate(stories,vectors,**c) for c in cases];environment=[audit_environment(r) for r in rows]
    for world in (60401,60402,60403):
        paired=[r for r in rows if r['world_seed']==world]
        stripped=lambda r:[w for w in r['events'] if not w['evaluation'].get('noise_observation',False)]
        assert stripped(paired[0])==stripped(paired[1])==stripped(paired[2])
        assert paired[0]['initial_memory']==paired[1]['initial_memory']==paired[2]['initial_memory']
    write_gzip(out/'trajectories.json.gz',rows);atomic_json(out/'sample_trajectory.json',rows[1])
    atomic_json(out/'environment_audit.json',environment)
    metrics=[];audits=[];sample=None
    for n,row in enumerate(rows):
        for seed in (11,23):
            methods=list(METHODS);methods=methods[n%5:]+methods[:n%5]
            for method in methods:
                head=heads[seed] if method=='adaptive_evolution' else None
                result=run(row,method,seed,head);check=audit(row,result)
                repeat=run(copy.deepcopy(row),method,seed,head)
                for k in ('trace','queries','lifetimes','episodes','acquisition'):assert result[k]==repeat[k],k
                for k,v in result['metrics'].items():
                    if k not in ('policy_seconds','timing_seconds'):assert v==repeat['metrics'][k],k
                filename=f'traces/{n}-{seed}-{method}.json.gz';result['metrics']['trace_file']=filename
                write_gzip(out/filename,result);metrics.append(result['metrics'])
                audits.append(dict(trajectory=n,method=method,seed=seed,**check))
                if n==1 and method=='adaptive_evolution' and seed==11:sample=result
    fairness=[]
    for n,row in enumerate(rows):
        rr=[r for r in metrics if r['environment_sha256']==row['environment_sha256']]
        for key in ('initial_sha256','input_sha256','query_sha256','capacity','observation_opportunities','queries'):
            assert len({r[key] for r in rr})==1,key
        assert all(r['policy_calls']['observe']==r['policy_calls']['evict'] and r['policy_calls']['retrieve']==r['policy_calls']['feedback']==204 for r in rr)
        fairness.append(dict(trajectory=n,passed=True,pressure=row['pressure'],active_memory_count=8,noise_memory_count=row['noise_memory_count'],
                             same_candidate_opportunities=True,actual_overflow_counts=[r['overflow_opportunities'] for r in rr]))
    summary=[];diagnostics=[]
    for pressure in PRESSURES:
        for method in METHODS:
            rr=[r for r in metrics if r['pressure']==pressure and r['method']==method and r['shift']]
            fields=['accuracy','stable_retention','stale_error','adaptation_lag','adaptation_success','eviction_rate_per_observation','noise_memory_count']
            summary.append(dict(pressure=pressure,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in fields}))
    for method in METHODS:
        rr=[r for r in metrics if r['method']==method];aa=[a for a in audits if a['method']==method]
        ep=[e for r in rr for e in read_gzip(out/r['trace_file'])['episodes'] if e['applicable']]
        diagnostics.append(dict(method=method,runs=len(rr),state_change_without_write_events=sum(r['state_change_without_write_events'] for r in rr),
            new_content_correct_answers=sum(a['correct_answers_unavailable_at_phase_start'] for a in aa),
            recoveries_after_miss=sum(r['acquisition_recoveries_after_miss'] for r in rr),episodes=len(ep),
            episodes_at_lag_floor=sum(e['lag_queries']==5 for e in ep)))
    assert all(d['new_content_correct_answers']>0 for d in diagnostics)
    atomic_json(out/'metrics.json',metrics);atomic_json(out/'summary.json',summary)
    atomic_json(out/'diagnostics.json',diagnostics);atomic_json(out/'fairness_audit.json',fairness)
    atomic_json(out/'trace_audit.json',audits);atomic_json(out/'replay_audit.json',dict(passed=True,runs=100,excluded_metrics=['policy_seconds','timing_seconds']))
    atomic_json(out/'sample_full_trace.json',sample)
    plots(out,rows,sample,summary);full_trace_page(out,sample);report(out,summary,diagnostics)
    assert hashes=={p.relative_to(ROOT).as_posix():sha(p) for p in paths+list(checkpoints.values())}
    atomic_json(out/'completion.json',dict(training_steps=0,api_calls=0,policy_runs=100,trajectory_count=10,
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(summary=summary,diagnostics=diagnostics),indent=2));print('Stopped at mini report; formal B1 not started.')


if __name__=='__main__':main()
