"""Read-only review of mini evidence. Does not rerun/train or edit the mini."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import snapshot_source
from src.memory.b1_policy import Policy,frozen_head
from scripts.b1_sanity_check import read_gzip,read,verify
from scripts.ame_validation_batches import sha


def plot_exact_probe(source,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probe=read(source/'controlled_probe.json')
    head=frozen_head(ROOT/'runs/evolving-memory-v1/checkpoints/learned-11.pt')
    curves={}
    for method in ('adaptive_rule','adaptive_evolution'):
        p=Policy(method,8,11,head if method=='adaptive_evolution' else None);p.bootstrap(probe['initial'])
        points=[]
        for e in probe['events']:
            old=p.items['0:1'];start=p.now
            target=.5+p.held_w*(old.topic==p.focus);valid=float(p.ledger[old.key]==old.item_id)
            for dt in np.linspace(0,e['time']-start,51):
                points.append([start+dt,target+(old.I-target)*np.exp(-p.held_k*dt),
                               valid+(old.C-valid)*np.exp(-4*p.held_k*dt),old.U*np.exp(-p.held_k*dt/20),old.T+dt])
            p.update_state(e)
            if e['kind']=='observe':p.observe(e);p.evict()
            if e['kind']=='query':p.retrieve(e);p.feedback('0:1',1.)
            if e['kind']=='feedback_probe':p.feedback('0:1',0.)
            old=p.items['0:1'];points.append([p.now,old.I,old.C,old.U,old.T])
            expected=next(r['state'] for r in probe['results'][method] if r['time']==e['time'] and r['event']==e['kind'])
            np.testing.assert_allclose(points[-1][1:],[expected[f] for f in ('I','C','U','T')],rtol=1e-12,atol=1e-12)
        curves[method]=np.array(points)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,(ax,field) in enumerate(zip(axes.flat,('I: importance','C: observed consistency','U: usage','T: elapsed time'))):
        for method,color in [('adaptive_rule','#178e78'),('adaptive_evolution','#6551b4')]:
            v=curves[method];ax.plot(v[:,0],v[:,i+1],label=method,color=color)
        for t in (10,20,30,42):ax.axvline(t,color='grey',alpha=.25,ls='--')
        ax.set(title=field,xlabel='Observed time');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Exact held-input flow + event jumps (controlled probe, not benchmark accuracy)\nTopic B at10; negative feedback at20; topic A at30; new content version at42',fontsize=12)
    fig.savefig(out/'memory_evolution_exact.png',dpi=150);fig.savefig(out/'memory_evolution_exact.svg');plt.close(fig)


def main():
    source=ROOT/'runs/ame-validation-v2/B1-sanity-v1';out=ROOT/'runs/ame-validation-v2/B1-sanity-review'
    out.mkdir(exist_ok=True,parents=True);verify(source)
    rows=read_gzip(source/'trajectories.json.gz');metrics=read(source/'metrics.json');summ=read(source/'summary.json')
    byhash={r['environment_sha256']:r for r in rows};phases=[]
    for m in metrics:
        row=byhash[m['environment_sha256']];trace=read_gzip(source/m['trace_file'])['trace']
        seen=set()
        for entry in row['events']:
            if entry['input']['kind']!='query':continue
            phase=entry['evaluation']['phase']
            if phase in seen:continue
            seen.add(phase);event_id=entry['input']['event_id'];resident=trace[event_id]['resident']
            distractor_fraction=sum(i['key'] in row['distractor_keys'] for i in resident)/8
            remaining_queries=[e for e in row['events'] if e['input']['kind']=='query' and e['evaluation']['phase']==phase]
            live={i['item_id'] for i in resident}
            ceiling=sum(e['evaluation']['target_id'] in live for e in remaining_queries)/len(remaining_queries)
            actual=[q for q in read_gzip(source/m['trace_file'])['queries'] if q['phase']==phase]
            assert abs(ceiling-np.mean([q['correct'] for q in actual]))<1e-12
            phases.append(dict(method=m['method'],world_seed=m['world_seed'],trajectory_seed=m['trajectory_seed'],
                stable_fraction=m['stable_fraction'],regime=m['regime'],policy_seed=m['policy_seed'],phase=phase,
                distractor_fraction=distractor_fraction,available_query_ceiling=ceiling,
                no_observations_inside_query_block=True))
    fixed=[r for r in phases if r['method']=='fixed_decay']
    assert min(r['distractor_fraction'] for r in fixed)>=.75
    diagnosis=dict(functional_checks_passed=True,formal_B1_ready=False,
        reason='Chronological order structurally disadvantages recency: six distractors immediately precede every query block. No observations inside a query block, so missed content cannot be reacquired; measured query-window adaptation is initial content availability, not online acquisition.',
        fixed_decay_phase_checks=len(fixed),fixed_decay_min_distractor_occupancy=min(r['distractor_fraction'] for r in fixed),
        all_methods_query_block_ceiling_exactly_matches_accuracy=True,
        phases=phases,
        next_protocol_changes=['Randomize/interleave relevant observations, distractors and queries with independent scheduling RNG; retain burst schedule as named stress condition.',
             'Balance or explicitly factor stable/volatile refresh opportunity; do not pass evaluator stable labels online.',
             'Include causal post-shift observations while measuring acquisition lag; keep genuinely no-reminder retention as a separate metric.',
             'Repeat mini with new reserved sanity seeds and protocol before formal training. Preserve current outputs and show all conditions, not only favorable schedules.'],
        scope='Posthoc design diagnosis of sanity evidence, not formal effect comparison; no new training or benchmark rollouts.')
    atomic_json(out/'protocol_diagnosis.json',diagnosis);plot_exact_probe(source,out)
    table='<table><tr><th>方法</th><th>准确率</th><th>稳定保留</th><th>旧版本错误</th><th>替换率</th><th>平均拒收次数</th></tr>'
    for r in summ:
        table+='<tr><td>'+r['method']+'</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ('accuracy','stable_retention','stale_retrieval','replacement_rate'))+f'<td>{r["rejected_write_count"]:.2f}</td></tr>'
    table+='</table>'
    page='''<!doctype html><meta charset="utf-8"><title>B1 sanity review</title><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.8 system-ui;color:#213b4d}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.callout{background:#fff0dc;padding:18px;border-radius:8px}</style>
<h1>B1 sanity check：实现通过，协议尚未通过</h1><p class="callout">8条条件轨迹 × 2策略种子 × 5方法，容量8；80次无训练评测及80次确定性重放。没有正式B1训练，也没有API调用。当前发现的日程偏差需要先修正并重新做mini，不能直接进入正式训练。</p>
<h2>框架结构</h2><p>独立80键公开事实生成器 → 相同初始快照与事件输入 → MemoryPolicy五接口 → 只读评估器 → 内容生存/版本/资源计数审计。参考答案与稳定标签仅供评估；事后反馈通过统一接口返回。</p>
<h2>策略说明</h2><p>fixed_decay为纯年龄指数；frequency统计返回次数；adaptive_evolution加载原123参数头的冻结11/23权重；adaptive_rule保留相同四状态方程、固定系数；immediate_context用当前主题、观察一致性和频率，无演化隐状态。所有方法同容量、相同候选拒收机会、同top1读取；旧v1共用的显式版本惩罚已对所有方法移除，旧成绩不与新表拼接。</p>
<h2>检查结果</h2><p>初始记忆/输入/查询哈希一致；各方法写入与溢出机会一致；真实替换+候选拒收=溢出机会；容量从未超过8。全部80条完整轨迹逐事件重建与重放相符。检查了因果时间流、旧版本错误可见、淘汰内容不可恢复、寿命右截尾，以及NumPy冻结头与原123参数头数值一致。</p>
<h2>描述性结果——不作为方法优越性证据</h2>'''+table+'''
<h2>发现的两项阻断问题</h2><ol><li><strong>事件排列系统性压低年龄策略。</strong>每阶段最后6条观察都是干扰事实。在全部96个固定衰减阶段检查中，查询开始时至少75%槽位是干扰。这是预设到达顺序的结果，不能解释成演化机制更优。</li><li><strong>“适应滞后”在当前日程中不具备想要的含义。</strong>40个查询组成一个没有新观察的连续块；内容缺失后，该块内没有获取新知识的机会。逐阶段验证发现，查询开始时的内容可用率恰好决定整块准确率。窗口过阈值更多反映查询顺序/初始存活，不能作为在线再适应速度。</li></ol>
<p>此外，固定50%稳定查询会改变不同事实比例下的单条访问频率；稳定事实只观察一次而易变事实得到修订；提前主题提示可靠；冻结旧头未经新分布训练；实际替换率也未匹配。即时规则在此mini中分数更高，但同样不能据此下研究结论。</p>
<h2>状态演化图</h2><p>预先指定的机制探针，显示精确分段流和事件跳变。没有容量竞争，不代表基准效果。I随主题离开降低、主题返回回升；C响应负反馈并在版本变旧后降低；U逐渐衰减；T在反馈时重置。已删除事实不能凭这些状态重建。</p><img src="memory_evolution_exact.png">
<h2>正式B1前的修正清单</h2><ol><li>用独立随机种子交错安排相关观察、干扰和查询；保留“干扰突发”作为命名压力条件，避免它垄断主实验。</li><li>独立控制内容刷新机会与稳定性；保持角色标签不可见。</li><li>测适应速度的阶段允许真实的修订/重观察到达；无提醒回访仍单列，不混成恢复。</li><li>协议先登记，再用未用于此mini的新sanity种子复查；不读取正式测试种子、不训练、不按正结果选日程。</li></ol>
<p><strong>当前停在汇报阶段，正式B1未启动。</strong></p>
<p><a href="../B1-sanity-v1/index.html">原始mini报告与完整指标</a> · <a href="../B1-sanity-v1/sample_trajectory.json">样例轨迹</a> · <a href="../B1-sanity-v1/metrics.json">metric JSON</a> · <a href="protocol_diagnosis.json">日程诊断</a> · <a href="tests.txt">测试记录</a></p>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    signature,archive=snapshot_source()
    atomic_json(out/'review_manifest.json',dict(source=signature,archive=archive,
        primary_completion_sha256=sha(source/'completion.json'),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='review_manifest.json'},
        formal_B1_started=False,training_steps=0,api_calls=0))
    print(json.dumps({k:v for k,v in diagnosis.items() if k!='phases'},indent=2))


if __name__=='__main__':main()
