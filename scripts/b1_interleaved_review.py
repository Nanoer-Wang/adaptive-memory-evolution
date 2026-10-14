"""Post-run read-only interpretation, preserving all registered mini endpoints."""
import sys,json,html
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import snapshot_source
from scripts.b1_interleaved_sanity import verify
from scripts.b1_sanity_check import read_gzip
from scripts.ame_validation_batches import read,sha


def main():
    source=ROOT/'runs/ame-validation-v2/B1-interleaved-sanity-v1';out=ROOT/'runs/ame-validation-v2/B1-interleaved-review'
    out.mkdir(exist_ok=True,parents=True);verify(source)
    rows=read(source/'metrics.json');endpoints=[];manual=read(source/'sample_full_trace.json')
    for m in rows:
        r=read_gzip(source/m['trace_file'])
        for episode in r['episodes']:
            if not episode['applicable']:continue
            qq=[q for q in r['queries'] if q['phase']==episode['phase'] and q['target_query']]
            tail=[q['correct'] for q in qq[-8:]]
            endpoints.append(dict(method=m['method'],pressure=m['pressure'],world_seed=m['world_seed'],policy_seed=m['policy_seed'],
                phase=episode['phase'],first_hit_recovered=episode['recovered'],first_hit_lag=episode['lag_queries'],
                final_eight_query_accuracy=float(np.mean(tail)),final_two_windows_good=sum(tail[:4])>=3 and sum(tail[4:])>=3,
                status='posthoc diagnostic only; does not replace the registered first-hit endpoint'))
    atomic_json(out/'durability_diagnostic.json',endpoints)
    grouped=[]
    for method in dict.fromkeys(m['method'] for m in rows):
        rr=[r for r in endpoints if r['method']==method]
        grouped.append(dict(method=method,first_hit_success=float(np.mean([r['first_hit_recovered'] for r in rr])),
                            terminal_success=float(np.mean([r['final_two_windows_good'] for r in rr])),
                            terminal_accuracy=float(np.mean([r['final_eight_query_accuracy'] for r in rr]))))
    # Explicitly record the evidence inspected by the assistant, never claim an
    # independent human review. Assertions keep the cited trace facts exact.
    f=manual['trace'];item=lambda n,k:next(i for i in f[n]['resident'] if i['item_id']==k)
    assert f[230]['event']['kind']=='context' and f[230]['focus']==1
    assert item(232,'21:1')['I']<item(230,'21:1')['I']
    assert f[234]['action']['action']=='reject' and f[234]['action']['candidate']=='4:2'
    assert f[259]['answer'] is None and f[283]['answer'] is None
    assert f[297]['action']['action']=='replace' and f[297]['action']['candidate']=='4:2'
    assert f[307]['answer']['item_id']=='4:2' and f[308]['feedback_value']==1.
    assert item(425,'21:1')['I']>item(423,'21:1')['I']
    notes=dict(reviewer='assistant trace inspection; not independent human adjudication',
        sample='world60401,medium,capacity8,policy_seed11,adaptive_evolution; selected before results',
        reviewed_event_ranges=[[229,236],[259,260],[283,284],[297,308],[422,426]],
        findings=[
            'Event230 changes topic to B after advancing with prior held inputs; I of retained21:1 drops2.0659->1.6324 by event232 without evicting it.',
            'New version4:2 at234 is rejected; queries259/283 miss with feedback0; repeated observation297 admits4:2; query307 succeeds with feedback1 at308. This is actual delayed reacquisition, not a query-block availability proxy.',
            'At return context423, retained21:1 has I0.5000; before query425 it rises to0.9210 while retaining the same content/episode. Feedback426 increases U and resets T.',
            'Selected version1 records show C=1 while consistent; some old versions are evicted rather than showing a smooth C decline. The displayed C curve is not manufactured to suggest more variation.',
            'Evicted content is absent; later recovery requires observed content. Full event trace is available alongside the checked segments.'],
        functional_protocol_checks_passed=True,formal_B1_started=False,
        remaining_limits=['first-hit threshold does not guarantee persistent adaptation','state affects retention ranking; replacement-matched causal attribution not established','frozen transfer weights, three paired shift worlds, no independent significance claim'])
    atomic_json(out/'trace_review.json',notes)
    table='<table><tr><th>方法</th><th>首次过阈值成功率</th><th>末尾两个窗口均过阈值</th><th>末尾8查询准确率</th></tr>'
    for r in grouped:table+='<tr><td>'+html.escape(r['method'])+'</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ['first_hit_success','terminal_success','terminal_accuracy'])+'</tr>'
    table+='</table>'
    page='''<!doctype html><meta charset="utf-8"><title>Interleaved mini review</title><style>body{max-width:1150px;margin:35px auto;padding:0 20px;font:16px/1.8 system-ui;color:#243b4c}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:15px;background:#edf5f4}</style><h1>交错流mini：两项旧缺陷已修正，正式B1未启动</h1>
<p class="note">10条条件轨迹、2策略种子、容量8、5方法，共100次无训练评测及100次完整重放；容量16仅作接口单元检查。观察—查询—反馈逐项交错，压力只改变噪声键池大小，查询和时钟保持一致。</p>
<h2>检查结果</h2><ul><li>活跃事实8条，噪声池2/8/16条；每阶段实际遍历全部噪声键。noise ratio是不同键数比，不是噪声事件占比。</li><li>全部方法同初始内容、候选写入机会、查询预算与输入；重复观察的原位刷新、候选拒收和驻留替换分别计数，实际溢出数不强行说成相同。</li><li>所有100次回放都有逐事件内容重建、容量与重放审计。无切换对照的适应lag为null；人工构造的指标测试能区分立即恢复、延迟恢复和未恢复。</li><li>本轮轨迹检查由助手进行，覆盖切换前后、拒收、失败反馈、重观察和恢复片段；不是独立人工评审。</li></ul>
<h2>一个真实恢复过程</h2><p>完整样例的事件234首次观察新版本4:2但拒收；事件259、283查询失败，反馈为0；事件297重新观察后保存，事件307回答正确，308得到正反馈。首次观察后第3次相关查询恢复，经过72.9时间单位。与旧查询块不同，这个成绩依赖期间实际到达的新证据。</p>
<p>另一个始终驻留的21:1，在事件230切换到B后，I从2.0659下降到事件232的1.6324；返回A时，事件423至425的I从0.5000回升至0.9210。这些变化发生在同一内容和驻留episode内，可以检查演化过程，而不把重新插入当成隐状态复活。</p>
<h2>适应指标需要怎样解释</h2><p>原登记lag测量首次连续两个窗口达标，具有实际事件意义，但不意味着后续一直保持正确。下表是事后补充的持续性诊断，不修改原端点，也不用于挑选方法。</p>'''+table+'''
<p>必须同时看整体准确率、稳定保留、旧版本错误与持续性。在本mini中，自适应保留了更多稳定信息，但中/高压力下首次过阈值比即时上下文规则更慢；固定衰减容易频繁替换，首次过阈值成功并不保证后续稳定。阶段中途还会再次修订事实，因此末尾表现同时包含对后续修订的重新适应，不能单独归因于恢复后遗忘；这张事后诊断表不替代原登记指标。</p>
<h2>状态图与完整证据</h2><img src="../B1-interleaved-sanity-v1/memory_evolution.png"><p>图中条目按预登记规则选择，不因曲线好看而筛选；淘汰处留空，重新出现需要观察。C保持1的选中记录意味着其存活期间保持观察一致性。</p>
<p><a href="../B1-interleaved-sanity-v1/event_trace.html">完整可读event trace</a> · <a href="../B1-interleaved-sanity-v1/sample_full_trace.json">完整trace JSON</a> · <a href="../B1-interleaved-sanity-v1/metrics.json">全部metrics JSON</a> · <a href="../B1-interleaved-sanity-v1/index.html">压力条件完整报告</a> · <a href="trace_review.json">助手逐事件检查记录</a> · <a href="durability_diagnostic.json">持续性诊断</a> · <a href="tests.txt">测试日志</a></p>
<h2>尚不能作出的结论</h2><p>I/C/U/T目前只通过保留排序影响答案；这个mini证明状态会响应事件，也修复了固定干扰块和无新观察的问题，但没有证明收益独立于动态替换。需要后续同替换预算、无演化规则与稳定保留约束的归因实验。三种压力只有3个配对世界，冻结头未在新日程训练，不进行显著性或普遍优越性宣称。</p><p><strong>当前停止在mini汇报，未启动正式B1。</strong></p>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    signature,archive=snapshot_source()
    atomic_json(out/'manifest.json',dict(source=signature,archive=archive,primary_completion_sha256=sha(source/'completion.json'),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},training_steps=0,formal_B1_started=False))
    print(json.dumps(grouped,indent=2))


if __name__=='__main__':main()
