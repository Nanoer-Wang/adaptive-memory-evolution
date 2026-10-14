"""Post-hoc content coverage audit; repeated events cannot hide missing content."""
import json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
raw={r['id']:r for r in json.loads((ROOT/'runs/role-relation-v1/examples.json').read_text(encoding='utf-8'))}
for version in [1,2]:
    out=ROOT/f'runs/evidence-history-v{version}'
    def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
    def save(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
    streams={s['id']:s for s in read('streams.json') if s['split']=='test'};pred=read('predictions.json');metrics=read('metrics.json');coverage=[]
    for p in pred:
        s=streams[p['stream']];needed=set();written=set();lost_events=0
        for e,skip in zip(s['events'],p['skip']):
            r=raw[e['id']];text=r['text2']
            if r['label']!=0:needed.add(text)
            if not skip:written.add(text)
            if r['label']!=0 and text not in written:lost_events+=1
        coverage.append(dict(stream=p['stream'],group=p['group'],method=p['method'],required_unique=len(needed),missing_unique=len(needed-written),missing_at_event=lost_events))
    save('content_coverage.json',coverage)
    aggregate=[]
    for m in metrics:
        cc=[r for r in coverage if r['method']==m['method']]
        if cc:aggregate.append(dict(method=m['method'],required_unique=sum(r['required_unique'] for r in cc),missing_unique=sum(r['missing_unique'] for r in cc),missing_at_event=sum(r['missing_at_event'] for r in cc)))
    save('coverage_summary.json',aggregate)
    groups=len({s['group'] for s in streams.values()});exact=next(m for m in metrics if m['method']=='exact_only');ema=next(m for m in metrics if m['method']=='ema_0.25')
    save('analysis.json',dict(test_source_groups=groups,extra_writes_saved_by_ema=exact['writes']-ema['writes'],ema_saving_relative_to_exact=(exact['writes']-ema['writes'])/exact['writes'],
        exact_repeat_ledger='All observed exact strings registered. A mistaken initial semantic skip can persist across repeats, so content_coverage separately counts distinct missing non-equivalent text and missing-at-event occurrences.',
        caution='Constructed public SICK replay, already used semantic data. Synthetic gaps and stationary anchor; target has no intrinsic elapsed-time dependence. Zero observed errors on small sample is not safe-skip certification. Different history audit pools across v1/v2; do not attribute cross-run changes solely to loss correction.'))
    if version==1:continue
    methods=['write_all','exact_only','protected_single','ema_0.25','static-23','gru-23','liquid-23'];names=['Write all','Exact only','Single evidence','EMA (tau=.25)','Static (seed23)','GRU (seed23)','Liquid (seed23)']
    mm={m['method']:m for m in metrics};vals=[mm[m]['writes'] for m in methods]
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ax.barh(names,vals,color=['#9aacbc','#758ca6','#dcaa64','#4b8e84','#839aad','#899b86','#b57a63']);ax.invert_yaxis();ax.set_xlabel('Writes / 3,180 replay events (lower is cheaper; errors reported separately)');ax.set_title('Guarded history replay: exact repeats explain most savings')
    for i,v in enumerate(vals):ax.text(v+15,i,str(v),va='center')
    ax.set_xlim(0,3500);fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    table=''.join(f'<tr><td>{m["method"]}</td><td>{m["writes"]}</td><td>{m["semantic_skips"]}</td><td>{m["semantic_errors"]}</td></tr>' for m in metrics if '_reset' not in m['method'] and '_unitdt' not in m['method'])
    params=read('parameters.json');timing=read('timing.json');speed={r['method']:r['seconds']/r['streams']/20*1e6 for r in timing}
    cost=''.join(f'<tr><td>{k}</td><td>{v["parameters"]}</td><td>{v["state_bytes"]}</td><td>{speed[k]:.1f}</td></tr>' for k,v in params.items() if k.endswith('-23'))
    ctable=''.join(f'<tr><td>{r["method"]}</td><td>{r["required_unique"]}</td><td>{r["missing_unique"]}</td><td>{r["missing_at_event"]}</td></tr>' for r in aggregate if r['method'] in ['exact_only','protected_single','ema_0.25','gru-23','liquid-23'])
    report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>保护证据后的历史决策实验</title><style>body{{max-width:1100px;margin:40px auto;font:17px/1.7 system-ui;color:#263749}}h1,h2{{color:#143e55}}img{{width:100%}}td,th{{padding:7px 14px;border-bottom:1px solid #ddd}}.note{{padding:18px;background:#eef5f2}}a{{color:#176a91}}</style>
<h1>从语义判断推进到历史决策</h1><p>CPU · 0 API 请求 · 2 轮机制实验 · 每种神经模型 3 个种子</p>
<p class="note">保护接口与历史比较已经跑通。确认的局部进展：精确去重后，简单 EMA 在本轮额外省掉 23 次写入（777→754，约 2.96%），未观察到额外内容漏存。液态、GRU 和静态神经模型在本轮均未选出满足操作点要求的额外语义跳过策略；不能把共同去重收益算作液态优势。</p>
<h2>实验具体是什么</h2><p>公开 SICK 中以固定旧句为锚，把候选描述构成每条 20 事件的回放；用有放回抽样产生精确重复，间隔取 0.25、1、4。该任务检验新描述应写入还是冗余：人工等义或此前精确观察过为冗余，其余新内容应保留。不是事实更新、真实人类会话或最终问答实验。</p>
<p>所有方案共享原始双向 NLI 六概率、角色等义分数及时间间隔；历史状态看不到标签、ID、来源组。非精确重复要同时满足双向蕴含≥0.9、矛盾&lt;0.1、角色等义≥0.5，历史只能进一步拒绝，不能推翻底层冲突证据。精确重复由共同账本处理。</p>
<p>第一轮用非重复事件训练；诊断发现 4,026 个训练事件中只有 528 个允许参与语义跳过，第二轮改为只在这些候选上训练。各神经模型同样修改；第二轮使用第一轮未回放的旧 SICK 校准部分作为历史审计。语义样本以前已被查看，因此仍是开发性机制实验，不是全新语义验证。两轮测试池不同，不能把跨轮分数变化当作目标修正的因果收益。</p>
<p>第二轮 159 条历史，3,180 事件，{groups} 个来源组。单条历史内部的顺序没有真实时间标签，跳过真值也不依赖时间间隔，所以这里不能证明连续物理时间的优势。</p>
<img src="comparison.png"><table><tr><th>方法</th><th>写入</th><th>额外语义跳过</th><th>语义错误</th></tr>{table}</table>
<h2>误判与内容覆盖</h2><p>本轮采用探索性的选择集语义误判≤5%、至少 20 个语义跳过操作点，阈值在测试前冻结。它不是之前的 1% 安全认证。精确重复不进入该误判分母，避免大量简单重复冲淡错误。EMA 的零错误仅来自 23 次额外跳过，样本太少，不足以认证低风险。</p>
<p>如果第一次错误丢弃，后面的精确重复不能把丢失掩盖掉。因此额外检查每条历史最终缺失的不同非等义内容，并统计各时刻缺失事件。所有评价原文仍保存；这是离线写入模拟，不执行实际永久删除。</p>
<table><tr><th>方法</th><th>应保存的不同内容</th><th>最终缺失</th><th>时刻缺失</th></tr>{ctable}</table>
<h2>资源计账与消融</h2><p>共同精确账本最多 20 个 SHA256 键，载荷上界 640 字节，另有固定锚文本；不含 Python 容器开销。神经额外状态 64 字节，参数量如下；EMA 使用一个标量。宽度相同但参数量不同，不能称严格参数匹配。时间仅计控制器，排除共同编码、账本与保护逻辑，不等于端到端加速。</p>
<table><tr><th>模型（种子23）</th><th>参数</th><th>历史状态字节</th><th>控制器 µs/事件</th></tr>{cost}</table>
<p>保存了每事件重置历史及液态转移使用单位时间的消融。第二轮神经方案全部退回精确去重，因此这些操作点无法证明隐藏状态贡献。第一轮 GRU 部分种子有少量额外跳过，液态仍没有稳定增益；完整结果不隐藏。</p>
<h2>可以如何继续</h2><p>把证据保护接口作为共同基础继续使用，EMA 作为必须保留的强历史基线。下一轮若研究液态连续性，应选有真实时间或可靠时间标签、需要跨事件积累才能判断的任务，避免只在当前句对即可决定的冗余判别上反复优化。精确去重、时间平滑和原始 NLI 均必须保留对照。</p>
<p><a href="../evidence-history-v1/metrics.json">第一轮完整结果</a> · <a href="protocol.json">第二轮冻结协议</a> · <a href="metrics.json">全部结果和消融</a> · <a href="content_coverage.json">逐历史内容覆盖</a> · <a href="streams.json">回放与间隔</a> · <a href="predictions.json">逐事件分数</a> · <a href="timing.json">耗时</a></p></html>'''
    (out/'report.html').write_text(report,encoding='utf-8');print(json.dumps(read('analysis.json')));print(json.dumps(aggregate[:5]));print(speed)
