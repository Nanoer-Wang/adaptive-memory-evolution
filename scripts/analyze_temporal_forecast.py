"""Chronological test analysis with paired week-block uncertainty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/temporal-forecast-v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
metrics=read('metrics.json');params=read('parameters.json');timing=read('timing.json');rows=read('examples.json');p=np.load(OUT/'predictions.npz');y=p['truth'];records=[rows[int(i)] for i in p['ids']]
condition=np.asarray([r['condition'] for r in records]);weeks=np.asarray([str(pd.Timestamp(r['target_time']).to_period('W')) for r in records]);families=['persistence','ema','ridge_history','last_mlp','gru','input_flow','liquid','liquid_reset','liquid_unitdt']
def names(f):
    if f in ['persistence','ema','ridge_history']:return [f]
    if f in ['liquid_reset','liquid_unitdt']:return [f'liquid-{s}_'+f.split('_',1)[1] for s in [11,23,42]]
    return [f'{f}-{s}' for s in [11,23,42]]
summary=[];errors={}
for f in families:
    nn=names(f);errors[f]=np.mean([np.abs(p[n]-y) for n in nn],0)
    for c in ['observed','half_missing']:
        ix=condition==c;ms=[r for r in metrics if r['method'] in nn and r['condition']==c]
        summary.append(dict(family=f,condition=c,n=int(ix.sum()),mae_mean=float(np.mean([m['mae'] for m in ms])),mae_seed_sd=float(np.std([m['mae'] for m in ms])),rmse_mean=float(np.mean([m['rmse'] for m in ms]))))
intervals=[];rng=np.random.default_rng(20261012)
for c in ['observed','half_missing']:
    ix=condition==c;ww=sorted(set(weeks[ix]));counts=np.asarray([sum(ix&(weeks==w)) for w in ww])
    for other in ['persistence','last_mlp','gru','input_flow','liquid_reset','liquid_unitdt']:
        difference=errors['liquid']-errors[other];sums=np.asarray([difference[ix&(weeks==w)].sum() for w in ww]);samples=[]
        for _ in range(3000):
            ids=rng.integers(0,len(ww),len(ww));samples.append(float(sums[ids].sum()/counts[ids].sum()))
        intervals.append(dict(condition=c,contrast='liquid minus '+other,mae_difference=float(difference[ix].mean()),week_blocks=len(ww),percentile95=np.percentile(samples,[2.5,97.5]).tolist()))
cost=[]
for f in ['last_mlp','gru','input_flow','liquid']:
    rr=[r for r in params if r['method'] in names(f)];tt=[r['microseconds_per_24h_window'] for r in timing if r['method'] in names(f)]
    cost.append(dict(family=f,parameters=rr[0]['parameters'],hidden_state_bytes=rr[0]['hidden_state_bytes'],microseconds_per_window_mean=float(np.mean(tt))))
save('analysis.json',dict(summary=summary,paired_week_intervals=intervals,cost=cost,
    uncertainty='Paired whole-week bootstrap of seed-mean absolute errors, NOT errors of a prediction ensemble. Conditional on this site/year/trained seeds; residual temporal correlation and multiple contrasts remain. Exploratory intervals, no noninferiority claim.',
    conclusion='History-bearing liquid has lower error than resetting it, with fewer parameters than GRU and similar numerical accuracy, but slower CPU runtime. Input-only flow competes well; no unique coupled-liquid or elapsed-time benefit established. Numeric diagnostic, not language-memory result.'))
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ff=['persistence','last_mlp','gru','input_flow','liquid','liquid_reset'];labels=['Last value','Last MLP','GRU','Input flow','Liquid','Liquid reset']
for ax,c,title in zip(axes,['observed','half_missing'],['Observed rows (natural gaps remain)','Additional 50% observation thinning']):
    vals=[next(r['mae_mean'] for r in summary if r['family']==f and r['condition']==c) for f in ff]
    ax.barh(labels,vals,color=['#a2acb6','#859aaa','#537e9a','#639987','#ba8250','#baaaaa']);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.15);ax.set_title(title);ax.set_xlabel('6-hour forecast MAE (micrograms / cubic metre)')
    for i,v in enumerate(vals):ax.text(v+.15,i,f'{v:.2f}',va='center')
fig.suptitle('Real timestamps: history helps, coupled-liquid advantage remains unproven',fontsize=13);fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
table=''.join(f'<tr><td>{r["family"]}</td><td>{r["condition"]}</td><td>{r["mae_mean"]:.2f}</td><td>{r["rmse_mean"]:.2f}</td></tr>' for r in summary)
ct=''.join(f'<tr><td>{r["family"]}</td><td>{r["parameters"]}</td><td>{r["hidden_state_bytes"]}</td><td>{r["microseconds_per_window_mean"]:.1f}</td></tr>' for r in cost)
it=''.join(f'<tr><td>{r["contrast"]}</td><td>{r["mae_difference"]:.2f}</td><td>[{r["percentile95"][0]:.2f}, {r["percentile95"][1]:.2f}]</td></tr>' for r in intervals if r['condition']=='half_missing')
audit=read('split_audit.json')
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实时间序列机制诊断</title><style>body{{max-width:1150px;margin:40px auto;font:17px/1.7 system-ui;color:#253749}}h1,h2{{color:#143f57}}td,th{{padding:7px 14px;border-bottom:1px solid #ddd}}img{{width:100%}}.note{{background:#eef5f2;padding:18px}}a{{color:#176a91}}</style>
<h1>连续历史机制：真实时间序列诊断</h1><p>2026-10-12 · UCI 北京 PM2.5 · CPU · 12 个神经模型 · 3 个基础基线 · 0 API 请求</p>
<p class="note">本轮确认一个可推进的局部方向：保留历史的液态状态优于每步清空状态。液态参数少于 GRU，数值误差接近，但 CPU 更慢；输入条件化流也很强，尚无状态耦合液态独有优势。此结论只针对数值时间序列，不是语言记忆成绩。</p>
<h2>为什么换任务</h2><p>此前固定锚句的冗余标签主要由当前句对决定，难以检验连续历史。现改为公开真实小时观测，用过去 24 小时的污染、气象与风向预测窗口结束 6 小时后的 PM2.5。源数据与时间戳真实，额外随机删掉 50% 观测是明确的合成压力条件，不称真实传感器故障。</p>
<p>2010–2012 训练 {audit['train']['n']} 个窗口条件样本，2013 验证 {audit['validation']['n']}，2014 测试 {audit['test']['n']}；每 6 小时取一个目标。输入与目标均落在同一划分内；标准化仅用训练年份。自然缺失行不观测，不做未来插值。所有模型只看过去 11 个特征、观测掩码、真实间隔和距目标时间，不看未来天气。</p>
<p>基线包括最后观测、验证集选时间常数的 EMA、看到完整 24 小时向量与缺失掩码的岭回归、仅末次观测 MLP、GRU、输入条件化流、状态耦合液态。神经模型同宽度16、同12轮和优化预算、三种子；验证均方误差选检查点。参数数目不同，完整报告，不称严格容量匹配。</p>
<img src="comparison.png"><table><tr><th>方法</th><th>条件</th><th>MAE</th><th>RMSE</th></tr>{table}</table>
<h2>历史贡献与不确定性</h2><p>以下为额外缺失50%条件下液态与对照的 MAE 差，负值表示液态更好。区间按整周配对重采样3,000次，使用三种子平均绝对误差，不把模型预测平均成额外集成。相邻时间仍相关，单站点/单测试年和多项比较限制结论。没有预先设定非劣界限，因此误差接近不能写成统计等效。</p><table><tr><th>对比</th><th>MAE差</th><th>探索性95%区间</th></tr>{it}</table>
<p>状态清空消融使用相同已训练权重，说明本模型利用了历史，但不是重新训练的无历史最优模型；末次观测 MLP 提供另一个对照。单位时间消融没有显示稳定优势，因此不能单凭本轮宣称真实间隔建模本身带来增益。简单输入流也必须保留为下一轮强对照。</p>
<h2>资源计账</h2><table><tr><th>模型</th><th>参数</th><th>隐藏状态字节</th><th>CPU µs/24小时窗口</th></tr>{ct}</table><p>全部递归隐藏状态64字节，液态的优势只是相对GRU更少参数；它并未减少本轮隐藏状态容量。计时含单窗口前向，不含数据读取及训练。液态四子步求解较慢，不能宣称全面节省计算资源。</p>
<h2>如何继续推进</h2><p>保留“紧凑连续状态能保留有用历史”作为下一轮方向，在参数预算匹配和多站点独立验证下检验质量—参数—延迟取舍，同时比较输入流、GRU 和液态。若输入流已足够，不把相同收益包装成液态耦合独有贡献。要回到语言记忆，需要单独验证语义证据到连续状态的接口，本次气象预测不能替代该证据。</p>
<p>97 项测试通过。单独测试最初遇到本机 NumPy/torch 初始化顺序崩溃，修正测试导入顺序后全量通过；实验自身先初始化 NumPy，训练已正常完成。数据、权重、标准化、协议、逐样本预测和耗时均保存。</p>
<p>来源：<a href="https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data">UCI Beijing PM2.5 Data</a>。归档下载哈希见 protocol.json。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="metrics.json">每种子指标</a> · <a href="analysis.json">区间与汇总</a> · <a href="predictions.npz">逐样本预测</a> · <a href="examples.json">时间窗口记录</a> · <a href="timing.json">耗时</a> · <a href="parameters.json">参数量</a></p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8');print(json.dumps(cost));print(json.dumps([r for r in intervals if r['condition']=='half_missing']));print(json.dumps([r for r in summary if r['condition']=='half_missing']))
