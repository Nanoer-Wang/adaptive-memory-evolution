"""Post-hoc diagnostics only: never modifies frozen thresholds or predictions."""
import json, html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/relation-verifier-v1'
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def save(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

records=[json.loads(s) for s in (OUT/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
examples={r['id']:r for r in read('examples.json')}
summary=read('summary.json');methods=list(records[0]['scores'])
grid=[]
for split in ('selection','certificate','fresh_paws','mrpc','revision'):
    rows=[r for r in records if r['split']==split]
    for m in methods:
        for t in (.5,.8,.9,.95,.99,.995,.999,.9995,.9999):
            selected=[r for r in rows if r['scores'][m]>=t]
            errors=sum(r['label']==0 for r in selected)
            grid.append(dict(split=split,method=m,threshold=t,n=len(rows),skips=len(selected),errors=errors,
                             wrong_among_skips=errors/len(selected) if selected else None))
save('diagnostic_grid.json',grid)
# Fixed descriptive cutoff; NOT a newly selected policy.
errors=[]
for r in records:
    if r['label']==0 and r['scores']['nli_bidirectional_guarded']>=.99:
        errors.append(dict(**r,text1=examples[r['id']]['text1'],text2=examples[r['id']]['text2']))
save('high_confidence_errors.json',errors)
t=read('timing.json')
timing=dict(total_scoring_wall_seconds=sum(b['wall_seconds'] for b in t['blocks']),
    cold_nli_seconds=sum(b['seconds'] for b in t['cold_nli_batches']),
    cold_directional_pairs=sum(b['pairs'] for b in t['cold_nli_batches']),
    cache_hits=sum(b['stats'][d]['cache_hits'] for b in t['blocks'] for d in ('forward','backward')),
    note='CPU batch throughput, not single-request latency; excludes initial model loading/download. No end-to-end write savings measured.')
save('analysis.json',dict(timing=timing,high_confidence_negative_count=len(errors),
    conclusion='No candidate passed selection. All policies retain all; zero errors is vacuous. NLI improves over cosine on PAWS but loses to lexical logistic; external MRPC ranking differs. No liquid-history result.'))
labels=['Cosine','Lexical logistic','LTC (3-seed mean)','One-way NLI','Mutual NLI','Mutual + veto']
names=['old_cosine','old_logistic','ltc_mean','nli_oneway','nli_bidirectional','nli_bidirectional_guarded']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,split,title in zip(axes,['fresh_paws','mrpc'],['Fresh PAWS / 1,500 pairs','External MRPC / 1,725 pairs']):
    d={r['method']:r['auc'] for r in summary if r['split']==split and r['policy']=='candidate'}
    d['ltc_mean']=float(np.mean([d[f'old_ltc-{seed}'] for seed in (11,23,42)]))
    vals=[d[n] for n in names]
    ax.barh(labels,vals,color=['#9aa8b4','#517399','#83956d','#e3b169','#d18a43','#b75839'])
    ax.set_xlim(.5,.9);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('ROC AUC (ranking, not safe-skip accuracy)')
    for i,v in enumerate(vals):ax.text(v+.003,i,f'{v:.3f}',va='center')
fig.suptitle('Semantic verification audit — no policy met the skip-risk criterion',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
tr=''.join(f'<tr><td>{r["split"]}</td><td>{r["method"]}</td><td>{r["auc"]:.4f}</td><td>{r["skips"]}</td></tr>' for r in summary if r['policy']=='candidate' and r['split'] in ('fresh_paws','mrpc'))
errhtml=''.join(f'<li><b>{html.escape(r["id"])}</b> · 分数 {r["scores"]["nli_bidirectional_guarded"]:.4f}<p>{html.escape(r["text1"])}</p><p>{html.escape(r["text2"])}</p></li>' for r in sorted(errors,key=lambda r:-r['scores']['nli_bidirectional_guarded'])[:15])
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语义关系可靠性实验</title>
<style>body{{max-width:1100px;margin:40px auto;font:17px/1.65 system-ui;color:#253348}}h1,h2{{color:#163d55}}img{{width:100%}}td,th{{padding:6px 15px;border-bottom:1px solid #ddd}}.note{{padding:18px;background:#fff1dc}}li{{margin:20px 0}}a{{color:#176a94}}</style>
<h1>先验证语义，再研究连续历史</h1><p>2026-10-12 · 8,722 对样本 · 8 个方法 · CPU · 0 API 请求</p>
<p class="note">本轮尚未建立可靠的自动跳过判据。8 个方法均未通过预先固定的选择条件，因此全部保留信息。跳过 0 条不代表语义判断准确，也不构成成本优势。</p>
<h2>改动与结果</h2><p>引入冻结 DeBERTa NLI 联合编码，分别判断两个方向的蕴含，另测数字与否定变化的否决规则。双向判断在两个数据集均优于单向；否决规则降低本轮总体排序能力，不能据此宣称提高可靠性。PAWS 上词法分类器仍领先；MRPC 上双向 NLI 排序更好，说明优势依赖数据分布。</p>
<img src="comparison.png"><table><tr><th>数据</th><th>方法</th><th>AUC</th><th>获准跳过</th></tr>{tr}</table>
<h2>协议与边界</h2><p>用旧 PAWS 选择集 3,850 对选固定网格阈值：至少跳过 100 对，且跳过集合中错误比例 ≤0.5%。随后用未使用过的 1,500 个句子连通分量代表进行独立校准，单侧 95% Clopper–Pearson 上界要求 ≤1%。本轮在选择阶段就无合格候选，未发生有效的低风险认证。</p>
<p>另取未使用的 1,500 个 PAWS 分量代表审计；它们来自原官方训练集剩余部分，并非官方测试集。MRPC 官方测试集 1,725 对为本项目新增外部数据。147 条 MemOps 修订沿用上轮，不能称为新验证。所有标签保留，不删除困难样本。句子隔离不排除文章相关、预训练污染或域偏移。MRPC 的等义标签不等于严格双向蕴含。</p>
<p>模型为 <a href="https://huggingface.co/cross-encoder/nli-deberta-v3-small">公开 NLI DeBERTa</a> 的固定量化 ONNX 版本；未验证与全精度模型的一致性。<a href="https://huggingface.co/datasets/nyu-mll/glue">MRPC 数据来源</a>。模型、数据哈希与版本见 protocol.json。</p>
<h2>高置信错误诊断</h2><p>以下为官方负标签且双向带守卫得分 ≥0.99 的事后诊断样例（按分数展示前 15 个，共 {len(errors)} 个）。0.99 只用于检查错误，不是新部署阈值。完整样例已保存。</p><ol>{errhtml}</ol>
<h2>运行记录与下一步</h2><p>评分累计耗时 {timing['total_scoring_wall_seconds']:.1f} 秒；冷 NLI 内核 {timing['cold_nli_seconds']:.1f} 秒、{timing['cold_directional_pairs']} 个方向句对；缓存命中 {timing['cache_hits']}。这是批处理测量，不是在线延迟，也不是端到端节省。</p>
<p>下一步应在训练侧补充关系角色、实体绑定和单向新增信息监督，另留新数据评价；不继续用本轮审计集选模型。允许不确定信息保留或延期决策。等语义层建立非零覆盖的可靠基准后，再固定该层，对比无历史、指数平滑、GRU 与液态连续状态，控制共同输入、参数与计算预算，并分别打乱顺序、时间间隔和重复次数检验历史贡献。</p>
<p>87 项测试通过（tests.txt）。默认系统临时目录权限导致首次测试夹具失败，指定新的项目内临时目录后全量通过。</p>
<p><a href="scores.jsonl">逐样本双向分数</a> · <a href="diagnostic_grid.json">完整诊断阈值表</a> · <a href="high_confidence_errors.json">高置信错误原文</a> · <a href="protocol.json">冻结协议</a> · <a href="timing.json">性能记录</a> · <a href="split_audit.json">划分审计</a></p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print(json.dumps(timing));print('high confidence errors',len(errors))
print([(r['id'],r['text1'],r['text2']) for r in sorted(errors,key=lambda r:-r['scores']['nli_bidirectional_guarded'])[:5]])
