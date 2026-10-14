"""Post-freeze paired source bootstrap, error audit and human-readable report."""
import json,html,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/role-relation-v1'
CLASSES=('equivalent','new_addition','new_weaker','contradiction','unresolved')
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def macro(cm):
    den=cm.sum(0)+cm.sum(1)
    return float(np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(5),where=den>0)))

metrics=read('metrics.json');test=[r for r in metrics if r['split']=='test']
rows=read('examples.json');lookup={r['id']:r for r in rows}
preds=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
groups=sorted({p['group'] for p in preds if p['split']=='test'});gi={g:i for i,g in enumerate(groups)}
cms={m:np.zeros((len(groups),5,5)) for m in ('nli_lexical','nli_lexical_roles')}
for p in preds:
    if p['split']=='test' and p['method'] in cms:cms[p['method']][gi[p['group']],p['label'],p['prediction']]+=1
rng=np.random.default_rng(20261012);deltas=[]
for _ in range(3000):
    ids=rng.integers(0,len(groups),len(groups))
    deltas.append(macro(cms['nli_lexical_roles'][ids].sum(0))-macro(cms['nli_lexical'][ids].sum(0)))
ci=dict(method='Paired original-caption/source-component bootstrap, 3000 resamples, seed 20261012; conditional on fitted models; no training-seed or domain uncertainty.',groups=len(groups),
        delta=macro(cms['nli_lexical_roles'].sum(0))-macro(cms['nli_lexical'].sum(0)),percentile95=np.percentile(deltas,[2.5,97.5]).tolist())
errors=[]
for p in preds:
    if p['split']=='test' and p['method']=='nli_lexical_roles' and p['label']!=p['prediction']:
        r=lookup[p['id']];errors.append(dict(**p,text1=r['text1'],text2=r['text2']))
save('test_errors.json',errors)
external=read('external/metrics.json');exttest=[r for r in external if r['split']=='test']
save('analysis.json',dict(role_increment=ci,test_errors=len(errors),history_admitted=False,
    source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('scripts/role_relation_external.py','scripts/analyze_role_relation.py')},
    caveat='SICK safety criterion underpowered from group counts, not a powered failed trial. External PAWS audit separately measures binary equivalence. No historical-state experiment run.'))
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
labels=['Frozen NLI','Supervised NLI','+ Lexical','+ Role bindings'];vals=[r['macro_f1'] for r in test]
axes[0].bar(labels,vals,color=['#a6b4c1','#6d94b5','#51877d','#c17a40']);axes[0].set_ylim(0,1);axes[0].set_title('Source-isolated SICK / 2,402 directed pairs');axes[0].set_ylabel('Five-class macro F1');axes[0].tick_params(axis='x',labelrotation=12)
for i,v in enumerate(vals):axes[0].text(i,v+.015,f'{v:.3f}',ha='center')
cm=np.asarray(test[-1]['confusion']);norm=cm/cm.sum(1,keepdims=True)
axes[1].imshow(norm,vmin=0,vmax=1,cmap='Blues');axes[1].set_xticks(range(5),['Equal','Addition','Weaker','Conflict','Other'],rotation=25);axes[1].set_yticks(range(5),['Equal','Addition','Weaker','Conflict','Other']);axes[1].set_xlabel('Predicted');axes[1].set_ylabel('Human directional labels');axes[1].set_title('Role-aware model: row-normalized confusion')
for i in range(5):
    for j in range(5):axes[1].text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
trs=''.join(f'<tr><td>{r["method"]}</td><td>{r["accuracy"]:.3f}</td><td>{r["macro_f1"]:.3f}</td><td>{r["report"]["contradiction"]["recall"]:.3f}</td></tr>' for r in test)
ext=''.join(f'<tr><td>{r["method"]}</td><td>{r["auc"]:.3f}</td><td>{r["skips"]}</td><td>{r["errors"]}</td></tr>' for r in exttest)
errs=''.join(f'<li><b>{html.escape(e["id"])}</b> · {CLASSES[e["label"]]} → {CLASSES[e["prediction"]]}<p>旧：{html.escape(e["text1"])}</p><p>新：{html.escape(e["text2"])}</p></li>' for e in sorted(errors,key=lambda e:-max(e['probabilities']))[:12])
lo,hi=ci['percentile95'];audit=read('split_audit.json');extcomplete=read('external/complete.json')
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>关系角色监督实验</title>
<style>body{{max-width:1100px;margin:40px auto;font:17px/1.7 system-ui;color:#263648}}h1,h2{{color:#153e57}}img{{width:100%}}td,th{{padding:8px 16px;border-bottom:1px solid #ddd}}.note{{padding:18px;background:#fff0d9}}li{{margin:22px 0}}a{{color:#1c688f}}</style>
<h1>关系角色监督：有效增量与仍未解决的可靠性</h1><p>2026-10-12 · CPU · 0 API 请求 · 90 项测试通过</p>
<p class="note">角色特征带来可测改善，但语义层尚未通过预设准入条件，未启动液态、GRU 或历史统计比较。不得把本轮增益归因于液态机制。</p>
<h2>实际补充了什么</h2><p>使用公开 SICK 的人工双向蕴含标注训练五类关系：等义、新信息严格增加、信息变弱、矛盾、其余未确定关系。旧→新无法推出但新→旧可以推出，才标为单向新增；中性不直接等于新增。正反样本同组，新增与变弱互换。SICK 611 对方向不一致样本完整保存，不强制编造标签；成绩仅适用于其余 9,229 个原始句对。</p>
<p>冻结句法模型抽取动作、施事、受事，主动被动语态映射到共同角色，并构造谓词—角色对应、三元组重合与角色交叉特征。模型使用人工关系标签学习这些自动角色特征；本轮没有新增人工角色金标，也没有微调大型语义编码器。角色互换不自动标成矛盾，两件事可能同时发生。</p>
<p>关系表示来自冻结双向 NLI，三种轻量分类器分别使用 NLI、NLI 加词法、NLI 加词法与角色。相同训练/选择数据，相同正则化搜索范围，仅由选择集宏 F1 选配置；测试前固定权重。原始 NLI 直接映射也保留作为基线。</p>
<img src="comparison.png"><table><tr><th>方法</th><th>准确率</th><th>宏 F1</th><th>矛盾召回</th></tr>{trs}</table>
<p>角色特征相对词法对照的宏 F1 增量为 {ci['delta']*100:.2f} 个百分点，按 101 个原始来源组配对重采样的 95% 区间为 [{lo*100:.2f}, {hi*100:.2f}] 个百分点。该区间条件于已训练模型，不包含训练变化与跨域不确定性，不能作为普遍优势的保证。</p>
<h2>数据独立性与统计能力</h2><p>将原句及来源描述的连通分量整体划分，训练/选择/校准/测试分别 {audit['rows']['train']} / {audit['rows']['selection']} / {audit['rows']['certificate']} / {audit['rows']['test']} 个方向样本；独立来源组为 356 / 84 / 60 / 101。这是自定义来源隔离划分，绝非官方 SICK 榜单结果。测试数据在拟合时未使用；来源描述仅用于分组，不作为模型输入。</p>
<p>协议要求宏 F1≥0.80、每类召回≥0.75、矛盾召回≥0.90，同时等义跳过需满足 1% 风险标准。关系成绩未达标。此外选择只有 84 个独立代表，达不到最少 100 次跳过；校准只有 60 组，即使全部零错，95% 单侧上界仍为 4.87%，无法认证 1%。这一可行性问题在拟合前记录于 feasibility_audit.json；因此不能将该校准结果描述为一次充分样本的安全性检验失败。要在零错误情况下达到该上界，至少需 299 个独立接受样本。</p>
<h2>另补独立 PAWS 验证</h2><p>为避免只在 SICK 报告改善，冻结模型后另用此前从未使用的 2,000 个 PAWS 句子组，每组一个样本，1,000 校准、1,000 测试。旧 PAWS 选择集只拟合阈值，不训练关系分类器。全部模型采用同一条件，不在新测试上调参。PAWS 仅支持等义二分类审计，不能据其负标签编造新增/矛盾金标；文章层相关仍无法排除。</p>
<table><tr><th>方法</th><th>等义 AUC</th><th>候选跳过</th><th>错误</th></tr>{ext}</table><p>主模型外部风险认证：{str(extcomplete['primary_certified'])}。零跳过不等于正确识别，也无成本收益。</p>
<h2>剩余错误（按预测置信度展示前 12 条）</h2><ol>{errs}</ol>
<h2>后续研究边界</h2><p>先补新增信息与矛盾召回，检查量词、属性修饰和多动作角色绑定；若开发继续使用本轮错误，必须另留新测试。未通过语义准入前不开展液态优势实验。通过后固定语义层，向所有历史模型提供相同关系分布、来源与时间间隔，在同一预算下比较无历史、指数统计、GRU、液态状态，并测试去掉时间、打乱顺序与重复来源的消融。</p>
<p>资料：<a href="https://zenodo.org/records/2787612">SICK 官方归档</a> · <a href="https://alt.qcri.org/semeval2014/task1/index.php?id=task-guidelines">任务定义</a> · <a href="https://spacy.io/models/en">句法模型</a>。SICK 为研究用途公开数据，遵守归档许可；预训练污染不能排除。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="metrics.json">完整分类指标</a> · <a href="predictions.jsonl">逐条预测</a> · <a href="parsed_roles.jsonl">角色抽取记录</a> · <a href="test_errors.json">全部测试错误</a> · <a href="analysis.json">配对区间</a> · <a href="timing.json">耗时</a> · <a href="external/metrics.json">独立 PAWS 结果</a></p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print(json.dumps(ci));print(json.dumps(exttest))
